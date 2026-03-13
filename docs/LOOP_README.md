# Loop Rollback Image Generation Pipeline

## Overview

This pipeline enables **iterative rollback-based image refinement** using the Infinity transformer model. The core idea is to regenerate tokens at specific scales multiple times (rollback), then merge the re-sampled tokens with the previously generated ones to improve image quality and consistency.

### Architecture

The Loop Rollback pipeline consists of:

1. **infinityLoopFloat.py** (`infinity/models/infinityLoopFloat.py`)
   - Extended Infinity transformer with rollback and merge logic
   - Saves per-scale state (tokens, accumulated features) for rollback
   - Supports multiple merge strategies for combining previous and re-generated tokens

2. **run_loop.py** (`tools/run_loop.py`)
   - CLI interface for the loop rollback workflow
   - Orchestrates base image generation and multi-scale rollback iterations
   - Handles output saving for each rollback configuration

3. **infer_loop.sh** (`scripts/infer_loop.sh`)
   - Bash wrapper with default parameters
   - Easy-to-customize entry point for users

## Usage

### Basic Loop Rollback Generation

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_loop.sh
```

Or with custom prompt and parameters:

```bash
python3 tools/run_loop.py \
  --prompt "Cute Shiba Inu wearing a space helmet." \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 \
  --tau 0.5 \
  --seed 1 \
  --save_file ./outputs/my_output/ \
  --rollback_merge_mode 6
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prompt` | str | `'a dog'` | Text prompt for image generation |
| `--rollback_merge_mode` | int | `-1` | Merge strategy for rollback tokens (see below) |
| `--rollback_schedule` | str (JSON) | `''` | Rollback rules as JSON list (see below) |
| `--cfg` | str | `'3'` | Classifier-free guidance scale (comma-separated for per-scale) |
| `--tau` | float | `1` | Temperature for sampling |
| `--seed` | int | `0` | Random seed for reproducibility |
| `--save_file` | str | `'./outputs'` | Output directory for generated images |
| `--pn` | str | Required | Parameter count: `'0.06M'`, `'0.25M'`, or `'1M'` |
| `--h_div_w_template` | float | `1.0` | Aspect ratio template (height / width) |

### Rollback Schedule Format

The `--rollback_schedule` parameter accepts a **JSON list of rule objects**. Each rule specifies:

| Field | Type | Description |
|-------|------|-------------|
| `scale` | int | The trigger scale — after generating this scale, rollback is triggered |
| `rollback_to` | int | The target scale to roll back to (must be < `scale`) |
| `times` | int | How many times to repeat the rollback-and-regenerate cycle |

**Single rule example:**
```bash
# Scale 4 → rollback to scale 2, repeat 3 times
--rollback_schedule '[{"scale":4,"rollback_to":2,"times":3}]'
```

This means: after generating scale 4, restore state to scale 2, then re-run scales 2→3→4. Repeat this 3 times.

**Multiple rules example:**
```bash
# Scale 4 → rollback to scale 2 (3 times), AND scale 6 → rollback to scale 4 (2 times)
--rollback_schedule '[{"scale":4,"rollback_to":2,"times":3},{"scale":6,"rollback_to":4,"times":2}]'
```

**Legacy format (backward compatible):**

The old dict format `{scale: times}` is still supported internally, where `rollback_to` defaults to `scale - 1`:
```python
# Old: rollback_schedule = {4: 1}
# Equivalent new: [{"scale": 4, "rollback_to": 3, "times": 1}]
```

### Merge Modes

The `--rollback_merge_mode` parameter controls how re-generated tokens are combined with the original tokens:

| Mode | Name | Formula | Description |
|------|------|---------|-------------|
| `-1` | Disabled | — | No merging; rollback is disabled |
| `0` | Replace | `down_feat` | Fully replace with new tokens |
| `1` | Average | `0.5 * (prev + new)` | Simple average of old and new |
| `2` | Geometric | `sqrt(clamp(prev * new, min=0))` | Geometric mean |
| `3` | CFG-weighted | `(1-cfg) * new + (1-cfg) * prev` | CFG-value weighted blend |
| `4` | Weighted variant 4 | Custom blend | Implementation-specific |
| `5` | Weighted variant 5 | Custom blend | Implementation-specific |
| `6` | Weighted variant 6 | Custom blend | Implementation-specific |

### Output

The pipeline generates:

- **output_base.jpg** — Base image (no rollback applied)
- **output_rollback_{desc}.jpg** — Rollback image with descriptive filename
  - e.g., `output_rollback_s4rb2x3.jpg` means: scale 4 → rollback to scale 2, 3 times

## Technical Details

### Rollback Mechanism

During autoregressive generation, the Infinity model produces tokens scale by scale. The rollback mechanism works as follows:

1. **State Saving:** At each scale that is a rollback target, the model saves its full state (accumulated features, tokens, KV cache length, etc.)
2. **Forward Pass:** Generate tokens normally for the current scale
3. **Rollback Decision:** If the current scale is a trigger in `rollback_schedule`, check if retries remain
4. **State Restoration:** Restore state to the `rollback_to` scale (which can be multiple scales back)
5. **KV Cache Truncation:** Truncate the KV cache to match the restored scale's cumulative token count
6. **Re-generation:** Re-run all scales from `rollback_to` up to the trigger scale
7. **Merge:** Combine the re-generated tokens with the original using the selected merge mode
8. **Repeat:** If more retries remain, repeat from step 4
9. **Continue:** Once all retries are exhausted, proceed to the next scale

```
Scale 0  →  Scale 1  →  Scale 2  →  Scale 3  →  Scale 4  → ... → Final Image
                            ↑                       │
                            └───────rollback─────────┘
                              restore state to scale 2
                              truncate KV cache
                              re-run scales 2→3→4
                              merge tokens
                              (repeat N times)
```

### Rollback Schedule

The rollback schedule is specified as a list of rule dicts:

```python
# Scale 4 → rollback to scale 2, repeat 3 times
rollback_schedule = [{"scale": 4, "rollback_to": 2, "times": 3}]

# Multiple rules: scale 4 → scale 2 (3x), scale 6 → scale 4 (2x)
rollback_schedule = [
    {"scale": 4, "rollback_to": 2, "times": 3},
    {"scale": 6, "rollback_to": 4, "times": 2},
]

# Legacy format still supported (rollback_to defaults to scale - 1):
rollback_schedule = {4: 1}  # equivalent to [{"scale": 4, "rollback_to": 3, "times": 1}]
```

From the CLI, pass as a JSON string:
```bash
--rollback_schedule '[{"scale":4,"rollback_to":2,"times":3}]'
```

### Scale Schedule

The scale schedule determines the spatial resolution at each generation step. It is derived from the dynamic resolution system:

```python
scale_schedule = dynamic_resolution_h_w[h_div_w_template][pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
```

Each scale `(t, h, w)` defines the patch count for that level.

### Token Merge in Detail

When rollback is enabled, re-generated indices are merged with original indices at the VAE level:

```python
# Pseudo-code for merge operation
prev_idx = scale_states[si]['indices']  # Original indices
cur_idx = newly_generated_indices       # Re-generated indices

if rollback_merge_mode == 0:
    merged = cur_idx                    # Full replacement
elif rollback_merge_mode == 1:
    merged = 0.5 * (prev_idx + cur_idx) # Average
elif rollback_merge_mode == 2:
    merged = sqrt(clamp(prev_idx * cur_idx, min=0))  # Geometric mean
# ... modes 3-6 use CFG-weighted variants
```

The merged indices are then converted to codes via the VAE quantizer and used for subsequent scales.

## Examples

### Example 1: Basic Rollback (scale 3 → scale 1)

```bash
python3 tools/run_loop.py \
  --prompt "A photograph of a cat sitting on a windowsill" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 \
  --seed 42 \
  --rollback_merge_mode 6 \
  --rollback_schedule '[{"scale":3,"rollback_to":1,"times":2}]' \
  --save_file ./outputs/cat_windowsill/
```

### Example 2: Multi-scale Rollback

```bash
python3 tools/run_loop.py \
  --prompt "An astronaut riding a horse on Mars, cinematic lighting" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 \
  --seed 1 \
  --rollback_merge_mode 1 \
  --rollback_schedule '[{"scale":4,"rollback_to":2,"times":3},{"scale":6,"rollback_to":4,"times":2}]' \
  --save_file ./outputs/astronaut/
```

### Example 3: Comparing Merge Modes with Deep Rollback

To compare different merge strategies with deep rollback (scale 5 → scale 2):

```bash
for mode in 0 1 2 3 4 5 6; do
  python3 tools/run_loop.py \
    --prompt "A beautiful sunset over the ocean" \
    --pn 1M \
    --model_path weights/infinity_2b_reg.pth \
    --vae_path weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --text_channels 2048 \
    --cfg 4 --tau 0.5 \
    --seed 1 \
    --rollback_merge_mode ${mode} \
    --rollback_schedule '[{"scale":5,"rollback_to":2,"times":3}]' \
    --save_file ./outputs/sunset_mode${mode}/
done
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** The rollback loop generates many images sequentially. GPU memory is cleared after each generation, but if issues persist:
- Reduce the number of scales or retries in the loop
- Use a smaller model (`--model_type infinity_2b`)
- Lower resolution (`--pn 0.25M`)

### Issue: Rollback has no visible effect

**Solution:** Ensure:
1. `--rollback_merge_mode` is set to a value ≥ 0 (not `-1`)
2. The scale schedule has enough scales for the rollback to operate on
3. Check the output images in `merge_*/` subdirectories

### Issue: All rollback outputs look identical

**Solution:** This can happen when the merge mode is too conservative. Try:
- `--rollback_merge_mode 0` (full replacement) for maximum variation
- Increasing `--tau` for more diverse sampling

## Performance Notes

- **Base Generation:** ~5–15 seconds per image (model-dependent)
- **Rollback Loop:** Default configuration generates 30 additional images (3 scales × 10 retries)
- **Memory:** GPU memory is explicitly freed after each generation via `torch.cuda.empty_cache()`
- **Disk:** Each image is ~50–200 KB (JPEG), full run produces ~30 MB

## Related Components

- **Batch Loop:** [BATCH_LOOP_README.md](BATCH_LOOP_README.md) — Sweep over many rollback schedules in one run (load model once)
- **P2P Editing:** [P2P_README.md](P2P_README.md) — Prompt-to-prompt token replacement
- **Rollback Details:** [ROLLBACK_README.md](../ROLLBACK_README.md) — Low-level rollback mechanism
- **Dynamic Resolution:** `infinity/utils/dynamic_resolution.py` — Scale schedule system
- **VAE Quantizer:** `infinity/models/bsq_vae/` — Binary spherical quantizer
- **Infinity Transformer:** `infinity/models/infinityLoopFloat.py` — Core model with rollback
