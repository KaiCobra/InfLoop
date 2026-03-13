# Loop Rollback Pipeline — Quick Start Guide

## Files

```
InfLoop/
├── infinity/
│   └── models/
│       └── infinityLoopFloat.py       # Core: Infinity model with rollback + merge
├── tools/
│   └── run_loop.py                    # CLI: Loop rollback pipeline entry point
├── scripts/
│   └── infer_loop.sh                  # Script: Ready-to-run inference wrapper
└── docs/
    ├── LOOP_README.md                 # Full documentation
    └── LOOP_QUICKSTART.md             # This file
```

## 30-Second Start

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_loop.sh
```

This will:
1. Generate a base image from the prompt ("Cute Shiba Inu wearing a space helmet.")
2. Apply rollback: scale 4 → rollback to scale 2, repeated 3 times
3. Merge re-generated tokens using mode 6
4. Save all images to `./outputs/Shiba_rollback/`

## What Each File Does

### 1. infinityLoopFloat.py (`infinity/models/infinityLoopFloat.py`)
- **Purpose:** Infinity transformer extended with rollback and merge logic
- **Key Method:** `autoregressive_infer_cfg()`
  - `rollback_schedule` — List of rule dicts specifying rollback behavior
  - `rollback_merge_mode` — How to combine old and re-generated tokens
- **New format:**
  ```python
  rollback_schedule = [
      {"scale": 4, "rollback_to": 2, "times": 3},  # scale 4 → back to scale 2, 3 times
      {"scale": 6, "rollback_to": 4, "times": 2},  # scale 6 → back to scale 4, 2 times
  ]
  ```
- **Mechanism:**
  1. Save state at each scale that is a rollback target
  2. After generating the trigger scale, restore state to the target scale
  3. Truncate KV cache to match the target scale's token count
  4. Re-run all scales from target to trigger, then merge
  5. Repeat for the specified number of times

### 2. run_loop.py (`tools/run_loop.py`)
- **Purpose:** CLI orchestrator for the loop rollback workflow
- **Workflow:**
  - Parse arguments
  - Load models (text encoder, VAE, Infinity)
  - Generate base image (no rollback) → `output1.jpg`
  - Loop over scales (1–3) × retries (1–10) → `merge_{mode}/output_{scale}_{retry}.jpg`
  - Clean up GPU memory after each generation

### 3. infer_loop.sh (`scripts/infer_loop.sh`)
- **Purpose:** Easy-to-modify bash wrapper
- **Key Parameters:**
  ```bash
  prompt="Cute Shiba Inu wearing a space helmet."
  rollback_merge_mode=6    # Merge strategy
  rollback_schedule='[{"scale":4,"rollback_to":2,"times":3}]'
  seed=1                   # Reproducibility
  save_dir="./outputs/Shiba_rollback/"
  ```

## Workflow Diagram

```
┌───────────────────────────────────────────────────────────┐
│             Loop Rollback Pipeline                        │
└───────────────────────────────────────────────────────────┘

┌─ STEP 1: BASE GENERATION ────────┐
│                                   │
│  Prompt → Infinity Model          │
│  Scale 0 → Scale 1 → ... → N     │
│       ↓                           │
│  Base Image (output1.jpg)         │
│                                   │
└───────────────────────────────────┘

┌─ STEP 2: ROLLBACK GENERATION ─────┐
│                                   │
│  With rollback_schedule:          │
│  [{"scale":4,"rollback_to":2,     │
│    "times":3}]                    │
│                                   │
│  Infinity Model                   │
│  Scale 0 → 1 → 2 → 3 → 4:       │
│    ┌────────────────────┐         │
│    │  Save state @ s=2  │         │
│    │  Generate s=2,3,4  │         │
│    │  Rollback to s=2   │──┐      │
│    │  Truncate KV cache │  │      │
│    │  Re-run s=2→3→4    │  │      │
│    │  Merge (mode M)    │←─┘      │
│    │  Repeat 3 times    │         │
│    └────────────────────┘         │
│  → Scale 5 → ... → N             │
│       ↓                           │
│  output_rollback_s4rb2x3.jpg      │
│                                   │
└───────────────────────────────────┘

Output Structure:
  outputs/
  ├── output_base.jpg                    ← Base image
  └── output_rollback_s4rb2x3.jpg        ← Rollback result
```

## Key Concepts

### Rollback
- **What:** Re-generate tokens from a target scale up to a trigger scale by restoring state
- **Why:** Different random samples across multiple scales produce variations; merging improves quality
- **How:** State is saved at rollback target scales; on trigger, state + KV cache are restored and scales re-run
- **Deep rollback:** Can jump back multiple scales (e.g., scale 6 → scale 2 re-runs 4 scales)

### Merge Modes
| Mode | Strategy | Best For |
|------|----------|----------|
| `0` | Full replacement | Maximum variation |
| `1` | Simple average | Smooth blending |
| `2` | Geometric mean | Balanced combination |
| `3–6` | Weighted variants | Fine-grained control |

### Scale Schedule
- The generation process is split into multiple scales (resolutions)
- Lower scales = coarse structure, higher scales = fine details
- Rollback at lower scales changes overall composition
- Rollback at higher scales refines local details

## Common Commands

### Quick Test
```bash
bash scripts/infer_loop.sh
```

### Custom Prompt
```bash
python3 tools/run_loop.py \
  --prompt "A golden retriever playing in autumn leaves" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 --seed 1 \
  --rollback_merge_mode 6 \
  --rollback_schedule '[{"scale":4,"rollback_to":2,"times":3}]' \
  --save_file ./outputs/golden_retriever/
```

### Deep Rollback (scale 6 → scale 2)
```bash
python3 tools/run_loop.py \
  --prompt "A cat sitting on a rooftop at sunset" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 --seed 1 \
  --rollback_merge_mode 6 \
  --rollback_schedule '[{"scale":6,"rollback_to":2,"times":2}]' \
  --save_file ./outputs/cat_rooftop/
```

### Multi-Rule Rollback
```bash
python3 tools/run_loop.py \
  --prompt "A futuristic cityscape at night" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 --seed 1 \
  --rollback_merge_mode 6 \
  --rollback_schedule '[{"scale":3,"rollback_to":1,"times":2},{"scale":6,"rollback_to":3,"times":3}]' \
  --save_file ./outputs/cityscape/
```

### No Rollback (Base Only)
```bash
python3 tools/run_loop.py \
  --prompt "A cat sitting on a rooftop" \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 --seed 1 \
  --save_file ./outputs/cat_rooftop_base/
```

### Compare All Merge Modes
```bash
for mode in 0 1 2 3 4 5 6; do
  python3 tools/run_loop.py \
    --prompt "A futuristic cityscape at night" \
    --pn 1M \
    --model_path weights/infinity_2b_reg.pth \
    --vae_path weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --text_channels 2048 \
    --cfg 4 --tau 0.5 --seed 1 \
    --rollback_merge_mode ${mode} \
    --rollback_schedule '[{"scale":4,"rollback_to":2,"times":3}]' \
    --save_file ./outputs/cityscape_mode${mode}/
done
```

## File Compatibility

| Component | Based On | Changes |
|-----------|----------|---------|
| `infinityLoopFloat.py` | `infinityFloat.py` | + Rollback state saving, merge logic |
| `run_loop.py` | `run_infinity.py` | + Multi-scale rollback loop, merge mode arg |
| `infer_loop.sh` | `infer.sh` | + `rollback_merge_mode` parameter |

## Validation

All files are:
- ✅ Syntactically correct
- ✅ Backward compatible (no breaking changes to existing code)
- ✅ Well-documented (docstrings and comments)

## Next Steps

1. **Test:** Run `bash scripts/infer_loop.sh` to verify basic functionality
2. **Customize:** Edit prompt and parameters in `scripts/infer_loop.sh`
3. **Experiment:** Try different `--rollback_merge_mode` values (0–6) to compare merge strategies
4. **Batch Sweep:** Use `bash scripts/infer_loop_batch.sh` to run many rollback schedules at once — see [BATCH_LOOP_README.md](BATCH_LOOP_README.md)
5. **Analyze:** Compare `output_base.jpg` with rollback outputs

## Troubleshooting

See [LOOP_README.md](LOOP_README.md) for detailed troubleshooting guide.
