# Batch Loop Rollback — Full Documentation

## Overview

The **Batch Loop Rollback** pipeline automates large-scale rollback experiments. It loads model weights **once**, then sweeps through an arbitrary number of rollback schedule configurations — generating one image per experiment — without reloading the model.

This is essential for systematic ablation studies: comparing different rollback depths, retry counts, multi-rule combinations, and merge modes.

### Architecture

```
InfLoop/
├── scripts/
│   ├── gen_schedules.py             # Generator: produce experiment configs
│   ├── schedules_batch_test.json    # Example: 14 hand-picked experiments
│   ├── schedules_sweep.json         # Example: auto-generated sweep (186 exps)
│   └── infer_loop_batch.sh          # Bash wrapper: launch batch inference
├── tools/
│   └── run_loop_batch.py            # Python runner: load once, infer many
└── docs/
    └── BATCH_LOOP_README.md         # This file
```

### Relationship to Other Pipelines

| Pipeline | Script | Python | Purpose |
|----------|--------|--------|---------|
| Single Loop | `infer_loop.sh` | `run_loop.py` | One rollback schedule, one image |
| **Batch Loop** | `infer_loop_batch.sh` | `run_loop_batch.py` | Many schedules, many images, load once |
| P2P | `infer_p2p.sh` | `run_p2p.py` | Prompt-to-prompt editing |

---

## Quick Start

### 1. Run with the hand-picked test set (14 experiments)

```bash
cd /home/avlab/Documents/InfLoop
bash scripts/infer_loop_batch.sh
```

This uses `scripts/schedules_batch_test.json` by default and saves results to `./outputs/batch_rollback_test/`.

### 2. Generate a larger sweep and run it

```bash
# Step 1: Generate schedule config
python scripts/gen_schedules.py \
  --scale_min 3 --scale_max 8 \
  --gap_min 1 --gap_max 2 \
  --times 1 2 3 \
  --multi_rule \
  -o scripts/schedules_sweep.json

# Step 2: Run batch inference
SCHEDULE_FILE=scripts/schedules_sweep.json bash scripts/infer_loop_batch.sh
```

### 3. Use a custom schedule file

```bash
SCHEDULE_FILE=scripts/my_custom_schedules.json bash scripts/infer_loop_batch.sh
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  Batch Loop Rollback Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────┐
  │  gen_schedules.py        │   ← (optional) auto-generate configs
  │  --scale_min 3           │
  │  --scale_max 8           │
  │  --times 1 2 3           │
  │  --multi_rule            │
  └───────────┬──────────────┘
              │ writes
              ▼
  ┌──────────────────────────┐
  │  schedules_sweep.json    │   ← JSON experiment list
  │  [                       │
  │    {"name":"s3rb2x1",    │
  │     "rules":[...]},      │
  │    {"name":"s4rb2x3",    │
  │     "rules":[...]},      │
  │    ...                   │
  │  ]                       │
  └───────────┬──────────────┘
              │ reads
              ▼
  ┌──────────────────────────────────────────────────────┐
  │  run_loop_batch.py                                   │
  │                                                      │
  │  ┌────────────────────────────────────────────────┐  │
  │  │  LOAD ONCE                                     │  │
  │  │  ├─ Text encoder (T5)                          │  │
  │  │  ├─ VAE (BSQ)                                  │  │
  │  │  └─ Infinity transformer                       │  │
  │  └────────────────────────────────────────────────┘  │
  │                      │                               │
  │  ┌───────────────────▼────────────────────────────┐  │
  │  │  GENERATE BASE IMAGE (optional, --skip_base)   │  │
  │  │  → output_base.jpg                             │  │
  │  └────────────────────────────────────────────────┘  │
  │                      │                               │
  │  ┌───────────────────▼────────────────────────────┐  │
  │  │  FOR EACH EXPERIMENT:                          │  │
  │  │    Experiment 1: s3rb2x1                       │  │
  │  │    → output_rollback_s3rb2x1.jpg               │  │
  │  │                                                │  │
  │  │    Experiment 2: s4rb2x3                       │  │
  │  │    → output_rollback_s4rb2x3.jpg               │  │
  │  │                                                │  │
  │  │    Experiment 3: s4rb2_s6rb4_x2                │  │
  │  │    → output_rollback_s4rb2_s6rb4_x2.jpg        │  │
  │  │    ...                                         │  │
  │  │  (GPU memory cleared after each image)         │  │
  │  └────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────┘
```

---

## File Reference

### 1. `scripts/gen_schedules.py` — Experiment Generator

Automatically generates a JSON schedule file by sweeping over parameter ranges.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scale_min` | `2` | Minimum trigger scale index |
| `--scale_max` | `10` | Maximum trigger scale index (1M has 13 scales, index 0–12) |
| `--gap_min` | `1` | Minimum rollback gap (`scale − rollback_to`) |
| `--gap_max` | `3` | Maximum rollback gap |
| `--times` | `1 2 3` | List of retry counts to sweep |
| `--multi_rule` | off | Also generate dual-rule combinations |
| `--triple_rule` | off | Also generate triple-rule combinations (warning: can be very large) |
| `-o` | `scripts/schedules_sweep.json` | Output file path |
| `--dry_run` | off | Print count only, do not write file |

#### Experiment Types

| Type | Description | Example |
|------|-------------|---------|
| **Single-rule** | One rollback rule | `s4rb2x3` — scale 4 → rollback to 2, 3 times |
| **Dual-rule** | Two non-overlapping rules | `s3rb2_s6rb4_x2` — two rollbacks in one run |
| **Triple-rule** | Three non-overlapping rules | `s3rb2_s5rb4_s8rb6_x1` |

Non-overlapping constraint: rule A's trigger scale ≤ rule B's rollback target, so the rollback regions don't interfere with each other.

#### Usage Examples

```bash
# Dry run: see how many experiments
python scripts/gen_schedules.py --dry_run
# → Single-rule experiments: 78

# Small focused sweep
python scripts/gen_schedules.py \
  --scale_min 3 --scale_max 6 \
  --gap_min 1 --gap_max 1 \
  --times 1 3 5 \
  -o scripts/schedules_small.json
# → 12 experiments

# Medium sweep with dual-rule combos
python scripts/gen_schedules.py \
  --scale_min 3 --scale_max 8 \
  --gap_min 1 --gap_max 2 \
  --times 1 2 3 \
  --multi_rule \
  -o scripts/schedules_sweep.json
# → 186 experiments (36 single + 150 dual)

# Full sweep (use with caution!)
python scripts/gen_schedules.py \
  --scale_min 2 --scale_max 10 \
  --gap_min 1 --gap_max 3 \
  --times 1 2 3 \
  --multi_rule --triple_rule \
  -o scripts/schedules_full.json
```

### 2. Schedule JSON Format

The JSON file is a list of experiment objects:

```json
[
  {
    "name": "s4rb2x3",
    "rules": [{"scale": 4, "rollback_to": 2, "times": 3}]
  },
  {
    "name": "s4rb2_s6rb4_x2",
    "rules": [
      {"scale": 4, "rollback_to": 2, "times": 2},
      {"scale": 6, "rollback_to": 4, "times": 2}
    ]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Human-readable label; used in output filename |
| `rules` | list | List of rollback rule objects |
| `rules[].scale` | int | Trigger scale index |
| `rules[].rollback_to` | int | Target scale to roll back to |
| `rules[].times` | int | Number of rollback-merge cycles |

You can also write schedule files by hand — just follow this format.

### 3. `tools/run_loop_batch.py` — Batch Runner

Loads all models once, then iterates over every experiment in the schedule.

#### Additional CLI Parameters (beyond standard model args)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--schedule_file` | str | `''` | Path to a JSON schedule file |
| `--schedules` | str | `''` | Inline JSON string (alternative to file) |
| `--skip_base` | flag | off | Skip generating the base (no-rollback) image |
| `--prompt` | str | `'a dog'` | Text prompt |
| `--save_file` | str | `'./outputs'` | Output directory |

#### Input Sources (priority order)

1. `--schedule_file` — path to a JSON file (preferred for large experiments)
2. `--schedules` — inline JSON string (convenient for small tests)

#### Output Files

```
outputs/batch_rollback_test/
├── output_base.jpg                        ← Base image (no rollback)
├── output_rollback_s3rb1x1.jpg            ← Single-rule result
├── output_rollback_s4rb2x3.jpg
├── output_rollback_s4rb2_s6rb4_x2.jpg     ← Dual-rule result
└── ...
```

Filenames are derived from the experiment `name` field.

### 4. `scripts/infer_loop_batch.sh` — Bash Wrapper

Pre-configured shell script. Key variables to customize:

```bash
# Which schedule file to use
SCHEDULE_FILE="${SCHEDULE_FILE:-scripts/schedules_batch_test.json}"

# Prompt
prompt="Cute Shiba Inu wearing a space helmet."

# Output directory
save_dir="./outputs/batch_rollback_test/"

# Merge strategy (applied to all experiments)
rollback_merge_mode=6

# Random seed
seed=1
```

Override the schedule file via environment variable:
```bash
SCHEDULE_FILE=scripts/schedules_sweep.json bash scripts/infer_loop_batch.sh
```

---

## Pre-built Schedule Files

### `schedules_batch_test.json` — Hand-picked (14 experiments)

Curated set covering representative rollback patterns:

| # | Name | Rules | What it tests |
|---|------|-------|---------------|
| 1–2 | `s3rb1x1`, `s3rb1x3` | Scale 3 → 1 | Early rollback, varying times |
| 3–4 | `s4rb2x1`, `s4rb2x3` | Scale 4 → 2 | Mid rollback, varying times |
| 5–6 | `s5rb3x1`, `s5rb3x3` | Scale 5 → 3 | Deeper rollback |
| 7–8 | `s6rb4x1`, `s6rb4x3` | Scale 6 → 4 | High-res rollback |
| 9 | `s4rb1x2` | Scale 4 → 1 | Large gap (3 scales) |
| 10 | `s5rb2x2` | Scale 5 → 2 | Large gap (3 scales) |
| 11 | `s6rb3x2` | Scale 6 → 3 | Large gap (3 scales) |
| 12 | `s4rb2_s6rb4_x2` | Scale 4→2 + Scale 6→4 | Dual-rule combo |
| 13 | `s3rb1_s5rb3_x2` | Scale 3→1 + Scale 5→3 | Dual-rule, earlier scales |
| 14 | `s4rb2_s6rb4_s8rb6_x1` | 3 rules | Triple-rule cascade |

### `schedules_sweep.json` — Auto-generated (186 experiments)

Generated with:
```bash
python scripts/gen_schedules.py \
  --scale_min 3 --scale_max 8 \
  --gap_min 1 --gap_max 2 \
  --times 1 2 3 \
  --multi_rule \
  -o scripts/schedules_sweep.json
```

Breakdown: 36 single-rule + 150 dual-rule experiments.

---

## Scale Index Reference

For `pn=1M`, `h_div_w_template=1.0` (square), the scale schedule has **13 scales** (index 0–12):

| Index | t | h × w | Resolution (×16) | Tokens (h×w) |
|-------|---|-------|-------------------|--------------|
| 0 | 1 | 1 × 1 | 16 × 16 | 1 |
| 1 | 2 | 2 × 2 | 32 × 32 | 4 |
| 2 | 3 | 4 × 4 | 64 × 64 | 16 |
| 3 | 4 | 6 × 6 | 96 × 96 | 36 |
| 4 | 5 | 8 × 8 | 128 × 128 | 64 |
| 5 | 6 | 12 × 12 | 192 × 192 | 144 |
| 6 | 7 | 16 × 16 | 256 × 256 | 256 |
| 7 | 9 | 20 × 20 | 320 × 320 | 400 |
| 8 | 11 | 24 × 24 | 384 × 384 | 576 |
| 9 | 13 | 32 × 32 | 512 × 512 | 1024 |
| 10 | 15 | 40 × 40 | 640 × 640 | 1600 |
| 11 | 17 | 48 × 48 | 768 × 768 | 2304 |
| 12 | 21 | 64 × 64 | 1024 × 1024 | 4096 |

**Interpretation:**
- Rollback at **low scales (0–4)** affects the overall composition and structure
- Rollback at **mid scales (5–8)** adjusts medium-frequency features (shapes, textures)
- Rollback at **high scales (9–12)** refines fine details

---

## Advanced Usage

### Custom schedule file by hand

Create `scripts/my_experiments.json`:

```json
[
  {
    "name": "early_rollback",
    "rules": [{"scale": 3, "rollback_to": 0, "times": 5}]
  },
  {
    "name": "late_refine",
    "rules": [{"scale": 9, "rollback_to": 7, "times": 2}]
  },
  {
    "name": "cascade_3stage",
    "rules": [
      {"scale": 3, "rollback_to": 1, "times": 2},
      {"scale": 6, "rollback_to": 4, "times": 2},
      {"scale": 9, "rollback_to": 7, "times": 1}
    ]
  }
]
```

Run:
```bash
SCHEDULE_FILE=scripts/my_experiments.json bash scripts/infer_loop_batch.sh
```

### Inline schedule (no file needed)

```bash
python3 tools/run_loop_batch.py \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_path weights/infinity_vae_d32reg.pth \
  --vae_type 32 \
  --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
  --text_channels 2048 \
  --cfg 4 --tau 0.5 --seed 1 \
  --rollback_merge_mode 6 \
  --prompt "A cat on a rooftop at sunset" \
  --save_file ./outputs/inline_test/ \
  --schedules '[{"name":"test1","rules":[{"scale":4,"rollback_to":2,"times":2}]},{"name":"test2","rules":[{"scale":6,"rollback_to":3,"times":1}]}]'
```

### Skip base image

If you already have the base image or only care about rollback results:

```bash
python3 tools/run_loop_batch.py \
  --skip_base \
  --schedule_file scripts/schedules_sweep.json \
  ... (other args)
```

### Sweep merge modes

To compare merge modes, run the same schedule file with different `rollback_merge_mode`:

```bash
for mode in 0 1 2 3 4 5 6; do
  python3 tools/run_loop_batch.py \
    --pn 1M \
    --model_path weights/infinity_2b_reg.pth \
    --vae_path weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --text_channels 2048 \
    --cfg 4 --tau 0.5 --seed 1 \
    --rollback_merge_mode ${mode} \
    --prompt "A futuristic cityscape at night" \
    --save_file ./outputs/merge_mode_${mode}/ \
    --schedule_file scripts/schedules_batch_test.json \
    --skip_base
done
```

Note: each `run_loop_batch.py` invocation loads the model once, so 7 merge modes = 7 model loads. This is unavoidable since merge mode is a global setting.

### Multi-prompt batch

Use `exp_prompt/prompts.jsonl` with a shell loop — each prompt gets its own output folder, but the model reloads per prompt:

```bash
while IFS= read -r line; do
  prompt=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt'])")
  # Create a safe directory name from the prompt
  dir_name=$(echo "$prompt" | tr ' ' '_' | tr -cd '[:alnum:]_' | head -c 50)
  
  python3 tools/run_loop_batch.py \
    --pn 1M \
    --model_path weights/infinity_2b_reg.pth \
    --vae_path weights/infinity_vae_d32reg.pth \
    --vae_type 32 \
    --text_encoder_ckpt weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --text_channels 2048 \
    --cfg 4 --tau 0.5 --seed 1 \
    --rollback_merge_mode 6 \
    --prompt "$prompt" \
    --save_file "./outputs/multi_prompt/${dir_name}/" \
    --schedule_file scripts/schedules_batch_test.json
done < exp_prompt/prompts.jsonl
```

---

## Performance & Resource Estimates

### Timing (RTX 4090, 2B model, 1M resolution)

| Operation | Time |
|-----------|------|
| Model loading (T5 + VAE + Infinity) | ~30–60s |
| Base image generation | ~10–15s |
| Per rollback experiment (varies by depth) | ~10–30s |
| 14 experiments (schedules_batch_test.json) | ~5–8 min |
| 186 experiments (schedules_sweep.json) | ~1–2 hours |

### Disk Usage

| Item | Size |
|------|------|
| Per image (JPEG) | ~50–200 KB |
| 14 experiments + base | ~1–3 MB |
| 186 experiments + base | ~10–40 MB |

### GPU Memory

- Model footprint: ~6–8 GB (2B model in bf16)
- Per-image inference: ~2–4 GB additional
- Memory is freed (`torch.cuda.empty_cache()`) after each image
- Peak usage: ~10–12 GB

---

## Troubleshooting

### "CUDA out of memory" during batch run

The model stays in GPU memory throughout. If individual experiments OOM:
- Reduce `--pn` (e.g., `0.25M` instead of `1M`)
- Use smaller rollback gaps (fewer intermediate scales to re-run)
- Reduce `times` in the schedule

### Batch run interrupted — how to resume?

Currently there is no built-in resume. Workaround:
1. Check which images were already generated in the output directory
2. Edit the schedule JSON to remove completed experiments
3. Re-run with `--skip_base`

### Output images all look the same

- Ensure `--rollback_merge_mode` ≥ 0 (not `-1`)
- Try different merge modes: `0` (replace) gives maximum variation
- Increase `--tau` for more diverse sampling
- The same `--seed` produces the same base tokens — rollback introduces variation through re-sampling

### gen_schedules.py produces too many experiments

Use `--dry_run` first to check the count, then narrow the ranges:
```bash
# Check count
python scripts/gen_schedules.py --scale_min 4 --scale_max 6 --times 1 3 --dry_run

# If OK, generate
python scripts/gen_schedules.py --scale_min 4 --scale_max 6 --times 1 3 -o scripts/schedules_small.json
```

---

## Related Documentation

- [LOOP_README.md](LOOP_README.md) — Single-run loop rollback (format details, merge modes, technical internals)
- [LOOP_QUICKSTART.md](LOOP_QUICKSTART.md) — Quick start for single-run loop
- [P2P_README.md](P2P_README.md) — Prompt-to-prompt editing pipeline
- [ROLLBACK_README.md](../ROLLBACK_README.md) — Low-level rollback mechanism
