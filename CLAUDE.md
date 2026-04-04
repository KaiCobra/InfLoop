# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Infinity** is a Visual AutoRegressive (VAR) text-to-image model using bitwise token prediction. This fork extends it with **training-free image editing** methods for an ACM MM 2026 paper. The editing research is the active development focus — the base Infinity model code is upstream and rarely modified.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation

# Training (rarely used in this fork)
SINGLE=1 bash scripts/train.sh          # single GPU
torchrun --nproc_per_node=8 train.py     # multi GPU

# Single-image editing inference
bash scripts/infer_p2p_edit.sh           # P2P-Edit (attention-masked editing)
bash scripts/infer_kv_edit.sh            # KV-Edit (KV cache editing)
bash scripts/infer_selfAttn_edit.sh      # Self-attention editing

# Batch PIE-Bench evaluation
bash scripts/batch_run_pie_edit.sh       # P2P-Edit on full PIE-Bench (700 images)
bash scripts/batch_run_kv_edit.sh        # KV-Edit on PIE-Bench
bash scripts/eval_pie.sh                 # Compute metrics on results

# Attention visualization
python -m attention_map.run --input exp_prompts/prompts.jsonl --output outputs/attention_maps

# LaTeX paper
cd ACM_MM_2026_latex && latexmk -pdf main.tex
```

## Architecture

### Editing Pipeline (the active research)

Each editing method follows the same pattern: **shell script** (parameters) → **tools/** (pipeline logic) → **infinity/models/** (model variant).

| Method | Script | Pipeline | Model |
|--------|--------|----------|-------|
| P2P-Edit | `scripts/infer_p2p_edit.sh` | `tools/run_p2p_edit.py` | `infinity/models/infinity_p2p_edit.py` |
| KV-Edit | `scripts/infer_kv_edit.sh` | `tools/run_kv_edit.py` | `infinity/utils/kv_cache_manager.py` |
| SelfAttn-Edit | `scripts/infer_selfAttn_edit.sh` | `tools/run_selfAttn_edit.py` | `infinity/models/infinity_selfAttn_edit.py` |
| Batch PIE | `scripts/batch_run_pie_edit.sh` | `tools/batch_run_pie_edit.py` → `tools/run_pie_edit.py` | (uses P2P-Edit model) |

**P2P-Edit pipeline** (`tools/run_p2p_edit.py`) — the primary method:
1. **Phase 1**: Generate source image, capture cross-attention maps and bitwise tokens
2. **Phase 1.5**: Build attention masks — IQR-filter across transformer blocks, then threshold to produce focus/preserve binary masks
3. **Phase 2**: Generate target image, replacing source tokens in background (low-attention) regions while letting edit regions (high-attention) generate freely

**Attention extraction** (`attention_map/extractor.py`): `CrossAttentionExtractor` hooks into transformer self/cross-attention layers to capture per-word, per-scale, per-block attention maps. Used by all editing pipelines.

### Dynamic Threshold (Binary/Ternary Search)

When a reference mask is available (e.g., PIE-Bench GT masks from `mapping_file.json`), the system uses ternary search to find the attention percentile threshold that maximizes IoU between the candidate attention mask and the reference mask. Implementation: `compute_attention_mask_dynamic_threshold()` in `tools/run_p2p_edit.py`. Falls back to fixed percentile when no reference mask is provided.

### Model Variants

Each `infinity/models/infinity*.py` file defines its own `class Infinity(nn.Module)` with a different `autoregressive_infer_cfg()` method. The variant is selected by the import in the corresponding `tools/run_*.py` script. Key differences:
- `infinity.py` — base generation (text-to-image)
- `infinity_p2p_edit.py` — adds source token injection + attention mask hooks
- `infinity_selfAttn_edit.py` — manipulates self-attention KV cache
- `infinityInject.py` — image injection variant

### Base Model (upstream, stable)

- `infinity/models/bsq_vae/` — Bitwise Scalar Quantized VAE tokenizer (d=16/24/32/64)
- `infinity/models/basic.py` — core transformer blocks (AdaLN, cross-attention, self-attention, MLP)
- `infinity/models/t5.py` — Flan-T5-XL text encoder wrapper
- `infinity/utils/` — dynamic resolution, checkpointing, distributed training, bitwise token storage

### Evaluation

- `tools/eval_pie_results.py` — computes PIE-Bench metrics (Structure Distance, PSNR, SSIM, LPIPS, CLIP whole/edited) against `mapping_file.json` ground truth
- PIE-Bench masks use RLE format: `(start, count)` pairs, row-major order, 512x512 resolution
- `tools/batch_run_pie_edit.py` reads `mapping_file.json` to pass GT masks to the dynamic threshold search

## Weights (not in repo)

Expected at `weights/`:
- `infinity_2b_reg.pth` — Infinity 2B model
- `infinity_vae_d32reg.pth` — BSQ-VAE (d=32)
- `models--google--flan-t5-xl/` — text encoder

## Paper

`ACM_MM_2026_latex/` — ACM MM 2026 submission. Main file: `main.tex`, sections in `tex/`, figures in `figs/`, tables in `tabs/`, algorithms in `equations/`. The paper describes the P2P-Edit method implemented in this codebase.

## Language

Code comments and docstrings are primarily in Traditional Chinese (繁體中文). The user communicates in Chinese.
