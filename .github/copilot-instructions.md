# Project Guidelines

## Overview

**Infinity** fork for training-free VAR image editing (ACM MM 2026 paper). The editing research is the active focus — base Infinity model code is upstream and rarely modified.

## Language

- User communicates in **Traditional Chinese (繁體中文)**. Respond in Chinese unless asked otherwise.
- Code comments and docstrings are primarily in Traditional Chinese.

## Environment

- **Conda env**: `Infinity` (activate before running anything)
- **Python**: 3.11, **GPU**: NVIDIA RTX 5090
- **Weights** (not in repo): expected at `weights/` — see `CLAUDE.md` for paths

## Key Commands

```bash
# Single-image editing (primary workflow)
bash scripts/infer_p2p_edit.sh

# Batch PIE-Bench evaluation (700 images)
bash scripts/batch_run_pie_edit.sh
bash scripts/eval_pie.sh              # compute metrics

# LaTeX paper
cd ACM_MM_2026_latex && latexmk -pdf main.tex
```

Full command reference: see `CLAUDE.md` § Key Commands.

## Architecture

Each editing method: **shell script** (params) → **`tools/`** (pipeline) → **`infinity/models/`** (model variant).

| Method | Script | Pipeline | Model |
|--------|--------|----------|-------|
| P2P-Edit (primary) | `scripts/infer_p2p_edit.sh` | `tools/run_p2p_edit.py` | `infinity/models/infinity_p2p_edit.py` |
| KV-Edit | `scripts/infer_kv_edit.sh` | `tools/run_kv_edit.py` | `infinity/utils/kv_cache_manager.py` |
| SelfAttn-Edit | `scripts/infer_selfAttn_edit.sh` | `tools/run_selfAttn_edit.py` | `infinity/models/infinity_selfAttn_edit.py` |

Detailed pipeline docs: `docs/P2P_EDIT_README.md`, `docs/KV_EDIT_README.md`, `docs/SELFATTN_EDIT_README.md`.

## Conventions

- **Do not modify** `infinity/models/basic.py`, `infinity/models/bsq_vae/`, or other upstream base model files unless explicitly asked.
- Each model variant (`infinity/models/infinity*.py`) defines its own `class Infinity(nn.Module)` with a custom `autoregressive_infer_cfg()`. The variant is selected by the import in the corresponding `tools/run_*.py`.
- Shell scripts are the user-facing entry points — keep them self-contained with all parameter defaults.
- Output images go under `outputs/`.

## Paper (ACM MM 2026)

- Location: `ACM_MM_2026_latex/` — `main.tex` includes sections from `tex/`, figures from `imgs/`, tables from `tabs/`, equations from `equations/`.
- Writing skills available: `.github/skills/how_to_write_conclusion.md`, `.github/skills/how_to_write_experiments.md`, `.claude/skills/ACM.md`.
- Paper format: ACM `sigconf`, 6–8 pages + 2 pages references only, double-blind anonymous.
- When editing LaTeX: keep captions concise, claims consistent across sections, and numbers matching the experiments.
