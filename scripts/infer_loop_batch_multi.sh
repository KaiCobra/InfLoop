#!/bin/bash

# Multi-Prompt Batch Loop Rollback
# ─────────────────────────────────────────────────────────────────────
# Reads prompts from a JSONL file, runs every rollback schedule for
# each prompt, and generates multiple images per setting (different seeds).
#
# Model weights are loaded ONCE for the entire run.
#
# Output structure:
#   save_dir/
#   ├── prompt_001_OPEN/
#   │   ├── prompt_info.json        ← prompt text + render targets
#   │   ├── base/
#   │   │   ├── seed_1.jpg ... seed_5.jpg
#   │   ├── s4rb2x3/
#   │   │   ├── seed_1.jpg ... seed_5.jpg
#   │   └── ...
#   ├── prompt_002_LATTE/
#   │   └── ...
#   └── experiment_summary.json     ← full run metadata
#
# Usage:
#   # 用預設的 prompts.jsonl + schedules_batch_test.json
#   bash scripts/infer_loop_batch_multi.sh
#
#   # 指定自己的 prompt 和 schedule 檔案:
#   PROMPT_FILE=exp_prompt/my_prompts.jsonl \
#   SCHEDULE_FILE=scripts/schedules_batch_test.json \
#   bash scripts/infer_loop_batch_multi.sh

# ── Model / VAE / T5 arguments ─────────────────────────────────────
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001
text_channels=2048
apply_spatial_patchify=0

# ── Loop Rollback parameters ───────────────────────────────────────
rollback_merge_mode=6            # Merge strategy
seed=1                           # Base seed (images use seed, seed+1, ..., seed+N-1)
num_images=5                     # Images per prompt×setting

# ── Input files ────────────────────────────────────────────────────
PROMPT_FILE="${PROMPT_FILE:-exp_prompt/prompts.jsonl}"
SCHEDULE_FILE="${SCHEDULE_FILE:-scripts/schedules_batch_test.json}"

# ── Output directory ───────────────────────────────────────────────
save_dir="./outputs/outputs_loop_exp/multi_prompt_test/"

# ── Run ────────────────────────────────────────────────────────────
python3 tools/run_loop_batch.py \
  --cfg ${cfg} \
  --tau ${tau} \
  --pn ${pn} \
  --model_path ${infinity_model_path} \
  --vae_type ${vae_type} \
  --vae_path ${vae_path} \
  --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
  --use_bit_label ${use_bit_label} \
  --model_type ${model_type} \
  --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
  --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
  --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
  --checkpoint_type ${checkpoint_type} \
  --text_encoder_ckpt ${text_encoder_ckpt} \
  --text_channels ${text_channels} \
  --apply_spatial_patchify ${apply_spatial_patchify} \
  --seed ${seed} \
  --save_file "${save_dir}" \
  --rollback_merge_mode ${rollback_merge_mode} \
  --schedule_file "${SCHEDULE_FILE}" \
  --prompt_file "${PROMPT_FILE}" \
  --num_images ${num_images}
