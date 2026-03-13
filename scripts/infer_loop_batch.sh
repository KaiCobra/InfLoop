#!/bin/bash

# Batch Loop Rollback — sweep over multiple rollback schedules
# Model weights are loaded ONCE; then every schedule in the JSON
# file is tested sequentially.
#
# Usage:
#   bash scripts/infer_loop_batch.sh
#   # 用預設的 schedules_batch_test.json
# 
# Simple Usage:
#     # 1. 先看看會產幾組
#     python scripts/gen_schedules.py --scale_min 3 --scale_max 8 --times 1 2 3 --multi_rule --dry_run
# 
#     # 2. 確認數量 OK，產出 JSON
#     python scripts/gen_schedules.py --scale_min 3 --scale_max 8 --times 1 2 3 --multi_rule -o scripts/schedules_sweep.json
# 
#     # 3. 跑批次推論（模型只 load 一次）
#     SCHEDULE_FILE=scripts/schedules_sweep.json bash scripts/infer_loop_batch.sh
#
# Adjust SCHEDULE_FILE below (or pass via env-var) to point at your
# own experiment list.

# ── Model / VAE / T5 arguments (same as infer_loop.sh) ──────────────────
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

# ── Loop Rollback parameters ────────────────────────────────────────────
rollback_merge_mode=5            # Merge strategy
seed=1                           # Random seed

# ── Schedule file: each entry is one experiment ─────────────────────────
SCHEDULE_FILE="${SCHEDULE_FILE:-scripts/schedules_batch_test.json}"

# ── Prompt & output ─────────────────────────────────────────────────────
prompt="A real photo of a Cute Shiba Inu wearing a space helmet and holding a sign says \"excellent space\"."
save_dir="./outputs/outputs_loop_exp/batch_rollback_test/"

# ── Run ─────────────────────────────────────────────────────────────────
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
  --prompt "${prompt}" \
  --seed ${seed} \
  --save_file "${save_dir}" \
  --rollback_merge_mode ${rollback_merge_mode} \
  --schedule_file "${SCHEDULE_FILE}"
