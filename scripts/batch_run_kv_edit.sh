#!/bin/bash
# ==============================================================================
# batch_run_kv_edit.sh
#
# 目的：批量跑 extracted_pie_bench，使用 KV-Edit 交錯式管線。
#       模型只載入一次，每個 case 逐 scale 交錯處理三個 phase。
#
# 執行：bash scripts/batch_run_kv_edit.sh
# ==============================================================================

# ── 模型設定 ──
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

# ── KV-Edit 核心參數 ──
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
num_full_replace_scales=2
kv_blend_ratio=1.0
kv_blend_scales=6
gradient_threshold=0.35

# ── Attention 設定 ──
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
save_attn_vis=1
seed=1

# ── 批量資料設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1"
output_dir="./outputs/outputs_loop_exp/pie_bench_kv_edit_results"

# 可選：只跑特定 category
categories=""
max_per_cat=-1
skip_existing=1

echo "================================================================"
echo " Batch PIE KV-Edit (Interleaved Pipeline)"
echo " bench_dir  : ${bench_dir}"
echo " output_dir : ${output_dir}"
echo "================================================================"

python3 tools/batch_run_kv_edit.py \
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
  --image_injection_scales ${image_injection_scales} \
  --inject_weights "${inject_weights}" \
  --num_full_replace_scales ${num_full_replace_scales} \
  --kv_blend_ratio ${kv_blend_ratio} \
  --kv_blend_scales ${kv_blend_scales} \
  --gradient_threshold ${gradient_threshold} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --save_attn_vis ${save_attn_vis} \
  --bench_dir "${bench_dir}" \
  --output_dir "${output_dir}" \
  ${categories:+--categories "${categories}"} \
  --max_per_cat ${max_per_cat} \
  --skip_existing ${skip_existing} \
  --seed ${seed}
