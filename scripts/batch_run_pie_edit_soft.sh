#!/bin/bash
# ==============================================================================
# batch_run_pie_edit_soft.sh
#
# 目的：
#   批量跑 extracted_pie_bench（Soft Blending 版本）。
#   與 batch_run_pie_edit.sh 的差異：
#     • 使用 soft blending 取代 hard token replacement
#     • 在 logits 空間（或機率空間）做加權混合，而非二值替換
#
# 執行：
#   bash scripts/batch_run_pie_edit_soft.sh
# ==============================================================================

# ── 模型設定（與 batch_run_pie_edit.sh 相同）──
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

# ── P2P-Edit 核心參數 ──
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
num_full_replace_scales=2
attn_threshold_percentile=80
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
p2p_token_replace_prob=0.0
use_cumulative_prob_mask=0
save_attn_vis=1
use_normalized_attn=0
seed=1

# ── Soft Blending 參數 ──
# blending 公式選擇：
#   none        = 停用（維持 hard replacement，等同原版 batch_run_pie_edit）
#   linear      = logits 空間線性混合：α * source_logits + (1-α) * target_logits
#   prob_linear = 機率空間線性混合：α * source_one_hot + (1-α) * softmax(target_logits)
#   geometric   = 幾何平均
#   slerp       = 球面插值（Spherical Linear Interpolation）
soft_blend_method="slerp"

# Source one-hot logits 的溫度
# 值越大 source 信號越硬（deterministic）；值越小越柔和
soft_blend_temperature=10.0

# Phase 1.7 fallback
phase17_fallback_replace_scales=4

# ── SA×CA DiffSegmenter 參數 ──
# 迭代精煉次數（0=停用，3=DiffSegmenter 建議值）
sa_refine_iterations=3
# Self-attention block 範圍
sa_block_start=0
sa_block_end=-1
# SA 擷取的最大 scale index（-1=全部；後期 scale 可能 OOM）
sa_max_scale=9

# ── 批量資料設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1"
output_dir="./outputs/outputs_loop_exp/pie_edit_soft_blending"

# 可選：只跑特定 category，逗號分隔；留空表示全部
categories=""

# 每個 category 最多跑幾個 case；-1 表示全部
max_per_cat=-1

# 1=若 target.jpg 已存在就跳過（可續跑）
skip_existing=1

echo "================================================================"
echo " Batch PIE P2P-Edit (Soft Blending)"
echo " bench_dir        : ${bench_dir}"
echo " output_dir       : ${output_dir}"
echo " blend_method     : ${soft_blend_method}"
echo " blend_temperature: ${soft_blend_temperature}"
echo "================================================================"

python3 tools/batch_run_pie_edit_soft.py \
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
  --attn_threshold_percentile ${attn_threshold_percentile} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --p2p_token_replace_prob ${p2p_token_replace_prob} \
  --use_cumulative_prob_mask ${use_cumulative_prob_mask} \
  --phase17_fallback_replace_scales ${phase17_fallback_replace_scales} \
  --save_attn_vis ${save_attn_vis} \
  --use_normalized_attn ${use_normalized_attn} \
  --soft_blend_method ${soft_blend_method} \
  --soft_blend_temperature ${soft_blend_temperature} \
  --sa_refine_iterations ${sa_refine_iterations} \
  --sa_block_start ${sa_block_start} \
  --sa_block_end ${sa_block_end} \
  --sa_max_scale ${sa_max_scale} \
  --bench_dir "${bench_dir}" \
  --output_dir "${output_dir}" \
  ${categories:+--categories "${categories}"} \
  --max_per_cat ${max_per_cat} \
  --skip_existing ${skip_existing} \
  --seed ${seed}
