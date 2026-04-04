#!/bin/bash
# ==============================================================================
# batch_run_pie_edit_selfattn_seg.sh
#
# 目的：
#   批量跑 extracted_pie_bench（DiffSegmenter-style 物件分割版本）。
#   與 batch_run_pie_edit.sh 的差異：
#     • Source gen 時同時擷取 cross-attention + self-attention maps
#     • 使用 DiffSegmenter 迭代精煉（SA × CA）產生物件 mask
#     • 不做 target gen — 只輸出 source.jpg + 物件 mask 視覺化
#
# 執行：
#   bash scripts/batch_run_pie_edit_selfattn_seg.sh
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

# ── Source Image Injection ──
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"

# ── Cross-Attention 參數 ──
num_full_replace_scales=2
attn_threshold_percentile=80
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
seed=1

# ── DiffSegmenter Self-Attention 參數 ──
# SA × CA 迭代次數（DiffSegmenter 建議 2~3 次）
sa_refine_iterations=3

# Self-attention 擷取 block 範圍
# 全部 block 會導致 OOM，建議只用 3~5 個 block
sa_block_start=2
sa_block_end=7

# SA 擷取的最大 scale index
# 後期 scale 累積大量 KV cache，attention matrix 太大會 OOM
# -1 = 全部（容易 OOM）；建議設 9 左右
sa_max_scale=9

# ── 批量資料設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1"
output_dir="./outputs/outputs_loop_exp/debug_selfattn_seg_filtered"

# 可選：只跑特定 category，逗號分隔；留空表示全部
categories=""

# 每個 category 最多跑幾個 case；-1 表示全部
max_per_cat=-1

# 1=若 selfattn_seg/meta.json 已存在就跳過（可續跑）
skip_existing=1

echo "================================================================"
echo " Batch PIE SelfattnSeg (DiffSegmenter-style)"
echo " bench_dir          : ${bench_dir}"
echo " output_dir         : ${output_dir}"
echo " sa_refine_iters    : ${sa_refine_iterations}"
echo " sa_block_range     : ${sa_block_start}~${sa_block_end}"
echo " sa_max_scale       : ${sa_max_scale}"
echo " attn_percentile    : ${attn_threshold_percentile}"
echo "================================================================"

python3 tools/batch_run_pie_edit_selfattn_seg.py \
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
