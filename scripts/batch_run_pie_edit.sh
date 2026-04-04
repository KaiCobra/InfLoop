#!/bin/bash
# ==============================================================================
# batch_run_pie_edit.sh
#
# 目的：
#   批量跑 extracted_pie_bench，模型只載入一次。
#   每個 case 會讀取 image.jpg + meta.json，並自動：
#   1) 移除 source/target prompt 中的 []
#   2) 取 prompt 差異詞作為 source_focus_words / target_focus_words
#
# 執行：
#   bash scripts/batch_run_pie_edit.sh
# ==============================================================================

# ── 模型設定（與 infer_p2p_edit.sh 相同）──
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

# ── P2P-Edit 核心參數（與 infer_p2p_edit.sh 對齊）──
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
num_full_replace_scales=2
attn_threshold_percentile=80
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
p2p_token_replace_prob=0.0
use_cumulative_prob_mask=1
save_attn_vis=1
use_normalized_attn=0
use_last_scale_mask=0
last_scale_majority_threshold=0.45
seed=1

# 閾值方法（1~8）
# 1 = 固定 percentile
# 2 = Dynamic threshold（ternary search + reference mask，需搭配 PIE-Bench）
# 3 = Otsu 最大類間方差法（無超參數）
# 4 = FFT 低通去噪 + Otsu
# 5 = Spectral Energy Ratio 自適應閾值
# 6 = Source Image Edge-Attention 跨頻譜相干性（需 source_image）
# 7 = GMM 雙高斯混合模型
# 8 = 複合方案（Edge-Coherent → Otsu → R_k fallback）
threshold_method=1

# Single-focus fallback（只有 target focus，無 source focus）時，
# Phase 1.7 以 source gen token 替換前幾個 scale，讓 attention 擷取時有結構參考
# 0 = 停用（純 free-gen）；建議值 4
phase17_fallback_replace_scales=4

# Dynamic threshold via binary search（使用 mapping_file.json 中的 GT mask 引導）
# 0 = 停用（使用固定 percentile），1 = 啟用（二分法搜尋）
use_dynamic_threshold=0
dynamic_threshold_iters=20

# ── 批量資料設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1"
output_dir="./outputs/outputs_loop_exp/pie_bench_correctedResults_threshold_method${threshold_method}"

# 可選：只跑特定 category，逗號分隔；留空表示全部 0~9 大任務
categories=""

# 每個 category 最多跑幾個 case；-1 表示全部
max_per_cat=-1

# 1=若 target.jpg 已存在就跳過（可續跑）
skip_existing=1

echo "================================================================"
echo " Batch PIE P2P-Edit"
echo " bench_dir  : ${bench_dir}"
echo " output_dir : ${output_dir}"
echo "================================================================"

python3 tools/batch_run_pie_edit.py \
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
  --threshold_method ${threshold_method} \
  --attn_threshold_percentile ${attn_threshold_percentile} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --p2p_token_replace_prob ${p2p_token_replace_prob} \
  --use_cumulative_prob_mask ${use_cumulative_prob_mask} \
  --phase17_fallback_replace_scales ${phase17_fallback_replace_scales} \
  --save_attn_vis ${save_attn_vis} \
  --use_normalized_attn ${use_normalized_attn} \
  --use_last_scale_mask ${use_last_scale_mask} \
  --last_scale_majority_threshold ${last_scale_majority_threshold} \
  --use_dynamic_threshold ${use_dynamic_threshold} \
  --dynamic_threshold_iters ${dynamic_threshold_iters} \
  --bench_dir "${bench_dir}" \
  --output_dir "${output_dir}" \
  ${categories:+--categories "${categories}"} \
  --max_per_cat ${max_per_cat} \
  --skip_existing ${skip_existing} \
  --seed ${seed}
