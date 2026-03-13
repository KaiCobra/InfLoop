#!/bin/bash
# ==============================================================================
# pie_edit.sh — PIE-Bench 批量 P2P-Edit 評估啟動腳本
#
# 功能說明：
#   批量處理 extracted_pie_bench 中的全部 700 個測試案例（10 個 category）。
#   模型只載入一次，依照原始資料夾結構儲存結果：
#     {output_dir}/{category}/{case_id}/source.jpg
#     {output_dir}/{category}/{case_id}/target.jpg
#
# 使用方式：
#   bash scripts/pie_edit.sh
#
# 可調整的旋鈕：
#   --categories     只跑指定 category（逗號分隔），預設全部
#   --max_per_cat    每個 category 最多幾個案例，預設 -1（全部）
#   --skip_existing  1 = 跳過已完成案例（續跑斷點），0 = 全部重跑
#   --save_attn_vis  1 = 儲存 attention 遮罩視覺化（較慢），預設 0
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

# ── Source Image Injection ──
# 前幾個 scale 使用 source image VAE codes 注入（全部為 0.0 = 100% source image）
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"

# ── P2P-Edit 核心參數 ──
num_full_replace_scales=2
attn_threshold_percentile=80
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
p2p_token_replace_prob=0.0

# ── 批量設定 ──
bench_dir="./outputs/outputs_loop_exp/extracted_pie_bench"
output_dir="./outputs/outputs_loop_exp/pie_bench_results_pieMask85"

# 只跑特定 category（逗號分隔資料夾名稱），留空白代表全部
# 範例：categories="0_random_140,1_change_object_80"
categories=""

# 每個 category 最多處理幾個案例（-1 = 全部）
max_per_cat=-1

# 1 = 若 target.jpg 已存在則跳過（可安全續跑）；0 = 全部重跑
skip_existing=1

# 是否儲存 attention 遮罩視覺化（1 = 開啟，批量評估時建議關閉）
save_attn_vis=1

# 是否使用 PIE-Bench 提供的 mask.png 做為 token 替換遮罩
# 0 = 使用 attention threshold（預設）；1 = 使用 PIE mask
use_pie_mask=0

# （需 use_pie_mask=1）白色（編輯）區域內以 attention 二次篩選
# 0 = 純 PIE mask（預設）；1 = PIE mask AND attention fallback
# 邏輯：final_mask = pie_bg_mask OR attn_replacement_mask
#   → 黑色區域全部保留 source；白色區域中 attention 未聚焦的 token 也保留 source
pie_mask_attn_fallback=0

# 亂數種子
seed=1

# ── 執行批量評估 ──
echo "================================================================"
echo " PIE-Bench 批量 P2P-Edit 評估"
echo " bench_dir  : ${bench_dir}"
echo " output_dir : ${output_dir}"
echo "================================================================"

python3 tools/run_pie_edit.py \
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
  --save_attn_vis ${save_attn_vis} \
  --bench_dir "${bench_dir}" \
  --output_dir "${output_dir}" \
  ${categories:+--categories "${categories}"} \
  --max_per_cat ${max_per_cat} \
  --skip_existing ${skip_existing} \
  --use_pie_mask ${use_pie_mask} \
  --pie_mask_attn_fallback ${pie_mask_attn_fallback} \
  --seed ${seed}
