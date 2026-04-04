#!/bin/bash
# ==============================================================================
# eval_pie.sh — 使用官方 PnPInversion evaluate.py 評估 PIE-Bench 結果
#
# 評估指標（官方 8 項）：
#   structure_distance          ↓  DINO ViT-B/8 key self-similarity MSE（全圖）
#   psnr_unedit_part            ↑  背景 PSNR
#   lpips_unedit_part           ↓  背景 LPIPS（SqueezeNet）
#   mse_unedit_part             ↓  背景 MSE
#   ssim_unedit_part            ↑  背景 SSIM
#   clip_similarity_source_image   CLIP(edited, source_prompt)
#   clip_similarity_target_image ↑  CLIP(edited, target_prompt)
#   clip_similarity_target_image_edit_part ↑  CLIP(edited_region, target_prompt)
#
# result_format 兩種模式：
#
#   "batch_run" — 我們的 batch_run_pie_edit.py 輸出格式：
#       result_dir/<category>/<case_id>/target.jpg  （需要重組）
#
#   "direct" — 已是官方 PnPInversion 格式（leditspp、fireflow 等比較方法）：
#       result_dir/annotation_images/<image_path>   （直接使用）
#
# 使用方式：
#   bash scripts/eval_pie.sh
# ==============================================================================

# ── 自動安裝評估套件 ──
echo "[Setup] 檢查並安裝評估套件..."
pip install -q lpips open-clip-torch


threshold_method=1

# ── 路徑設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1/"
result_dir="outputs/outputs_loop_exp/pie_bench_correctedResults_threshold_method${threshold_method}"
output_csv="./outputs/eval_pie_bench_correctedResults_threshold_method${threshold_method}/per_case.csv"
summary_json="./outputs/eval_pie_bench_correctedResults_threshold_method${threshold_method}/summary.json"

# ── 過濾設定 ──
# 只評估指定 category（逗號分隔），留空白代表全部
# 範例：categories="0_random_140,1_change_object_80"
categories=""

# 每個 category 最多評估幾個案例（-1 = 全部）
max_per_cat=-1

# 若 target.jpg 不存在則跳過（1 = 跳過，0 = 報錯）
skip_missing=1

# ── 模型設定 ──
# LPIPS backbone：alex（最快）/ vgg（標準）/ squeeze
lpips_net="alex"

# CLIP model（open_clip 格式）
# 選項：ViT-B-32 / ViT-L-14 / ViT-H-14
clip_model="ViT-L-14"
clip_pretrained="openai"

# 是否跳過 Structure Distance 計算（DINO）
# 0 = 開啟（預設）；1 = 關閉（較小數据集快速評估時可用）
no_structure_dist=0

# ── 裝置 ──
# 留空白 = 自動偵測；或填 cuda / cpu
device=""

# ── 執行評估 ──
echo "================================================================"
echo " PIE-Bench 定量評估"
echo " bench_dir  : ${bench_dir}"
echo " result_dir : ${result_dir}"
echo " output_csv : ${output_csv}"
echo "================================================================"

python3 tools/eval_pie_results.py \
  --bench_dir     "${bench_dir}" \
  --result_dir    "${result_dir}" \
  --output_csv    "${output_csv}" \
  --summary_json  "${summary_json}" \
  --max_per_cat   ${max_per_cat} \
  --skip_missing  ${skip_missing} \
  --lpips_net     ${lpips_net} \
  --clip_model    ${clip_model} \
  --clip_pretrained ${clip_pretrained} \
  $([[ ${no_structure_dist} -eq 1 ]] && echo "--no_structure_dist") \
  ${categories:+--categories "${categories}"} \
  ${device:+--device ${device}}
