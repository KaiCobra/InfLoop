#!/bin/bash
# ==============================================================================
# eval_pie.sh — PIE-Bench 結果定量評估啟動腳本
#
# 評估指標：
#   背景保留（Background Preservation，基於 mask.png 遮罩）：
#     PSNR  ↑  比較 source vs edited 的背景區域
#     SSIM  ↑  比較 source vs edited 的背景區域
#     LPIPS ↓  比較 source vs edited 的背景區域
#
#   結構保留（Structure Preservation，使用全圖）：
#     StructDist ↓  DINO ViT-S/8 key-feature MSE（source vs edited）
#
#   編輯對齊（Edit Alignment）：
#     CLIP sim ↑  edited image（target.jpg）vs target_prompt 文字
#
# 輸出：
#   outputs/eval_pie/per_case.csv     — 每個案例的詳細數字
#   outputs/eval_pie/summary.json     — 每個 category 的平均 + 全域平均
#
# 使用方式：
#   bash scripts/eval_pie.sh
#
# 前置需求（首次執行自動安裝）：
#   pip install lpips open-clip-torch
# ==============================================================================

# ── 自動安裝評估套件 ──
echo "[Setup] 檢查並安裝評估套件..."
pip install -q lpips open-clip-torch

# ── 路徑設定 ──
bench_dir="./outputs/outputs_loop_exp/extracted_pie_bench"
result_dir="./outputs/outputs_loop_exp/pie_bench_results_pieMask"
output_csv="./outputs/eval_pieMask/per_case.csv"
summary_json="./outputs/eval_pieMask/summary.json"

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
