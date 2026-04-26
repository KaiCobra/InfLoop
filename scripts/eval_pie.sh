#!/bin/bash
# ==============================================================================
# eval_pie.sh — 使用官方 PnPInversion MetricsCalculator 評估 PIE-Bench 結果
#
# 評估指標（官方 8 項）：
#   structure_distance          ↓  DINO ViT-B/8 key self-similarity MSE（全圖）
#   psnr_unedit_part            ↑  背景 PSNR (torchmetrics)
#   lpips_unedit_part           ↓  背景 LPIPS（SqueezeNet, torchmetrics）
#   mse_unedit_part             ↓  背景 MSE (torchmetrics)
#   ssim_unedit_part            ↑  背景 SSIM (torchmetrics)
#   clip_similarity_source_image   CLIP(source, source_prompt) (torchmetrics CLIPScore)
#   clip_similarity_target_image ↑  CLIP(target, target_prompt)     — CLIPw
#   clip_similarity_target_image_edit_part ↑  CLIP(target*mask, target_prompt) — CLIPe
#
# 模型固定（與官方 PnPInversion 一致）：
#   LPIPS: SqueezeNet
#   CLIP:  openai/clip-vit-large-patch14
#   DINO:  dino_vitb8 (ViT-B/8)
#
# 使用方式：
#   bash scripts/eval_pie.sh
# ==============================================================================

# 強制使用本地快取，避免 HuggingFace 網路逾時
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── 自動安裝評估套件 ──
echo "[Setup] 檢查並安裝評估套件..."
pip install -q torchmetrics open-clip-torch lpips hpsv2 image-reward


threshold_method=13

# ── 路徑設定 ──
bench_dir="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1/"
result_dir="outputs/outputs_loop_exp/pie_bench_correctedResults_threshold_method${threshold_method}_CV0.5-1.0_k0.2-0.5"
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

# 是否跳過 Structure Distance 計算（DINO）
# 0 = 開啟（預設）；1 = 關閉（較小數据集快速評估時可用）
no_structure_dist=0

# ── 裝置 ──
# 留空白 = 自動偵測；或填 cuda / cpu
device=""

# ── 執行評估 ──
echo "================================================================"
echo " PIE-Bench 定量評估（官方 PnPInversion 指標）"
echo " bench_dir  : ${bench_dir}"
echo " result_dir : ${result_dir}"
echo " output_csv : ${output_csv}"
echo " 模型：LPIPS=SqueezeNet  CLIP=clip-vit-large-patch14  DINO=ViT-B/8"
echo "================================================================"

python3 tools/eval_pie_results.py \
  --bench_dir     "${bench_dir}" \
  --result_dir    "${result_dir}" \
  --output_csv    "${output_csv}" \
  --summary_json  "${summary_json}" \
  --max_per_cat   ${max_per_cat} \
  --skip_missing  ${skip_missing} \
  $([[ ${no_structure_dist} -eq 1 ]] && echo "--no_structure_dist") \
  ${categories:+--categories "${categories}"} \
  ${device:+--device ${device}}
