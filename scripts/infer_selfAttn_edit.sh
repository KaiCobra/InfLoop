#!/bin/bash
# ==============================================================================
# KV-Edit on Infinity ── 圖像編輯管線
#
# 核心原理（KV-Edit paper 2502.17363）：
#   Phase 1  Source forward：
#     擷取 source prompt 生成過程中每個 transformer block、每個 scale 的
#     self-attention K/V，作為 frozen background memory。
#
#   Phase 2  Target forward：
#     • Background tokens（mask 白色）→ 使用 source K/V（frozen memory）
#     • Foreground tokens（mask 黑色）→ 使用 target 當前 K/V（自由生成）
#     • 所有 token 的 Q 均 attend 到合併後的 K/V
#     → 無 bitwise token 替換，純粹 K/V memory 重用。
#
# Mask 定義（--mask_image）：
#   白色（255） = background = frozen，保留 source 結構
#   黑色（  0） = foreground = editing area，target 自由生成
#
# 執行方式：
#   bash scripts/infer_selfAttn_edit.sh
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

# （可選）Source image：
#   設定路徑 → Phase 1 使用 VAE encoder 連續特徵輔助 source gen
#   留空白   → Phase 1 純以 source_prompt 自由生成
source_image=""
# source_image="./path/to/source.jpg"

# Source image VAE feature 注入強度（僅在 source_image 非空時生效）
image_injection_scales=2   # 前幾個 scale weight=0（完全注入）
inject_weights=""           # 若空白，由 image_injection_scales 自動生成 schedule

# ── Mask（前景/背景分割）──
# 白色（255）= background（frozen，使用 source K/V）
# 黑色（  0）= foreground（editing，target 自由生成）
# 留空白     = 未提供 mask → 全部視為 foreground（等同純 target 生成）
mask_image="./outputs/outputs_loop_exp/extracted_pie_bench/8_change_background_80/811000000000/mask.png"
mask_threshold=0.5

# ── KV 擷取/注入的 scale 範圍 ──
# kv_scale_start：從第幾個 scale 開始做 KV-Edit（前面的 scale 完全為 target 自由生成）
# kv_scale_end  ：到第幾個 scale 結束（-1 = 最後一個 scale）
kv_scale_start=0
kv_scale_end=-1

# CFG batch 中擷取哪個 batch 的 K/V（0 = conditioned）
attn_batch_idx=0

# ── Prompt 設定 ──

# 範例 1：背景替換
source_prompt="a woman wearing a hat and gloves is walking on a snow covered path."
target_prompt="a woman wearing a hat and gloves is walking on a leaves covered path."

# 範例 2：物件替換
# source_prompt="a little carton sheep in a white background."
# target_prompt="a little carton sheep in a forest background."

# 範例 3：風格/主體替換
# source_prompt="A oil paint of Girl with a Pearl Earring."
# target_prompt="A oil paint of Green Frog with a Pearl Earring."

# 範例 4：配合 PIE-Bench change_background
# source_prompt="a woman wearing a hat and gloves is walking on a snow covered path."
# target_prompt="a woman wearing a hat and gloves is walking on a sandy beach path."

# ── 輸出目錄 ──
current_time=$(date +"%Y%m%d_%H%M%S")
save_file="./outputs/outputs_loop_exp/kv_edit_${current_time}/"

# ── 執行 KV-Edit 管線 ──
python3 tools/run_selfAttn_edit.py \
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
  --source_prompt "${source_prompt}" \
  --target_prompt "${target_prompt}" \
  ${source_image:+--source_image "${source_image}"} \
  ${source_image:+--image_injection_scales ${image_injection_scales}} \
  ${inject_weights:+--inject_weights "${inject_weights}"} \
  ${mask_image:+--mask_image "${mask_image}"} \
  --mask_threshold ${mask_threshold} \
  --kv_scale_start ${kv_scale_start} \
  --kv_scale_end ${kv_scale_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --save_file ${save_file} \
  --seed 1 \
  --h_div_w_template 1.000
