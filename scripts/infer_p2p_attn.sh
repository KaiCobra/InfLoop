#!/bin/bash
# ==============================================================================
# P2P-Attn (Prompt-to-Prompt + Attention-Guided) 圖像編輯管線
#
# 功能說明：
#   在原有 P2P 管線的基礎上加入 cross-attention 引導的空間遮罩，
#   專為「同場景、局部文字內容不同」的 prompt pair 設計。
#
# 與 P2P-Edit / batch_run_pie_edit 的關係：
#   • P2P-Attn 為純 attention-guided 模式，不注入 source image VAE codes。
#   • 若需要 source image 結構保留 + 多種閾值方法（Otsu / Method 14 / Last-scale 等），
#     請改用 scripts/infer_p2p_edit.sh 或 scripts/batch_run_pie_edit.sh。
#
# 設計邏輯：
#   1. 前 N 個 scale（num_full_replace_scales）：100% source token 替換（保留結構）
#   2. 第 N+1 個 scale 之後：
#      - 擷取 source 生成的 cross-attention map
#      - 找出 focus_words 對應的高 attention 區域
#      - 高 attention 區域（focus）→ 不替換，讓 target 自由渲染
#      - 低 attention 區域（背景）→ 替換為 source token，保留場景結構
#
# 執行方式：
#   bash scripts/infer_p2p_attn.sh
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

# ── P2P-Attn 核心參數 ──

# 前幾個 scale 做 100% source token 替換（保留整體結構）
# 建議值：3~5（越大 = 結構保留越強，但 target 彈性越小）
num_full_replace_scales=0

# Attention 閾值百分位數
# 高於此百分位的空間位置被視為「focus 區域」（不替換）
# 75 = 前 25% 最強 attention 視為 focus 區域
# 數值越小 = focus 區域越大（更保守）
# 數值越大 = focus 區域越小（僅鎖定最核心位置）
attn_threshold_percentile=80

# 用於計算 attention 遮罩的 transformer block 起始/結束 index
# -1 = 自動（起始 = 模型深度的 1/2，結束 = 最後一個 block）
attn_block_start=2
attn_block_end=-1

# CFG 設定下擷取哪個 batch 的 attention
# 0 = conditioned（source prompt 對應的注意力，通常是我們想要的）
# 1 = unconditioned（null prompt 對應的注意力）
attn_batch_idx=0

# Fallback 機率替換（當某個 scale 無 attention 遮罩時使用）
p2p_token_replace_prob=0.5

# Token + 遮罩資料儲存路徑
p2p_token_file="./tokens_p2p_attn.pkl"

# 是否儲存 attention 遮罩視覺化（白色 = 替換區域，黑色 = focus 區域）
save_attn_vis=1

seed=1

# ── Prompt 設定 ──
# source_focus_words / target_focus_words：分別指定各自 prompt 中的 focus 詞彙
# 高 attention 區域取 union → 兩張圖的物件區域都受保護

# source_prompt="A Picasso style painting of a cat with a cowboy hat chasing a rat at a gym."
# target_prompt="A Vincent van Gogh style painting of a cat with a cowboy hat chasing a rat at a gym."

# source_prompt="A girl, facial features extreme close up."
# target_prompt="A smiling girl, facial features extreme close up."

source_prompt="A cartoon style elephant holding an umbrella on the hill."
target_prompt="A cartoon style giraffe holding an umbrella on the hill."
source_focus_words="elephant"
target_focus_words="giraffe"

# source_prompt="A train platform sign that reads \"PLEASE STAND BEHIND LINE\" as a train approaches."
# target_prompt="A train platform sign that reads \"DESTINATION: LONDON\" as a train approaches."
# source_focus_words="PLEASE STAND BEHIND LINE"
# target_focus_words="DESTINATION LONDON"

# source_prompt="Cute Shiba Inu wearing a space helmet and pink T-shirt in a bedroom holding a sign reads \"GET OUT\"."
# target_prompt="Cute Shiba Inu wearing a space helmet and blue T-shirt in a bedroom holding a sign reads \"TEEN SPIRIT\"."
# source_focus_words="GET OUT"
# target_focus_words="TEEN SPIRIT"

# source_prompt="A waterpaint of a Cute capybara sipping bubble tea in a cozy cafe."
# target_prompt="A 3D rendered model of a Cute capybara sipping bubble tea in a cozy cafe."

# source_prompt="A black dog inside a dimly lit backstage theater hallway."
# target_prompt="A orange cat inside a Brightly backstage theater hallway."

# source_prompt="A oil painting of a dog laying down on the ground."
# target_prompt="A real photographic of a dog laying down on the ground."

# source_prompt="A photograph of a cat playing a guitar at a cozy street café terrace."
# target_prompt="A photograph of a fox playing a guitar at a cozy street café terrace."
# source_focus_words="cat"
# target_focus_words="fox"

# ── 輸出目錄 ──
current_time=$(date +"%Y%m%d_%H%M%S")
save_file="./outputs/p2p_attn_${current_time}/"

# ── 執行 P2P-Attn 管線 ──
python3 tools/run_p2p_attn.py \
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
  --source_focus_words "${source_focus_words}" \
  --target_focus_words "${target_focus_words}" \
  --num_full_replace_scales ${num_full_replace_scales} \
  --attn_threshold_percentile ${attn_threshold_percentile} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --p2p_token_replace_prob ${p2p_token_replace_prob} \
  --p2p_token_file ${p2p_token_file} \
  --save_attn_vis ${save_attn_vis} \
  --save_file ${save_file} \
  --seed ${seed}
