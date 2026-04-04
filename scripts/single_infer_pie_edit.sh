#!/bin/bash
# ==============================================================================
# P2P-Edit (Source Image Injection + Attention-Guided P2P) 圖像編輯管線
#
# 功能說明：
#   接收「source image + source prompt + target prompt」作為輸入，
#   利用 source image 的 VAE codes 注入前幾個 scale，確保結構高度保留，
#   再由 attention 遮罩引導 target 生成，允許局部語義改變。
#
# 設計邏輯：
#   [Phase 0]  Source image → VAE encoder → per-scale bit tokens
#   [Phase 1]  Source gen：前 N scale 注入 source image (weight=0) + 存 token + 擷取 attention
#   [Phase 1.5] Source attention → focus mask
#   [Phase 1.7] Target 自由生成 → target focus mask
#   [Phase 1.9] union(source_focus, target_focus) → replacement mask
#   [Phase 2]  Target gen：
#              • image 模式：前 N scale 100% source image VAE codes 注入（連續特徵層級）
#              • p2p   模式：前 N scale 100% source gen token 替換（離散 token 層級）
#              • 後面 scale：attention 遮罩引導替換
#
# Ablation study（--target_injection_source）：
#   p2p   = source gen 的離散 token 注入（受 source prompt 語義影響）
#   image = source image 直接 VAE 注入（像素最精確，無 prompt 語義偏差）
#
# 執行方式：
#   bash scripts/infer_p2p_edit.sh
# ==============================================================================

# ── 模型設定（與 infer_p2p_attn.sh 相同）──
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

# Source image 路徑（將被 VAE 編碼為 bit tokens）
# Source image 路徑（將被 VAE 編碼為離散 token）
# • 設定路徑 → P2P-Edit 模式：source gen 注入 + P2P token 來自 source image 編碼
# • 留空白   → 純 P2P-Attn 模式：P2P token 來自 source gen 採樣（與 run_p2p_attn.py 相同）
# source_image="./outputs/outputs_loop_exp/extracted_pie_bench/2_add_object_80/211000000000/image.jpg"
dir="./outputs/outputs_loop_exp/extracted_pie_bench/2_add_object_80/212000000005/"
source_image="${dir}image.jpg"
# source_image=""  # 用此行可切換為純 P2P-Attn 模式

# 前幾個 scale 使用 source image 注入（weight=0 → 100% source image）
# 建議：2（粗略結構 scale）；較大值可強制更多 scale 保留 source 圖像結構
image_injection_scales=3

# 各 scale 的注入強度列表（共 13 個，對應 pn=1M 的 13 個 scale）
# 0.0 = 100% source image；1.0 = 100% 自由生成；中間值 = 線性混合
# 若不為空，將覆蓋 image_injection_scales 的二值 schedule
# 範例：前 2 scale 完全注入，其餘自由生成
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"

# Ablation study：target 生成前幾個 scale 的 injection 來源
# ⚠ target 生成現在固定使用純 P2P-Attn 模式（不注入 source image），
#   確保 target prompt 能自由改變內容。
#   以下變數保留供未來 ablation 參考，目前不傳入模型。
# target_injection_source="image"  # 已停用

# ── Attention 遮罩參數 ──

# 前幾個 scale 做 100% source token 替換（p2p 模式時才生效）
# 應與 image_injection_scales 保持一致
num_full_replace_scales=2

# Attention 閾值百分位數
# 高於此百分位的空間位置被視為「focus 區域」（不替換）
# 75 = 前 25% 最強 attention 視為 focus 區域
attn_threshold_percentile=80

# 用於計算 attention 遮罩的 transformer block 起始/結束 index
# -1 = 自動（起始 = 模型深度的 1/2，結束 = 最後一個 block）
attn_block_start=2
attn_block_end=-1

# CFG 設定下擷取哪個 batch 的 attention
# 0 = conditioned（通常是我們想要的）
attn_batch_idx=0

# Fallback 機率替換（當某個 scale 無 attention 遮罩時使用）
p2p_token_replace_prob=0.0

# 是否啟用跨尺度累積機率遮罩
# 1 = 當前 scale 使用前面所有 scale mask 疊加平均（灰階=替換機率）
# 0 = 維持單一 scale 對單一 mask（bool）
use_cumulative_prob_mask=1

# Single-focus fallback（只有 target focus，無 source focus）時，
# Phase 1.7 以 source gen token 替換前幾個 scale，讓 attention 擷取時有結構參考
# 0 = 停用（純 free-gen）；建議值 4
phase17_fallback_replace_scales=4

# Debug mode：儲存所有中間過程產出的圖片（source gen、Phase 1.7 guided gen、Phase 1.7 fallback gen）
# 0 = 關閉（只儲存 source.jpg + target.jpg）；1 = 開啟
debug_mode=1
# Token + 遮罩資料儲存路徑
p2p_token_file="./tokens_p2p_edit.pkl"

# 是否儲存 attention 遮罩視覺化
save_attn_vis=1
use_normalized_attn=0

# ── Prompt 設定 ──
# 使用python 讀取meta.json裡的 source_prompt、target_prompt(使用 remove_prompt_brackets 清掉 '[' & ']')，然後使用source_prompt、target_prompt 裡的文字，使用different word 來找出 focus_words（以空格分隔），並傳入 source_focus_words、target_focus_words
# def remove_prompt_brackets(prompt: str) -> str:
#     """移除 []，保留括號內文字。"""
#     if not prompt:
#         return ""
#     return re.sub(r"\[([^\]]*)\]", r"\1", prompt).strip()


meta_file="${dir}meta.json"
readarray -t _PROMPT_DATA < <(python3 - "${meta_file}" <<'PY'
import json
import re
import sys

meta_path = sys.argv[1]
with open(meta_path, 'r', encoding='utf-8') as f:
  meta = json.load(f)


def clean_prompt(text: str) -> str:
  text = text or ""
  text = re.sub(r'[\[\]\(\)（）]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text


def ordered_diff_words(a: str, b: str) -> str:
  b_words = set(b.split())
  result = []
  seen = set()
  for word in a.split():
    if word not in b_words and word not in seen:
      result.append(word)
      seen.add(word)
  return ' '.join(result)


source = clean_prompt(meta.get('source_prompt', ''))
target = clean_prompt(meta.get('target_prompt', ''))

print(source)
print(target)
print(ordered_diff_words(source, target))
print(ordered_diff_words(target, source))
PY
)

source_prompt="${_PROMPT_DATA[0]}"
target_prompt="${_PROMPT_DATA[1]}"
source_focus_words="${_PROMPT_DATA[2]}"
target_focus_words="${_PROMPT_DATA[3]}"

# ── 輸出目錄 ──
current_time=$(date +"%Y%m%d_%H%M%S")
save_file="./outputs/p2p_edit_${current_time}/"

# ── 執行 P2P-Edit 管線 ──
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
  --source_prompt "${source_prompt}" \
  --target_prompt "${target_prompt}" \
  --source_focus_words "${source_focus_words}" \
  --target_focus_words "${target_focus_words}" \
  --source_image "${source_image}" \
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
  --debug_mode ${debug_mode} \
  --save_attn_vis ${save_attn_vis} \
  --use_normalized_attn ${use_normalized_attn} \
  --save_file ${save_file} \
  --seed 1 \
  --h_div_w_template 1.000
