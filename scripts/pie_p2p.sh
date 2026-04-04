#!/bin/bash
# ==============================================================================
# PIE-P2P：批次 Prompt-to-Prompt 圖像生成（PIE-Bench 資料集）
#
# 功能說明：
#   讀取 PIE-Bench mapping_file.json，對每筆資料的 original_prompt（source）
#   和 editing_prompt（target）各生成一張圖片，使用 P2P-Attn pipeline。
#
#   - 模型只載入一次，批次處理全部資料（節省重複載入時間）
#   - 輸出結構：output_dir/{image_key}/source.jpg + target.jpg
#   - 支援斷點續跑：已存在 target.jpg 的筆數自動跳過
#   - Focus words 自動從 blended_word 欄位解析
#
# 執行方式：
#   bash scripts/pie_p2p.sh
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

# ── P2P-Attn 參數（與 infer_p2p_attn.sh 相同）──

# 前幾個 scale 做 100% source token 替換（保留整體結構）
num_full_replace_scales=0

# Attention 閾值百分位數
# 高於此百分位的空間位置被視為「focus 區域」（不替換）
attn_threshold_percentile=72

# 用於計算 attention 遮罩的 transformer block 起始/結束 index
# -1 = 自動（起始 = 模型深度的 1/2，結束 = 最後一個 block）
attn_block_start=2
attn_block_end=-1

# CFG 設定下擷取哪個 batch 的 attention
# 0 = conditioned（source prompt 對應的注意力）
attn_batch_idx=0

# Fallback 機率替換（當某個 scale 無 attention 遮罩時使用）
p2p_token_replace_prob=0.5

# ── 資料集與輸出設定 ──

# PIE-Bench mapping_file.json 路徑
json_file="./outputs/outputs_loop_exp/PIE-Bench_v1-20260314T125823Z-3-001/PIE-Bench_v1/mapping_file.json"

# 輸出根目錄（每筆資料依 key 建立子目錄）
output_dir="./outputs/outputs_loop_exp/pie_p2p"

# 處理範圍（0-based index）
# start_idx=0：從第一筆開始
# end_idx=-1：處理到最後一筆；設正整數可限制筆數（用於測試）
start_idx=0
end_idx=-1

# 斷點續跑：1 = 跳過已存在 target.jpg 的筆數；0 = 強制重跑全部
skip_existing=1

# ── 執行批次 PIE-P2P 管線 ──
python3 tools/run_pie_p2p.py \
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
  --num_full_replace_scales ${num_full_replace_scales} \
  --attn_threshold_percentile ${attn_threshold_percentile} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --p2p_token_replace_prob ${p2p_token_replace_prob} \
  --json_file "${json_file}" \
  --output_dir "${output_dir}" \
  --start_idx ${start_idx} \
  --end_idx ${end_idx} \
  --skip_existing ${skip_existing} \
  --seed 1
