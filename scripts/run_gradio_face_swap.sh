#!/bin/bash
# ==============================================================================
# run_gradio_face_swap.sh
#
# 啟動 Face-Swap pipeline 的 Gradio demo page。
# UI 會給以下控制項：
#   - prompt T_t
#   - 一張 source face image
#   - λ₁ / λ₂ sliders（phase 2 線性混合：new = λ₁·e_I + λ₂·proj(e_A)）
#   - subject word（預設 "boy"）
#   - seed
#
# 模型只會在啟動時載入一次，重複按 Run 共用同一份 weights。
# B 圖會依 (prompt, seed) 快取在 work_dir 下，prompt 不變時 Run 不會重生 B。
#
# 前置：請先確認 AdaFace server 已在 127.0.0.1:8000 啟動：
#   curl -s http://127.0.0.1:8000/health
#
# 執行：
#   bash scripts/run_gradio_face_swap.sh
# ==============================================================================

# ── 模型設定（與 batch_run_pie_edit_faceSwap.sh 相同）──
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
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
num_full_replace_scales=2
attn_threshold_percentile=80
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
p2p_token_replace_prob=0.0
use_cumulative_prob_mask=1
save_attn_vis=0          # demo 預設不存 attn 視覺化（避免每次 Run 都生成大量檔案）
use_normalized_attn=0
use_last_scale_mask=0
last_scale_majority_threshold=0.45
threshold_method=1
absolute_high=0.7
absolute_low=0.3
seed=1

# ── Demo 預設值（UI 上仍可覆蓋）──
default_prompt="a boy turned his head to his left over the shuolder and tilted up"
default_subject="boy"

# ── Gradio server ──
host="0.0.0.0"
port=7860
share=0                # 1=產生 *.gradio.live 公開連結
adaface_url="http://127.0.0.1:8000"
work_dir="./outputs/gradio_face_swap"
identity_cache_dir="./weights/identities"   # learned 模式從這裡讀 v_A.pt；Train 按鈕也寫到這
debug_face_op=0

mkdir -p "${work_dir}"

echo "================================================================"
echo " Face-Swap Gradio Demo"
echo " host:port  : ${host}:${port}  share=${share}"
echo " work_dir   : ${work_dir}"
echo " adaface_url: ${adaface_url}"
echo "================================================================"

python3 tools/gradio_face_swap.py \
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
  --save_attn_vis ${save_attn_vis} \
  --use_normalized_attn ${use_normalized_attn} \
  --use_last_scale_mask ${use_last_scale_mask} \
  --last_scale_majority_threshold ${last_scale_majority_threshold} \
  --absolute_high ${absolute_high} \
  --absolute_low ${absolute_low} \
  --seed ${seed} \
  --host "${host}" \
  --port ${port} \
  --share ${share} \
  --work_dir "${work_dir}" \
  --adaface_url "${adaface_url}" \
  --identity_cache_dir "${identity_cache_dir}" \
  --default_prompt "${default_prompt}" \
  --default_subject "${default_subject}" \
  --debug_face_op ${debug_face_op}
