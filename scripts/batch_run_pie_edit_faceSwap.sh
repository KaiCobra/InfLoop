#!/bin/bash
# ==============================================================================
# batch_run_pie_edit_faceSwap.sh
#
# Face-swap 批量管線。模型只載入一次。
# 每個 identity（例如 face/smith/）會：
#   1. 用固定 prompt T_t 產出 base 圖 B（cache 到 output_dir/<identity>/B.jpg）
#   2. 透過 AdaFace HTTP server 計算 e_A（多張 source 平均）/ e_B（B 的 embedding）
#   3. 對 prompt T_t 在 subject token (預設 "boy") 位置做：
#        phase 1 : 不動
#        phase 1.7: e_I -= proj(e_B)        （proj=repeat 4×→2048-d, scale 到原 norm）
#        phase 2 : e_I  = proj(e_A)
#   4. 跑 P2P-Edit pipeline，source image=B，輸出 target.jpg 為 face-swap 結果
#
# 前置作業：請先確認 AdaFace server 已在 127.0.0.1:8000 啟動：
#   curl -s http://127.0.0.1:8000/health
#
# 執行：
#   bash scripts/batch_run_pie_edit_faceSwap.sh
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

# ── P2P-Edit 核心參數（與 batch_run_pie_edit.sh 對齊）──
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

# 閾值方法（同 batch_run_pie_edit.sh，1=固定 percentile，13=Meta-Adaptive，14=Absolute）
threshold_method=1
absolute_high=0.7
absolute_low=0.3

# ── Face-Swap 設定 ──
face_root="./face"
identities=""               # csv 留空=face_root 下全部子資料夾
prompt_t="a boy turned his head to his left over the shoulder and tilted up"
subject_word="boy"
adaface_url="http://127.0.0.1:8000"
regen_B=0                   # 1=強制重生 B；0=有 cache 就用
debug_face_op=1             # 1=印出每個 token 操作前後 norm

# Phase 2 線性混合：new = lam1 * e_I + lam2 * proj(e_A)
# lam1=0, lam2=1 → 完全 replace（預設、與舊行為一致）
# lam1=1, lam2=0 → 完全保留原 token（無 face swap）
lam1=-0.2
lam2=1.0

# Phase 2: 改用 Textual-Inversion 學好的 v_A（先跑 scripts/optimize_face_token.sh）
# 1=讀 weights/identities/<id>/v_A.pt 直接寫入 boy 位置（lam1/lam2 將被忽略）
# 0=用 AdaFace linear 路徑（預設）
use_learned_v_A=0
identity_cache_dir="./weights/identities"

# get current time
timeStamp=$(date +%Y%m%d_%H%M%S)

# 輸出目錄（依需要改 tag）
output_dir="./outputs/outputs_loop_exp/face_exp2/face_swap_threshold_method${threshold_method}_${timeStamp}_${lam1}_${lam2}"

# 1=若 target.jpg 已存在就跳過（可續跑）
skip_existing=1

echo "================================================================"
echo " Face-Swap Batch P2P-Edit"
echo " face_root  : ${face_root}"
echo " identities : ${identities:-<all>}"
echo " prompt_t   : ${prompt_t}"
echo " subject    : ${subject_word}"
echo " adaface_url: ${adaface_url}"
echo " output_dir : ${output_dir}"
echo "================================================================"

python3 tools/batch_run_pie_edit_faceSwap.py \
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
  --face_root "${face_root}" \
  --output_dir "${output_dir}" \
  ${identities:+--identities "${identities}"} \
  --prompt_t "${prompt_t}" \
  --subject_word "${subject_word}" \
  --adaface_url "${adaface_url}" \
  --regen_B ${regen_B} \
  --debug_face_op ${debug_face_op} \
  --lam1 ${lam1} \
  --lam2 ${lam2} \
  --use_learned_v_A ${use_learned_v_A} \
  --identity_cache_dir "${identity_cache_dir}" \
  --skip_existing ${skip_existing} \
  --seed ${seed} \
  --absolute_high ${absolute_high} \
  --absolute_low ${absolute_low}
