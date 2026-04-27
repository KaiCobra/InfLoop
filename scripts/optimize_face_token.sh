#!/bin/bash
# ==============================================================================
# optimize_face_token.sh
#
# 對每個 identity 跑 Textual Inversion，把 prompt 中的 subject token (預設 "boy")
# 在 Infinity 自身 embedding 空間裡優化成代表該 identity 的 v_A，
# 存到 weights/identities/<id>/v_A.pt。
#
# 之後 batch / gradio face-swap 啟用 --use_learned_v_A=1 即可使用。
#
# 執行：
#   bash scripts/optimize_face_token.sh
# ==============================================================================

# ── 模型設定（同 batch_run_pie_edit_faceSwap.sh）──
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

# ── Textual-Inversion 設定 ──
face_root="./face"
identities=""                       # csv 留空=全部 face_root 子資料夾
identity_cache_dir="./weights/identities"
prompt_t="a boy turned his head to his left over the shuolder and tilted up"
subject_word="boy"
steps=200
lr=1e-3
l2_reg=1e-4
log_every=20
seed=1
regen=0                             # 1=覆寫已存在的 v_A.pt

echo "================================================================"
echo " Textual Inversion: face-swap v_A optimizer"
echo " face_root          : ${face_root}"
echo " identities         : ${identities:-<all>}"
echo " identity_cache_dir : ${identity_cache_dir}"
echo " prompt_t           : ${prompt_t}"
echo " subject_word       : ${subject_word}"
echo " steps / lr / l2    : ${steps} / ${lr} / ${l2_reg}"
echo " regen              : ${regen}"
echo "================================================================"

mkdir -p "${identity_cache_dir}"

python3 tools/optimize_face_token.py \
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
  --face_root "${face_root}" \
  ${identities:+--identities "${identities}"} \
  --identity_cache_dir "${identity_cache_dir}" \
  --prompt_t "${prompt_t}" \
  --subject_word "${subject_word}" \
  --steps ${steps} \
  --lr ${lr} \
  --l2_reg ${l2_reg} \
  --log_every ${log_every} \
  --seed ${seed} \
  --regen ${regen}
