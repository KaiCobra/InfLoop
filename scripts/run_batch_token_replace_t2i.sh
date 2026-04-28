#!/bin/bash
# ==============================================================================
# run_batch_token_replace_t2i.sh
#
# 純 T2I + token-embedding 替換 批量測試（不走 P2P-Edit 三階段）
#   • 60 prompts × 10 identities
#   • 每個 prompt 先生一張 no-replace baseline
#   • 每個 (identity, prompt) 把 boy/girl/man/woman token 在 T5 hidden state
#     用 proj(e_A) 整段替換後再生一張
#
# 前置：AdaFace server 在 127.0.0.1:8000
# ==============================================================================

# ── Infinity 模型設定（與 gradio demo 對齊）──
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

# ── 批次設定 ──
PROMPTS_FILE="./posePrompt/t2i_pose_prompts.jsonl"
FACE_ROOT="/media/avlab/ee303_4T/faces_dataset_small"
N_FACES=10
GEN_SEED=42
FACE_PICK_SEED=0
ADAFACE_URL="http://127.0.0.1:8000"

# ── 輸出 ──
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./outputs/batch_token_replace_t2i_${TS}"

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo " Batch Token-Replacement T2I"
echo " output_dir : ${OUTPUT_DIR}"
echo " prompts    : ${PROMPTS_FILE}"
echo " face_root  : ${FACE_ROOT}  (n_faces=${N_FACES})"
echo " adaface    : ${ADAFACE_URL}"
echo "================================================================"

PYTHON_BIN="${PYTHON_BIN:-/home/avlab/anaconda3/envs/infinity-clean/bin/python}"
"${PYTHON_BIN}" tools/batch_token_replace_t2i.py \
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
  --prompts_file "${PROMPTS_FILE}" \
  --face_root "${FACE_ROOT}" \
  --n_faces ${N_FACES} \
  --output_dir "${OUTPUT_DIR}" \
  --adaface_url "${ADAFACE_URL}" \
  --gen_seed ${GEN_SEED} \
  --face_pick_seed ${FACE_PICK_SEED}
