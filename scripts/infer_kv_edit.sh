#!/bin/bash
# ==============================================================================
# infer_kv_edit.sh
#
# KV-Edit 交錯式管線：逐 scale 縱向處理 source → phase17 → target
# 支援 KV cache 結構注入 + dynamic attention mask
#
# 執行：bash scripts/infer_kv_edit.sh
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
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
num_full_replace_scales=2
kv_blend_ratio=0.3
kv_blend_scales=8
gradient_threshold=0.3

# ── Attention 設定 ──
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
save_attn_vis=1

# ── 輸入 ──
source_image="path/to/source_image.jpg"
source_prompt="a photo of a cat sitting on a couch"
target_prompt="a photo of a dog sitting on a couch"
source_focus_words="cat"
target_focus_words="dog"
save_dir="./outputs/kv_edit_test"
seed=1

echo "================================================================"
echo " KV-Edit Interleaved Pipeline"
echo " source: ${source_image}"
echo " save:   ${save_dir}"
echo "================================================================"

python3 tools/run_kv_edit.py \
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
  --source_image "${source_image}" \
  --source_prompt "${source_prompt}" \
  --target_prompt "${target_prompt}" \
  --source_focus_words "${source_focus_words}" \
  --target_focus_words "${target_focus_words}" \
  --save_dir "${save_dir}" \
  --image_injection_scales ${image_injection_scales} \
  --inject_weights "${inject_weights}" \
  --num_full_replace_scales ${num_full_replace_scales} \
  --kv_blend_ratio ${kv_blend_ratio} \
  --kv_blend_scales ${kv_blend_scales} \
  --gradient_threshold ${gradient_threshold} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --save_attn_vis ${save_attn_vis} \
  --seed ${seed}
