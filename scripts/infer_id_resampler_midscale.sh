#!/bin/bash
# ==============================================================================
# Mid-scale IDResampler inference:
#   - id_scale_start..id_scale_end: sks → IDResampler(AdaFace)
#   - other scales: remove sks token from text condition
# ==============================================================================

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
seed=1

resampler_ckpt="weights/id_resampler/resampler_step010000.pt"
# 自動掃所有中途 checkpoint；找不到 glob 時 fallback 到 resampler_ckpt。
resampler_ckpt_glob="weights/id_resampler/resampler_step*.pt"
resampler_n_tokens=1
resampler_n_id_ctx=4
resampler_n_layers=2
resampler_n_heads=8
resampler_use_prompt_ctx=1
resampler_anchor_word="person"
resampler_delta_max_norm=-1.0
resampler_out_norm_match="none"
resampler_residual_base="orig"
resampler_match_orig_norm=0

# 0-based；4..6 = 第 5/6/7 個 scale
id_scale_start=4
id_scale_end=6
inject_alpha=1.0

prompt="The sks woman is looking directly at the camera with a slight smile. The background is bright."
face_image="/media/avlab/8TB/gemma4/VGGFace2HQ_data/vgg_face_parts_extracted/1/1/n000002/0001_01.jpg"
subject_token="sks"
n_samples=4

out_dir="outputs/id_resampler_midscale"
out_prefix=""
adaface_url="http://127.0.0.1:8000"

top_k=900
top_p=0.97
cfg_insertion_layer=-5

mkdir -p "${out_dir}"

echo "================================================================"
echo " Mid-scale IDResampler inference"
echo " prompt        : ${prompt}"
echo " face_image    : ${face_image}"
echo " ckpt glob     : ${resampler_ckpt_glob}"
echo " id scales     : ${id_scale_start}..${id_scale_end} (0-based)"
echo " non-id scales : remove ${subject_token}"
echo " out_dir       : ${out_dir}"
echo "================================================================"

python3 tools/infer_id_resampler_midscale.py \
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
  --seed ${seed} \
  --prompt "${prompt}" \
  --face_image "${face_image}" \
  --subject_token "${subject_token}" \
  --n_samples ${n_samples} \
  --out_dir "${out_dir}" \
  --out_prefix "${out_prefix}" \
  --adaface_url "${adaface_url}" \
  --resampler_ckpt "${resampler_ckpt}" \
  --resampler_ckpt_glob "${resampler_ckpt_glob}" \
  --resampler_n_tokens ${resampler_n_tokens} \
  --resampler_n_id_ctx ${resampler_n_id_ctx} \
  --resampler_n_layers ${resampler_n_layers} \
  --resampler_n_heads ${resampler_n_heads} \
  --resampler_use_prompt_ctx ${resampler_use_prompt_ctx} \
  --resampler_anchor_word "${resampler_anchor_word}" \
  --resampler_delta_max_norm ${resampler_delta_max_norm} \
  --resampler_out_norm_match "${resampler_out_norm_match}" \
  --resampler_residual_base "${resampler_residual_base}" \
  --resampler_match_orig_norm ${resampler_match_orig_norm} \
  --id_scale_start ${id_scale_start} \
  --id_scale_end ${id_scale_end} \
  --inject_alpha ${inject_alpha} \
  --top_k ${top_k} \
  --top_p ${top_p} \
  --cfg_insertion_layer ${cfg_insertion_layer} 2>&1 | tee "${out_dir}/infer_all.log"
