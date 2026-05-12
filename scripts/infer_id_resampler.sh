#!/bin/bash
# ==============================================================================
# infer_id_resampler.sh
#
# 純 text-to-image inference + sks → MLP(AdaFace(face)) 替換。
# 不走 P2P-Edit、沒有 base image B；就是普通的 Infinity T2I，但 prompt 中 "sks"
# 對應的 T5 output embedding 會被 IDResampler 產出的 token 蓋掉。
#
# 前置：
#   • AdaFace server 在 127.0.0.1:8000：curl -s http://127.0.0.1:8000/health
#   • 訓練好的 IDResampler ckpt（resampler_ckpt 留空就會用初始權重，相當於沒有 ID）
#
# 執行：
#   bash scripts/infer_id_resampler.sh
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
seed=1

# ── IDResampler ──
resampler_ckpt="weights/id_resampler/resampler.pt"
resampler_n_tokens=1
resampler_n_id_ctx=4
resampler_n_layers=2
resampler_n_heads=8
resampler_use_prompt_ctx=1
resampler_anchor_word="person"
# manifold guards：ckpt config 內若有設定會覆寫這裡
resampler_delta_max_norm=-1.0
resampler_out_norm_match="none"
resampler_residual_base="orig"           # 通常 ckpt config 會覆寫；新訓練 ckpt 預期為 orig

# ── 輸入：prompt + 一張 face image ──
prompt="The sks woman is looking directly at the camera with a slight smile. The background is bright."
face_image="/media/avlab/8TB/gemma4/VGGFace2HQ_data/vgg_face_parts_extracted/1/1/n000002/0001_01.jpg"
subject_token="sks"
n_samples=4                              # 同一 (prompt, face) 連跑幾張不同 seed

# ── 輸出 ──
out_dir="outputs/id_resampler_infer"
out_prefix=""                            # 空=用 face image 檔名

# ── AdaFace ──
adaface_url="http://127.0.0.1:8000"

# ── Diagnostic / inference 行為 ──
# baseline_mode:
#   no_inject   = 不替換 sks，純 prompt → 看 pipeline 本身會不會崩
#   anchor_only = 把 sks 換成 Resampler 的 anchor（純 'person' T5 emb），跳過 Resampler
#                → 看替換機制本身是否健康
#   resampler   = 用訓練好的 Resampler 替換（現行行為）
baseline_mode="resampler"
# inject_alpha: out[sks] = (1-α)*orig + α*new
#   α=0 等同 no_inject；α=0.3-0.5 是 textual inversion 類方法的甜蜜點；α=1.0 全替換
inject_alpha=1.0
resampler_match_orig_norm=0              # 1=把 replacement 先 rescale 到 ||orig_sks|| 再 mix

# ── Sampling ──
top_k=900
top_p=0.97
cfg_insertion_layer=-5
debug=0

mkdir -p "${out_dir}"

echo "================================================================"
echo " Infer with sks → IDResampler(AdaFace) replacement"
echo " prompt        : ${prompt}"
echo " face_image    : ${face_image}"
echo " subject_token : ${subject_token}"
echo " n_samples     : ${n_samples}  seed_start=${seed}"
echo " resampler_ckpt: ${resampler_ckpt}"
echo " baseline_mode : ${baseline_mode}  inject_alpha=${inject_alpha}"
echo " residual_base : ${resampler_residual_base}  match_orig_norm=${resampler_match_orig_norm}"
echo " out_dir       : ${out_dir}"
echo "================================================================"

python3 tools/infer_id_resampler.py \
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
  --resampler_n_tokens ${resampler_n_tokens} \
  --resampler_n_id_ctx ${resampler_n_id_ctx} \
  --resampler_n_layers ${resampler_n_layers} \
  --resampler_n_heads ${resampler_n_heads} \
  --resampler_use_prompt_ctx ${resampler_use_prompt_ctx} \
  --resampler_anchor_word "${resampler_anchor_word}" \
  --resampler_delta_max_norm ${resampler_delta_max_norm} \
  --resampler_out_norm_match "${resampler_out_norm_match}" \
  --resampler_residual_base "${resampler_residual_base}" \
  --baseline_mode "${baseline_mode}" \
  --inject_alpha ${inject_alpha} \
  --resampler_match_orig_norm ${resampler_match_orig_norm} \
  --top_k ${top_k} \
  --top_p ${top_p} \
  --cfg_insertion_layer ${cfg_insertion_layer} \
  --debug ${debug}
