#!/bin/bash
# ==============================================================================
# train_id_resampler.sh
#
# 訓練 IDResampler（VGGFace2HQ + Gemma4 caption 版）：
#   AdaFace e_A (512-d) → IDResampler → N 個 t5-output token，
#   覆蓋 prompt 過 T5 後 "sks" 對應位置的 last_hidden_state。
#
# 凍結 Infinity / VAE / T5；只更新 Resampler。teacher-forced bitwise CE。
#
# 資料：caption JSON 結構（一個 identity 對應一個 JSON）
#   [{ "image_path": ".../n000002/0001_01.jpg",
#      "description": "The sks woman is looking ..." }, ...]
#
# 前置：
#   • AdaFace server 在 127.0.0.1:8000：curl -s http://127.0.0.1:8000/health
#   • json_root 下遞迴有 *_gemma4_captions.json
#
# 訓出來的 ckpt 直接給 gradio_face_swap.py 用：
#   --resampler_ckpt weights/id_resampler/resampler.pt
# ==============================================================================

# ── 模型設定（與 run_gradio_face_swap.sh 對齊）──
pn=0.25M                                 # 訓練解析度建議用 0.25M（快、省 VRAM）
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

# ── Dataset ──
# 遞迴搜 *_gemma4_captions.json；每個 JSON 一個 identity，內含多筆 (image_path, description)
json_root="/media/avlab/8TB/gemma4/VGGFace2HQ_data/vgg_face_parts_extracted"
include_batch_sample=0                   # 1=連 *_gemma4_captions_batch_sample.json 也納入
max_jsons=-1                             # 隨機取前 N 個 JSON；-1=全部
max_entries=-1                           # 攤平後再隨機抽 M 個 entry；-1=全部
subject_token="sks"                      # caption 內的 placeholder
bsc_cache_dir="weights/bsc_cache"        # x_BLC/gt_BL 的 disk cache 根目錄（依 pn 分槽）
adaface_cache_dir="weights/adaface_cache"
adaface_url="http://127.0.0.1:8000"
warm_cache_only=0                        # 1=只跑一次 dataset 把 cache 填滿就結束（推薦先跑這個）

# ── Resampler 結構 ──
resampler_n_tokens=1
resampler_n_id_ctx=4
resampler_n_layers=2
resampler_n_heads=8
resampler_use_prompt_ctx=1
resampler_anchor_word="person"
# ── Resampler manifold guards（forward 內生效）──
# delta_max_norm: >0 = hard-clamp Resampler delta 的 per-token L2；-1 = 不限
# out_norm_match: 'none' | 'anchor' | 'base' = 強制 out 的 per-token norm 對齊指定目標
resampler_delta_max_norm=-1.0
resampler_out_norm_match="none"
# residual_base:
#   orig   = 學 prompt 當下 sks token 的小殘差（推薦）
#   anchor = 用 anchor word contextual emb 當基底（舊行為）
resampler_residual_base="orig"
train_scale_start=4                     # 0-based；4..6 = 第 5/6/7 個 scale
train_scale_end=6

# ── Optim ──
steps=10000
lr=1e-5
weight_decay=1e-4
grad_accum=1
grad_clip=1.0
lr_schedule="cosine"                     # cosine | constant
# direction / norm regularizers（對抗 manifold drift）
l2_anchor=1e-2                           # L2(out − anchor).mean()；0=關
cos_anchor=1.0                           # (1 − cos(out, base))；對抗方向漂移
norm_penalty=1e-1                        # (||out|| − ||base||)^2；對抗 norm 炸大
log_every=500
save_every=1000

# ── 輸入/輸出 ──
out_ckpt="weights/id_resampler/resampler.pt"
resume_ckpt=""

mkdir -p "$(dirname "${out_ckpt}")"
mkdir -p "${bsc_cache_dir}" "${adaface_cache_dir}"

echo "================================================================"
echo " Train IDResampler (VGGFace2HQ + Gemma4 captions)"
echo " json_root        : ${json_root}"
echo " include_batch    : ${include_batch_sample}"
echo " max_jsons/entries: ${max_jsons} / ${max_entries}"
echo " subject_token    : ${subject_token}"
echo " adaface_url      : ${adaface_url}"
echo " bsc_cache_dir    : ${bsc_cache_dir}"
echo " adaface_cache_dir: ${adaface_cache_dir}"
echo " warm_cache_only  : ${warm_cache_only}"
echo " out_ckpt         : ${out_ckpt}"
echo " resume_ckpt      : ${resume_ckpt:-(none)}"
echo " steps/lr/wd      : ${steps} / ${lr} / ${weight_decay}"
echo " resampler        : tokens=${resampler_n_tokens} id_ctx=${resampler_n_id_ctx}"
echo "                    layers=${resampler_n_layers} heads=${resampler_n_heads}"
echo "                    use_prompt_ctx=${resampler_use_prompt_ctx} anchor=${resampler_anchor_word}"
echo " residual_base    : ${resampler_residual_base}"
echo " train scales     : ${train_scale_start}..${train_scale_end} (0-based)"
echo "================================================================"

python3 tools/train_id_resampler.py \
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
  --json_root "${json_root}" \
  --include_batch_sample ${include_batch_sample} \
  --max_jsons ${max_jsons} \
  --max_entries ${max_entries} \
  --subject_token "${subject_token}" \
  --bsc_cache_dir "${bsc_cache_dir}" \
  --adaface_cache_dir "${adaface_cache_dir}" \
  --adaface_url "${adaface_url}" \
  --warm_cache_only ${warm_cache_only} \
  --resampler_n_tokens ${resampler_n_tokens} \
  --resampler_n_id_ctx ${resampler_n_id_ctx} \
  --resampler_n_layers ${resampler_n_layers} \
  --resampler_n_heads ${resampler_n_heads} \
  --resampler_use_prompt_ctx ${resampler_use_prompt_ctx} \
  --resampler_anchor_word "${resampler_anchor_word}" \
  --resampler_delta_max_norm ${resampler_delta_max_norm} \
  --resampler_out_norm_match "${resampler_out_norm_match}" \
  --resampler_residual_base "${resampler_residual_base}" \
  --train_scale_start ${train_scale_start} \
  --train_scale_end ${train_scale_end} \
  --resume_ckpt "${resume_ckpt}" \
  --steps ${steps} \
  --lr ${lr} \
  --weight_decay ${weight_decay} \
  --grad_accum ${grad_accum} \
  --grad_clip ${grad_clip} \
  --lr_schedule ${lr_schedule} \
  --l2_anchor ${l2_anchor} \
  --cos_anchor ${cos_anchor} \
  --norm_penalty ${norm_penalty} \
  --log_every ${log_every} \
  --save_every ${save_every} \
  --out_ckpt "${out_ckpt}"
