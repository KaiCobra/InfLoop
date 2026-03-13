#!/bin/bash

# Loop Rollback Image Generation Pipeline
# This script runs the loop rollback workflow:
# 1. Generate a base image from the given prompt
# 2. Apply rollback at multiple scales with multiple retry counts
# 3. Merge results using the specified merge mode
#
# The rollback mechanism regenerates tokens at certain scales
# and merges them with previously generated tokens to improve
# image quality and consistency.

# Set arguments for inference
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

# Loop Rollback specific parameters
rollback_merge_mode=6            # Merge strategy: -1=disabled, 0=replace, 1=avg, 2=geometric, 3..6=weighted variants
seed=1                           # Random seed for reproducibility

# Rollback schedule (JSON format):
# Each rule: {"scale": <trigger>, "rollback_to": <target>, "times": <count>}
# Example: scale 4 → rollback to scale 2, repeat 3 times
rollback_schedule='[{"scale":4,"rollback_to":2,"times":1}]'

# Prompt
prompt="Cute Shiba Inu wearing a space helmet and holding a sign says \"excellent space\"."

# Output directory
save_dir="./outputs/Shiba_rollback/"

# Run Loop Rollback pipeline
python3 tools/run_loop.py \
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
--prompt "${prompt}" \
--seed ${seed} \
--save_file ${save_dir} \
--rollback_merge_mode ${rollback_merge_mode} \
--rollback_schedule "${rollback_schedule}"
