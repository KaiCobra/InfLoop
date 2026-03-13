#!/bin/bash

# Prompt-to-Prompt (P2P) Image Editing Pipeline
# This script runs the p2p editing workflow:
# 1. Generate source image and save bitwise tokens
# 2. Generate target image with source token guidance

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

# P2P specific parameters
num_source_scales=4              # Number of scales to extract tokens from source
p2p_token_replace_prob=1       # Probability of replacing target tokens with source tokens
p2p_token_file="./tokens_p2p.pkl"  # File to save/load extracted tokens

# Example source and target prompts
# source_prompt="A photograph of a cat playing a guitar at a gym."
# target_prompt="A photograph of a cat playing a guitar at a Library."

# source_prompt="A train platform sign that reads \"DIVISION POINT\" as a train approaches. A male commuter wearing a dark coat stands nearby."
# target_prompt="A train platform sign that reads \"DESTINATION\" as a train approaches. A female commuter wearing a white coat stands nearby."

# source_prompt="A black dog inside a dimly lit backstage theater hallway."
# target_prompt="A orange cat inside a Brightly backstage theater hallway."

source_prompt="A cartoon style elephant holding an umbrella on the hill."
target_prompt="A cartoon style giraffe holding an umbrella on the hill."

# source_prompt="A train platform sign that reads \"PLEASE STAND BEHIND LINE\" as a train approaches."
# target_prompt="A train platform sign that reads \"DESTINATION: LONDON\" as a train approaches."

# source_prompt="A crowded subway platform at rush hour with an overhead LED sign that reads \"NEXT TRAIN\"."
# target_prompt="A crowded subway platform at rush hour with an overhead LED sign that reads \"NO ESCAPE\"."

# source_prompt="Cute Shiba Inu wearing a space helmet and pink T-shirt in a bedroom holding a sign reads \"GET OUT\"."
# target_prompt="Cute Shiba Inu wearing a space helmet and blue T-shirt in a bedroom holding a sign reads \"TEEN SPIRIT\"."


# source_prompt="""A cinematic, ultra-realistic close-up photograph of a train platform sign at blue hour. The rectangular white metal sign fills most of the frame, occupying about 70% of the image. The camera is positioned directly in front of it at eye level. The sign has bold red capital letters printed clearly across the surface, with a clean matte finish and slightly worn edges.

# It reads: \"PLEASE STAND BEHIND LINE\".

# In the shallow background, softly blurred, a train approaches with glowing headlights and subtle motion blur. Faint silhouettes of commuters stand behind a yellow safety line. Cool ambient lighting, soft mist in the air, 85mm lens, shallow depth of field, sharp focus on the sign, highly detailed, realistic lighting.
# """

# target_prompt="""A cinematic, ultra-realistic close-up photograph of a train platform sign at blue hour. The rectangular white metal sign fills most of the frame, occupying about 70% of the image. The camera is positioned directly in front of it at eye level. The sign has bold red capital letters printed clearly across the surface, with a clean matte finish and slightly worn edges.

# It reads: \"DESTINATION: LONDON\".

# In the shallow background, softly blurred, a train approaches with glowing headlights and subtle motion blur. Faint silhouettes of commuters stand behind a yellow safety line. Cool ambient lighting, soft mist in the air, 85mm lens, shallow depth of field, sharp focus on the sign, highly detailed, realistic lighting.
# """

# Run P2P pipeline
python3 tools/run_p2p.py \
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
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--source_prompt "${source_prompt}" \
--target_prompt "${target_prompt}" \
--seed 1 \
--save_file ./outputs/p2p/ \
--num_source_scales ${num_source_scales} \
--p2p_token_replace_prob ${p2p_token_replace_prob} \
--p2p_token_file ${p2p_token_file}




