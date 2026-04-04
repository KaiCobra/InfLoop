#!/bin/bash
# ==============================================================================
# P2P-Edit + SAM3 動態 Threshold 管線
#
# 流程：
#   1. SAM3 segmentation：偵測 source image 中 focus 區域的 mask 面積比例
#   2. 動態計算 attn_threshold_percentile
#   3. 執行 P2P-Edit
#
# 執行方式：
#   bash scripts/infer_p2p_edit_sam3.sh
# ==============================================================================

# ── SAM3 環境 ──
SAM3_PYTHON="/home/avlab/sam3/.venv/bin/python"
SAM3_DIR="/home/avlab/sam3"
sam_boost=1.0

# ── 模型設定 ──
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
vae_type=32
vae_path=weights/infinity_vae_d32_reg.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001
text_channels=2048
apply_spatial_patchify=0

# ── P2P-Edit 核心參數 ──
source_image="./input/mona_lisa.jpg"
image_injection_scales=2
inject_weights="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
# Phase 2：留空 = 不注入 source image feature（原始行為）
phase2_inject_weights=""
num_full_replace_scales=2

# ── Attention 遮罩參數（attn_threshold_percentile 將由 SAM3 動態決定）──
attn_block_start=2
attn_block_end=-1
attn_batch_idx=0
p2p_token_replace_prob=0.0
p2p_token_file="./tokens_p2p_edit.pkl"
save_attn_vis=1

# ── Prompt 設定 ──
source_prompt="A painting of a woman with dark hair"
target_prompt="A painting of a woman with white hair"
source_focus_words="dark"
target_focus_words="white"
sam3_prompt=$source_focus_words

# ── 輸出目錄 ──
current_time=$(date +"%Y%m%d_%H%M%S")
save_file="./outputs/p2p_edit_sam3_${current_time}/"

# ==============================================================================
# Phase 1: SAM3 Segmentation → 計算 edit 區域比例
# ==============================================================================
echo "========================================"
echo "[Phase 1] Running SAM3 segmentation..."
echo "  Image: ${source_image}"
echo "  SAM3 Prompt: ${sam3_prompt}"
echo "========================================"

# 先在 InfLoop 目錄下解析絕對路徑（避免 cd 後相對路徑失效）
source_image_abs="$(realpath "${source_image}")"
save_file_abs="$(realpath -m "${save_file}")"

# 用 SAM3 的 venv 跑 basic.py 存視覺化圖 (sam_mask.jpg) 到輸出目錄
mkdir -p "${save_file_abs}"
(cd "${SAM3_DIR}" && ${SAM3_PYTHON} basic.py \
  --image "${source_image_abs}" \
  --prompt "${sam3_prompt}" \
  --save_dir "${save_file_abs}")

# 用 SAM3 的 venv 跑 segmentation，只輸出 mask 面積比例
edit_ratio=$(cd "${SAM3_DIR}" && ${SAM3_PYTHON} -c "
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

image = Image.open('${source_image_abs}')
model = build_sam3_image_model()
processor = Sam3Processor(model)
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt='${sam3_prompt}')

masks, scores = output['masks'], output['scores']
if len(masks) == 0:
    print('0.0')
else:
    # 取分數最高的 mask，計算面積比例
    best_idx = scores.argmax().item() if isinstance(scores, torch.Tensor) else int(np.argmax(scores))
    mask = masks[best_idx]
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    ratio = mask_np.sum() / mask_np.size * 100
    print(f'{ratio:.2f}')
")

if [ -z "${edit_ratio}" ]; then
    echo "[Warning] SAM3 returned empty result, using default threshold=80"
    edit_ratio="20.0"
fi

echo "[Phase 1 Result] Edit region ratio: ${edit_ratio}%"

# ==============================================================================
# Phase 2: 動態計算 attn_threshold_percentile
# ==============================================================================
# 邏輯：
#   - edit 區域佔比小（如 5%）→ threshold 高（如 95）→ 精準聚焦小區域
#   - edit 區域佔比大（如 30%）→ threshold 低（如 70）→ 允許更廣的改動
#   - 公式：threshold = 100 - edit_ratio，clamp 到 [70, 98]

attn_threshold_percentile=$(python3 -c "
ratio = float(${edit_ratio})
threshold = 100.0 - ratio
if threshold == 100:
    threshold = 85
print(threshold)
")

echo "========================================"
echo "[Phase 2] Dynamic threshold calculation"
echo "  Edit ratio: ${edit_ratio}%"
echo "  → attn_threshold_percentile: ${attn_threshold_percentile}"
echo "========================================"

# ==============================================================================
# Phase 3: 執行 P2P-Edit
# ==============================================================================
echo "========================================"
echo "[Phase 3] Running P2P-Edit..."
echo "========================================"

/home/avlab/anaconda3/envs/infinity-clean/bin/python tools/run_p2p_edit_fuse.py \
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
  --source_prompt "${source_prompt}" \
  --target_prompt "${target_prompt}" \
  --source_focus_words "${source_focus_words}" \
  --target_focus_words "${target_focus_words}" \
  --source_image "${source_image}" \
  --image_injection_scales ${image_injection_scales} \
  --inject_weights "${inject_weights}" \
  --phase2_inject_weights "${phase2_inject_weights}" \
  --num_full_replace_scales ${num_full_replace_scales} \
  --attn_threshold_percentile ${attn_threshold_percentile} \
  --attn_block_start ${attn_block_start} \
  --attn_block_end ${attn_block_end} \
  --attn_batch_idx ${attn_batch_idx} \
  --p2p_token_replace_prob ${p2p_token_replace_prob} \
  --p2p_token_file ${p2p_token_file} \
  --save_attn_vis ${save_attn_vis} \
  --sam_mask "${save_file_abs}/sam_binary.png" \
  --sam_boost ${sam_boost} \
  --save_file ${save_file} \
  --seed 1

echo "========================================"
echo "[Done] Output saved to: ${save_file}"
echo "  Used attn_threshold_percentile: ${attn_threshold_percentile}"
echo "  Edit region ratio from SAM3: ${edit_ratio}%"
echo "========================================"
