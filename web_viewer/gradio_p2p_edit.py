"""
P2P-Edit Gradio Demo — 模型常駐記憶體，互動式圖像編輯

啟動方式：
    cd /home/avlab/Documents/InfLoop
    python web_viewer/gradio_p2p_edit.py

模型只在啟動時載入一次，之後每次互動都直接推論。
"""

import os
import sys
import time
import datetime
import zipfile

# 將專案根目錄加入 Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image
from torch.cuda.amp import autocast

torch._dynamo.config.cache_size_limit = 64

# ── 從 run_p2p_edit.py 匯入核心元件 ──
from tools.run_p2p_edit import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
    encode_image_to_raw_features,
    encode_image_to_scale_tokens,
    find_focus_token_indices,
    parse_focus_words_arg,
    merge_focus_terms,
    derive_focus_terms_from_prompt_diff,
    collect_attention_text_masks,
    combine_and_store_masks,
    build_cumulative_replacement_prob_masks,
    CrossAttentionExtractor,
    _iqr_filtered_mean,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# ============================================================
# 固定模型參數（與 infer_p2p_edit.sh 一致）
# ============================================================
MODEL_CONFIG = dict(
    pn='1M',
    model_type='infinity_2b',
    model_path='weights/infinity_2b_reg.pth',
    vae_type=32,
    vae_path='weights/infinity_vae_d32reg.pth',
    cfg=4.0,
    tau=0.5,
    text_encoder_ckpt='weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001',
    text_channels=2048,
    use_scale_schedule_embedding=0,
    use_bit_label=1,
    add_lvl_embeding_only_first_block=1,
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    apply_spatial_patchify=0,
    checkpoint_type='torch',
    cfg_insertion_layer=0,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    bf16=1,
)

# ============================================================
# 全域模型物件（啟動時載入一次）
# ============================================================
text_tokenizer = None
text_encoder = None
vae = None
infinity = None


def load_models():
    """載入所有模型到 GPU，只呼叫一次。"""
    global text_tokenizer, text_encoder, vae, infinity

    print("=" * 60)
    print("[Gradio] 載入模型中 ...")
    print("=" * 60)

    # 建立 args-like 物件供 load 函式使用
    class Args:
        pass
    args = Args()
    for k, v in MODEL_CONFIG.items():
        setattr(args, k, v)
    args.cache_dir = '/dev/shm'
    args.enable_model_cache = 0
    args.use_flex_attn = 0
    args.h_div_w_template = 1.0

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    print("=" * 60)
    print("[Gradio] 模型載入完成！")
    print("=" * 60)


# ============================================================
# 推論主函式
# ============================================================

def decode_scale_images(vae, summed_codes_list):
    """將各 scale 的 summed_codes 解碼為 RGB PIL Image 列表。"""
    images = []
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            for codes in summed_codes_list:
                codes_gpu = codes.to('cuda')
                decoded = vae.decode(codes_gpu.squeeze(-3))  # [1, 3, H, W], range [-1, 1]
                img = decoded[0].permute(1, 2, 0).float().cpu().numpy()
                img = (img + 1) / 2  # [-1,1] → [0,1]
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                images.append(Image.fromarray(img))
    return images


def make_attention_map_images(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """Post-IQR attention heatmap（與 mask 計算一致）。委派給 make_postiqr_attn_images。"""
    return make_postiqr_attn_images(extractor, focus_token_indices, scale_schedule, attn_block_indices)


def _collect_per_scale_spatial(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """
    內部共用：從 extractor 取各 scale 的 per-block spatial attention（shape: H×W ndarray）。
    回傳 {si: [(H,W), ...]}，每個 si 對應所有有效 block 的 attention 列表。
    """
    result = {}
    for si, (_, h, w) in enumerate(scale_schedule):
        blocks = []
        for block_idx in attn_block_indices:
            block_maps = extractor.attention_maps.get(block_idx, [])
            if si >= len(block_maps):
                continue
            attn_tensor = block_maps[si]          # [1, num_heads, L, k_len]
            attn_agg = attn_tensor.mean(dim=1)[0] # [L, k_len]
            focus_indices = [i for i in focus_token_indices if i < attn_agg.shape[-1]]
            if not focus_indices:
                continue
            word_attn = attn_agg[:, focus_indices].mean(dim=-1).float()  # [L]
            expected_L = h * w
            if word_attn.shape[0] == expected_L:
                spatial = word_attn.numpy().reshape(h, w)
            else:
                side = int(word_attn.shape[0] ** 0.5)
                if side * side == word_attn.shape[0]:
                    spatial = word_attn.numpy().reshape(side, side)
                else:
                    continue
            blocks.append(spatial)
        if blocks:
            result[si] = blocks
    return result


def _attn_to_heatmap(attn_map, h, w):
    """(H,W) float → PIL RGB jet heatmap，nearest upscale × 16。"""
    mn, mx = attn_map.min(), attn_map.max()
    normalized = (attn_map - mn) / (mx - mn + 1e-8)
    upscaled = cv2.resize(normalized.astype(np.float32), (w * 16, h * 16),
                          interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((upscaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def _attn_to_value_image(attn_map, h, w, target_size=896, max_cell=72):
    """
    將 (H,W) attention 矩陣轉成「帶數值標註」的 PIL 圖。
    每個 cell 上會疊上對應的 normalized 數值（0.000–1.000，2~3 位小數）。

    Args:
        attn_map: (H, W) ndarray float — raw attention values
        h, w: token grid 的高寬（與 attn_map.shape 相同）
        target_size: 期望輸出影像的最大邊長（px），用來決定 cell_px
        max_cell: 單格最大 px（避免小 scale 過度放大）

    Returns:
        PIL.Image RGB
    """
    if attn_map is None or attn_map.size == 0:
        return Image.fromarray(np.zeros((h * 16, w * 16, 3), dtype=np.uint8))

    # 自適應 cell 大小：大 scale 縮小、小 scale 放大
    cell_px = max(8, min(max_cell, target_size // max(h, w)))
    H_px, W_px = h * cell_px, w * cell_px

    mn = float(attn_map.min())
    mx = float(attn_map.max())
    normalized = (attn_map - mn) / (mx - mn + 1e-8)

    upscaled = cv2.resize(normalized.astype(np.float32), (W_px, H_px),
                          interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap((upscaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # cell 太小時不疊文字（會看不到、且太密集）
    if cell_px < 18:
        return Image.fromarray(img_rgb)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # 自動字體大小：每 cell ~ 0.25–0.55
    font_scale = max(0.3, min(0.6, cell_px / 90.0))
    thickness = max(1, cell_px // 32)
    decimals = 3 if cell_px >= 36 else 2

    for i in range(h):
        for j in range(w):
            v = float(normalized[i, j])
            text = f"{v:.{decimals}f}"
            (tw, th_), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cx = j * cell_px + (cell_px - tw) // 2
            cy = i * cell_px + (cell_px + th_) // 2
            # 黑色描邊 + 白色填字，保證在 jet 任何顏色上都清楚
            cv2.putText(img_rgb, text, (cx, cy), font, font_scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img_rgb, text, (cx, cy), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    return Image.fromarray(img_rgb)


def _compute_preiqr_matrices(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """
    回傳 {si: (H, W) ndarray float}，pre-IQR：所有 block 直接平均（naive mean）。
    """
    if not focus_token_indices or not extractor or not extractor.attention_maps:
        return {}
    per_scale = _collect_per_scale_spatial(extractor, focus_token_indices,
                                           scale_schedule, attn_block_indices)
    matrices = {}
    for si in per_scale:
        if not per_scale[si]:
            continue
        stack = np.stack(per_scale[si], axis=0)  # (num_blocks, H, W)
        matrices[si] = stack.mean(axis=0).astype(np.float32)
    return matrices


def _compute_postiqr_matrices(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """
    回傳 {si: (H, W) ndarray float}，post-IQR：移除離群 block 後平均（與 mask 計算一致）。
    """
    if not focus_token_indices or not extractor or not extractor.attention_maps:
        return {}
    per_scale = _collect_per_scale_spatial(extractor, focus_token_indices,
                                           scale_schedule, attn_block_indices)
    matrices = {}
    for si in per_scale:
        if not per_scale[si]:
            continue
        stack_t = torch.tensor(np.stack(per_scale[si], axis=0), dtype=torch.float32)
        filtered, _, _ = _iqr_filtered_mean(stack_t)  # (H, W) ndarray
        matrices[si] = np.asarray(filtered, dtype=np.float32)
    return matrices


def make_attn_value_images_from_matrices(matrices, scale_schedule):
    """
    將 {si: (H, W) ndarray} 轉成逐 scale 的 PIL 數值標註圖。
    沒有矩陣的 scale 用全黑佔位。
    """
    images = []
    for si, (_, h, w) in enumerate(scale_schedule):
        if si in matrices and matrices[si] is not None:
            images.append(_attn_to_value_image(matrices[si], h, w))
        else:
            images.append(Image.fromarray(np.zeros((h * 16, w * 16, 3), dtype=np.uint8)))
    return images


def save_attn_value_txts(matrices, scale_schedule, save_dir, prefix):
    """
    將每個 scale 的原始 attention 矩陣存成 .txt（高精度科學記號），
    方便事後檢查精確數值。回傳 [path, ...]。
    """
    paths = []
    for si, (_, h, w) in enumerate(scale_schedule):
        if si not in matrices or matrices[si] is None:
            continue
        m = np.asarray(matrices[si], dtype=np.float64)
        path = os.path.join(save_dir, f"{prefix}_scale_{si:02d}_{h}x{w}.txt")
        mn, mx = float(m.min()), float(m.max())
        header = (
            f"scale_index={si}  shape=({h},{w})  "
            f"min={mn:.6e}  max={mx:.6e}  mean={float(m.mean()):.6e}\n"
            f"raw attention values (row-major):"
        )
        np.savetxt(path, m, fmt="%.6e", header=header)
        paths.append(path)
    return paths


def make_preiqr_attn_images(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """
    IQR 過濾前的 attention heatmap：所有 block 直接平均（naive mean），
    不做 IQR outlier 移除。
    """
    if not focus_token_indices or not extractor.attention_maps:
        return []
    per_scale = _collect_per_scale_spatial(extractor, focus_token_indices,
                                           scale_schedule, attn_block_indices)
    images = []
    for si, (_, h, w) in enumerate(scale_schedule):
        if si not in per_scale or not per_scale[si]:
            images.append(Image.fromarray(np.zeros((h * 16, w * 16, 3), dtype=np.uint8)))
            continue
        stack = np.stack(per_scale[si], axis=0)   # (num_blocks, H, W)
        mean_map = stack.mean(axis=0)              # simple mean，無 IQR
        images.append(_attn_to_heatmap(mean_map, h, w))
    return images


def make_postiqr_attn_images(extractor, focus_token_indices, scale_schedule, attn_block_indices):
    """
    IQR 過濾後（但尚未 threshold）的 attention heatmap：
    以 _iqr_filtered_mean 移除離群 block 再平均，與 mask 計算所用的 attention 一致。
    """
    if not focus_token_indices or not extractor.attention_maps:
        return []
    per_scale = _collect_per_scale_spatial(extractor, focus_token_indices,
                                           scale_schedule, attn_block_indices)
    images = []
    for si, (_, h, w) in enumerate(scale_schedule):
        if si not in per_scale or not per_scale[si]:
            images.append(Image.fromarray(np.zeros((h * 16, w * 16, 3), dtype=np.uint8)))
            continue
        stack_t = torch.tensor(np.stack(per_scale[si], axis=0), dtype=torch.float32)
        filtered, n_out, n_use = _iqr_filtered_mean(stack_t)  # (H, W) ndarray
        images.append(_attn_to_heatmap(filtered, h, w))
    return images


def make_attention_mask_images(text_masks, scale_schedule):
    """
    將各 scale 的 bool attention mask 轉成 PIL Image 列表。
    白色 = focus 區域（True），黑色 = 背景（False）。
    每個 scale 的空間大小 upscale 到像素空間（nearest，每 token 16 像素）。
    沒有 mask 的 scale 填充黑色圖。
    """
    images = []
    for si, (_, h, w) in enumerate(scale_schedule):
        H_px, W_px = h * 16, w * 16
        if si in text_masks:
            mask = text_masks[si].astype(np.uint8) * 255  # bool → 0/255
            mask_up = cv2.resize(mask, (W_px, H_px), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(mask_up, cv2.COLOR_GRAY2RGB)
        else:
            rgb = np.zeros((H_px, W_px, 3), dtype=np.uint8)
        images.append(Image.fromarray(rgb))
    return images


def make_zip(save_dir, zip_name, file_paths):
    """將 file_paths 列表打包成 ZIP，回傳 ZIP 路徑。"""
    zip_path = os.path.join(save_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for fpath in file_paths:
            zf.write(fpath, os.path.basename(fpath))
    return zip_path


def run_p2p_edit(
    source_image_pil,
    source_prompt,
    target_prompt,
    source_focus_words,
    target_focus_words,
    source_keep_words,
    inject_weights_str,
    num_full_replace_scales,
    attn_threshold_percentile,
    threshold_method,
    use_cumulative_prob_mask,
    h_div_w_template,
    seed,
    show_attn_vis=False,
    cv_min=0.0,
    cv_max=1.0,
    k_min=0.2,
    k_max=0.5,
    absolute_high=0.7,
    absolute_low=0.3,
):
    """執行一次 P2P-Edit 推論。
    回傳 (source, target,
           vae, preiqr_attn, postiqr_attn, mask,
           preiqr_val, postiqr_val,
           vae_zip, preiqr_zip, attn_zip, mask_zip,
           preiqr_val_zip, postiqr_val_zip, attn_txt_zip,
           p17_vae, p17_preiqr_attn, p17_postiqr_attn, p17_mask,
           p17_preiqr_val, p17_postiqr_val,
           p17_vae_zip, p17_preiqr_zip, p17_attn_zip, p17_mask_zip,
           p17_preiqr_val_zip, p17_postiqr_val_zip, p17_attn_txt_zip,
           status)
    """
    _empty = (None, None,
              [], [], [], [],
              [], [],
              None, None, None, None,
              None, None, None,
              [], [], [], [],
              [], [],
              None, None, None, None,
              None, None, None,
              "請上傳 source image。")

    # 無 source image → fallback 到 p2p_attn 模式（純 prompt 驅動，不做 image injection）
    attn_fallback_mode = (source_image_pil is None)

    t_start = time.time()

    # ── 參數準備 ──
    cfg = MODEL_CONFIG['cfg']
    tau = MODEL_CONFIG['tau']
    pn = MODEL_CONFIG['pn']
    seed = int(seed)
    num_full_replace_scales = int(num_full_replace_scales)
    threshold_method = int(threshold_method)
    phase17_fallback_replace_scales = 4
    attn_block_start = 2
    attn_block_end = -1

    # Scale schedule
    h_div_w = float(h_div_w_template)
    scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)

    # Attention block 範圍
    depth = len(infinity.unregistered_blocks)
    abs_start = (depth // 2) if attn_block_start < 0 else min(attn_block_start, depth - 1)
    abs_end = (depth - 1) if attn_block_end < 0 else min(attn_block_end, depth - 1)
    attn_block_indices = list(range(abs_start, abs_end + 1))

    # Focus / keep words
    source_focus_words_list = parse_focus_words_arg(source_focus_words)
    target_focus_words_list = parse_focus_words_arg(target_focus_words)
    source_keep_words_list  = parse_focus_words_arg(source_keep_words)

    # Auto diff
    auto_src, auto_tgt = derive_focus_terms_from_prompt_diff(source_prompt, target_prompt)
    source_focus_words_list = merge_focus_terms(source_focus_words_list, auto_src)
    target_focus_words_list = merge_focus_terms(target_focus_words_list, auto_tgt)

    source_focus_token_indices = find_focus_token_indices(
        text_tokenizer, source_prompt, source_focus_words_list, verbose=True,
    ) if source_focus_words_list else []
    target_focus_token_indices = find_focus_token_indices(
        text_tokenizer, target_prompt, target_focus_words_list, verbose=True,
    ) if target_focus_words_list else []
    source_keep_token_indices = find_focus_token_indices(
        text_tokenizer, source_prompt, source_keep_words_list, verbose=True,
    ) if source_keep_words_list else []

    # 需要 patch attention 的 token 集合（focus + keep）
    need_attn = bool(source_focus_token_indices or source_keep_token_indices)

    device_cuda = torch.device('cuda')

    # ── Phase 0：編碼 Source Image（attn fallback 模式無圖，全部跳過）──
    if attn_fallback_mode:
        source_pil = None
        image_raw_features = None
        image_scale_tokens = {}
        inject_schedule = None
    else:
        source_pil = source_image_pil.convert('RGB')
        image_raw_features = encode_image_to_raw_features(
            vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
            device=device_cuda, apply_spatial_patchify=False,
        )
        image_scale_tokens = encode_image_to_scale_tokens(
            vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
            device=device_cuda, apply_spatial_patchify=False,
        )

        # Inject schedule：優先使用手動填入的 inject_weights；否則全 0（100% source image）
        inject_weights_str = inject_weights_str.strip()
        if inject_weights_str:
            parsed_w = [float(x) for x in inject_weights_str.split()]
            if len(parsed_w) != total_scales:
                err = list(_empty)
                err[-1] = (
                    f"inject_weights 長度 {len(parsed_w)} 與 scale 總數 {total_scales} 不符。"
                )
                return tuple(err)
            inject_schedule = parsed_w
        else:
            inject_schedule = [0.0] * total_scales  # 全 scale 100% source image

    # ── Phase 1：Source 生成 + Attention 擷取 ──
    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    source_extractor = CrossAttentionExtractor(
        model=infinity, block_indices=attn_block_indices,
        batch_idx=0, aggregate_method="mean",
    )
    if need_attn:
        source_extractor.register_patches()

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_img_tensor = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=cfg, tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                vae_type=MODEL_CONFIG['vae_type'],
                sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    if need_attn:
        source_extractor.remove_patches()

    # ── Phase 1.5：Source Focus Mask ──
    source_text_masks = {}
    source_low_attn_masks = {}
    source_keep_masks = {}
    if len(source_extractor.attention_maps) > 0:
        if source_focus_token_indices:
            source_text_masks = collect_attention_text_masks(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                threshold_percentile=attn_threshold_percentile,
                threshold_method=threshold_method,
                label="source", low_attn=False,
                cv_min=cv_min, cv_max=cv_max, k_min=k_min, k_max=k_max,
                absolute_high=absolute_high, absolute_low=absolute_low,
            )
            source_low_attn_masks = collect_attention_text_masks(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                threshold_percentile=attn_threshold_percentile,
                threshold_method=threshold_method,
                label="source_preserve", low_attn=True,
                cv_min=cv_min, cv_max=cv_max, k_min=k_min, k_max=k_max,
                absolute_high=absolute_high, absolute_low=absolute_low,
            )
        if source_keep_token_indices:
            source_keep_masks = collect_attention_text_masks(
                extractor=source_extractor,
                focus_token_indices=source_keep_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                threshold_percentile=attn_threshold_percentile,
                threshold_method=threshold_method,
                label="source_keep", low_attn=False,
                cv_min=cv_min, cv_max=cv_max, k_min=k_min, k_max=k_max,
                absolute_high=absolute_high, absolute_low=absolute_low,
            )

    # ── Phase 1.6：Preserve Storage（低 attention 背景 + keep words）──
    phase17_storage = None
    has_low_attn_preserve = bool(source_low_attn_masks and image_scale_tokens)
    has_keep_preserve     = bool(source_keep_masks and p2p_token_storage.tokens)

    if has_low_attn_preserve or has_keep_preserve:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

        if has_low_attn_preserve:
            for si, low_mask in source_low_attn_masks.items():
                if si not in image_scale_tokens:
                    continue
                phase17_storage.tokens[si] = image_scale_tokens[si].clone()
                phase17_storage.masks[si] = torch.tensor(
                    low_mask, dtype=torch.bool
                ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()

        if has_keep_preserve:
            for si, keep_mask in source_keep_masks.items():
                if si not in p2p_token_storage.tokens:
                    continue
                if si in phase17_storage.masks:
                    existing = phase17_storage.masks[si].squeeze().numpy().astype(bool)
                    combined = existing | keep_mask
                    phase17_storage.masks[si] = torch.tensor(
                        combined, dtype=torch.bool
                    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()
                else:
                    phase17_storage.tokens[si] = p2p_token_storage.tokens[si].clone()
                    phase17_storage.masks[si] = torch.tensor(
                        keep_mask, dtype=torch.bool
                    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()

    # ── Phase 1.7：Target 生成 → Target Focus Mask ──
    target_text_masks = {}
    p17_summed_codes_list = []   # Phase 1.7 逐 scale VAE codes 快照
    p17_target_extractor = None  # Phase 1.7 attention extractor（用於事後視覺化）
    if target_focus_token_indices:
        target_extractor = CrossAttentionExtractor(
            model=infinity, block_indices=attn_block_indices,
            batch_idx=0, aggregate_method="mean",
        )
        target_extractor.register_patches()

        _phase17_storage = phase17_storage
        _phase17_use_mask = (phase17_storage is not None)
        _phase17_full_replace = num_full_replace_scales
        if _phase17_storage is None and phase17_fallback_replace_scales > 0 and p2p_token_storage.tokens:
            _phase17_storage = p2p_token_storage
            _phase17_use_mask = False
            _phase17_full_replace = phase17_fallback_replace_scales

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _phase17_img = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder,
                    target_prompt,
                    g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                    cfg_list=cfg, tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                    vae_type=MODEL_CONFIG['vae_type'],
                    sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                    enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                    p2p_token_storage=_phase17_storage,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=_phase17_use_mask,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=_phase17_full_replace,
                    inject_image_features=None,
                    inject_schedule=None,
                )
        # Phase 1.7 結束，在 Phase 2 覆寫前先把 codes 快照存起來
        p17_summed_codes_list = list(getattr(infinity, 'summed_codes_list_for_vis', []))
        del _phase17_img

        target_extractor.remove_patches()
        p17_target_extractor = target_extractor
        target_text_masks = collect_attention_text_masks(
            extractor=target_extractor,
            focus_token_indices=target_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=attn_threshold_percentile,
            threshold_method=threshold_method,
            label="target",
            cv_min=cv_min, cv_max=cv_max, k_min=k_min, k_max=k_max,
            absolute_high=absolute_high, absolute_low=absolute_low,
        )
        torch.cuda.empty_cache()

    # ── Phase 1.9：合併 Mask + 覆寫 storage ──
    if source_text_masks or target_text_masks:
        combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=num_full_replace_scales,
        )

    if use_cumulative_prob_mask and p2p_token_storage.masks:
        p2p_token_storage.masks = build_cumulative_replacement_prob_masks(
            masks=p2p_token_storage.masks,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
        )

    if image_scale_tokens:
        for si_tok, tok in image_scale_tokens.items():
            p2p_token_storage.tokens[si_tok] = tok

    # attn fallback：source 是 Phase 1 生成出來的，轉成 PIL 供顯示/存檔
    if attn_fallback_mode:
        src_np = source_img_tensor.cpu().numpy()
        if src_np.dtype != np.uint8:
            src_np = np.clip(src_np, 0, 255).astype(np.uint8)
        source_pil = Image.fromarray(cv2.cvtColor(src_np, cv2.COLOR_BGR2RGB))

    del source_img_tensor
    torch.cuda.empty_cache()

    # ── Phase 2：Target 生成 ──
    has_mask = len(p2p_token_storage.masks) > 0

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_img_tensor = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=cfg, tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                vae_type=MODEL_CONFIG['vae_type'],
                sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=has_mask,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=num_full_replace_scales,
                inject_image_features=None,
                inject_schedule=None,
            )

    # ── 輸出圖片轉換 ──
    target_np = target_img_tensor.cpu().numpy()
    if target_np.dtype != np.uint8:
        target_np = np.clip(target_np, 0, 255).astype(np.uint8)
    # OpenCV BGR → RGB for Gradio
    target_rgb = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)

    # ── 逐 scale BSQ-VAE 解碼圖 ──
    scale_imgs = []
    summed_codes_list = getattr(infinity, 'summed_codes_list_for_vis', [])
    if summed_codes_list:
        scale_imgs = decode_scale_images(vae, summed_codes_list)

    # ── 視覺化（Attention Map / Mask），可由 show_attn_vis 控制是否計算 ──
    preiqr_attn_imgs = []
    attn_imgs        = []
    preiqr_val_imgs  = []
    postiqr_val_imgs = []
    mask_imgs        = []
    p17_scale_imgs       = []
    p17_preiqr_attn_imgs = []
    p17_attn_imgs        = []
    p17_preiqr_val_imgs  = []
    p17_postiqr_val_imgs = []
    p17_mask_imgs        = []
    preiqr_matrices_src  = {}
    postiqr_matrices_src = {}
    p17_preiqr_matrices  = {}
    p17_postiqr_matrices = {}

    if show_attn_vis:
        # ── 計算 Source 端 Pre/Post-IQR attention matrices ──
        preiqr_matrices_src = _compute_preiqr_matrices(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
        )
        postiqr_matrices_src = _compute_postiqr_matrices(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
        )

        preiqr_attn_imgs = make_preiqr_attn_images(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
        )
        attn_imgs = make_attention_map_images(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
        )
        preiqr_val_imgs  = make_attn_value_images_from_matrices(preiqr_matrices_src,  scale_schedule)
        postiqr_val_imgs = make_attn_value_images_from_matrices(postiqr_matrices_src, scale_schedule)
        mask_imgs = make_attention_mask_images(source_text_masks, scale_schedule)

        # ── Phase 1.7 視覺化 ──
        p17_scale_imgs = decode_scale_images(vae, p17_summed_codes_list) if p17_summed_codes_list else []
        if p17_target_extractor is not None:
            p17_preiqr_matrices = _compute_preiqr_matrices(
                extractor=p17_target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
            )
            p17_postiqr_matrices = _compute_postiqr_matrices(
                extractor=p17_target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
            )
            p17_preiqr_attn_imgs = make_preiqr_attn_images(
                extractor=p17_target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
            )
            p17_attn_imgs = make_attention_map_images(
                extractor=p17_target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
            )
        p17_preiqr_val_imgs  = make_attn_value_images_from_matrices(p17_preiqr_matrices,  scale_schedule)
        p17_postiqr_val_imgs = make_attn_value_images_from_matrices(p17_postiqr_matrices, scale_schedule)
        p17_mask_imgs = make_attention_mask_images(target_text_masks, scale_schedule)

    # ── 同時存檔 ──
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "outputs", f"gradio_p2p_edit_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    source_pil.save(os.path.join(
        save_dir,
        "source_generated.jpg" if attn_fallback_mode else "source_input.jpg",
    ))
    cv2.imwrite(os.path.join(save_dir, "target.jpg"), target_np)
    def _save_imgs(imgs, pattern, paths_list):
        for si, img in enumerate(imgs):
            p = os.path.join(save_dir, pattern.format(si=si))
            img.save(p); paths_list.append(p)

    vae_paths = []
    _save_imgs(scale_imgs, "scale_{si:02d}_vae.jpg", vae_paths)
    vae_zip = make_zip(save_dir, "vae_scales.zip", vae_paths) if vae_paths else None

    # Attention / Mask 存檔（僅在 show_attn_vis 時才有圖可存）
    preiqr_zip = attn_zip = mask_zip = None
    preiqr_val_zip = postiqr_val_zip = attn_txt_zip = None
    p17_vae_zip = p17_preiqr_zip = p17_attn_zip = p17_mask_zip = None
    p17_preiqr_val_zip = p17_postiqr_val_zip = p17_attn_txt_zip = None

    if show_attn_vis:
        preiqr_paths, attn_paths, mask_paths = [], [], []
        preiqr_val_paths, postiqr_val_paths = [], []
        _save_imgs(preiqr_attn_imgs,  "scale_{si:02d}_attn_preiqr.jpg",           preiqr_paths)
        _save_imgs(attn_imgs,         "scale_{si:02d}_attn_postiqr.jpg",          attn_paths)
        _save_imgs(mask_imgs,         "scale_{si:02d}_mask.png",                  mask_paths)
        _save_imgs(preiqr_val_imgs,   "scale_{si:02d}_attn_preiqr_values.png",    preiqr_val_paths)
        _save_imgs(postiqr_val_imgs,  "scale_{si:02d}_attn_postiqr_values.png",   postiqr_val_paths)

        attn_txt_paths = []
        attn_txt_paths += save_attn_value_txts(preiqr_matrices_src,  scale_schedule, save_dir, "attn_preiqr_raw")
        attn_txt_paths += save_attn_value_txts(postiqr_matrices_src, scale_schedule, save_dir, "attn_postiqr_raw")

        preiqr_zip      = make_zip(save_dir, "attn_preiqr_maps.zip",      preiqr_paths)      if preiqr_paths      else None
        attn_zip        = make_zip(save_dir, "attn_postiqr_maps.zip",     attn_paths)        if attn_paths        else None
        mask_zip        = make_zip(save_dir, "attn_masks.zip",            mask_paths)        if mask_paths        else None
        preiqr_val_zip  = make_zip(save_dir, "attn_preiqr_values.zip",    preiqr_val_paths)  if preiqr_val_paths  else None
        postiqr_val_zip = make_zip(save_dir, "attn_postiqr_values.zip",   postiqr_val_paths) if postiqr_val_paths else None
        attn_txt_zip    = make_zip(save_dir, "attn_raw_matrices.zip",     attn_txt_paths)    if attn_txt_paths    else None

        p17_vae_paths, p17_preiqr_paths, p17_attn_paths, p17_mask_paths = [], [], [], []
        p17_preiqr_val_paths, p17_postiqr_val_paths = [], []
        _save_imgs(p17_scale_imgs,         "p17_scale_{si:02d}_vae.jpg",                  p17_vae_paths)
        _save_imgs(p17_preiqr_attn_imgs,   "p17_scale_{si:02d}_attn_preiqr.jpg",          p17_preiqr_paths)
        _save_imgs(p17_attn_imgs,          "p17_scale_{si:02d}_attn_postiqr.jpg",         p17_attn_paths)
        _save_imgs(p17_mask_imgs,          "p17_scale_{si:02d}_mask.png",                 p17_mask_paths)
        _save_imgs(p17_preiqr_val_imgs,    "p17_scale_{si:02d}_attn_preiqr_values.png",   p17_preiqr_val_paths)
        _save_imgs(p17_postiqr_val_imgs,   "p17_scale_{si:02d}_attn_postiqr_values.png",  p17_postiqr_val_paths)

        p17_attn_txt_paths = []
        p17_attn_txt_paths += save_attn_value_txts(p17_preiqr_matrices,  scale_schedule, save_dir, "p17_attn_preiqr_raw")
        p17_attn_txt_paths += save_attn_value_txts(p17_postiqr_matrices, scale_schedule, save_dir, "p17_attn_postiqr_raw")

        p17_vae_zip         = make_zip(save_dir, "p17_vae_scales.zip",            p17_vae_paths)         if p17_vae_paths         else None
        p17_preiqr_zip      = make_zip(save_dir, "p17_attn_preiqr_maps.zip",      p17_preiqr_paths)      if p17_preiqr_paths      else None
        p17_attn_zip        = make_zip(save_dir, "p17_attn_postiqr_maps.zip",     p17_attn_paths)        if p17_attn_paths        else None
        p17_mask_zip        = make_zip(save_dir, "p17_attn_masks.zip",            p17_mask_paths)        if p17_mask_paths        else None
        p17_preiqr_val_zip  = make_zip(save_dir, "p17_attn_preiqr_values.zip",    p17_preiqr_val_paths)  if p17_preiqr_val_paths  else None
        p17_postiqr_val_zip = make_zip(save_dir, "p17_attn_postiqr_values.zip",   p17_postiqr_val_paths) if p17_postiqr_val_paths else None
        p17_attn_txt_zip    = make_zip(save_dir, "p17_attn_raw_matrices.zip",     p17_attn_txt_paths)    if p17_attn_txt_paths    else None

    del target_img_tensor, target_np, p2p_token_storage
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    mode_str = "P2P-Attn fallback（無 source image，純 prompt 驅動）" if attn_fallback_mode else "P2P-Edit（source image injection）"
    status = (
        f"完成！耗時 {elapsed:.1f}s\n"
        f"模式: {mode_str}\n"
        f"Source focus: {source_focus_words_list}  |  keep: {source_keep_words_list}\n"
        f"Target focus: {target_focus_words_list}\n"
        f"Threshold method: {threshold_method}  |  percentile: {attn_threshold_percentile}\n"
        f"Scale schedule: {total_scales} scales, h/w={h_div_w}  |  full_replace={num_full_replace_scales}\n"
        f"結果已存於: {save_dir}/"
    )
    return (source_pil, Image.fromarray(target_rgb),
            scale_imgs, preiqr_attn_imgs, attn_imgs, mask_imgs,
            preiqr_val_imgs, postiqr_val_imgs,
            vae_zip, preiqr_zip, attn_zip, mask_zip,
            preiqr_val_zip, postiqr_val_zip, attn_txt_zip,
            p17_scale_imgs, p17_preiqr_attn_imgs, p17_attn_imgs, p17_mask_imgs,
            p17_preiqr_val_imgs, p17_postiqr_val_imgs,
            p17_vae_zip, p17_preiqr_zip, p17_attn_zip, p17_mask_zip,
            p17_preiqr_val_zip, p17_postiqr_val_zip, p17_attn_txt_zip,
            status)


# ============================================================
# Gradio UI
# ============================================================

_RATIO_CHOICES = [0.333, 0.4, 0.5, 0.571, 0.666, 0.75, 0.8,
                  1.0, 1.25, 1.333, 1.5, 1.75, 2.0, 2.5, 3.0]
_RATIO_CHOICES_NP = np.array(_RATIO_CHOICES)


def auto_aspect_ratio(pil_img):
    """根據上傳圖片的實際尺寸，自動選擇最接近的 H/W 比例。"""
    if pil_img is None:
        return gr.update()
    w, h = pil_img.size
    ratio = h / w
    nearest = float(_RATIO_CHOICES_NP[np.argmin(np.abs(_RATIO_CHOICES_NP - ratio))])
    return nearest


def auto_focus_words(src_prompt, tgt_prompt):
    """根據 source / target prompt 差異自動推導 focus words，供即時更新用。"""
    auto_src, auto_tgt = derive_focus_terms_from_prompt_diff(src_prompt, tgt_prompt)
    src_str = ", ".join(auto_src) if auto_src else ""
    tgt_str = ", ".join(auto_tgt) if auto_tgt else ""
    return src_str, tgt_str


def build_ui():
    with gr.Blocks(title="P2P-Edit Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# P2P-Edit 互動式圖像編輯")
        gr.Markdown("上傳 source image，設定 prompt 與參數，即可產生編輯結果。模型已常駐記憶體。")

        with gr.Row():
            # ── 左欄：輸入 ──
            with gr.Column(scale=1):
                source_image = gr.Image(
                    label="Source Image（拖曳上傳；留空會 fallback 到 P2P-Attn 純 prompt 模式）",
                    type="pil",
                    height=350,
                )
                source_prompt = gr.Textbox(
                    label="Source Prompt",
                    value="A oil paint of Girl with a Pearl Earring.",
                )
                target_prompt = gr.Textbox(
                    label="Target Prompt",
                    value="A oil paint of Green Frog with a Pearl Earring.",
                )
                source_focus_words = gr.Textbox(
                    label="Source Focus Words（自動推導，可手動覆蓋）",
                    value="Girl",
                    placeholder="以逗號分隔多個 phrase",
                )
                target_focus_words = gr.Textbox(
                    label="Target Focus Words（自動推導，可手動覆蓋）",
                    value="Green Frog",
                    placeholder="以逗號分隔多個 phrase",
                )
                source_keep_words = gr.Textbox(
                    label="Source Keep Words（高 attention 區域強制保留 source token）",
                    value="",
                    placeholder="選填，以逗號分隔",
                )
                seed = gr.Number(label="Seed", value=1, precision=0)

            # ── 右欄：參數 ──
            with gr.Column(scale=1):
                inject_weights_str = gr.Textbox(
                    label="Inject Weights（13 個浮點數，空格分隔；0.0=100% source image，1.0=自由生成）",
                    value="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
                    placeholder="留空則全部設為 0.0",
                )
                num_full_replace_scales = gr.Slider(
                    minimum=0, maximum=6, step=1, value=2,
                    label="Num Full Replace Scales（前 N scale 強制 100% P2P token 替換）",
                )
                threshold_method = gr.Dropdown(
                    choices=[
                        (f"1 – Fixed Percentile", 1),
                        (f"2 – Dynamic (Ternary Search, PIE-Bench GT)", 2),
                        (f"3 – Otsu（無超參，推薦）", 3),
                        (f"4 – FFT + Otsu", 4),
                        (f"5 – Spectral Energy Ratio", 5),
                        (f"6 – Edge-Attention Coherence（需 source image）", 6),
                        (f"7 – GMM 雙高斯混合", 7),
                        (f"8 – 複合（Edge→Otsu→R_k fallback）", 8),
                        (f"9 – IPR 逆參與率", 9),
                        (f"10 – Shannon Entropy", 10),
                        (f"11 – Block Consensus Voting", 11),
                        (f"12 – Kneedle / Elbow Detection", 12),
                        (f"13 – Meta-Adaptive（CV 變異係數）", 13),
                        (f"14 – Absolute（正規化 >high / <low）", 14),
                    ],
                    value=13,
                    label="Threshold Method（Attention 遮罩二值化策略）",
                )
                attn_threshold_percentile = gr.Slider(
                    minimum=0, maximum=100, step=1, value=20,
                    label="Attention Threshold Percentile（Method 1 專用；愈低 → focus 區域愈大）",
                )
                use_cumulative_prob_mask = gr.Checkbox(
                    label="使用跨尺度累積機率遮罩 (Cumulative Prob Mask)",
                    value=False,
                )
                show_attn_vis = gr.Checkbox(
                    label="顯示 Attention Map / Mask 視覺化（取消勾選可大幅加速）",
                    value=False,
                )
                gr.Markdown("**Method 13 Meta-Adaptive 參數**")
                with gr.Row():
                    cv_min = gr.Number(label="CV min", value=0.0, precision=3)
                    cv_max = gr.Number(label="CV max", value=1.0, precision=3)
                with gr.Row():
                    k_min = gr.Number(label="k min", value=0.2, precision=3)
                    k_max = gr.Number(label="k max", value=0.5, precision=3)
                gr.Markdown("**Method 14 Absolute 參數**")
                with gr.Row():
                    absolute_high = gr.Number(label="Absolute high（focus >= 此值）", value=0.7, precision=3)
                    absolute_low  = gr.Number(label="Absolute low（preserve < 此值）", value=0.3, precision=3)
                h_div_w_template = gr.Dropdown(
                    choices=[
                        ("1:3 超橫 (0.333)",  0.333),
                        ("2:5 (0.4)",          0.4),
                        ("1:2 (0.5)",          0.5),
                        ("4:7 (0.571)",        0.571),
                        ("2:3 (0.666)",        0.666),
                        ("3:4 (0.75)",         0.75),
                        ("4:5 (0.8)",          0.8),
                        ("1:1 正方 (1.0)",     1.0),
                        ("5:4 (1.25)",         1.25),
                        ("4:3 (1.333)",        1.333),
                        ("3:2 (1.5)",          1.5),
                        ("7:4 (1.75)",         1.75),
                        ("2:1 (2.0)",          2.0),
                        ("5:2 (2.5)",          2.5),
                        ("3:1 超直 (3.0)",     3.0),
                    ],
                    value=1.0,
                    label="H / W 比例（圖片長寬比）",
                )

                with gr.Row():
                    run_btn  = gr.Button("開始編輯", variant="primary", size="lg")
                    stop_btn = gr.Button("停止",      variant="stop",    size="lg")

        gr.Markdown("---")

        with gr.Row():
            source_output = gr.Image(label="Source 重建 / 生成", height=400)
            target_output = gr.Image(label="Target 編輯結果", height=400)

        with gr.Row():
            gr.Markdown("### 逐 Scale BSQ-VAE 解碼圖")
            vae_zip_file = gr.File(label="下載全部 VAE 圖（ZIP）", file_count="single", height=60)
        scale_gallery = gr.Gallery(
            label="各 Scale VAE 解碼結果（由粗到細）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### 逐 Scale Attention Map — Pre-IQR（所有 Block 直接平均）")
            preiqr_zip_file = gr.File(label="下載全部 Pre-IQR（ZIP）", file_count="single", height=60)
        preiqr_gallery = gr.Gallery(
            label="各 Scale Pre-IQR Attention Map（Jet 熱力圖）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### 逐 Scale Attention Map — Post-IQR（IQR 過濾後，與 Mask 一致）")
            attn_zip_file = gr.File(label="下載全部 Post-IQR（ZIP）", file_count="single", height=60)
        attn_gallery = gr.Gallery(
            label="各 Scale Post-IQR Attention Map（Jet 熱力圖）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### 逐 Scale Attention Map — Pre-IQR（疊上精確數值，normalized 0–1）")
            preiqr_val_zip_file = gr.File(label="下載全部 Pre-IQR 數值圖（ZIP）", file_count="single", height=60)
        preiqr_val_gallery = gr.Gallery(
            label="各 Scale Pre-IQR Attention Map + 數值",
            columns=4,
            height=320,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### 逐 Scale Attention Map — Post-IQR（疊上精確數值，normalized 0–1）")
            postiqr_val_zip_file = gr.File(label="下載全部 Post-IQR 數值圖（ZIP）", file_count="single", height=60)
        postiqr_val_gallery = gr.Gallery(
            label="各 Scale Post-IQR Attention Map + 數值",
            columns=4,
            height=320,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### 原始 Attention 矩陣（Pre/Post-IQR raw values, .txt）")
            attn_txt_zip_file = gr.File(label="下載全部原始矩陣（ZIP，.txt）", file_count="single", height=60)

        with gr.Row():
            gr.Markdown("### 逐 Scale Attention Mask（二值化 Focus Mask）")
            mask_zip_file = gr.File(label="下載全部 Attention Mask（ZIP）", file_count="single", height=60)
        mask_gallery = gr.Gallery(
            label="各 Scale Attention Mask（白=focus，黑=背景）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        gr.Markdown("---")
        gr.Markdown("## Phase 1.7 — Target 生成中間結果")

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale BSQ-VAE 解碼圖")
            p17_vae_zip_file = gr.File(label="下載全部（ZIP）", file_count="single", height=60)
        p17_scale_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale VAE 解碼結果",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale Attention Map — Pre-IQR")
            p17_preiqr_zip_file = gr.File(label="下載全部 Pre-IQR（ZIP）", file_count="single", height=60)
        p17_preiqr_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale Pre-IQR Attention Map",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale Attention Map — Post-IQR（Target Focus Words）")
            p17_attn_zip_file = gr.File(label="下載全部 Post-IQR（ZIP）", file_count="single", height=60)
        p17_attn_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale Post-IQR Attention Map（Jet 熱力圖）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale Attention Map — Pre-IQR（疊上精確數值）")
            p17_preiqr_val_zip_file = gr.File(label="下載全部 Pre-IQR 數值圖（ZIP）", file_count="single", height=60)
        p17_preiqr_val_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale Pre-IQR Attention Map + 數值",
            columns=4,
            height=320,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale Attention Map — Post-IQR（疊上精確數值）")
            p17_postiqr_val_zip_file = gr.File(label="下載全部 Post-IQR 數值圖（ZIP）", file_count="single", height=60)
        p17_postiqr_val_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale Post-IQR Attention Map + 數值",
            columns=4,
            height=320,
            object_fit="contain",
        )

        with gr.Row():
            gr.Markdown("### Phase 1.7 原始 Attention 矩陣（.txt）")
            p17_attn_txt_zip_file = gr.File(label="下載全部原始矩陣（ZIP，.txt）", file_count="single", height=60)

        with gr.Row():
            gr.Markdown("### Phase 1.7 逐 Scale Attention Mask（二值化 Target Focus Mask）")
            p17_mask_zip_file = gr.File(label="下載全部（ZIP）", file_count="single", height=60)
        p17_mask_gallery = gr.Gallery(
            label="Phase 1.7 各 Scale Attention Mask（白=focus，黑=背景）",
            columns=6,
            height=280,
            object_fit="contain",
        )

        status_box = gr.Textbox(label="執行狀態", lines=4, interactive=False)

        # ── 上傳圖片時自動選擇最接近的長寬比 ──
        source_image.change(
            fn=auto_aspect_ratio,
            inputs=[source_image],
            outputs=[h_div_w_template],
        )

        # ── prompt 變更時自動推導 focus words ──
        for prompt_comp in [source_prompt, target_prompt]:
            prompt_comp.change(
                fn=auto_focus_words,
                inputs=[source_prompt, target_prompt],
                outputs=[source_focus_words, target_focus_words],
            )

        # ── 開始編輯 ──
        run_event = run_btn.click(
            fn=run_p2p_edit,
            inputs=[
                source_image,
                source_prompt,
                target_prompt,
                source_focus_words,
                target_focus_words,
                source_keep_words,
                inject_weights_str,
                num_full_replace_scales,
                attn_threshold_percentile,
                threshold_method,
                use_cumulative_prob_mask,
                h_div_w_template,
                seed,
                show_attn_vis,
                cv_min,
                cv_max,
                k_min,
                k_max,
                absolute_high,
                absolute_low,
            ],
            outputs=[
                source_output, target_output,
                scale_gallery, preiqr_gallery, attn_gallery, mask_gallery,
                preiqr_val_gallery, postiqr_val_gallery,
                vae_zip_file, preiqr_zip_file, attn_zip_file, mask_zip_file,
                preiqr_val_zip_file, postiqr_val_zip_file, attn_txt_zip_file,
                p17_scale_gallery, p17_preiqr_gallery, p17_attn_gallery, p17_mask_gallery,
                p17_preiqr_val_gallery, p17_postiqr_val_gallery,
                p17_vae_zip_file, p17_preiqr_zip_file, p17_attn_zip_file, p17_mask_zip_file,
                p17_preiqr_val_zip_file, p17_postiqr_val_zip_file, p17_attn_txt_zip_file,
                status_box,
            ],
        )

        # ── 停止按鈕：取消正在執行或排隊中的推論 ──
        stop_btn.click(fn=None, cancels=[run_event])

        demo.queue()

    return demo


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    load_models()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
