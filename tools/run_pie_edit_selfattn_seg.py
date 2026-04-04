#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pie_edit_selfattn_seg.py — DiffSegmenter-style object segmentation
using cross-attention + self-attention maps from the Infinity model.

核心概念（對齊 DiffSegmenter, IEEE TIP 2025）：
  • Cross-attention map → 初始語義 score map（text token 對應的空間位置）
  • Self-attention map  → 物件完整形狀（同一物件的 pixel 互相 attend）
  • 迭代精煉公式：A_ca^(n) = A_sa @ A_ca^(n-1)
    → 語義信號沿著 self-attention 的物件邊界「擴散」，得到完整的物件遮罩

Pipeline:
  Phase 0  : 編碼 Source Image（VAE features）
  Phase 1  : Source Gen（image injection）+ CrossAttention & SelfAttention 擷取
  Phase 1.5: DiffSegmenter 迭代精煉（SA × CA）
  Output   : source.jpg + 每個 scale 的 refined object mask + 視覺化

不包含 target gen — 此 pipeline 專注於物件定位。
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (
    add_common_arguments,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
    encode_image_to_raw_features,
    find_focus_token_indices,
    _iqr_filtered_mean,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from attention_map.extractor import CrossAttentionExtractor, SelfAttentionExtractor


# ============================================================
# DiffSegmenter Core Algorithm
# ============================================================

def extract_cross_attn_map(
    extractor: CrossAttentionExtractor,
    block_indices: List[int],
    scale_idx: int,
    focus_token_indices: List[int],
    spatial_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    提取 focus token 的 cross-attention map，跨 block IQR 過濾後取平均。

    Returns: [H, W] numpy array, or None.
    """
    maps_per_block = []
    for bidx in block_indices:
        m = extractor.extract_word_attention(
            bidx, scale_idx, focus_token_indices, spatial_size,
        )
        if m is not None:
            maps_per_block.append(m)
    if not maps_per_block:
        return None
    if len(maps_per_block) == 1:
        return maps_per_block[0]
    attn_stack = torch.tensor(np.stack(maps_per_block), dtype=torch.float32)
    filtered_attn, _, _ = _iqr_filtered_mean(attn_stack)
    return filtered_attn


def extract_intra_scale_self_attn(
    extractor: SelfAttentionExtractor,
    block_indices: List[int],
    scale_idx: int,
    spatial_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    提取 intra-scale self-attention matrix [H*W, H*W]。

    AR 模型的 self-attention（含 KV cache）shape 為 [Lq, Lk]，
    其中 Lq = 當前 scale 的 token 數，Lk = 累積所有 scale 的 token 數。

    取最後 H*W 列（當前 scale 的 key），得到 [H*W, H*W] spatial affinity matrix。
    跨 block IQR 過濾取平均。

    Returns: [H*W, H*W] numpy array, or None.
    """
    H, W = spatial_size
    target_len = H * W

    matrices = []
    for bidx in block_indices:
        if bidx not in extractor.attention_maps:
            continue
        attn_list = extractor.attention_maps[bidx]
        if scale_idx >= len(attn_list):
            continue

        attn = attn_list[scale_idx]  # [1, num_heads, Lq, Lk]
        attn_agg = extractor.aggregate_heads(attn)  # [1, Lq, Lk]
        sa = attn_agg[0].float().cpu().numpy()  # [Lq, Lk]

        Lq, Lk = sa.shape
        if Lq < target_len or Lk < target_len:
            continue

        # Intra-scale: 取最後 target_len 行和最後 target_len 列
        intra = sa[-target_len:, -target_len:]  # [H*W, H*W]
        matrices.append(intra)

    if not matrices:
        return None
    if len(matrices) == 1:
        return matrices[0]
    attn_stack = torch.tensor(np.stack(matrices), dtype=torch.float32)
    filtered_attn, _, _ = _iqr_filtered_mean(attn_stack)
    return filtered_attn


def diffsegmenter_refine(
    cross_attn_map: np.ndarray,
    self_attn_matrix: np.ndarray,
    num_iterations: int = 3,
) -> List[np.ndarray]:
    """
    DiffSegmenter-style 迭代精煉：

        A_ca^(n) = A_sa @ A_ca^(n-1)

    Self-attention 將語義信號（cross-attention）沿物件邊界擴散，
    使得 focus word 的 attention 不再只集中在高亮區域，
    而是擴展到完整物件。

    Args:
        cross_attn_map:   [H, W] — 初始 cross-attention map
        self_attn_matrix: [H*W, H*W] — intra-scale self-attention matrix
        num_iterations:   迭代次數（DiffSegmenter 建議 2~3 次）

    Returns:
        List of [H, W] arrays — 每次迭代結果（index 0 = 原始 CA）
    """
    H, W = cross_attn_map.shape
    ca_flat = cross_attn_map.reshape(-1).astype(np.float64)  # [H*W]

    # Normalize SA rows to sum to 1 (row-stochastic)
    sa = self_attn_matrix.astype(np.float64)
    sa_sum = sa.sum(axis=-1, keepdims=True)
    sa_sum = np.clip(sa_sum, 1e-8, None)
    sa_norm = sa / sa_sum

    results = [cross_attn_map.copy()]  # iteration 0 = raw CA

    refined = ca_flat.copy()
    for _ in range(num_iterations):
        refined = sa_norm @ refined
        # Min-max normalize 防止數值爆炸
        r_min, r_max = refined.min(), refined.max()
        if r_max - r_min > 1e-8:
            refined = (refined - r_min) / (r_max - r_min)
        results.append(refined.reshape(H, W))

    return results


# ============================================================
# Visualization Helpers
# ============================================================

def save_heatmap(arr: np.ndarray, path: str) -> None:
    """存 2D array 為彩色 heatmap（JET colormap）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr_f = arr.astype(np.float64)
    a_min, a_max = arr_f.min(), arr_f.max()
    if a_max - a_min > 1e-8:
        arr_f = (arr_f - a_min) / (a_max - a_min)
    img_gray = (arr_f * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    h, w = img_color.shape[:2]
    scale = max(1, 256 // max(h, w))
    if scale > 1:
        img_color = cv2.resize(
            img_color, (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imwrite(path, img_color)


def save_binary_mask(mask: np.ndarray, path: str) -> None:
    """存 boolean mask 為黑白圖。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (mask.astype(np.uint8) * 255)
    h, w = img.shape[:2]
    scale = max(1, 256 // max(h, w))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)


def save_overlay(source_img: np.ndarray, mask: np.ndarray,
                 path: str, alpha: float = 0.4) -> None:
    """
    將 refined mask 疊加到 source image 上做視覺化。
    mask 的 True 區域以紅色半透明覆蓋。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h_src, w_src = source_img.shape[:2]
    # Resize mask to source image size
    mask_resized = cv2.resize(
        mask.astype(np.uint8), (w_src, h_src),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    overlay = source_img.copy()
    # 紅色覆蓋 (BGR: [0, 0, 255])
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[mask_resized] = cv2.addWeighted(
        overlay, 1 - alpha, red, alpha, 0,
    )[mask_resized]
    cv2.imwrite(path, overlay)


# ============================================================
# run_one_case — 主流程
# ============================================================

def run_one_case(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    source_image_path: str,
    source_prompt: str,
    target_prompt: str,
    source_focus_words: List[str],
    target_focus_words: List[str],
    save_dir: str,
    args,
    scale_schedule: list,
    attn_block_indices: List[int],
    total_scales: int,
    device_cuda: torch.device,
    mask_path: Optional[str] = None,
    blended_words: Optional[List[str]] = None,
    case_source_dir: Optional[str] = None,
) -> bool:
    """
    DiffSegmenter-style object segmentation using cross+self attention.

    只做 source gen + attention 擷取 + SA×CA 迭代精煉。
    不做 target gen。

    輸出：
      - source.jpg
      - selfattn_seg/scale{NN}_ca_raw.png          — 原始 cross-attention
      - selfattn_seg/scale{NN}_refined_iter{N}.png  — 每次迭代結果
      - selfattn_seg/scale{NN}_mask_object.png      — 物件 mask（白=物件）
      - selfattn_seg/scale{NN}_overlay.png           — 疊加到 source image
      - selfattn_seg/meta.json                       — 元資料
    """
    os.makedirs(save_dir, exist_ok=True)

    # 決定 focus words：優先 source，其次 target
    focus_words_list = source_focus_words if source_focus_words else target_focus_words
    focus_prompt = source_prompt if source_focus_words else target_prompt
    focus_token_indices = (
        find_focus_token_indices(text_tokenizer, focus_prompt, focus_words_list)
        if focus_words_list else []
    )

    if not focus_token_indices:
        print("    [SelfattnSeg] No focus token indices found, skip")
        return False

    # SA block range（可與 CA block range 不同）
    sa_block_start = int(getattr(args, 'sa_block_start', 0))
    sa_block_end = int(getattr(args, 'sa_block_end', -1))
    depth = len(infinity.unregistered_blocks)
    sa_block_start = max(0, sa_block_start)
    sa_block_end = (depth - 1) if sa_block_end < 0 else min(sa_block_end, depth - 1)
    sa_block_indices = list(range(sa_block_start, sa_block_end + 1))

    sa_iterations = int(getattr(args, 'sa_refine_iterations', 3))
    threshold_pct = float(args.attn_threshold_percentile)
    num_full_replace = int(args.num_full_replace_scales)
    sa_max_scale = int(getattr(args, 'sa_max_scale', -1))  # -1 = 全部

    # ─── Phase 0: Encode source image ───
    source_pil = Image.open(source_image_path).convert('RGB')
    image_raw_features = encode_image_to_raw_features(
        vae=vae,
        pil_img=source_pil,
        scale_schedule=scale_schedule,
        device=device_cuda,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )

    # inject_schedule
    if args.inject_weights.strip():
        inject_schedule = [float(w) for w in args.inject_weights.strip().split()]
    else:
        inject_schedule = [
            0.0 if si < args.image_injection_scales else 1.0
            for si in range(total_scales)
        ]

    # ─── Phase 1: Source gen + Cross/Self Attention extraction ───
    p2p_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    cross_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=int(args.attn_batch_idx),
        aggregate_method="mean",
    )
    # sa_max_scale: 限制 SA 擷取的最大 scale，避免 OOM
    # 後期 scale 累積大量 KV cache，attention matrix 會佔滿 GPU 記憶體
    sa_capture_end = sa_max_scale if sa_max_scale >= 0 else -1
    self_extractor = SelfAttentionExtractor(
        model=infinity,
        block_indices=sa_block_indices,
        batch_idx=int(args.attn_batch_idx),
        aggregate_method="mean",
        capture_attention=True,
        capture_scale_start=num_full_replace,  # 只擷取需要的 scale
        capture_scale_end=sa_capture_end,
        capture_kv=False,
        inject_kv=False,
    )

    cross_extractor.register_patches()
    self_extractor.register_patches()

    print(f"    [Phase 1] Source gen (CA blocks {attn_block_indices[0]}-"
          f"{attn_block_indices[-1]}, SA blocks {sa_block_indices[0]}-"
          f"{sa_block_indices[-1]})")

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                p2p_token_storage=p2p_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    cross_extractor.remove_patches()
    self_extractor.remove_patches()

    # Save source image
    source_np = source_image.cpu().numpy()
    if source_np.dtype != np.uint8:
        source_np = np.clip(source_np, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'source.jpg'), source_np)

    # ─── Phase 1.5: DiffSegmenter iterative refinement ───
    vis_dir = os.path.join(save_dir, 'selfattn_seg')
    refined_masks: Dict[int, np.ndarray] = {}
    refined_maps: Dict[int, np.ndarray] = {}
    scale_info: Dict[str, dict] = {}

    sa_max_label = "all" if sa_max_scale < 0 else str(sa_max_scale)
    print(f"    [Phase 1.5] DiffSegmenter SA×CA refinement "
          f"({sa_iterations} iters, threshold={threshold_pct}%, "
          f"SA blocks {sa_block_indices[0]}-{sa_block_indices[-1]}, "
          f"sa_max_scale={sa_max_label})")

    last_refined_mask: Optional[np.ndarray] = None  # 用於上採樣 fallback

    for si in range(num_full_replace, total_scales):
        _, H, W = scale_schedule[si]

        # Cross-attention map for focus words
        ca_map = extract_cross_attn_map(
            cross_extractor, attn_block_indices, si,
            focus_token_indices, (H, W),
        )
        if ca_map is None:
            print(f"      scale {si}: no cross-attention map, skip")
            scale_info[str(si)] = {
                "spatial_size": [H, W], "status": "no_ca_map",
            }
            continue

        # 超過 sa_max_scale 的 scale：上採樣上一個成功的 refined mask
        beyond_sa_range = (sa_max_scale >= 0 and si > sa_max_scale)

        sa_matrix = None
        if not beyond_sa_range:
            sa_matrix = extract_intra_scale_self_attn(
                self_extractor, sa_block_indices, si, (H, W),
            )

        if sa_matrix is not None:
            # ── DiffSegmenter iterative refinement ──
            iteration_results = diffsegmenter_refine(
                ca_map, sa_matrix, num_iterations=sa_iterations,
            )

            for iter_idx, iter_map in enumerate(iteration_results):
                label = "ca_raw" if iter_idx == 0 else f"refined_iter{iter_idx}"
                save_heatmap(
                    iter_map,
                    os.path.join(vis_dir, f'scale{si:02d}_{label}.png'),
                )

            final_map = iteration_results[-1]
            thr = np.percentile(final_map, threshold_pct)
            mask = (final_map >= thr)
            refined_masks[si] = mask
            refined_maps[si] = final_map
            last_refined_mask = mask  # 記住最新的 refined mask

            # SA matrix heatmap（小 scale 才存）
            if H * W <= 1024:
                save_heatmap(
                    sa_matrix,
                    os.path.join(vis_dir, f'scale{si:02d}_sa_matrix.png'),
                )

            status = "refined"
            print(f"      scale {si} ({H}x{W}): obj={mask.mean()*100:.1f}% [SA refined]")

        elif last_refined_mask is not None:
            # ── Upsample fallback：從上一個成功 scale 的 mask 上採樣 ──
            upsampled = cv2.resize(
                last_refined_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            refined_masks[si] = upsampled
            refined_maps[si] = ca_map
            mask = upsampled

            save_heatmap(ca_map, os.path.join(vis_dir, f'scale{si:02d}_ca_raw.png'))
            reason = "beyond_sa_max" if beyond_sa_range else "no_sa_data"
            status = f"upsampled_{reason}"
            print(f"      scale {si} ({H}x{W}): obj={mask.mean()*100:.1f}% "
                  f"[upsampled from prev scale]")

        else:
            # ── 完全 fallback：只用 raw CA threshold ──
            save_heatmap(ca_map, os.path.join(vis_dir, f'scale{si:02d}_ca_raw.png'))
            thr = np.percentile(ca_map, threshold_pct)
            mask = (ca_map >= thr)
            refined_masks[si] = mask
            refined_maps[si] = ca_map

            status = "ca_only_fallback"
            print(f"      scale {si} ({H}x{W}): obj={mask.mean()*100:.1f}% "
                  f"[CA threshold only]")

        save_binary_mask(mask, os.path.join(vis_dir, f'scale{si:02d}_mask_object.png'))
        save_overlay(source_np, mask,
                     os.path.join(vis_dir, f'scale{si:02d}_overlay.png'))
        scale_info[str(si)] = {
            "spatial_size": [H, W],
            "status": status,
            "obj_ratio": float(mask.mean()),
        }

    # Save metadata
    meta = {
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "focus_words": focus_words_list,
        "focus_token_indices": focus_token_indices,
        "sa_refine_iterations": sa_iterations,
        "sa_max_scale": sa_max_scale,
        "threshold_percentile": threshold_pct,
        "ca_block_range": [attn_block_indices[0], attn_block_indices[-1]],
        "sa_block_range": [sa_block_indices[0], sa_block_indices[-1]],
        "num_full_replace_scales": num_full_replace,
        "scales": scale_info,
    }
    os.makedirs(vis_dir, exist_ok=True)
    with open(os.path.join(vis_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Cleanup
    del source_image, source_np, p2p_storage
    torch.cuda.empty_cache()

    return True
