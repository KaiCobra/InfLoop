#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pie_edit.py — PIE-Bench 批量 P2P-Edit 評估腳本

功能：
  • 載入模型一次，批量處理 PIE-Bench 資料集的 700 個測試案例
  • 依照原始資料夾結構輸出結果（{output_dir}/{category}/{case_id}/）
  • 每個案例輸出：source.jpg、target.jpg（可選：attn_masks/）

使用方式：
  bash scripts/pie_edit.sh
  或
  python3 tools/run_pie_edit.py --bench_dir <path> --output_dir <path> [options]
"""

import os
import re
import sys
import json
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from PIL import Image

# ── 確保工作目錄在 sys.path 中 ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (
    find_focus_token_indices,
    collect_attention_text_masks,
    combine_and_store_masks,
    _save_masks_to_dir,
    gen_one_img,
    encode_image_to_raw_features,
    encode_image_to_scale_tokens,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    add_common_arguments,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from attention_map.extractor import CrossAttentionExtractor


# ============================================================
# 工具函式
# ============================================================

def clean_target_prompt(prompt: str) -> str:
    """
    去掉 PIE-Bench target_prompt 中的方括號標記，保留內容。

    範例：
        'a slanted [rusty] mountain [motorcycle] in front of a [fence]'
        → 'a slanted rusty mountain motorcycle in front of a fence'
    """
    return re.sub(r'\[([^\]]*)\]', r'\1', prompt).strip()


def extract_focus_words(blended_words: List[str]) -> Tuple[str, str]:
    """
    從 PIE-Bench blended_words 提取 source 和 target focus 詞彙。

    格式：['bicycle,motorcycle', 'building,fence', 'road,']
    •  左側非空 → source focus
    •  右側非空 → target focus
    •  空白側  → 新增/刪除操作（忽略空白側）

    回傳：(source_focus_str, target_focus_str) 以空格分隔
    """
    src_parts: List[str] = []
    tgt_parts: List[str] = []
    for bw in blended_words:
        parts = bw.split(',', 1)
        src = parts[0].strip()
        tgt = parts[1].strip() if len(parts) > 1 else ''
        if src:
            src_parts.append(src)
        if tgt:
            tgt_parts.append(tgt)
    return ' '.join(src_parts), ' '.join(tgt_parts)


def load_and_resize_pie_masks(
    mask_path: str,
    scale_schedule: list,
    num_full_replace_scales: int,
) -> Tuple[Dict[int, np.ndarray], float]:
    """
    載入 PIE-Bench mask.png，縮放至各 scale 的空間尺寸，
    轉換為 replacement_mask（與 combine_and_store_masks 輸出慣例相同）：

      PIE mask 慣例：255（白）= 編輯區域，  0（黑）= 背景保留
      replacement_mask：True = 背景（替換為 source token），False = 編輯區（保留 target gen）

    只回傳 si >= num_full_replace_scales 的 scale
    （前幾個 scale 由 p2p_attn_full_replace_scales 全部替換，不需個別遮罩）。

    回傳：
      - 各 scale replacement mask（True=背景保留 source）
      - 原始 mask 的白色比例（[0, 1]）
    """
    mask_arr = np.array(Image.open(mask_path).convert('L'))  # [H, W] uint8
    white_ratio = float((mask_arr >= 128).mean())
    result: Dict[int, np.ndarray] = {}
    for si, (_, h, w) in enumerate(scale_schedule):
        if si < num_full_replace_scales:
            continue
        if mask_arr.shape != (h, w):
            resized = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            resized = mask_arr.copy()
        # 白(255)=編輯→False，黑(0)=背景→True
        result[si] = resized < 128
    return result, white_ratio


# ============================================================
# 單一案例處理
# ============================================================

def run_one_case(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    source_image_path: str,
    source_prompt: str,
    target_prompt: str,
    source_focus_words: str,
    target_focus_words: str,
    save_dir: str,
    args,
    scale_schedule: list,
    attn_block_indices: List[int],
    total_scales: int,
    device_cuda: torch.device,
    mask_path: Optional[str] = None,
) -> bool:
    """
    執行一個 P2P-Edit 案例的完整 7 phase 管線。
    mask_path: PIE-Bench mask.png 路徑。不為 None 時啟用 PIE mask 模式：
      • 跳過 attention 閾值遮罩計算（Phase 1.5）
      • 以 PIE mask 等比例縮放作為各 scale 的 replacement mask
    成功回傳 True，失敗拋出例外。
    """
    use_pie_mask = (mask_path is not None)
    os.makedirs(save_dir, exist_ok=True)

    source_focus_words_list = (
        [w for w in source_focus_words.split() if w] if source_focus_words.strip() else []
    )
    target_focus_words_list = (
        [w for w in target_focus_words.split() if w] if target_focus_words.strip() else []
    )

    # ── Focus Token Indices ──
    source_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, source_prompt, source_focus_words_list)
        if source_focus_words_list else []
    )
    target_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, target_prompt, target_focus_words_list)
        if target_focus_words_list else []
    )

    # ─────────────────────────────────────
    # Phase 0：編碼 Source Image
    # ─────────────────────────────────────
    source_pil_img = Image.open(source_image_path).convert('RGB')

    image_raw_features = encode_image_to_raw_features(
        vae=vae,
        pil_img=source_pil_img,
        scale_schedule=scale_schedule,
        device=device_cuda,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )
    image_scale_tokens = encode_image_to_scale_tokens(
        vae=vae,
        pil_img=source_pil_img,
        scale_schedule=scale_schedule,
        device=device_cuda,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )

    # 建立 inject_schedule
    if args.inject_weights.strip():
        parsed_w = [float(w) for w in args.inject_weights.strip().split()]
        if len(parsed_w) != total_scales:
            raise ValueError(
                f"--inject_weights 長度 ({len(parsed_w)}) 與 scale 總數 ({total_scales}) 不符。"
            )
        inject_schedule = parsed_w
    else:
        inject_schedule = [
            0.0 if si < args.image_injection_scales else 1.0
            for si in range(total_scales)
        ]

    # ── 預先載入 PIE mask（各 scale，用於 Phase 1.6 + 1.9）──
    pie_scale_masks: Dict[int, np.ndarray] = {}
    pie_mask_white_ratio: float = 0.0
    pie_mask_forced_off: bool = False
    if use_pie_mask:
        pie_scale_masks, pie_mask_white_ratio = load_and_resize_pie_masks(
            mask_path, scale_schedule, args.num_full_replace_scales
        )
        # 若 mask 幾乎全白（包含全白），視為不可用：回退到 source/target attention mask
        if pie_mask_white_ratio >= 0.85:
            pie_mask_forced_off = True
            use_pie_mask = False
            pie_scale_masks = {}
            print(
                f"    ⚠ PIE mask 白色比例過高 ({pie_mask_white_ratio * 100:.2f}%)，"
                "改用 source/target attention mask"
            )

    # ─────────────────────────────────────
    # Phase 1：Source 生成 + Attention 擷取
    # PIE mask 模式（純淨）：不需要 attention → 跳過 register_patches
    # PIE mask + attn_fallback 模式：仍需要 attention → 正常掛 hook
    # ─────────────────────────────────────
    need_source_attn = source_focus_token_indices and (
        not use_pie_mask or args.pie_mask_attn_fallback
    )
    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    source_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
    )
    if need_source_attn:
        source_extractor.register_patches()

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
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    if need_source_attn:
        source_extractor.remove_patches()

    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'source.jpg'), img_np)

    # ─────────────────────────────────────
    # Phase 1.5：Dual Attention Masks
    # 純淨 PIE mask 模式：跳過；fallback 模式：正常計算（用於二次篩選）
    # ─────────────────────────────────────
    source_text_masks: Dict[int, np.ndarray] = {}
    source_low_attn_masks: Dict[int, np.ndarray] = {}
    run_source_attn = need_source_attn and len(source_extractor.attention_maps) > 0
    if run_source_attn:
        source_text_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source",
            low_attn=False,
        )
        source_low_attn_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source_preserve",
            low_attn=True,
        )

    # ─────────────────────────────────────
    # Phase 1.6：建立 Phase 1.7 Preserve Storage
    # PIE mask 模式：以 pie_scale_masks（黑=背景=True）作為 preserve 遮罩
    # 一般模式：使用 source low-attention mask（同以前）
    # ─────────────────────────────────────
    phase17_storage: Optional[BitwiseTokenStorage] = None
    if use_pie_mask and pie_scale_masks and image_scale_tokens:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
        for si, preserve_mask in pie_scale_masks.items():
            if si not in image_scale_tokens:
                continue
            phase17_storage.tokens[si] = image_scale_tokens[si].clone()
            # preserve_mask True=背景=在 Phase 1.7 中錨定為 source image token
            phase17_storage.masks[si] = (
                torch.tensor(preserve_mask, dtype=torch.bool)
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()   # [1, 1, h, w, 1]
            )
    elif source_low_attn_masks and image_scale_tokens:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
        for si, low_mask in source_low_attn_masks.items():
            if si not in image_scale_tokens:
                continue
            phase17_storage.tokens[si] = image_scale_tokens[si].clone()
            mask_tensor = (
                torch.tensor(low_mask, dtype=torch.bool)
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1)   # [1, 1, h, w, 1]
            )
            phase17_storage.masks[si] = mask_tensor.cpu()

    # ─────────────────────────────────────
    # Phase 1.7：Target Guided Gen → Target Focus Mask
    # PIE mask 模式：執行有引導的 target gen（背景錨定），但不收集 attention
    # 一般模式：同以前，收集 target attention 作為 focus mask
    # ─────────────────────────────────────
    target_text_masks: Dict[int, np.ndarray] = {}
    # PIE mask 模式只要有 phase17_storage 就執行；一般模式需要 target_focus_token_indices
    run_phase17 = (phase17_storage is not None) and (
        use_pie_mask or bool(target_focus_token_indices)
    )
    if run_phase17:
        # PIE fallback 模式也需要 target attention；純淨 PIE 模式不需要
        need_target_attn = (
            not use_pie_mask or args.pie_mask_attn_fallback
        ) and bool(target_focus_token_indices)
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
        ) if need_target_attn else None
        if need_target_attn:
            target_extractor.register_patches()

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _ = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder,
                    target_prompt,
                    g_seed=args.seed,
                    gt_leak=0, gt_ls_Bl=None,
                    cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=args.enable_positive_prompt,
                    p2p_token_storage=phase17_storage,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=True,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=args.num_full_replace_scales,
                    inject_image_features=None,
                    inject_schedule=None,
                )

        if need_target_attn:
            target_extractor.remove_patches()
            target_text_masks = collect_attention_text_masks(
                extractor=target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=args.num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                threshold_percentile=args.attn_threshold_percentile,
                label="target",
            )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ─────────────────────────────────────
    # Phase 1.9：寫入 Replacement Mask + 覆寫 storage.tokens
    # PIE mask 模式：直接將 pie_scale_masks 寫入 p2p_token_storage.masks
    # 一般模式：combine source∪target attention mask（同以前）
    # ─────────────────────────────────────
    if use_pie_mask and pie_scale_masks:
        use_attn_fallback = bool(args.pie_mask_attn_fallback) and (
            bool(source_text_masks) or bool(target_text_masks)
        )
        if use_attn_fallback:
            # Step 1：先用 attention 計算基礎 replacement mask（True = 保留 source）
            combine_and_store_masks(
                source_text_masks=source_text_masks,
                target_text_masks=target_text_masks,
                scale_schedule=scale_schedule,
                p2p_token_storage=p2p_token_storage,
                num_full_replace_scales=args.num_full_replace_scales,
            )
            # Step 2：疊加 PIE 約束：黑色區域強制保留 source（OR 運算）
            #   final_mask = pie_bg OR attn_replacement
            #   → 黑色(pie_bg=True)：永遠保留 source
            #   → 白色(pie_bg=False)：由 attention 決定
            for si, pie_bg in pie_scale_masks.items():
                pie_bg_t = (
                    torch.tensor(pie_bg, dtype=torch.bool)
                    .unsqueeze(0).unsqueeze(0).unsqueeze(-1)   # [1, 1, h, w, 1]
                )
                if si in p2p_token_storage.masks:
                    p2p_token_storage.masks[si] = (
                        pie_bg_t | p2p_token_storage.masks[si]
                    ).cpu()
                else:
                    # 此 scale 沒有 attention mask → fallback 到純 PIE mask
                    p2p_token_storage.masks[si] = pie_bg_t.cpu()
        else:
            # 純淨 PIE mask 模式：直接寫入 pie_scale_masks
            for si, repl_mask in pie_scale_masks.items():
                p2p_token_storage.masks[si] = (
                    torch.tensor(repl_mask, dtype=torch.bool)
                    .unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()   # [1, 1, h, w, 1]
                )
    elif source_text_masks or target_text_masks:
        combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=args.num_full_replace_scales,
        )

    for si_tok, tok in image_scale_tokens.items():
        p2p_token_storage.tokens[si_tok] = tok

    # ─────────────────────────────────────
    # Attention / Mask 視覺化（批量模式預設關閉）
    # 對齊 infer_p2p_edit：盡量輸出 source / target / combined（每個 scale）
    # ─────────────────────────────────────
    if args.save_attn_vis:
        attn_vis_dir = os.path.join(save_dir, 'attn_masks')
        if source_text_masks:
            _save_masks_to_dir(
                bool_masks=source_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, 'source'),
                file_prefix='source_focus',
                invert=True,
            )
        if source_low_attn_masks:
            _save_masks_to_dir(
                bool_masks=source_low_attn_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, 'phase17_preserve'),
                file_prefix='preserve',
                invert=False,
            )
        if target_text_masks:
            _save_masks_to_dir(
                bool_masks=target_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, 'target'),
                file_prefix='target_focus',
                invert=True,
            )
        if (mask_path is not None) and (pie_mask_forced_off or (use_pie_mask and pie_scale_masks)):
            # 仍輸出 PIE mask 縮放結果，方便對照（即使本次被判定為白色過高而停用）
            if pie_mask_forced_off:
                pie_scale_masks_vis, _ = load_and_resize_pie_masks(
                    mask_path, scale_schedule, args.num_full_replace_scales
                )
            else:
                pie_scale_masks_vis = pie_scale_masks
            _save_masks_to_dir(
                bool_masks=pie_scale_masks_vis,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, 'pie_mask'),
                file_prefix='pie_bg',
                invert=False,
            )
        if p2p_token_storage.masks:
            combined_vis = {
                si: ~m.squeeze().numpy()
                for si, m in p2p_token_storage.masks.items()
            }
            _save_masks_to_dir(
                bool_masks=combined_vis,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, 'combined'),
                file_prefix='combined_focus',
                invert=True,
            )

    # ─────────────────────────────────────
    # Phase 2：Target 生成
    # ─────────────────────────────────────
    has_mask = len(p2p_token_storage.masks) > 0

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=has_mask,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=args.num_full_replace_scales,
                inject_image_features=None,
                inject_schedule=None,
            )

    img_np = target_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'target.jpg'), img_np)

    # 清理顯存
    del source_image, target_image, img_np, p2p_token_storage
    if phase17_storage is not None:
        del phase17_storage
    torch.cuda.empty_cache()

    return True


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PIE-Bench 批量 P2P-Edit 評估（模型只載入一次）'
    )
    add_common_arguments(parser)

    # ── 批量設定 ──
    parser.add_argument('--bench_dir', type=str, required=True,
                        help='extracted_pie_bench 根目錄')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出根目錄（會依照原始資料夾結構存放結果）')
    parser.add_argument('--categories', type=str, default='',
                        help='只跑指定 category（逗號分隔資料夾名稱）；預設全部')
    parser.add_argument('--max_per_cat', type=int, default=-1,
                        help='每個 category 最多處理幾個案例（-1 = 全部）')
    parser.add_argument('--skip_existing', type=int, default=1, choices=[0, 1],
                        help='若 target.jpg 已存在則跳過（預設：1）')

    # ── P2P-Edit 參數 ──
    parser.add_argument('--num_full_replace_scales', type=int, default=2,
                        help='前幾個 scale 做 100%% source token 替換')
    parser.add_argument('--attn_threshold_percentile', type=float, default=80.0,
                        help='Attention 閾值百分位數（預設：80）')
    parser.add_argument('--attn_block_start', type=int, default=2,
                        help='Attention block 起始 index（-1 = 自動後半段）')
    parser.add_argument('--attn_block_end', type=int, default=-1,
                        help='Attention block 結束 index（-1 = 最後一個）')
    parser.add_argument('--attn_batch_idx', type=int, default=0,
                        help='0 = conditioned batch')
    parser.add_argument('--p2p_token_replace_prob', type=float, default=0.0,
                        help='Fallback 機率替換（無遮罩時）')
    parser.add_argument('--save_attn_vis', type=int, default=0, choices=[0, 1],
                        help='儲存 attention 遮罩視覺化（批量模式預設關閉）')

    # ── Source Image Injection 參數 ──
    parser.add_argument('--image_injection_scales', type=int, default=2,
                        help='前幾個 scale 使用 source image 注入')
    parser.add_argument('--inject_weights', type=str, default='',
                        help='各 scale 注入強度（空格分隔）。預設由 image_injection_scales 生成')
    parser.add_argument('--use_pie_mask', type=int, default=0, choices=[0, 1],
                        help='使用 PIE-Bench 提供的 mask.png 作為 token 篩選遮罩（1=開啟）'
                             '。開啟時以 mask 決定編輯/保留區域。')
    parser.add_argument('--pie_mask_attn_fallback', type=int, default=0, choices=[0, 1],
                        help='（需 use_pie_mask=1）在白色（編輯）區域內，'
                             '以 attention mask 二次篩選「真正需要編輯」的 token；'
                             '白色區域中 attention 未聚焦的 token 仍保留 source。'
                             '組合公式：final_mask = pie_bg OR attn_replacement')

    args = parser.parse_args()

    # 解析 cfg（支援 "4" 和 "4,6" 兩種格式）
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print('\n' + '=' * 80)
    print('PIE-Bench 批量 P2P-Edit 評估')
    print('=' * 80)
    print(f'bench_dir      : {args.bench_dir}')
    print(f'output_dir     : {args.output_dir}')
    print(f'skip_existing  : {bool(args.skip_existing)}')
    print(f'max_per_cat    : {args.max_per_cat if args.max_per_cat > 0 else "全部"}')
    print(f'full_replace   : {args.num_full_replace_scales} scales')
    print(f'attn_percentile: {args.attn_threshold_percentile}')
    print(f'use_pie_mask   : {bool(args.use_pie_mask)}')
    print(f'pie_attn_fallbk: {bool(args.pie_mask_attn_fallback)}')
    print('=' * 80 + '\n')

    # ── 載入模型（只載入一次）──
    print('[Init] 載入模型中（只執行一次）...')
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    print('[Init] 模型載入完成。\n')

    # ── Scale Schedule ──
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)

    # ── Attention Block 範圍 ──
    depth = len(infinity.unregistered_blocks)
    attn_block_start = (
        (depth // 2) if args.attn_block_start < 0
        else min(args.attn_block_start, depth - 1)
    )
    attn_block_end = (
        (depth - 1) if args.attn_block_end < 0
        else min(args.attn_block_end, depth - 1)
    )
    attn_block_indices = list(range(attn_block_start, attn_block_end + 1))
    print(f'[Config] Scale={total_scales}  AttnBlock={attn_block_start}~{attn_block_end}  Depth={depth}\n')

    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 決定要跑的 categories ──
    if args.categories.strip():
        cat_names = [c.strip() for c in args.categories.split(',') if c.strip()]
    else:
        cat_names = sorted(
            d for d in os.listdir(args.bench_dir)
            if os.path.isdir(os.path.join(args.bench_dir, d))
        )

    total_done = 0
    total_skip = 0
    total_err  = 0

    for cat_name in cat_names:
        cat_dir = os.path.join(args.bench_dir, cat_name)
        if not os.path.isdir(cat_dir):
            print(f'[Warning] Category 資料夾不存在，跳過：{cat_dir}')
            continue

        case_ids = sorted(
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
        )
        if args.max_per_cat > 0:
            case_ids = case_ids[:args.max_per_cat]

        cat_done = 0
        cat_skip = 0
        cat_err  = 0
        print(f'\n{"─" * 60}')
        print(f'[Category] {cat_name}  ({len(case_ids)} 個案例)')
        print(f'{"─" * 60}')

        for idx, case_id in enumerate(case_ids):
            case_dir  = os.path.join(cat_dir, case_id)
            meta_path = os.path.join(case_dir, 'meta.json')
            img_path  = os.path.join(case_dir, 'image.jpg')

            # 基本路徑檢查
            if not os.path.exists(meta_path):
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ 找不到 meta.json，跳過')
                cat_err += 1
                continue
            if not os.path.exists(img_path):
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ 找不到 image.jpg，跳過')
                cat_err += 1
                continue

            save_dir = os.path.join(args.output_dir, cat_name, case_id)

            # 跳過已處理案例（必須存在且大小 > 0，避免中途中斷的空檔被跳過）
            target_out = os.path.join(save_dir, 'target.jpg')
            if args.skip_existing and os.path.exists(target_out) and os.path.getsize(target_out) > 0:
                cat_skip += 1
                total_skip += 1
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ↓ 已存在，跳過')
                continue

            # 讀取 meta
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception as e:
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ✗ meta.json 讀取失敗：{e}')
                cat_err += 1
                total_err += 1
                continue

            source_prompt = meta.get('source_prompt', '')
            raw_target    = meta.get('target_prompt', '')
            target_prompt = clean_target_prompt(raw_target)
            blended_words = meta.get('blended_words', [])
            source_focus_words, target_focus_words = extract_focus_words(blended_words)

            print(f'\n  [{idx+1}/{len(case_ids)}] {case_id}')
            print(f'    src : {source_prompt}')
            print(f'    tgt : {target_prompt}')
            print(f'    focus ← "{source_focus_words}"  → "{target_focus_words}"')

            # PIE mask 路徑（若啟用）
            pie_mask_path: Optional[str] = None
            if args.use_pie_mask:
                _mpath = os.path.join(case_dir, 'mask.png')
                if os.path.exists(_mpath):
                    pie_mask_path = _mpath
                else:
                    print(f'    ⚠ mask.png 不存在，fallback 至 attention mask 模式')

            try:
                run_one_case(
                    infinity=infinity,
                    vae=vae,
                    text_tokenizer=text_tokenizer,
                    text_encoder=text_encoder,
                    source_image_path=img_path,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_focus_words=source_focus_words,
                    target_focus_words=target_focus_words,
                    save_dir=save_dir,
                    args=args,
                    scale_schedule=scale_schedule,
                    attn_block_indices=attn_block_indices,
                    total_scales=total_scales,
                    device_cuda=device_cuda,
                    mask_path=pie_mask_path,
                )
                cat_done  += 1
                total_done += 1
                print(f'    ✓  → {save_dir}/target.jpg')

            except Exception as exc:
                cat_err  += 1
                total_err += 1
                print(f'    ✗  Error：{exc}')
                traceback.print_exc()
                torch.cuda.empty_cache()

        print(f'\n  [Category 小計] {cat_name}：✓{cat_done}  ↓{cat_skip}  ✗{cat_err}')

    print('\n' + '=' * 80)
    print(f'PIE-Bench 批量評估完成')
    print(f'  ✓ 成功：{total_done}')
    print(f'  ↓ 跳過：{total_skip}')
    print(f'  ✗ 錯誤：{total_err}')
    print(f'  結果目錄：{args.output_dir}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
