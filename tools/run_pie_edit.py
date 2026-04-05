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
import time
import argparse
import difflib
import traceback
from dataclasses import asdict, dataclass
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
    derive_focus_terms_from_prompt_diff,
    parse_focus_words_arg,
    _iqr_filtered_mean,
    collect_attention_text_masks,
    collect_attention_text_masks_dynamic,
    collect_last_scale_attention_mask,
    combine_and_store_masks,
    build_cumulative_replacement_prob_masks,
    _mask_tensor_to_prob_map,
    _save_prob_masks_to_dir,
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
from attention_map.extractor import CrossAttentionExtractor, AttentionCacheInjector


# ============================================================
# 工具函式
# ============================================================

def decode_rle_mask(rle: List[int], hw: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    解碼 PIE-Bench mapping_file.json 中的 mask RLE。
    格式：(start, count) 交替排列，foreground=True。
    回傳 bool 陣列 [H, W]，True = 編輯區域。
    """
    h, w = hw
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(rle), 2):
        start = rle[i]
        count = rle[i + 1]
        end = min(start + count, h * w)
        flat[start:end] = 1
    return flat.reshape(h, w).astype(bool)


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


@dataclass
class AttentionCachePair:
    source_term: str
    target_term: str
    source_token_indices: List[int]
    target_token_indices: List[int]


def parse_blended_word_pairs(blended_words: List[str]) -> List[Tuple[str, str]]:
    """將 PIE-Bench blended_words 轉為 (source_term, target_term) 配對列表。"""
    pairs: List[Tuple[str, str]] = []
    for item in blended_words or []:
        left, right = (item.split(',', 1) + [''])[:2]
        pairs.append((left.strip(), right.strip()))
    return pairs


def sanitize_term_for_path(term: str) -> str:
    """將詞彙轉成可作為資料夾名稱的安全字串。"""
    safe = re.sub(r'[^\w\-.]+', '_', term.strip(), flags=re.UNICODE)
    safe = safe.strip('._')
    return safe or 'empty'


def build_attention_cache_pairs(
    text_tokenizer,
    source_prompt: str,
    target_prompt: str,
    blended_words: List[str],
) -> Tuple[List[AttentionCachePair], Dict[str, List[int]]]:
    """
    依 blended_words 建立 source→target token 對齊。

    Returns:
        - 可用於 attention cache replace 的配對列表（僅保留 source/target 兩側皆存在者）
        - 需要輸出的 source term -> source token indices（包含刪除詞，方便存 map）
    """
    cache_pairs: List[AttentionCachePair] = []
    source_terms_to_export: Dict[str, List[int]] = {}

    for src_term, tgt_term in parse_blended_word_pairs(blended_words):
        src_token_indices = (
            find_focus_token_indices(text_tokenizer, source_prompt, [src_term], verbose=False)
            if src_term else []
        )
        tgt_token_indices = (
            find_focus_token_indices(text_tokenizer, target_prompt, [tgt_term], verbose=False)
            if tgt_term else []
        )

        if src_term and src_token_indices:
            source_terms_to_export[src_term] = src_token_indices

        if src_term and tgt_term and src_token_indices and tgt_token_indices:
            cache_pairs.append(
                AttentionCachePair(
                    source_term=src_term,
                    target_term=tgt_term,
                    source_token_indices=src_token_indices,
                    target_token_indices=tgt_token_indices,
                )
            )

    return cache_pairs, source_terms_to_export


def _extract_per_head_token_attention(
    extractor: CrossAttentionExtractor,
    block_idx: int,
    scale_idx: int,
    token_indices: List[int],
) -> Optional[torch.Tensor]:
    """從原始 attention cache 取出某一組 token 的 per-head 空間圖。"""
    attn_list = extractor.attention_maps.get(block_idx)
    if attn_list is None or scale_idx >= len(attn_list):
        return None

    attn_tensor = attn_list[scale_idx]  # [1, H, L, K]
    valid_token_indices = [idx for idx in token_indices if idx < attn_tensor.shape[-1]]
    if not valid_token_indices:
        return None

    per_head = attn_tensor[0, :, :, valid_token_indices].mean(dim=-1)  # [H, L]
    return per_head.detach().cpu().float()


def build_attention_cache_from_source(
    extractor: CrossAttentionExtractor,
    cache_pairs: List[AttentionCachePair],
    source_terms_to_export: Dict[str, List[int]],
    scale_schedule: List[Tuple[int, int, int]],
    attn_block_indices: List[int],
) -> Tuple[Dict[int, Dict[int, Dict[int, torch.Tensor]]], Dict[str, Dict[int, np.ndarray]]]:
    """
    從 source 生成得到的 attention maps 建立：
      1. target replace 用的 per-block/per-scale attention cache
      2. 每個 source term 的視覺化 heatmap
    """
    replacement_maps: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
    export_maps: Dict[str, Dict[int, np.ndarray]] = {}

    total_scales = len(scale_schedule)

    for pair in cache_pairs:
        for block_idx in attn_block_indices:
            for scale_idx in range(total_scales):
                src_map = _extract_per_head_token_attention(
                    extractor=extractor,
                    block_idx=block_idx,
                    scale_idx=scale_idx,
                    token_indices=pair.source_token_indices,
                )
                if src_map is None:
                    continue
                replacement_maps.setdefault(block_idx, {}).setdefault(scale_idx, {})
                for tgt_token_idx in pair.target_token_indices:
                    replacement_maps[block_idx][scale_idx][tgt_token_idx] = src_map.clone()

    for term, token_indices in source_terms_to_export.items():
        export_maps[term] = {}
        for scale_idx, (_, h, w) in enumerate(scale_schedule):
            block_maps: List[np.ndarray] = []
            for block_idx in attn_block_indices:
                attn_map = extractor.extract_word_attention(
                    block_idx=block_idx,
                    scale_idx=scale_idx,
                    token_indices=token_indices,
                    spatial_size=(h, w),
                )
                if attn_map is not None:
                    block_maps.append(attn_map)
            if not block_maps:
                continue
            filtered_attn, _, _ = _iqr_filtered_mean(
                torch.tensor(np.stack(block_maps), dtype=torch.float32)
            )
            export_maps[term][scale_idx] = filtered_attn.astype(np.float32)

    return replacement_maps, export_maps


def save_attention_cache_maps(
    export_maps: Dict[str, Dict[int, np.ndarray]],
    scale_schedule: List[Tuple[int, int, int]],
    save_root: str,
) -> None:
    """將每個 source term 的 attention cache heatmap 存成每 scale 一張圖。"""
    if not export_maps:
        return

    os.makedirs(save_root, exist_ok=True)

    for term, scale_maps in export_maps.items():
        term_dir = os.path.join(save_root, sanitize_term_for_path(term))
        os.makedirs(term_dir, exist_ok=True)

        for scale_idx, (_, h, w) in enumerate(scale_schedule):
            attn_map = scale_maps.get(scale_idx)
            if attn_map is None:
                attn_map = np.zeros((h, w), dtype=np.float32)

            attn_min = float(attn_map.min())
            attn_max = float(attn_map.max())
            if attn_max > attn_min:
                norm = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                norm = np.zeros_like(attn_map, dtype=np.float32)

            gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
            vis_size = max(256, h * 4, w * 4)
            gray = cv2.resize(gray, (vis_size, vis_size), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            cv2.imwrite(
                os.path.join(term_dir, f'scale{scale_idx + 1:02d}_{h}x{w}.png'),
                heatmap,
            )


def save_attention_cache_metadata(
    save_dir: str,
    cache_pairs: List[AttentionCachePair],
    source_terms_to_export: Dict[str, List[int]],
    args,
    total_scales: int,
) -> None:
    """將 attention cache 對齊資訊存成 JSON，方便後續對照。"""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        'enabled': bool(args.use_attn_cache),
        'phase': args.attn_cache_phase,
        'max_scale_1_based': min(int(args.attn_cache_max_scale), total_scales),
        'cache_pairs': [asdict(item) for item in cache_pairs],
        'source_terms_to_export': source_terms_to_export,
    }
    with open(os.path.join(save_dir, 'alignments.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_case_reference_symlink(case_source_dir: str, save_dir: str) -> None:
    """在輸出案例資料夾中建立回指原始 extracted_pie_bench 案例的符號連結。"""
    if not case_source_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    link_path = os.path.join(save_dir, 'source_case_dir')
    target_path = os.path.abspath(case_source_dir)

    if os.path.islink(link_path):
        if os.path.realpath(link_path) == target_path:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        return

    os.symlink(target_path, link_path)


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


def map_white_ratio_to_threshold(
    white_ratio: float,
    thr_min: float = 65.0,
    thr_max: float = 92.0,
) -> float:
    """
    將白色比例（0~1）反向線性映射到 attention threshold。
      white=0.0 -> thr_max
      white=1.0 -> thr_min
    """
    ratio = float(np.clip(white_ratio, 0.0, 1.0))
    threshold = thr_max - ratio * (thr_max - thr_min)
    return float(np.clip(threshold, thr_min, thr_max))


def dilate_true_region(mask_bool: np.ndarray, expand_percent: float) -> np.ndarray:
    """
    對 bool mask 的 True 區域做膨脹（向外擴張）。
    expand_percent: 以短邊比例計算半徑（例如 2.0 = 2% * min(h, w)）。
    """
    if expand_percent <= 0:
        return mask_bool

    h, w = mask_bool.shape
    radius = int(round(min(h, w) * (expand_percent / 100.0)))
    if radius <= 0:
        return mask_bool

    k = radius * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def expand_storage_masks_inplace(
    masks: Dict[int, torch.Tensor],
    expand_percent: float,
) -> None:
    """
    對 p2p_token_storage.masks（True=保留 source）逐 scale 向外擴張 True 區域。
    """
    if expand_percent <= 0 or not masks:
        return

    for si, mask_t in list(masks.items()):
        if mask_t.dtype == torch.bool:
            mask_np = mask_t.squeeze().cpu().numpy().astype(bool)  # [h, w]
            expanded = dilate_true_region(mask_np, expand_percent)
            masks[si] = (
                torch.tensor(expanded, dtype=torch.bool)
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()
            )
        else:
            mask_np = mask_t.squeeze().cpu().numpy().astype(np.float32)  # [h, w]
            h, w = mask_np.shape
            radius = int(round(min(h, w) * (expand_percent / 100.0)))
            if radius > 0:
                k = radius * 2 + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            mask_np = np.clip(mask_np, 0.0, 1.0)
            masks[si] = (
                torch.tensor(mask_np, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()
            )


def build_full_p2p_alignment(
    text_tokenizer,
    source_prompt: str,
    target_prompt: str,
    blended_words: Optional[List[str]] = None,
) -> Dict[int, int]:
    """
    Prompt-to-Prompt 風格的完整 token 對齊。

    使用 difflib.SequenceMatcher 先找出共同 token（自動對齊），
    再用 blended_words 覆蓋 swap token 的映射。

    回傳：target_token_idx -> source_token_idx
    不在 dict 中的 target token = 新增 token，保留 target 自己的 attention。
    """
    src_enc = text_tokenizer(
        text=[source_prompt], max_length=512,
        padding='max_length', truncation=True, return_tensors='pt',
    )
    tgt_enc = text_tokenizer(
        text=[target_prompt], max_length=512,
        padding='max_length', truncation=True, return_tensors='pt',
    )

    src_ids = src_enc.input_ids[0].tolist()
    tgt_ids = tgt_enc.input_ids[0].tolist()
    src_len = sum(src_enc.attention_mask[0].tolist())
    tgt_len = sum(tgt_enc.attention_mask[0].tolist())

    src_ids_real = src_ids[:src_len]
    tgt_ids_real = tgt_ids[:tgt_len]

    # 1. SequenceMatcher 找出共同 token（common tokens）
    matcher = difflib.SequenceMatcher(None, src_ids_real, tgt_ids_real)
    alignment: Dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        src_start, tgt_start, size = block
        for offset in range(size):
            alignment[tgt_start + offset] = src_start + offset

    # 2. blended_words 覆蓋 swap token 映射
    if blended_words:
        for bw in blended_words:
            parts = bw.split(',', 1)
            src_word = parts[0].strip()
            tgt_word = parts[1].strip() if len(parts) > 1 else ''
            if not src_word or not tgt_word:
                continue

            src_indices = find_focus_token_indices(
                text_tokenizer, source_prompt, [src_word], verbose=False,
            )
            tgt_indices = find_focus_token_indices(
                text_tokenizer, target_prompt, [tgt_word], verbose=False,
            )
            if not src_indices or not tgt_indices:
                continue

            for i, tgt_i in enumerate(tgt_indices):
                src_i = src_indices[min(i, len(src_indices) - 1)]
                alignment[tgt_i] = src_i

    return alignment


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
    ref_mask_rle: Optional[List[int]] = None,
) -> bool:
    """
    執行一個 P2P-Edit 案例的完整 7 phase 管線。
    mask_path: PIE-Bench mask.png 路徑。不為 None 時啟用 PIE mask 模式：
      • 跳過 attention 閾值遮罩計算（Phase 1.5）
      • 以 PIE mask 等比例縮放作為各 scale 的 replacement mask
    ref_mask_rle: PIE-Bench mapping_file.json 中的 mask RLE（(start,count) 對）。
      若提供且 args.use_dynamic_threshold=1，則使用二分法搜尋閾值。
    成功回傳 True，失敗拋出例外。
    """
    pie_mode = int(args.use_pie_mask) if (mask_path is not None) else 0
    use_pie_mask = (pie_mode == 1)
    use_pie_ratio_threshold = (pie_mode == 2)
    os.makedirs(save_dir, exist_ok=True)
    ensure_case_reference_symlink(case_source_dir or '', save_dir)

    # ── Reference mask for dynamic threshold (binary search) ──
    use_dynamic_thr = bool(getattr(args, 'use_dynamic_threshold', 0))
    dynamic_thr_iters = int(getattr(args, 'dynamic_threshold_iters', 20))
    threshold_method = int(getattr(args, 'threshold_method', 1))
    ref_mask_2d: Optional[np.ndarray] = None
    if use_dynamic_thr and ref_mask_rle is not None and len(ref_mask_rle) >= 2:
        ref_mask_2d = decode_rle_mask(ref_mask_rle, hw=(512, 512))
        fg_ratio = ref_mask_2d.sum() / ref_mask_2d.size * 100
        print(f"    [DynThresh] Reference mask decoded: {ref_mask_2d.sum()} fg pixels ({fg_ratio:.1f}%)")

    attn_cache_enabled = bool(args.use_attn_cache) and int(args.attn_cache_max_scale) > 0
    attn_cache_phase = args.attn_cache_phase if attn_cache_enabled else 'off'
    attn_cache_apply_scales = set(range(min(int(args.attn_cache_max_scale), total_scales)))
    attention_cache_root = os.path.join(save_dir, 'attention_cache')

    cache_pairs: List[AttentionCachePair] = []
    source_terms_to_export: Dict[str, List[int]] = {}
    replacement_maps: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = {}
    if attn_cache_enabled:
        cache_pairs, source_terms_to_export = build_attention_cache_pairs(
            text_tokenizer=text_tokenizer,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            blended_words=blended_words or [],
        )
        save_attention_cache_metadata(
            save_dir=attention_cache_root,
            cache_pairs=cache_pairs,
            source_terms_to_export=source_terms_to_export,
            args=args,
            total_scales=total_scales,
        )
        print(
            f"    [AttnCache] phase={attn_cache_phase}, scales=1~{min(int(args.attn_cache_max_scale), total_scales)}, "
            f"replace_pairs={len(cache_pairs)}, export_terms={len(source_terms_to_export)}"
        )

    source_focus_words_list = source_focus_words
    target_focus_words_list = target_focus_words

    # ── Focus Token Indices ──
    source_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, source_prompt, source_focus_words_list)
        if source_focus_words_list else []
    )
    target_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, target_prompt, target_focus_words_list)
        if target_focus_words_list else []
    )
    has_source_focus = bool(source_focus_token_indices)
    has_target_focus = bool(target_focus_token_indices)
    single_focus_fallback = has_source_focus ^ has_target_focus
    single_focus_side = "source" if has_source_focus else ("target" if has_target_focus else "none")
    if single_focus_fallback:
        print(
            f"    ℹ Single-focus fallback: 僅使用 {single_focus_side} focus mask，"
            "跳過 Phase 1.7"
        )

    # ─────────────────────────────────────
    # Phase 0：編碼 Source Image
    # ─────────────────────────────────────
    source_pil_img = Image.open(source_image_path).convert('RGB')
    source_image_np_for_threshold = np.array(source_pil_img)  # (H, W, 3) uint8

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

    # ── 預先載入 PIE mask（mode=1: 直接當 replacement；mode=2: 只取 white ratio）──
    pie_scale_masks: Dict[int, np.ndarray] = {}
    pie_mask_white_ratio: float = 0.0
    pie_mask_forced_off: bool = False
    case_attn_threshold = float(args.attn_threshold_percentile)
    if use_pie_mask:
        pie_scale_masks, pie_mask_white_ratio = load_and_resize_pie_masks(
            mask_path, scale_schedule, args.num_full_replace_scales
        )
        # 若 mask 全白，視為不可用：回退到 source/target attention mask
        if pie_mask_white_ratio >= (1.0 - 1e-6):
            pie_mask_forced_off = True
            use_pie_mask = False
            pie_scale_masks = {}
            print(
                f"    ⚠ PIE mask 幾乎全白 ({pie_mask_white_ratio * 100:.2f}%)，"
                "改用 source/target attention mask"
            )
    elif use_pie_ratio_threshold:
        _, pie_mask_white_ratio = load_and_resize_pie_masks(
            mask_path, scale_schedule, args.num_full_replace_scales
        )
        case_attn_threshold = map_white_ratio_to_threshold(
            pie_mask_white_ratio, thr_min=65.0, thr_max=95.0
        )
        print(
            f"    ℹ PIE ratio→threshold: white={pie_mask_white_ratio * 100:.2f}% "
            f"-> attn_threshold_percentile={case_attn_threshold:.2f}"
        )

    if single_focus_fallback and use_pie_mask:
        use_pie_mask = False
        pie_scale_masks = {}
        print("    ℹ Single-focus fallback：忽略提供的 PIE mask，改用單邊 attention mask")

    # ─────────────────────────────────────
    # Phase 1：Source 生成 + Attention 擷取
    # PIE mask 模式（純淨）：不需要 attention → 跳過 register_patches
    # PIE mask + attn_fallback 模式：仍需要 attention → 正常掛 hook
    # ─────────────────────────────────────
    use_full_p2p = attn_cache_enabled and getattr(args, 'attn_cache_align_mode', 'blended') == 'full_p2p'
    need_source_attn = (
        bool(source_focus_token_indices) and (
            not use_pie_mask or args.pie_mask_attn_fallback or attn_cache_enabled
        )
    ) or use_full_p2p  # full_p2p 模式永遠需要 source attention
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

    if attn_cache_enabled and source_extractor.attention_maps and source_terms_to_export:
        replacement_maps, export_maps = build_attention_cache_from_source(
            extractor=source_extractor,
            cache_pairs=cache_pairs,
            source_terms_to_export=source_terms_to_export,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
        )
        save_attention_cache_maps(
            export_maps=export_maps,
            scale_schedule=scale_schedule,
            save_root=os.path.join(attention_cache_root, 'source_terms'),
        )

    # ── Full P2P alignment（full_p2p 模式）──
    full_p2p_alignment: Dict[int, int] = {}
    if use_full_p2p and source_extractor.attention_maps:
        full_p2p_alignment = build_full_p2p_alignment(
            text_tokenizer=text_tokenizer,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            blended_words=blended_words,
        )
        print(
            f"    [FullP2P] alignment={len(full_p2p_alignment)} token pairs, "
            f"scales=0~{min(int(args.attn_cache_max_scale), total_scales) - 1}"
        )

    # ─────────────────────────────────────
    # Phase 1.5：Dual Attention Masks
    # 純淨 PIE mask 模式：跳過；fallback 模式：正常計算（用於二次篩選）
    # ─────────────────────────────────────
    source_text_masks: Dict[int, np.ndarray] = {}
    source_low_attn_masks: Dict[int, np.ndarray] = {}
    use_last_scale = bool(getattr(args, 'use_last_scale_mask', 0))
    run_source_attn = need_source_attn and len(source_extractor.attention_maps) > 0
    if run_source_attn:
        if use_dynamic_thr and ref_mask_2d is not None:
            # ── Dynamic threshold via binary search (Algorithm 1) ──
            source_text_masks = collect_attention_text_masks_dynamic(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=args.num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                ref_mask=ref_mask_2d,
                max_iters=dynamic_thr_iters,
                fallback_percentile=case_attn_threshold,
                label="source",
                low_attn=False,
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
            source_low_attn_masks = collect_attention_text_masks_dynamic(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=args.num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                ref_mask=ref_mask_2d,
                max_iters=dynamic_thr_iters,
                fallback_percentile=case_attn_threshold,
                label="source_preserve",
                low_attn=True,
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
        else:
            # ── Fixed percentile threshold (fallback) ──
            _collect_fn = collect_last_scale_attention_mask if use_last_scale else collect_attention_text_masks
            _collect_kwargs = dict(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
                threshold_percentile=case_attn_threshold,
                label="source",
                low_attn=False,
                use_normalized_attn=bool(getattr(args, 'use_normalized_attn', 0)),
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
            if use_last_scale:
                _collect_kwargs["start_scale"] = args.num_full_replace_scales
                _collect_kwargs["majority_threshold"] = float(getattr(args, 'last_scale_majority_threshold', 0.5))
            else:
                _collect_kwargs["num_full_replace_scales"] = args.num_full_replace_scales
            source_text_masks = _collect_fn(**_collect_kwargs)

            _collect_kwargs_low = dict(
                extractor=source_extractor,
                focus_token_indices=source_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
                threshold_percentile=case_attn_threshold,
                label="source_preserve",
                low_attn=True,
                use_normalized_attn=bool(getattr(args, 'use_normalized_attn', 0)),
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
            if use_last_scale:
                _collect_kwargs_low["start_scale"] = args.num_full_replace_scales
                _collect_kwargs_low["majority_threshold"] = float(getattr(args, 'last_scale_majority_threshold', 0.5))
            else:
                _collect_kwargs_low["num_full_replace_scales"] = args.num_full_replace_scales
            source_low_attn_masks = _collect_fn(**_collect_kwargs_low)

    # ─────────────────────────────────────
    # Phase 1.6：建立 Phase 1.7 Preserve Storage
    # PIE mask 模式：以 pie_scale_masks（黑=背景=True）作為 preserve 遮罩
    # 一般模式：使用 source low-attention mask（同以前）
    # ─────────────────────────────────────
    phase17_storage: Optional[BitwiseTokenStorage] = None
    if (not single_focus_fallback) and use_pie_mask and pie_scale_masks and image_scale_tokens:
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
    elif (not single_focus_fallback) and source_low_attn_masks and image_scale_tokens:
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
    run_phase17 = (not single_focus_fallback) and (phase17_storage is not None) and (
        use_pie_mask or has_target_focus
    )
    if run_phase17:
        use_phase17_attn_cache_blended = (
            attn_cache_enabled and not use_full_p2p
            and attn_cache_phase in ('phase17', 'both') and bool(replacement_maps)
        )
        use_phase17_full_p2p = (
            use_full_p2p and attn_cache_phase in ('phase17', 'both') and bool(full_p2p_alignment)
        )
        # PIE fallback 模式也需要 target attention；純淨 PIE 模式不需要
        need_target_attn = (
            not use_pie_mask or args.pie_mask_attn_fallback
        ) and has_target_focus
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
            capture_attention=need_target_attn,
            replacement_maps=replacement_maps if use_phase17_attn_cache_blended else None,
            replace_scales=sorted(attn_cache_apply_scales) if use_phase17_attn_cache_blended else None,
        ) if (need_target_attn or use_phase17_attn_cache_blended) else None
        if target_extractor is not None:
            target_extractor.register_patches()

        # Full P2P injector：套在 extractor 之上（後掛先卸）
        phase17_injector: Optional[AttentionCacheInjector] = None
        if use_phase17_full_p2p:
            phase17_injector = AttentionCacheInjector(
                model=infinity,
                source_attention_maps=source_extractor.attention_maps,
                block_indices=attn_block_indices,
                token_alignment=full_p2p_alignment,
                max_scale=min(int(args.attn_cache_max_scale), total_scales) - 1,
                batch_idx=args.attn_batch_idx,
            )
            phase17_injector.register_patches()

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _phase17_img = gen_one_img(
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
        if getattr(args, 'debug_mode', 0):
            _dbg_np = _phase17_img.cpu().numpy()
            if _dbg_np.dtype != np.uint8:
                _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, 'phase17_guided.jpg'), _dbg_np)
            del _dbg_np
        del _phase17_img

        # 卸載順序：先卸 injector（後掛先卸），再卸 extractor
        if phase17_injector is not None:
            phase17_injector.remove_patches()
        if target_extractor is not None:
            target_extractor.remove_patches()
        if need_target_attn and target_extractor is not None:
            if use_dynamic_thr and ref_mask_2d is not None:
                target_text_masks = collect_attention_text_masks_dynamic(
                    extractor=target_extractor,
                    focus_token_indices=target_focus_token_indices,
                    scale_schedule=scale_schedule,
                    num_full_replace_scales=args.num_full_replace_scales,
                    attn_block_indices=attn_block_indices,
                    ref_mask=ref_mask_2d,
                    max_iters=dynamic_thr_iters,
                    fallback_percentile=case_attn_threshold,
                    label="target",
                    low_attn=False,
                    threshold_method=threshold_method,
                    source_image_np=source_image_np_for_threshold,
                )
            else:
                _tgt_collect_fn = collect_last_scale_attention_mask if use_last_scale else collect_attention_text_masks
                _tgt_kwargs = dict(
                    extractor=target_extractor,
                    focus_token_indices=target_focus_token_indices,
                    scale_schedule=scale_schedule,
                    attn_block_indices=attn_block_indices,
                    threshold_percentile=case_attn_threshold,
                    label="target",
                    use_normalized_attn=bool(getattr(args, 'use_normalized_attn', 0)),
                    threshold_method=threshold_method,
                    source_image_np=source_image_np_for_threshold,
                )
                if use_last_scale:
                    _tgt_kwargs["start_scale"] = args.num_full_replace_scales
                    _tgt_kwargs["majority_threshold"] = float(getattr(args, 'last_scale_majority_threshold', 0.5))
                else:
                    _tgt_kwargs["num_full_replace_scales"] = args.num_full_replace_scales
                target_text_masks = _tgt_collect_fn(**_tgt_kwargs)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ─────────────────────────────────────
    # Single-focus fallback（target-only）：
    # 跳過 Phase 1.7 時，若只有 target focus，仍需獨立擷取 target mask
    # ─────────────────────────────────────
    if single_focus_fallback and (single_focus_side == 'target') and has_target_focus:
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
            capture_attention=True,
        )
        target_extractor.register_patches()
        # Single-focus fallback：改用 source image 連續特徵注入前 N scale 作為結構錨定
        _fallback_scales = getattr(args, 'phase17_fallback_replace_scales', 0)
        # 建立 Phase 1.7 fallback 的 inject_schedule：
        # 前 _fallback_scales 個 scale 100% 注入 source image 連續特徵（weight=0.0），
        # 其餘 scale 自由生成（weight=1.0），讓 target 內容在細節 scale 自然顯現。
        # NOTE: Phase 1 的結構錨定靠的是連續特徵注入（summed_codes），
        #       而非離散 token 替換。必須傳入 inject_image_features 才能重現結構。
        _fb_inject_schedule: Optional[List[float]] = None
        _fb_inject_features = None
        if _fallback_scales > 0 and image_raw_features is not None:
            _fb_inject_schedule = (
                [0.0] * min(_fallback_scales, total_scales)
                + [1.0] * max(0, total_scales - _fallback_scales)
            )
            _fb_inject_features = image_raw_features
            print(f"  ℹ Phase 1.7 single-focus fallback: source image 注入前 {_fallback_scales} 個 scale（連續特徵）")
        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _fallback17_img = gen_one_img(
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
                    p2p_token_storage=None,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=False,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=0,
                    inject_image_features=_fb_inject_features,
                    inject_schedule=_fb_inject_schedule,
                )
        if getattr(args, 'debug_mode', 0):
            _dbg_np = _fallback17_img.cpu().numpy()
            if _dbg_np.dtype != np.uint8:
                _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, 'phase17_fallback.jpg'), _dbg_np)
            del _dbg_np
        del _fallback17_img
        target_extractor.remove_patches()
        if use_dynamic_thr and ref_mask_2d is not None:
            target_text_masks = collect_attention_text_masks_dynamic(
                extractor=target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                num_full_replace_scales=args.num_full_replace_scales,
                attn_block_indices=attn_block_indices,
                ref_mask=ref_mask_2d,
                max_iters=dynamic_thr_iters,
                fallback_percentile=case_attn_threshold,
                label="target",
                low_attn=False,
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
        else:
            _fb_collect_fn = collect_last_scale_attention_mask if use_last_scale else collect_attention_text_masks
            _fb_kwargs = dict(
                extractor=target_extractor,
                focus_token_indices=target_focus_token_indices,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
                threshold_percentile=case_attn_threshold,
                label="target",
                use_normalized_attn=bool(getattr(args, 'use_normalized_attn', 0)),
                threshold_method=threshold_method,
                source_image_np=source_image_np_for_threshold,
            )
            if use_last_scale:
                _fb_kwargs["start_scale"] = args.num_full_replace_scales
                _fb_kwargs["majority_threshold"] = float(getattr(args, 'last_scale_majority_threshold', 0.5))
            else:
                _fb_kwargs["num_full_replace_scales"] = args.num_full_replace_scales
            target_text_masks = _fb_collect_fn(**_fb_kwargs)
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

    if args.use_cumulative_prob_mask and p2p_token_storage.masks:
        p2p_token_storage.masks = build_cumulative_replacement_prob_masks(
            masks=p2p_token_storage.masks,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
        )

    # 每個 scale 的 replacement mask（True=保留 source）向外擴張
    expand_storage_masks_inplace(
        p2p_token_storage.masks,
        expand_percent=float(args.mask_expand_percent),
    )

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
            has_prob_mask = any(m.dtype != torch.bool for m in p2p_token_storage.masks.values())
            if has_prob_mask:
                combined_prob_vis = {
                    si: _mask_tensor_to_prob_map(m)
                    for si, m in p2p_token_storage.masks.items()
                }
                _save_prob_masks_to_dir(
                    prob_masks=combined_prob_vis,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, 'combined'),
                    file_prefix='combined_replace_prob',
                )
            else:
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
    phase2_attn_cache_blended = (
        attn_cache_enabled and not use_full_p2p
        and attn_cache_phase in ('phase2', 'both') and bool(replacement_maps)
    )
    phase2_full_p2p = (
        use_full_p2p and attn_cache_phase in ('phase2', 'both') and bool(full_p2p_alignment)
    )
    phase2_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
        capture_attention=False,
        replacement_maps=replacement_maps if phase2_attn_cache_blended else None,
        replace_scales=sorted(attn_cache_apply_scales) if phase2_attn_cache_blended else None,
    ) if phase2_attn_cache_blended else None
    if phase2_extractor is not None:
        phase2_extractor.register_patches()

    # Full P2P injector for Phase 2
    phase2_injector: Optional[AttentionCacheInjector] = None
    if phase2_full_p2p:
        phase2_injector = AttentionCacheInjector(
            model=infinity,
            source_attention_maps=source_extractor.attention_maps,
            block_indices=attn_block_indices,
            token_alignment=full_p2p_alignment,
            max_scale=min(int(args.attn_cache_max_scale), total_scales) - 1,
            batch_idx=args.attn_batch_idx,
        )
        phase2_injector.register_patches()

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

    # 卸載順序：先卸 injector，再卸 extractor
    if phase2_injector is not None:
        phase2_injector.remove_patches()
    if phase2_extractor is not None:
        phase2_extractor.remove_patches()

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

    # ── 批量設定（bench_dir / output_dir 選填；未提供時走單一案例模式）──
    parser.add_argument('--bench_dir', type=str, default=None,
                        help='批量模式：extracted_pie_bench 根目錄')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='批量模式：輸出根目錄（會依照原始資料夾結構存放結果）')
    parser.add_argument('--categories', type=str, default='',
                        help='只跑指定 category（逗號分隔資料夾名稱）；預設全部')
    parser.add_argument('--max_per_cat', type=int, default=-1,
                        help='每個 category 最多處理幾個案例（-1 = 全部）')
    parser.add_argument('--skip_existing', type=int, default=1, choices=[0, 1],
                        help='若 target.jpg 已存在則跳過（預設：1）')

    # ── 單一案例模式（不使用 bench_dir 時）──
    parser.add_argument('--source_image', type=str, default=None,
                        help='單一模式：source 圖片路徑')
    parser.add_argument('--source_prompt', type=str, default='',
                        help='單一模式：source prompt')
    parser.add_argument('--target_prompt', type=str, default='',
                        help='單一模式：target prompt')
    parser.add_argument('--source_focus_words', type=str, default='',
                        help='單一模式：source focus 詞彙（空格分隔）')
    parser.add_argument('--target_focus_words', type=str, default='',
                        help='單一模式：target focus 詞彙（空格分隔）')
    parser.add_argument('--save_file', type=str, default='./outputs/pie_edit_single',
                        help='單一模式：輸出目錄')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='單一模式：PIE mask.png 路徑（可選）')

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
    parser.add_argument('--use_pie_mask', type=int, default=0, choices=[0, 1, 2],
                        help='PIE mask 使用模式：0=關閉、1=直接作為 replacement mask、'
                            '2=僅用白色比例反推 attn_threshold_percentile（65~92 反向線性）。')
    parser.add_argument('--pie_mask_attn_fallback', type=int, default=0, choices=[0, 1],
                        help='（需 use_pie_mask=1）在白色（編輯）區域內，'
                             '以 attention mask 二次篩選「真正需要編輯」的 token；'
                             '白色區域中 attention 未聚焦的 token 仍保留 source。'
                             '組合公式：final_mask = pie_bg OR attn_replacement')
    parser.add_argument('--mask_expand_percent', type=float, default=0.0,
                        help='每個 scale 對最終 replacement mask 向外擴張 True 區域的比例（%% of min(H,W)）。')
    parser.add_argument('--use_attn_cache', type=int, default=0, choices=[0, 1],
                        help='是否啟用 blended_words 對齊的 attention cache / replace（Prompt-to-Prompt 風格）。')
    parser.add_argument('--attn_cache_phase', type=str, default='phase2',
                        choices=['phase17', 'phase2', 'both'],
                        help='在哪個 target generation phase 套用 attention cache。')
    parser.add_argument('--attn_cache_max_scale', type=int, default=13,
                        help='從 scale1 開始，最多套用到第幾個 scale（1-based；需 >0 才會生效）。')
    parser.add_argument('--attn_cache_align_mode', type=str, default='blended',
                        choices=['blended', 'full_p2p'],
                        help='Attention cache 對齊模式：'
                             'blended = 僅 blended_words 指定的 token 做替換（現有行為）；'
                             'full_p2p = 完整 Prompt-to-Prompt 對齊'
                             '（所有共同 token + swap token 都注入 source attention）。')
    parser.add_argument('--phase17_fallback_replace_scales', type=int, default=4,
                        help='Single-focus fallback（只有 target focus）時，'
                             'Phase 1.7 以 source gen token 替換前幾個 scale（0=停用）。預設：4')
    parser.add_argument('--use_normalized_attn', type=int, default=0, choices=[0, 1],
                        help='使用 z-score normalized threshold 取代固定 percentile（0=停用，1=啟用）')
    parser.add_argument('--threshold_method', type=int, default=1, choices=list(range(1, 14)),
                        help='閾值方法：1=固定percentile 2=dynamic ternary 3=Otsu 4=FFT+Otsu '
                             '5=SpectralEnergy 6=EdgeCoherence 7=GMM 8=Composite '
                             '9=IPR 10=Entropy 11=BlockConsensus 12=Kneedle '
                             '13=MetaAdaptive。預設：1')
    parser.add_argument('--use_last_scale_mask', type=int, default=0, choices=[0, 1],
                        help='僅從最後一個 scale 提取 attention mask，再向前逐步推導各 scale（0=停用，1=啟用）')
    parser.add_argument('--last_scale_majority_threshold', type=float, default=0.5,
                        help='Last-scale mask 向前推導時的多數投票閾值（預設 0.5 = 50%%）')
    parser.add_argument('--debug_mode', type=int, default=0, choices=[0, 1],
                        help='1=儲存所有中間過程圖片（phase17_guided.jpg、phase17_fallback.jpg）。預設：0')

    args = parser.parse_args()
    # 檢查模式：必須有 bench_dir 或 source_image
    if args.bench_dir is None and not args.source_image:
        parser.error('請提供 --bench_dir（批量模式）或 --source_image（单一模式）')
    if args.bench_dir is not None and args.output_dir is None:
        parser.error('批量模式需要指定 --output_dir')
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    if args.mask_expand_percent < 0:
        raise ValueError('--mask_expand_percent 必須 >= 0')
    if args.attn_cache_max_scale < 0:
        raise ValueError('--attn_cache_max_scale 必須 >= 0')

    print('\n' + '=' * 80)
    print('PIE-Bench 批量 P2P-Edit 評估')
    print('=' * 80)
    print(f'bench_dir      : {args.bench_dir}')
    print(f'output_dir     : {args.output_dir}')
    print(f'skip_existing  : {bool(args.skip_existing)}')
    print(f'max_per_cat    : {args.max_per_cat if args.max_per_cat > 0 else "全部"}')
    print(f'full_replace   : {args.num_full_replace_scales} scales')
    print(f'attn_percentile: {args.attn_threshold_percentile}')
    print(f'use_pie_mask   : {args.use_pie_mask} (0=off,1=direct,2=ratio->thr)')
    print(f'pie_attn_fallbk: {bool(args.pie_mask_attn_fallback)}')
    print(f'mask_expand_pct: {args.mask_expand_percent}')
    print(f'cum_prob_mask : {bool(args.use_cumulative_prob_mask)}')
    print(f'use_attn_cache : {bool(args.use_attn_cache)}')
    print(f'attn_cache_ph  : {args.attn_cache_phase}')
    print(f'attn_cache_max : {args.attn_cache_max_scale}')
    print(f'attn_cache_mode: {args.attn_cache_align_mode}')
    print(f'ph17_fb_scales : {args.phase17_fallback_replace_scales}')
    print(f'debug_mode     : {bool(args.debug_mode)}')
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

    # ══════════════════════════════════════════════════════
    # 單一案例模式
    # ══════════════════════════════════════════════════════
    if args.bench_dir is None:
        save_dir = os.path.abspath(args.save_file)
        os.makedirs(save_dir, exist_ok=True)

        src_focus_list = [w for w in args.source_focus_words.strip().split() if w]
        tgt_focus_list = [w for w in args.target_focus_words.strip().split() if w]

        # 若兩邊 focus 都未指定，自動從 prompt diff 推導
        if not src_focus_list and not tgt_focus_list:
            src_focus_list, tgt_focus_list = derive_focus_terms_from_prompt_diff(
                args.source_prompt, args.target_prompt
            )

        pie_mask_path: Optional[str] = (
            args.mask_path
            if getattr(args, 'mask_path', None) and os.path.exists(args.mask_path)
            else None
        )

        print(f'[單一模式]')
        print(f'  source_image  : {args.source_image}')
        print(f'  source_prompt : {args.source_prompt}')
        print(f'  target_prompt : {args.target_prompt}')
        print(f'  src_focus     : {src_focus_list}')
        print(f'  tgt_focus     : {tgt_focus_list}')
        print(f'  save_file     : {save_dir}')

        run_one_case(
            infinity=infinity,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            source_image_path=args.source_image,
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            source_focus_words=src_focus_list,
            target_focus_words=tgt_focus_list,
            save_dir=save_dir,
            args=args,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
            total_scales=total_scales,
            device_cuda=device_cuda,
            mask_path=pie_mask_path,
            blended_words=None,
            case_source_dir=None,
        )
        print(f'\n✓ 完成！結果儲存於： {save_dir}')
        return

    # ══════════════════════════════════════════════════════
    # 批量模式
    # ══════════════════════════════════════════════════════
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
            ensure_case_reference_symlink(case_dir, save_dir)

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
            src_diff_terms, tgt_diff_terms = derive_focus_terms_from_prompt_diff(
                source_prompt, target_prompt
            )
            source_focus_words = src_diff_terms
            target_focus_words = tgt_diff_terms

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
                t_start = time.time()
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
                    blended_words=blended_words,
                    case_source_dir=case_dir,
                )
                elapsed = time.time() - t_start
                # 儲存 timing.json
                timing_path = os.path.join(save_dir, 'timing.json')
                with open(timing_path, 'w') as f_t:
                    json.dump({'inference_sec': round(elapsed, 3)}, f_t)
                cat_done  += 1
                total_done += 1
                print(f'    ✓  → {save_dir}/target.jpg  ({elapsed:.1f}s)')

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
