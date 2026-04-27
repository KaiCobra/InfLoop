#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pie_edit_faceSwap.py — Face-Swap 版本的 P2P-Edit 單案例管線

與 tools/run_pie_edit.py 的差異：
  • 三個 phase（1 / 1.7 / 2）改成接收「預編碼好的」T5 text embedding
    (kv_compact, lens, cu_seqlens_k, Ltext)，繞過內部的 encode_prompt 呼叫
  • 不再使用 source_prompt / target_prompt 字串路徑生成；
    所有 phase 的 prompt 字面內容都是同一個固定 prompt T_t，
    只有 subject token (e.g. "boy") 的 embedding 被修改

模型 (infinity/models/infinity_p2p_edit.py) 完全沿用，不修改。
"""

import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (  # noqa: E402
    find_focus_token_indices,
    collect_attention_text_masks,
    collect_last_scale_attention_mask,
    combine_and_store_masks,
    build_cumulative_replacement_prob_masks,
    _mask_tensor_to_prob_map,
    _save_prob_masks_to_dir,
    _save_masks_to_dir,
    encode_image_to_raw_features,
    encode_image_to_scale_tokens,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage  # noqa: E402
from attention_map.extractor import CrossAttentionExtractor  # noqa: E402


# ============================================================
# Pre-encoded prompt 版本的單張生成
# ============================================================

def gen_one_img_kv(
    infinity_test,
    vae,
    text_cond_tuple,                         # (kv_compact, lens, cu_seqlens_k, Ltext)
    cfg_list=(),
    tau_list=(),
    negative_text_cond_tuple=None,
    scale_schedule=None,
    top_k: int = 900,
    top_p: float = 0.97,
    cfg_sc: int = 3,
    cfg_exp_k: float = 0.0,
    cfg_insertion_layer: int = -5,
    vae_type: int = 0,
    gumbel: int = 0,
    softmax_merge_topk: int = -1,
    gt_leak: int = -1,
    gt_ls_Bl=None,
    g_seed: Optional[int] = None,
    sampling_per_bits: int = 1,
    p2p_token_storage=None,
    p2p_token_replace_prob: float = 0.0,
    p2p_use_mask: bool = False,
    p2p_save_tokens: bool = True,
    p2p_attn_full_replace_scales: int = 0,
    inject_image_features=None,
    inject_schedule=None,
):
    """gen_one_img 的精簡版本：直接使用預編碼的 text_cond_tuple，不再呼叫 encode_prompt。"""
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=g_seed,
            B=1,
            negative_label_B_or_BLT=negative_text_cond_tuple,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=gumbel,
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            p2p_token_storage=p2p_token_storage,
            p2p_token_replace_prob=p2p_token_replace_prob,
            p2p_use_mask=p2p_use_mask,
            p2p_save_tokens=p2p_save_tokens,
            p2p_attn_full_replace_scales=p2p_attn_full_replace_scales,
            inject_image_features=inject_image_features,
            inject_schedule=inject_schedule,
        )
    img = img_list[0]
    del img_list
    return img


# ============================================================
# 單一 case：face-swap 版本
# ============================================================

def run_one_case_faceSwap(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,                            # 保留參數簽名，便於與其他 helper 介面一致
    source_image_path: str,                  # = B 的路徑
    prompt_text: str,                        # T_t（用於 find_focus_token_indices）
    subject_word: str,                       # "boy"
    kv_phase1: tuple,                        # T_t 原始 embedding
    kv_phase17: tuple,                       # T_t 中 e_I -= proj(e_B)
    kv_phase2: tuple,                        # T_t 中 e_I = proj(e_A)
    save_dir: str,
    args,
    scale_schedule: list,
    attn_block_indices: List[int],
    total_scales: int,
    device_cuda: torch.device,
) -> bool:
    """執行 face-swap 版本的 P2P-Edit 7 phase 管線（B 為 source image）。

    Phase 對應：
      • Phase 1：使用 kv_phase1（T_t 原樣）→ 重建 source-like 圖、抓 cross-attn
      • Phase 1.7：使用 kv_phase17（boy token -= e_B）→ 引導生成（背景錨定）
      • Phase 2：使用 kv_phase2（boy token = e_A）→ 最終 face-swap 結果

    全程不再呼叫 encode_prompt；attention focus 的 token index 由 prompt_text + subject_word
    透過 find_focus_token_indices 取得。
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Reference for dynamic threshold not used in face-swap pipeline ──
    use_dynamic_thr = False
    threshold_method = int(getattr(args, "threshold_method", 1))
    case_attn_threshold = float(args.attn_threshold_percentile)

    # ── Focus tokens：source/target focus 都用 subject_word（同一個 token 位置）──
    focus_words = [subject_word] if subject_word else []
    focus_token_indices = (
        find_focus_token_indices(text_tokenizer, prompt_text, focus_words)
        if focus_words else []
    )
    has_focus = bool(focus_token_indices)
    if not has_focus:
        print(f"    ⚠ subject_word='{subject_word}' 在 prompt 中找不到對應 token，"
              "後續 attention mask 將為空（face swap 效果可能不明顯）。")

    # ─────────────────────────────────────
    # Phase 0：編碼 Source Image (= B)
    # ─────────────────────────────────────
    source_pil_img = Image.open(source_image_path).convert("RGB")
    source_image_np_for_threshold = np.array(source_pil_img)

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

    if args.inject_weights.strip():
        parsed_w = [float(w) for w in args.inject_weights.strip().split()]
        if len(parsed_w) != total_scales:
            raise ValueError(
                f"--inject_weights 長度 ({len(parsed_w)}) 與 scale 總數 ({total_scales}) 不符"
            )
        inject_schedule = parsed_w
    else:
        inject_schedule = [
            0.0 if si < args.image_injection_scales else 1.0
            for si in range(total_scales)
        ]

    # ─────────────────────────────────────
    # Phase 1：Source 生成 + Attention 擷取（kv_phase1）
    # ─────────────────────────────────────
    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device="cpu")
    source_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
    )
    if has_focus:
        source_extractor.register_patches()

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_image = gen_one_img_kv(
                infinity, vae,
                text_cond_tuple=kv_phase1,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    if has_focus:
        source_extractor.remove_patches()

    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "source.jpg"), img_np)

    # ─────────────────────────────────────
    # Phase 1.5：Source attention masks（focus / preserve）
    # ─────────────────────────────────────
    source_text_masks: Dict[int, np.ndarray] = {}
    source_low_attn_masks: Dict[int, np.ndarray] = {}
    use_last_scale = bool(getattr(args, "use_last_scale_mask", 0))
    run_source_attn = has_focus and len(source_extractor.attention_maps) > 0

    if run_source_attn:
        _collect_fn = collect_last_scale_attention_mask if use_last_scale else collect_attention_text_masks
        common_kwargs = dict(
            extractor=source_extractor,
            focus_token_indices=focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
            threshold_percentile=case_attn_threshold,
            use_normalized_attn=bool(getattr(args, "use_normalized_attn", 0)),
            threshold_method=threshold_method,
            source_image_np=source_image_np_for_threshold,
            absolute_high=float(getattr(args, "absolute_high", 0.7)),
            absolute_low=float(getattr(args, "absolute_low", 0.3)),
        )

        focus_kwargs = dict(common_kwargs, label="source", low_attn=False)
        preserve_kwargs = dict(common_kwargs, label="source_preserve", low_attn=True)
        if use_last_scale:
            focus_kwargs["start_scale"] = args.num_full_replace_scales
            focus_kwargs["majority_threshold"] = float(getattr(args, "last_scale_majority_threshold", 0.5))
            preserve_kwargs["start_scale"] = args.num_full_replace_scales
            preserve_kwargs["majority_threshold"] = float(getattr(args, "last_scale_majority_threshold", 0.5))
        else:
            focus_kwargs["num_full_replace_scales"] = args.num_full_replace_scales
            preserve_kwargs["num_full_replace_scales"] = args.num_full_replace_scales

        source_text_masks = _collect_fn(**focus_kwargs)
        source_low_attn_masks = _collect_fn(**preserve_kwargs)

    # ─────────────────────────────────────
    # Phase 1.6：Phase 1.7 preserve storage（用 source low-attn mask 錨定 background）
    # ─────────────────────────────────────
    phase17_storage: Optional[BitwiseTokenStorage] = None
    if source_low_attn_masks and image_scale_tokens:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device="cpu")
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
    # Phase 1.7：Target guided gen（kv_phase17）→ Target focus mask
    # ─────────────────────────────────────
    target_text_masks: Dict[int, np.ndarray] = {}
    if phase17_storage is not None and has_focus:
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
            capture_attention=True,
        )
        target_extractor.register_patches()

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _phase17_img = gen_one_img_kv(
                    infinity, vae,
                    text_cond_tuple=kv_phase17,
                    g_seed=args.seed,
                    gt_leak=0, gt_ls_Bl=None,
                    cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    p2p_token_storage=phase17_storage,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=True,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=args.num_full_replace_scales,
                    inject_image_features=None,
                    inject_schedule=None,
                )
        if getattr(args, "debug_mode", 0):
            _dbg_np = _phase17_img.cpu().numpy()
            if _dbg_np.dtype != np.uint8:
                _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, "phase17_guided.jpg"), _dbg_np)
            del _dbg_np
        del _phase17_img

        target_extractor.remove_patches()

        _tgt_collect_fn = collect_last_scale_attention_mask if use_last_scale else collect_attention_text_masks
        _tgt_kwargs = dict(
            extractor=target_extractor,
            focus_token_indices=focus_token_indices,
            scale_schedule=scale_schedule,
            attn_block_indices=attn_block_indices,
            threshold_percentile=case_attn_threshold,
            label="target",
            use_normalized_attn=bool(getattr(args, "use_normalized_attn", 0)),
            threshold_method=threshold_method,
            source_image_np=source_image_np_for_threshold,
            absolute_high=float(getattr(args, "absolute_high", 0.7)),
            absolute_low=float(getattr(args, "absolute_low", 0.3)),
        )
        if use_last_scale:
            _tgt_kwargs["start_scale"] = args.num_full_replace_scales
            _tgt_kwargs["majority_threshold"] = float(getattr(args, "last_scale_majority_threshold", 0.5))
        else:
            _tgt_kwargs["num_full_replace_scales"] = args.num_full_replace_scales
        target_text_masks = _tgt_collect_fn(**_tgt_kwargs)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ─────────────────────────────────────
    # Phase 1.9：寫入 replacement mask（source ∪ target focus）
    # ─────────────────────────────────────
    if source_text_masks or target_text_masks:
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

    # ─────────────────────────────────────
    # Attention / Mask 視覺化
    # ─────────────────────────────────────
    if args.save_attn_vis:
        attn_vis_dir = os.path.join(save_dir, "attn_masks")
        if source_text_masks:
            _save_masks_to_dir(
                bool_masks=source_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "source"),
                file_prefix="source_focus",
                invert=True,
            )
        if source_low_attn_masks:
            _save_masks_to_dir(
                bool_masks=source_low_attn_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "phase17_preserve"),
                file_prefix="preserve",
                invert=False,
            )
        if target_text_masks:
            _save_masks_to_dir(
                bool_masks=target_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "target"),
                file_prefix="target_focus",
                invert=True,
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
                    vis_dir=os.path.join(attn_vis_dir, "combined"),
                    file_prefix="combined_replace_prob",
                )
            else:
                combined_vis = {
                    si: ~m.squeeze().numpy()
                    for si, m in p2p_token_storage.masks.items()
                }
                _save_masks_to_dir(
                    bool_masks=combined_vis,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "combined"),
                    file_prefix="combined_focus",
                    invert=True,
                )

    # ─────────────────────────────────────
    # Phase 2：最終 face-swap 生成（kv_phase2）
    # ─────────────────────────────────────
    has_mask = len(p2p_token_storage.masks) > 0
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_image = gen_one_img_kv(
                infinity, vae,
                text_cond_tuple=kv_phase2,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
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
    cv2.imwrite(os.path.join(save_dir, "target.jpg"), img_np)

    del source_image, target_image, img_np, p2p_token_storage
    if phase17_storage is not None:
        del phase17_storage
    torch.cuda.empty_cache()

    return True
