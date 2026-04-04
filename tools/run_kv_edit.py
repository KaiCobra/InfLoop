#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_kv_edit.py — KV-Edit 交錯式管線

核心理念（對比原始 P2P-Edit 管線）：
  原始：source gen (scale 0→12) → phase 1.7 (scale 0→12) → target gen (scale 0→12)
  KV-Edit：每個 scale 縱向處理三個 phase：
    scale 0: source → phase17 → target
    scale 1: source → phase17 → target
    ...
    scale 12: source → phase17 → target

優勢：
  1. 每個 scale 結束後，source 的 self-attention KV cache 可以注入到 target，
     讓 target 在中間 scale 保留結構資訊。
  2. Dynamic mask：使用 cross-attention peak + gradient flood fill 取代固定 percentile。
  3. 三個 phase 的 bitwise token 可以即時交換，不需等待整個 phase 完成。

用法：
  python3 tools/run_kv_edit.py \\
      --source_image path/to/image.jpg \\
      --source_prompt "..." \\
      --target_prompt "..." \\
      --source_focus_words "word1 word2" \\
      --target_focus_words "word1 word2" \\
      --save_dir ./outputs/kv_edit_test
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (
    add_common_arguments,
    encode_prompt,
    encode_image_to_raw_features,
    encode_image_to_scale_tokens,
    find_focus_token_indices,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    _save_masks_to_dir,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.kv_cache_manager import KVCacheManager
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from infinity.models.basic import CrossAttnBlock
from attention_map.extractor import CrossAttentionExtractor


# ============================================================
# Dynamic Mask：gradient-based flood fill from attention peak
# ============================================================

def compute_dynamic_mask(
    attn_map: np.ndarray,
    spatial_size: Tuple[int, int],
    min_ratio: float = 0.02,
    max_ratio: float = 0.85,
    gradient_threshold: float = 0.3,
) -> np.ndarray:
    """
    從 cross-attention map 的 peak 開始，以 attention score 強度向外擴散，
    生成動態 spatial mask。

    不使用固定 percentile，而是：
      1. 找到 attention map 中最大值的位置（peak）
      2. 從 peak 開始 BFS flood fill
      3. 擴散條件：鄰居的 attention score >= peak * adaptive_ratio
         或 鄰居與當前位置的 attention 差異（梯度）小於 gradient_threshold
      4. 自動確保 mask 面積在 [min_ratio, max_ratio] 之間

    Args:
        attn_map: [H, W] attention map（已 normalize 到 0~1 或任意正值）
        spatial_size: (H, W) 輸出尺寸
        min_ratio: mask 最小面積比例
        max_ratio: mask 最大面積比例
        gradient_threshold: 梯度擴散閾值（相對於 peak 值）

    Returns:
        bool mask [H, W]，True = 編輯區域（高 attention）
    """
    H, W = spatial_size

    # Resize attn_map to target spatial size
    if attn_map.shape != (H, W):
        attn_map = cv2.resize(
            attn_map.astype(np.float32), (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    peak_val = float(attn_map.max())
    if peak_val <= 0:
        return np.zeros((H, W), dtype=bool)

    # Normalize to [0, 1]
    attn_norm = attn_map / peak_val

    # Binary search for the best threshold that gives a reasonable mask area
    # Start from the peak and lower the threshold until we have enough coverage
    best_mask = None
    best_score = -1.0

    # Try multiple thresholds: from high (tight around peak) to low (broader)
    for threshold_frac in np.linspace(0.8, 0.05, 20):
        # Flood fill from peak with this threshold
        mask = _gradient_flood_fill(
            attn_norm, threshold_frac, gradient_threshold
        )
        area_ratio = float(mask.sum()) / (H * W)

        if min_ratio <= area_ratio <= max_ratio:
            # Score: prefer thresholds that give medium coverage
            # and higher attention within the mask
            mean_attn_in_mask = float(attn_norm[mask].mean()) if mask.any() else 0
            score = mean_attn_in_mask - abs(area_ratio - 0.15) * 0.5
            if score > best_score:
                best_score = score
                best_mask = mask

    if best_mask is None:
        # Fallback: simple threshold at median
        threshold = float(np.median(attn_norm[attn_norm > 0])) if (attn_norm > 0).any() else 0.5
        best_mask = attn_norm >= threshold
        area = float(best_mask.sum()) / (H * W)
        if area < min_ratio:
            # Too small: use top min_ratio pixels
            flat = attn_norm.flatten()
            k = max(1, int(min_ratio * H * W))
            thresh = np.sort(flat)[-k]
            best_mask = attn_norm >= thresh
        elif area > max_ratio:
            flat = attn_norm.flatten()
            k = max(1, int(max_ratio * H * W))
            thresh = np.sort(flat)[-k]
            best_mask = attn_norm >= thresh

    return best_mask


def _gradient_flood_fill(
    attn_norm: np.ndarray,
    threshold: float,
    gradient_threshold: float,
) -> np.ndarray:
    """
    從 attention peak 開始 BFS flood fill。

    擴散條件（OR）：
      1. 鄰居 attn >= threshold（絕對閾值）
      2. |鄰居 attn - 當前 attn| < gradient_threshold（梯度平緩 = 同區域）

    Args:
        attn_norm: [H, W] normalized attention（0~1）
        threshold: 絕對值閾值
        gradient_threshold: 梯度閾值
    """
    H, W = attn_norm.shape
    mask = np.zeros((H, W), dtype=bool)

    # Find peak
    peak_idx = np.unravel_index(attn_norm.argmax(), attn_norm.shape)
    peak_h, peak_w = peak_idx

    # BFS from peak
    from collections import deque
    queue = deque()
    queue.append((peak_h, peak_w))
    mask[peak_h, peak_w] = True

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        ch, cw = queue.popleft()
        cur_val = attn_norm[ch, cw]

        for dh, dw in neighbors:
            nh, nw = ch + dh, cw + dw
            if 0 <= nh < H and 0 <= nw < W and not mask[nh, nw]:
                nval = attn_norm[nh, nw]
                # 擴散條件：attention 夠高 OR 梯度夠平緩
                if nval >= threshold or abs(float(nval - cur_val)) < gradient_threshold:
                    mask[nh, nw] = True
                    queue.append((nh, nw))

    return mask


def compute_dynamic_mask_with_gt(
    attn_map: np.ndarray,
    gt_mask: np.ndarray,
    spatial_size: Tuple[int, int],
    gradient_threshold: float = 0.3,
    max_iters: int = 20,
) -> np.ndarray:
    """
    使用 GT mask 引導的二分法搜尋最佳 threshold，再以 gradient flood fill 生成 mask。

    演算法：
      1. 將 attn_map resize 到 GT mask 尺寸（nearest interpolation）
      2. 二分法搜尋 percentile threshold：
         - selected = attn >= threshold 的像素
         - outside = selected 且在 GT mask 外的像素數
         - inside  = selected 且在 GT mask 內的像素數
         - 若 outside > inside → threshold 太低（選到太多背景）→ 提高 percentile
         - 若 outside <= inside → 嘗試降低 percentile（擴大覆蓋）
      3. 用找到的 threshold 在 spatial_size 上做 gradient flood fill

    Args:
        attn_map: [H, W] attention map
        gt_mask: [H_gt, W_gt] bool, True = 編輯區域（PIE-Bench 白色區域）
        spatial_size: (H, W) 輸出尺寸
        gradient_threshold: 梯度擴散閾值
        max_iters: 二分法最大迭代次數

    Returns:
        bool mask [H, W]，True = 編輯區域
    """
    H, W = spatial_size
    H_gt, W_gt = gt_mask.shape[:2]

    # Step 1: Resize attn_map 到 GT mask 尺寸（nearest）
    attn_at_gt = cv2.resize(
        attn_map.astype(np.float32), (W_gt, H_gt),
        interpolation=cv2.INTER_NEAREST,
    )
    peak_val = float(attn_at_gt.max())
    if peak_val <= 0:
        return np.zeros((H, W), dtype=bool)
    attn_norm_gt = attn_at_gt / peak_val

    # 只看正值像素做 percentile
    positive_values = attn_norm_gt[attn_norm_gt > 0]
    if len(positive_values) == 0:
        return np.zeros((H, W), dtype=bool)

    # Step 2: 二分法搜尋 percentile
    low_pct, high_pct = 0.0, 100.0
    best_pct = 50.0

    for _ in range(max_iters):
        mid = (low_pct + high_pct) / 2.0
        threshold = float(np.percentile(positive_values, mid))
        selected = attn_norm_gt >= threshold

        outside = int(np.sum(selected & ~gt_mask))
        inside = int(np.sum(selected & gt_mask))

        if outside > inside:
            # 選到太多背景 → 提高 threshold（更嚴格）
            low_pct = mid
        else:
            # outside <= inside → 可接受，嘗試降低以擴大覆蓋
            best_pct = mid
            high_pct = mid

    # Step 3: 用找到的 threshold 在 spatial_size 做 flood fill
    best_threshold = float(np.percentile(positive_values, best_pct))

    if attn_map.shape != (H, W):
        attn_target = cv2.resize(
            attn_map.astype(np.float32), (W, H),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        attn_target = attn_map.astype(np.float32)

    peak_val_target = float(attn_target.max())
    if peak_val_target <= 0:
        return np.zeros((H, W), dtype=bool)
    attn_norm_target = attn_target / peak_val_target

    mask = _gradient_flood_fill(attn_norm_target, best_threshold, gradient_threshold)

    # 面積安全檢查
    area_ratio = float(mask.sum()) / (H * W)
    if area_ratio < 0.02:
        # Mask 太小，fallback 到不使用 GT 的版本
        mask = compute_dynamic_mask(
            attn_map, spatial_size, gradient_threshold=gradient_threshold,
        )

    return mask


# ============================================================
# Per-Scale Generator：封裝單一 scale 的生成邏輯
# ============================================================

class PerScaleGenerator:
    """
    封裝 Infinity 模型的 per-scale 生成邏輯。

    替代原本 autoregressive_infer_cfg() 的整體迴圈，
    允許外部管線逐 scale 控制三個 phase 的交錯執行。
    """

    def __init__(
        self,
        model,
        vae,
        text_cond_tuple,
        scale_schedule: List[Tuple[int, int, int]],
        cfg_list: List[float],
        tau_list: List[float],
        cfg_insertion_layer: List[int],
        vae_type: int = 32,
        g_seed: Optional[int] = None,
        top_k: int = 900,
        top_p: float = 0.97,
        apply_spatial_patchify: bool = False,
        sampling_per_bits: int = 1,
        negative_text_cond_tuple=None,
    ):
        self.model = model
        self.vae = vae
        self.scale_schedule = scale_schedule
        self.cfg_list = cfg_list
        self.tau_list = tau_list
        self.vae_type = vae_type
        self.top_k = top_k
        self.top_p = top_p
        self.apply_spatial_patchify = apply_spatial_patchify
        self.sampling_per_bits = sampling_per_bits

        if apply_spatial_patchify:
            self.vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            self.vae_scale_schedule = list(scale_schedule)

        self.num_scales = len(scale_schedule)
        self.B = 1

        # RNG
        if g_seed is not None:
            model.rng.manual_seed(g_seed)
        self.rng = model.rng if g_seed is not None else None

        # ── Text conditioning ──
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
        if any(np.array(cfg_list) != 1):
            self.bs = 2 * self.B
            if negative_text_cond_tuple is None:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = model.cfg_uncond[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_un, lens_un, cu_un, max_un = negative_text_cond_tuple
                kv_compact = torch.cat((kv_compact, kv_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_un)
        else:
            self.bs = self.B

        kv_compact = model.text_norm(kv_compact)
        sos = cond_BD = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
        kv_compact = model.text_proj_for_ca(kv_compact)
        self.ca_kv = (kv_compact, cu_seqlens_k, max_seqlen_k)

        with torch.amp.autocast('cuda', enabled=False):
            self.cond_BD_or_gss = model.shared_ada_lin(cond_BD.float()).float().contiguous()

        self.sos = sos
        self.initial_last_stage = (
            sos.unsqueeze(1).expand(self.bs, 1, -1)
            + model.pos_start.expand(self.bs, 1, -1)
        )

        # ── CFG insertion layers ──
        leng = len(model.unregistered_blocks)
        self.abs_cfg_insertion_layers = []
        self.add_cfg_on_logits = False
        for item in cfg_insertion_layer:
            if item == 0:
                self.add_cfg_on_logits = True
            elif item < 0:
                self.abs_cfg_insertion_layers.append(leng + item)
            else:
                raise ValueError(f'Invalid cfg_insertion_layer: {item}')

    def init_phase_state(self) -> dict:
        """初始化一個新 phase 的生成狀態。"""
        return {
            'summed_codes': 0,
            'last_stage': self.initial_last_stage.clone(),
            'cur_L': 0,
            'idx_Bld_list': [],
        }

    def enable_kv_caching(self):
        """啟用模型的 KV caching。"""
        for b in self.model.unregistered_blocks:
            sa = b.sa if isinstance(b, CrossAttnBlock) else b.attn
            sa.kv_caching(True)

    def disable_kv_caching(self):
        """停用模型的 KV caching。"""
        for b in self.model.unregistered_blocks:
            sa = b.sa if isinstance(b, CrossAttnBlock) else b.attn
            sa.kv_caching(False)

    def generate_one_scale(
        self,
        si: int,
        state: dict,
        p2p_token_storage: Optional[BitwiseTokenStorage] = None,
        p2p_save_tokens: bool = False,
        p2p_use_mask: bool = False,
        p2p_attn_full_replace_scales: int = 0,
        inject_image_features: Optional[torch.Tensor] = None,
        inject_schedule: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        生成單一 scale 的 token。

        Args:
            si: scale index
            state: 生成狀態 dict（由 init_phase_state() 或上一個 scale 返回）
            p2p_token_storage: 用於 token 替換
            p2p_save_tokens: 是否儲存生成的 token
            p2p_use_mask: 是否使用 spatial mask 替換
            p2p_attn_full_replace_scales: 前 N 個 scale 100% 替換
            inject_image_features: source image 連續特徵
            inject_schedule: 注入權重

        Returns:
            (idx_Bld, updated_state)
            idx_Bld: [B, 1, h, w, d] 生成的 token indices
        """
        model = self.model
        vae = self.vae
        B = self.B
        bs = self.bs
        pn = self.scale_schedule[si]

        summed_codes = state['summed_codes']
        last_stage = state['last_stage']
        cur_L = state['cur_L']

        cfg = self.cfg_list[si]
        cur_L += np.array(pn).prod()

        # ── Forward through transformer blocks ──
        layer_idx = 0
        for block_idx, b in enumerate(model.block_chunks):
            if model.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = model.add_lvl_embeding(last_stage, si, self.scale_schedule)
            if not model.add_lvl_embeding_only_first_block:
                last_stage = model.add_lvl_embeding(last_stage, si, self.scale_schedule)

            for m in b.module:
                last_stage = m(
                    x=last_stage,
                    cond_BD=self.cond_BD_or_gss,
                    ca_kv=self.ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=self.scale_schedule,
                    rope2d_freqs_grid=model.rope2d_freqs_grid,
                    scale_ind=si,
                )
                if (cfg != 1) and (layer_idx in self.abs_cfg_insertion_layers):
                    last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                    last_stage = torch.cat((last_stage, last_stage), 0)
                layer_idx += 1

        # ── Sample tokens ──
        if (cfg != 1) and self.add_cfg_on_logits:
            logits_BlV = model.get_logits(last_stage, self.sos).mul(1/self.tau_list[si])
            logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
        else:
            logits_BlV = model.get_logits(last_stage[:B], self.sos[:B]).mul(1/self.tau_list[si])

        if model.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            from infinity.models.infinity_p2p_edit import sample_with_top_k_top_p_also_inplace_modifying_logits_
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV, rng=self.rng,
                top_k=self.top_k or model.top_k,
                top_p=self.top_p or model.top_p,
                num_samples=1,
            )[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            raise NotImplementedError("Only bit_label mode is supported in KV-Edit")

        assert pn[0] == 1
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        if model.apply_spatial_patchify:
            idx_Bld = idx_Bld.permute(0, 3, 1, 2)
            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
            idx_Bld = idx_Bld.permute(0, 2, 3, 1)
        idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d]

        # ── P2P Token save ──
        if p2p_token_storage is not None and p2p_save_tokens:
            p2p_token_storage.save_tokens(si, idx_Bld.clone())

        # ── P2P Token replacement ──
        if p2p_token_storage is not None and p2p_token_storage.has_tokens_for_scale(si):
            source_indices = p2p_token_storage.load_tokens(si, idx_Bld.device)
            if source_indices is not None:
                _, _, h_src, w_src, _ = source_indices.shape
                _, _, h_cur, w_cur, _ = idx_Bld.shape
                if (h_src, w_src) != (h_cur, w_cur):
                    si_f = source_indices.squeeze(1).permute(0, 3, 1, 2).float()
                    si_f = F.interpolate(si_f, size=(h_cur, w_cur), mode='nearest')
                    source_indices = si_f.permute(0, 2, 3, 1).unsqueeze(1).long()
                if source_indices.shape[0] == 1 and B > 1:
                    source_indices = source_indices.expand(B, -1, -1, -1, -1)

                _inject_full = (
                    inject_image_features is not None
                    and inject_schedule is not None
                    and si < len(inject_schedule)
                    and inject_schedule[si] == 0.0
                )
                _force_full = (
                    p2p_attn_full_replace_scales > 0 and si < p2p_attn_full_replace_scales
                ) or _inject_full

                if _force_full:
                    idx_Bld = source_indices.clone()
                elif p2p_use_mask and p2p_token_storage.has_mask_for_scale(si):
                    spatial_mask = p2p_token_storage.load_mask(si, idx_Bld.device)
                    if spatial_mask is not None:
                        if spatial_mask.shape[2] != h_cur or spatial_mask.shape[3] != w_cur:
                            sm = spatial_mask.squeeze(1).permute(0, 3, 1, 2).float()
                            sm = F.interpolate(sm, size=(h_cur, w_cur), mode='nearest')
                            spatial_mask = sm.permute(0, 2, 3, 1).unsqueeze(1)
                        if spatial_mask.dtype == torch.bool:
                            replace_mask = spatial_mask
                        else:
                            rand = torch.rand(B, 1, h_cur, w_cur, 1, device=idx_Bld.device)
                            replace_mask = rand < spatial_mask.float().clamp(0, 1)
                        idx_Bld = torch.where(replace_mask, source_indices, idx_Bld)

        # ── Convert to codes and accumulate ──
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
        num_stages_minus_1 = self.num_scales - 1

        if si != num_stages_minus_1:
            if (inject_image_features is not None
                    and inject_schedule is not None
                    and si < len(inject_schedule)
                    and inject_schedule[si] < 1.0):
                inject_w = inject_schedule[si]
                img_feat = inject_image_features.to(codes.device)
                if img_feat.shape[0] == 1 and B > 1:
                    img_feat = img_feat.expand(B, -1, -1, -1, -1)
                interp_codes = F.interpolate(
                    codes, size=self.vae_scale_schedule[-1],
                    mode=vae.quantizer.z_interplote_up,
                )
                summed_codes = (summed_codes + interp_codes) if isinstance(summed_codes, torch.Tensor) else interp_codes
                if img_feat.shape[-3:] != summed_codes.shape[-3:]:
                    img_feat = F.interpolate(
                        img_feat, size=summed_codes.shape[-3:],
                        mode=vae.quantizer.z_interplote_up,
                    )
                summed_codes = summed_codes * inject_w + img_feat * (1.0 - inject_w)
            else:
                interp = F.interpolate(
                    codes, size=self.vae_scale_schedule[-1],
                    mode=vae.quantizer.z_interplote_up,
                )
                summed_codes = (summed_codes + interp) if isinstance(summed_codes, torch.Tensor) else interp

            next_stage = F.interpolate(
                summed_codes, size=self.vae_scale_schedule[si+1],
                mode=vae.quantizer.z_interplote_up,
            )
            next_stage = next_stage.squeeze(-3)
            if model.apply_spatial_patchify:
                next_stage = torch.nn.functional.pixel_unshuffle(next_stage, 2)
            next_stage = next_stage.reshape(*next_stage.shape[:2], -1)
            next_stage = torch.permute(next_stage, [0, 2, 1])
            next_stage = model.word_embed(model.norm0_ve(next_stage))
            next_stage = next_stage.repeat(bs // B, 1, 1)
        else:
            summed_codes = (summed_codes + codes) if isinstance(summed_codes, torch.Tensor) else codes
            next_stage = None  # 最後一個 scale，不需要 next_stage

        # 更新 state
        new_state = {
            'summed_codes': summed_codes,
            'last_stage': next_stage,
            'cur_L': cur_L,
            'idx_Bld_list': state['idx_Bld_list'] + [idx_Bld.detach()],
        }
        return idx_Bld, new_state

    def decode_image(self, summed_codes) -> torch.Tensor:
        """將累積的 codes 解碼為圖片。"""
        if self.vae_type != 0:
            img = self.vae.decode(summed_codes.squeeze(-3))
        else:
            raise NotImplementedError("Only vae_type != 0 supported")
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return img


# ============================================================
# 交錯式 KV-Edit Pipeline
# ============================================================

def run_kv_edit_pipeline(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    source_prompt: str,
    target_prompt: str,
    source_image_path: str,
    source_focus_words: List[str],
    target_focus_words: List[str],
    save_dir: str,
    args,
    scale_schedule: List[Tuple[int, int, int]],
    attn_block_indices: List[int],
    total_scales: int,
    device_cuda: torch.device,
    gt_mask_path: Optional[str] = None,
) -> bool:
    """
    KV-Edit 交錯式管線主函式。

    流程：
      Phase 0: 編碼 source image
      For each scale si:
        Phase 1: Source gen scale si（存 KV + tokens + attention）
        Phase 1.5: 計算 dynamic mask（從 cross-attention peak flood fill）
        Phase 1.7: Phase17 gen scale si（直接使用 source KV cache）
        Phase 2: Target gen scale si（直接使用 source KV cache + dynamic mask）
      Phase 3: 解碼最終圖片

    Args:
        gt_mask_path: GT mask 圖片路徑（PIE-Bench mask.png）。
                      若提供，使用二分法搜尋最佳 threshold，
                      使 attention mask 內外比例與 GT mask 對齊。

    Returns:
        True if success
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"KV-Edit Interleaved Pipeline")
    print(f"{'='*80}")
    print(f"  source_prompt: {source_prompt}")
    print(f"  target_prompt: {target_prompt}")
    print(f"  source_focus:  {source_focus_words}")
    print(f"  target_focus:  {target_focus_words}")

    num_full_replace_scales = int(getattr(args, 'num_full_replace_scales', 2))
    image_injection_scales = int(getattr(args, 'image_injection_scales', 2))
    kv_blend_ratio = float(getattr(args, 'kv_blend_ratio', 0.3))
    kv_blend_scales = int(getattr(args, 'kv_blend_scales', 8))
    gradient_threshold = float(getattr(args, 'gradient_threshold', 0.3))

    # ── Phase 0: 編碼 source image ──
    source_pil = Image.open(source_image_path).convert('RGB')
    image_raw_features = encode_image_to_raw_features(
        vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
        device=device_cuda, apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )
    image_scale_tokens = encode_image_to_scale_tokens(
        vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
        device=device_cuda, apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )

    # ── Inject schedule ──
    if args.inject_weights.strip():
        inject_schedule = [float(w) for w in args.inject_weights.strip().split()]
    else:
        inject_schedule = [
            0.0 if si < image_injection_scales else 1.0
            for si in range(total_scales)
        ]

    # ── GT Mask（資料集提供的編輯區域遮罩）──
    gt_mask = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask_img = np.array(Image.open(gt_mask_path).convert('L'))
        gt_mask = gt_mask_img >= 128  # True = 編輯區域（PIE-Bench 白色）
        print(f"  [GT Mask] loaded from {gt_mask_path} "
              f"({gt_mask.shape[1]}x{gt_mask.shape[0]}, "
              f"edit area={float(gt_mask.sum())/gt_mask.size*100:.1f}%)")

    # ── Focus token indices ──
    source_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, source_prompt, source_focus_words)
        if source_focus_words else []
    )
    target_focus_token_indices = (
        find_focus_token_indices(text_tokenizer, target_prompt, target_focus_words)
        if target_focus_words else []
    )

    # ── Text encoding ──
    source_text_cond = encode_prompt(text_tokenizer, text_encoder, source_prompt,
                                     bool(args.enable_positive_prompt))
    target_text_cond = encode_prompt(text_tokenizer, text_encoder, target_prompt,
                                     bool(args.enable_positive_prompt))

    cfg_list = args.cfg if isinstance(args.cfg, list) else [args.cfg] * total_scales
    tau_list = args.tau if isinstance(args.tau, list) else [args.tau] * total_scales

    # ── 建立三個 PerScaleGenerator（共用同一個 model）──
    source_gen = PerScaleGenerator(
        model=infinity, vae=vae,
        text_cond_tuple=source_text_cond,
        scale_schedule=scale_schedule,
        cfg_list=cfg_list, tau_list=tau_list,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type, g_seed=args.seed,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )
    phase17_gen = PerScaleGenerator(
        model=infinity, vae=vae,
        text_cond_tuple=target_text_cond,
        scale_schedule=scale_schedule,
        cfg_list=cfg_list, tau_list=tau_list,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type, g_seed=args.seed,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )
    target_gen = PerScaleGenerator(
        model=infinity, vae=vae,
        text_cond_tuple=target_text_cond,
        scale_schedule=scale_schedule,
        cfg_list=cfg_list, tau_list=tau_list,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type, g_seed=args.seed,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
    )

    # ── 狀態管理 ──
    kv_mgr = KVCacheManager()
    source_state = source_gen.init_phase_state()
    phase17_state = phase17_gen.init_phase_state()
    target_state = target_gen.init_phase_state()

    # Token storages
    source_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
    # For phase17: use source image tokens in background regions
    phase17_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
    # For target: use source image tokens with dynamic mask
    target_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    # Dynamic masks collected per-scale
    dynamic_masks: Dict[int, np.ndarray] = {}  # si -> bool mask [h, w]

    # ── Cross-Attention Extractor（掛在模型上）──
    source_extractor = CrossAttentionExtractor(
        model=infinity, block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx, aggregate_method="mean",
    )
    target_extractor = CrossAttentionExtractor(
        model=infinity, block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx, aggregate_method="mean",
    )

    print(f"\n[KV-Edit] Starting interleaved generation ({total_scales} scales)")
    print(f"  num_full_replace: {num_full_replace_scales}")
    print(f"  kv_offload_after: {kv_blend_scales} (source KV → CPU for si >= {kv_blend_scales})")
    print(f"  gradient_thresh:  {gradient_threshold}")
    print(f"  phase17/target:   直接使用 source KV（不維護獨立 KV 歷史）")
    if gt_mask is not None:
        print(f"  gt_mask:          loaded ({gt_mask.shape[1]}x{gt_mask.shape[0]}, edit area={float(gt_mask.sum())/gt_mask.size*100:.1f}%)")

    t_start = time.time()

    def _cleanup_extractors():
        """Ensure all monkey-patches are removed from the model."""
        if source_extractor.original_forwards:
            source_extractor.remove_patches()
        if target_extractor.original_forwards:
            target_extractor.remove_patches()

    # 啟用 KV caching 一次即可（三個 generator 共用同一模型）
    source_gen.enable_kv_caching()

    with autocast(dtype=torch.bfloat16):
      with torch.no_grad():
        try:
            for si in range(total_scales):
                _, h_si, w_si = scale_schedule[si]
                # si >= kv_blend_scales 時不再需要 GPU 上做 blend，
                # 將 KV cache 快照 offload 到 CPU 以節省 GPU 記憶體
                offload = (si >= kv_blend_scales)

                print(f"\n{'─'*60}")
                print(f"[Scale {si}/{total_scales-1}] {h_si}x{w_si}" +
                      (" (KV offload→CPU)" if offload else ""))
                print(f"{'─'*60}")

                # ═══════════════════════════════════════
                # Step 1: Source Generation
                # ═══════════════════════════════════════
                kv_mgr.restore_kv_cache(infinity, 'source')
                source_extractor.register_patches()

                src_idx, source_state = source_gen.generate_one_scale(
                    si=si, state=source_state,
                    p2p_token_storage=source_token_storage,
                    p2p_save_tokens=True,
                    p2p_use_mask=False,
                    inject_image_features=image_raw_features,
                    inject_schedule=inject_schedule,
                )
                source_extractor.remove_patches()
                kv_mgr.save_kv_cache(infinity, 'source', offload_to_cpu=offload)
                kv_mgr.save_gen_state(
                    'source', source_state['summed_codes'],
                    source_state['last_stage'], source_state['cur_L'],
                    source_state['idx_Bld_list'], si,
                )
                print(f"  [Source] scale {si} done")

                # ═══════════════════════════════════════
                # Step 2: Compute Dynamic Mask from source attention
                # ═══════════════════════════════════════
                if si >= num_full_replace_scales and source_focus_token_indices:
                    # 從 source extractor 取出 cross-attention map
                    src_attn_map = _extract_aggregated_attention(
                        source_extractor, source_focus_token_indices,
                        attn_block_indices, si, (h_si, w_si),
                    )
                    if src_attn_map is not None:
                        # Dynamic flood fill mask（有 GT mask 時用二分法校準 threshold）
                        if gt_mask is not None:
                            edit_mask = compute_dynamic_mask_with_gt(
                                src_attn_map, gt_mask, (h_si, w_si),
                                gradient_threshold=gradient_threshold,
                            )
                        else:
                            edit_mask = compute_dynamic_mask(
                                src_attn_map, (h_si, w_si),
                                gradient_threshold=gradient_threshold,
                            )
                        # edit_mask: True = 編輯區域（high attention to source focus）
                        # 對 phase17：背景（~edit_mask）要錨定 source token
                        # 對 target：背景（~edit_mask）要替換為 source token
                        dynamic_masks[si] = edit_mask

                        # 存入 phase17 storage: True = 保留 source（=背景）
                        preserve_mask = ~edit_mask
                        phase17_token_storage.masks[si] = (
                            torch.tensor(preserve_mask, dtype=torch.bool)
                            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                        )

                        # 存入 target storage: True = 保留 source（=背景）
                        target_token_storage.masks[si] = (
                            torch.tensor(preserve_mask, dtype=torch.bool)
                            .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                        )

                        area_pct = float(edit_mask.sum()) / (h_si * w_si) * 100
                        print(f"  [Mask] dynamic mask: edit area = {area_pct:.1f}%")

                # 前 N 個 scale：source tokens 直接寫入 storage
                if si in image_scale_tokens:
                    phase17_token_storage.tokens[si] = image_scale_tokens[si].clone()
                    target_token_storage.tokens[si] = image_scale_tokens[si].clone()

                # ═══════════════════════════════════════
                # Step 3: Phase 1.7 (Target guided with source structure)
                # ═══════════════════════════════════════
                # Phase17 直接使用 source 的 KV cache（不維護獨立 KV 歷史）
                kv_mgr.restore_kv_cache(infinity, 'source')

                p17_idx, phase17_state = phase17_gen.generate_one_scale(
                    si=si, state=phase17_state,
                    p2p_token_storage=phase17_token_storage,
                    p2p_save_tokens=False,
                    p2p_use_mask=(si >= num_full_replace_scales),
                    p2p_attn_full_replace_scales=num_full_replace_scales,
                    inject_image_features=None,
                    inject_schedule=None,
                )
                print(f"  [Phase17] scale {si} done (source KV)")

                # ═══════════════════════════════════════
                # Step 4: Target Generation (final output)
                # ═══════════════════════════════════════
                # Target 直接使用 source 的 KV cache（不維護獨立 KV 歷史）
                kv_mgr.restore_kv_cache(infinity, 'source')

                # 如果有 target focus，也掛 extractor 收集 attention
                if target_focus_token_indices:
                    target_extractor.register_patches()

                tgt_idx, target_state = target_gen.generate_one_scale(
                    si=si, state=target_state,
                    p2p_token_storage=target_token_storage,
                    p2p_save_tokens=False,
                    p2p_use_mask=(si >= num_full_replace_scales),
                    p2p_attn_full_replace_scales=num_full_replace_scales,
                    inject_image_features=None,
                    inject_schedule=None,
                )

                if target_focus_token_indices:
                    target_extractor.remove_patches()

                    # 也用 target attention 更新 mask（union with source mask）
                    if si >= num_full_replace_scales:
                        tgt_attn_map = _extract_aggregated_attention(
                            target_extractor, target_focus_token_indices,
                            attn_block_indices, si, (h_si, w_si),
                        )
                        if tgt_attn_map is not None:
                            if gt_mask is not None:
                                tgt_edit_mask = compute_dynamic_mask_with_gt(
                                    tgt_attn_map, gt_mask, (h_si, w_si),
                                    gradient_threshold=gradient_threshold,
                                )
                            else:
                                tgt_edit_mask = compute_dynamic_mask(
                                    tgt_attn_map, (h_si, w_si),
                                    gradient_threshold=gradient_threshold,
                                )
                            # Union with source mask（兩邊的 edit 區域都不應替換 source）
                            if si in dynamic_masks:
                                combined = dynamic_masks[si] | tgt_edit_mask
                                dynamic_masks[si] = combined
                            else:
                                dynamic_masks[si] = tgt_edit_mask

                print(f"  [Target] scale {si} done (source KV)")
        finally:
            _cleanup_extractors()

    # ── Disable KV caching ──
    source_gen.disable_kv_caching()

    # ── Decode images ──
    print(f"\n[KV-Edit] Decoding images...")

    source_img = source_gen.decode_image(source_state['summed_codes'])
    target_img = target_gen.decode_image(target_state['summed_codes'])

    # Save
    src_np = source_img[0].cpu().numpy()
    tgt_np = target_img[0].cpu().numpy()
    cv2.imwrite(os.path.join(save_dir, 'source.jpg'), src_np)
    cv2.imwrite(os.path.join(save_dir, 'target.jpg'), tgt_np)

    # Save masks
    if args.save_attn_vis and dynamic_masks:
        mask_dir = os.path.join(save_dir, 'dynamic_masks')
        os.makedirs(mask_dir, exist_ok=True)
        for si, mask in dynamic_masks.items():
            _, h_si, w_si = scale_schedule[si]
            vis = (mask.astype(np.uint8) * 255)
            vis = cv2.resize(vis, (max(256, w_si*8), max(256, h_si*8)),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(mask_dir, f'scale{si:02d}_{h_si}x{w_si}.png'),
                vis,
            )

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"KV-Edit complete ({elapsed:.1f}s)")
    print(f"  source: {os.path.join(save_dir, 'source.jpg')}")
    print(f"  target: {os.path.join(save_dir, 'target.jpg')}")
    print(f"{'='*80}\n")

    # Cleanup
    kv_mgr.clear_all()
    del source_state, phase17_state, target_state
    torch.cuda.empty_cache()

    return True


# ============================================================
# Attention 工具函式
# ============================================================

def _extract_aggregated_attention(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    attn_block_indices: List[int],
    scale_idx: int,
    spatial_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    從 extractor 取出指定 focus token 的聚合 attention map。

    Returns:
        np.ndarray [H, W] 或 None
    """
    if not extractor.attention_maps:
        return None

    H, W = spatial_size
    block_maps = []

    for block_idx in attn_block_indices:
        attn_map = extractor.extract_word_attention(
            block_idx=block_idx,
            scale_idx=scale_idx,
            token_indices=focus_token_indices,
            spatial_size=(H, W),
        )
        if attn_map is not None:
            block_maps.append(attn_map)

    if not block_maps:
        return None

    # 簡單平均所有 block（可日後改 IQR filter）
    stacked = np.stack(block_maps, axis=0)
    return stacked.mean(axis=0).astype(np.float32)


# ============================================================
# PIE-Bench 批量介面
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
    """PIE-Bench 相容介面：呼叫 KV-Edit 交錯式管線。"""
    return run_kv_edit_pipeline(
        infinity=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        source_image_path=source_image_path,
        source_focus_words=source_focus_words,
        target_focus_words=target_focus_words,
        save_dir=save_dir,
        args=args,
        scale_schedule=scale_schedule,
        attn_block_indices=attn_block_indices,
        total_scales=total_scales,
        device_cuda=device_cuda,
        gt_mask_path=mask_path,
    )


# ============================================================
# CLI 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='KV-Edit 交錯式管線')
    add_common_arguments(parser)

    # Source / Target
    parser.add_argument('--source_image', type=str, required=True)
    parser.add_argument('--source_prompt', type=str, required=True)
    parser.add_argument('--target_prompt', type=str, required=True)
    parser.add_argument('--source_focus_words', type=str, default='')
    parser.add_argument('--target_focus_words', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./outputs/kv_edit_output')

    # KV-Edit 參數
    parser.add_argument('--num_full_replace_scales', type=int, default=2)
    parser.add_argument('--image_injection_scales', type=int, default=2)
    parser.add_argument('--inject_weights', type=str, default='')
    parser.add_argument('--kv_blend_ratio', type=float, default=0.3,
                        help='Source KV cache 混入 target 的比例（0=不混，1=完全取代）')
    parser.add_argument('--kv_blend_scales', type=int, default=8,
                        help='前幾個 scale 啟用 KV blending')
    parser.add_argument('--gradient_threshold', type=float, default=0.3,
                        help='Dynamic mask 的梯度擴散閾值')

    # Attention
    parser.add_argument('--attn_block_start', type=int, default=2)
    parser.add_argument('--attn_block_end', type=int, default=-1)
    parser.add_argument('--attn_batch_idx', type=int, default=0)

    # 輸出
    parser.add_argument('--save_attn_vis', type=int, default=1, choices=[0, 1])

    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # ── 載入模型 ──
    print("[Init] Loading models...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    print("[Init] Model load complete.\n")

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)

    depth = len(infinity.unregistered_blocks)
    attn_block_start = (depth // 2) if args.attn_block_start < 0 else min(args.attn_block_start, depth - 1)
    attn_block_end = (depth - 1) if args.attn_block_end < 0 else min(args.attn_block_end, depth - 1)
    attn_block_indices = list(range(attn_block_start, attn_block_end + 1))

    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_focus = args.source_focus_words.strip().split() if args.source_focus_words.strip() else []
    target_focus = args.target_focus_words.strip().split() if args.target_focus_words.strip() else []

    run_kv_edit_pipeline(
        infinity=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        source_image_path=args.source_image,
        source_focus_words=source_focus,
        target_focus_words=target_focus,
        save_dir=args.save_dir,
        args=args,
        scale_schedule=scale_schedule,
        attn_block_indices=attn_block_indices,
        total_scales=total_scales,
        device_cuda=device_cuda,
    )


if __name__ == '__main__':
    main()
