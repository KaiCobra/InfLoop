#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Face identity visual adapter for Infinity.

This file defines the model-side pieces only:
  - SAM3FacePyramidAdapter: SAM3 pyramid + face masks -> id tokens per scale
  - VisualCrossAttentionBranch: zero-gated residual visual cross-attention
  - hook helpers to connect the branch to an existing Infinity instance

No dataloader or training loop lives here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_scale_key(scale_idx: int) -> str:
    return str(int(scale_idx))


class SAM3FacePyramidAdapter(nn.Module):
    """Convert SAM3 feature pyramid into pose-agnostic identity tokens.

    Args:
        sam_dim: SAM3 pyramid channel dimension. SAM3 ViT-H config uses 1024.
        model_dim: Infinity hidden dimension, e.g. 2048 for 2B.
        num_regions: number of face parsing regions in mask tensor.
        num_id_tokens: output identity tokens per scale.
        target_scales: 0-based Infinity scale indices to produce tokens for.
    """

    def __init__(
        self,
        sam_dim: int = 1024,
        model_dim: int = 2048,
        adapter_dim: int = 1024,
        num_regions: int = 10,
        num_id_tokens: int = 12,
        num_layers: int = 2,
        num_heads: int = 8,
        target_scales: Sequence[int] = tuple(range(4, 13)),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sam_dim = int(sam_dim)
        self.model_dim = int(model_dim)
        self.adapter_dim = int(adapter_dim)
        self.num_regions = int(num_regions)
        self.num_id_tokens = int(num_id_tokens)
        self.target_scales = tuple(int(x) for x in target_scales)

        self.in_proj = nn.Linear(self.sam_dim, self.adapter_dim)
        self.region_embed = nn.Parameter(torch.randn(self.num_regions, self.adapter_dim) * 0.02)
        self.level_embed = nn.Parameter(torch.randn(16, self.adapter_dim) * 0.02)
        self.scale_embed = nn.Parameter(torch.randn(32, self.adapter_dim) * 0.02)
        self.id_queries = nn.Parameter(torch.randn(self.num_id_tokens, self.adapter_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.adapter_dim,
            nhead=num_heads,
            dim_feedforward=self.adapter_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.query_norm = nn.LayerNorm(self.adapter_dim)
        self.query_attn = nn.MultiheadAttention(
            self.adapter_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(self.adapter_dim)
        self.out_proj = nn.Linear(self.adapter_dim, self.model_dim)

        # Keep initial visual influence small. The Infinity branch gate is also
        # zero-init, so the whole pipeline starts as exact no-op.
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        pyramid: Sequence[torch.Tensor],
        face_masks: Optional[torch.Tensor] = None,
        target_scales: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Return {scale_idx: id_tokens}, each shaped (B, N_id, model_dim).

        face_masks:
            Optional tensor (B, R, H, W). If omitted, a single all-ones global
            region is used and repeated to ``num_regions``.
        """
        if not pyramid:
            raise ValueError("pyramid must be non-empty")
        B = pyramid[0].shape[0]
        device = pyramid[0].device
        dtype = pyramid[0].dtype
        masks = self._prepare_masks(face_masks, B, device, dtype)

        region_tokens = []
        for level, feat in enumerate(pyramid):
            if feat.dim() != 4:
                raise ValueError(f"pyramid[{level}] must be (B,D,H,W), got {tuple(feat.shape)}")
            pooled = self._masked_region_pool(feat, masks)                    # (B, R, D)
            tok = self.in_proj(pooled)
            tok = tok + self.region_embed.unsqueeze(0)
            tok = tok + self.level_embed[level].view(1, 1, -1)
            region_tokens.append(tok)
        context = torch.cat(region_tokens, dim=1)                              # (B, L_ctx, A)
        context = self.encoder(context)

        scales = tuple(int(x) for x in (target_scales or self.target_scales))
        out: Dict[int, torch.Tensor] = {}
        for scale_idx in scales:
            q = self.id_queries.unsqueeze(0).expand(B, -1, -1)
            q = q + self.scale_embed[scale_idx].view(1, 1, -1)
            q = self.query_norm(q)
            h, _ = self.query_attn(q, context, context, need_weights=False)
            h = self.out_proj(self.out_norm(h))
            out[scale_idx] = h
        return out

    def _prepare_masks(
        self,
        face_masks: Optional[torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if face_masks is None:
            return torch.ones(B, self.num_regions, 1, 1, device=device, dtype=dtype)
        if face_masks.dim() == 3:
            face_masks = face_masks.unsqueeze(1)
        if face_masks.dim() != 4:
            raise ValueError(f"face_masks must be (B,R,H,W), got {tuple(face_masks.shape)}")
        face_masks = face_masks.to(device=device, dtype=dtype)
        if face_masks.size(0) == 1 and B > 1:
            face_masks = face_masks.expand(B, -1, -1, -1)
        if face_masks.size(0) != B:
            raise ValueError(f"face_masks batch must be 1 or {B}, got {face_masks.size(0)}")
        if face_masks.size(1) < self.num_regions:
            pad = self.num_regions - face_masks.size(1)
            face_masks = F.pad(face_masks, (0, 0, 0, 0, 0, pad), value=0)
        elif face_masks.size(1) > self.num_regions:
            face_masks = face_masks[:, : self.num_regions]
        return face_masks

    def _masked_region_pool(self, feat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, D, H, W = feat.shape
        masks = F.interpolate(masks, size=(H, W), mode="nearest")
        denom = masks.flatten(2).sum(dim=-1).clamp_min(1.0)                    # (B, R)
        pooled = torch.einsum("bdhw,brhw->brd", feat, masks) / denom.unsqueeze(-1)
        return pooled


class VisualCrossAttentionBranch(nn.Module):
    """Zero-gated visual cross-attention residual branch."""

    def __init__(
        self,
        model_dim: int = 2048,
        num_heads: int = 16,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor, id_tokens: torch.Tensor) -> torch.Tensor:
        if id_tokens.dim() != 3:
            raise ValueError(f"id_tokens must be (B,N,C), got {tuple(id_tokens.shape)}")
        if id_tokens.size(0) == 1 and x.size(0) > 1:
            id_tokens = id_tokens.expand(x.size(0), -1, -1)
        elif id_tokens.size(0) != x.size(0):
            if x.size(0) % id_tokens.size(0) == 0:
                id_tokens = id_tokens.repeat_interleave(x.size(0) // id_tokens.size(0), dim=0)
            else:
                raise ValueError(
                    f"id_tokens batch {id_tokens.size(0)} cannot match x batch {x.size(0)}"
                )
        h, _ = self.attn(self.norm(x), id_tokens.to(x.dtype), id_tokens.to(x.dtype), need_weights=False)
        return x + self.gate.to(x.dtype) * self.out_norm(h)


def attach_visual_identity_adapter(
    infinity_model: nn.Module,
    block_indices: Sequence[int] = tuple(range(16, 32)),
    scale_start: int = 4,
    scale_end: int = 12,
    model_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> nn.Module:
    """Attach trainable visual cross-attn branches to an Infinity instance.

    The function registers branches as ``infinity_model.visual_identity_branches``
    and stores hook handles in ``infinity_model.visual_identity_hook_handles``.
    At runtime set ``infinity_model.visual_id_tokens_by_scale`` to a dict:

        {scale_idx: Tensor[B, N_id, C]}

    before calling ``forward`` or the AR scale loop.
    """
    detach_visual_identity_adapter(infinity_model)

    if model_dim is None:
        model_dim = int(getattr(infinity_model, "C"))
    if num_heads is None:
        num_heads = int(getattr(infinity_model, "num_heads"))

    branches = nn.ModuleDict()
    handles = []
    block_set = {int(x) for x in block_indices}
    scale_start = int(scale_start)
    scale_end = int(scale_end)

    for block_idx in sorted(block_set):
        branch = VisualCrossAttentionBranch(model_dim=model_dim, num_heads=num_heads)
        branches[_to_scale_key(block_idx)] = branch

    infinity_model.visual_identity_branches = branches.to(next(infinity_model.parameters()).device)
    infinity_model.visual_id_tokens_by_scale = {}
    infinity_model.visual_identity_enabled = True
    infinity_model.visual_identity_scale_range = (scale_start, scale_end)

    for block_idx, block in enumerate(getattr(infinity_model, "unregistered_blocks")):
        if block_idx not in block_set:
            continue

        def _make_hook(idx: int):
            def _hook(module, args, kwargs, output):
                if not getattr(infinity_model, "visual_identity_enabled", False):
                    return output
                scale_idx = int(kwargs.get("scale_ind", 0))
                s0, s1 = getattr(infinity_model, "visual_identity_scale_range", (scale_start, scale_end))
                if scale_idx < s0 or scale_idx > s1:
                    return output
                tokens_by_scale = getattr(infinity_model, "visual_id_tokens_by_scale", None) or {}
                id_tokens = tokens_by_scale.get(scale_idx)
                if id_tokens is None:
                    return output
                branch = infinity_model.visual_identity_branches[_to_scale_key(idx)]
                return branch(output, id_tokens.to(output.device))
            return _hook

        handles.append(block.register_forward_hook(_make_hook(block_idx), with_kwargs=True))

    infinity_model.visual_identity_hook_handles = handles
    return infinity_model


def detach_visual_identity_adapter(infinity_model: nn.Module) -> None:
    handles = getattr(infinity_model, "visual_identity_hook_handles", [])
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    infinity_model.visual_identity_hook_handles = []
    if hasattr(infinity_model, "visual_identity_enabled"):
        infinity_model.visual_identity_enabled = False


def set_visual_id_tokens(infinity_model: nn.Module, tokens_by_scale: Dict[int, torch.Tensor]) -> None:
    infinity_model.visual_id_tokens_by_scale = {
        int(k): v for k, v in tokens_by_scale.items()
    }


def visual_identity_trainable_parameters(infinity_model: nn.Module):
    if hasattr(infinity_model, "visual_identity_branches"):
        yield from infinity_model.visual_identity_branches.parameters()
