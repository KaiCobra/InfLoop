#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
id_resampler.py — Attention-based MLP (Perceiver Resampler 風) 把 AdaFace ID
embedding 映射到 T5 output (last_hidden_state) 空間，產出 N 個 token，**直接**
覆蓋 prompt 經 T5 後 subject token 對應位置的 embedding。

設計重點（output 端替換 + Resampler with prompt context）：
  • 不需要在 prompt 裡放 'sks' 之類的稀有 token；直接拿 prompt 跑 T5，再用
    Resampler 產出的 token 覆蓋 subject token 那幾個位置的 last_hidden_state。
  • Resampler 的 cross-attention context = AdaFace 衍生的 n_id_ctx 個 token
    （+ 可選的 prompt T5 output），所以 ID token 既看 ID 特徵，也能感知 prompt。
  • 透過「anchor + small delta」初始化，未訓練時輸出 ≈ anchor（例如 'person' 的
    T5 embedding），訓練早期不會把 prompt 嵌入空間炸壞。

Shape contract（輸入 / 輸出）：
  IDResampler.forward(id_feat, prompt_ctx=None, prompt_mask=None) →
    out: (B, n_tokens, t5_dim)
  其中：
    id_feat:     (B, 512)  或 (512,)  — AdaFace L2-normalized embedding
    prompt_ctx:  (B, L_p, t5_dim)     — T5 last_hidden_state（可選；use_prompt_ctx=True 時用）
    prompt_mask: (B, L_p) bool        — True = 有效 token，False = padding
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ============================================================
# Resampler block: pre-norm cross-attn → self-attn → FFN
# ============================================================

class ResamplerBlock(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q_xa = nn.LayerNorm(d)
        self.norm_ctx = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_q_sa = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_q_ff = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * mlp_ratio),
            nn.GELU(),
            nn.Linear(d * mlp_ratio, d),
        )

    def forward(
        self,
        q: torch.Tensor,                       # (B, N, d)
        ctx: torch.Tensor,                     # (B, M, d)
        ctx_mask: Optional[torch.Tensor] = None,  # (B, M) bool, True=valid
    ) -> torch.Tensor:
        # nn.MultiheadAttention.key_padding_mask: True = positions to *ignore*.
        if ctx_mask is not None:
            kp_mask = ~ctx_mask
        else:
            kp_mask = None

        nq = self.norm_q_xa(q)
        nctx = self.norm_ctx(ctx)
        q_xa, _ = self.cross_attn(
            nq, nctx, nctx, key_padding_mask=kp_mask, need_weights=False
        )
        q = q + q_xa

        nq = self.norm_q_sa(q)
        q_sa, _ = self.self_attn(nq, nq, nq, need_weights=False)
        q = q + q_sa

        q = q + self.ffn(self.norm_q_ff(q))
        return q


# ============================================================
# IDResampler: 512-d AdaFace → N tokens of t5_dim
# ============================================================

class IDResampler(nn.Module):
    """Maps AdaFace id embedding → N tokens in T5 output space.

    Forward shapes
    --------------
    id_feat:     (B, id_dim)  — L2-normalized AdaFace embedding，dim=1 也接受。
    prompt_ctx:  (B, L_p, t5_dim) optional — 把 prompt 的 T5 last_hidden_state 餵進來
                 當 cross-attn 的額外 context；ID token 因此能感知 prompt 語意。
    prompt_mask: (B, L_p) bool optional — True 表示 valid token；padding 處設 False。

    out: (B, n_tokens, t5_dim) — 直接寫到 T5 output 對應 subject token 位置。
    """

    def __init__(
        self,
        id_dim: int = 512,
        t5_dim: int = 2048,
        n_tokens: int = 1,
        n_id_ctx: int = 4,
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        use_prompt_ctx: bool = True,
        anchor_emb: Optional[torch.Tensor] = None,
        delta_init_std: float = 1e-3,
        delta_max_norm: Optional[float] = None,
        out_norm_match: str = "none",
        residual_base: str = "anchor",
    ) -> None:
        """
        delta_max_norm:
            若給浮點，會把 self.delta_proj(q) 的 per-token L2 norm clamp 到不超過此值。
            None = 不限。對抗訓練時 delta 漸漸炸大跑出 valid manifold。
        out_norm_match:
            'none'   = 不做事
            'anchor' = 強制 out 的 per-token norm == anchor 的 per-token norm
            'base'   = 強制 out 的 per-token norm == 實際 residual base 的 norm
            雖然 Infinity 的 text_norm 是 RMSNorm scale-invariant，但這個約束會在
            training 時對 delta_proj weight 的梯度產生 implicit regularization 效果。
        residual_base:
            'anchor' = 舊行為，out = anchor + delta
            'orig'   = 差分學習，out = prompt 當下 sks T5 output + delta
        """
        super().__init__()
        self.id_dim = id_dim
        self.t5_dim = t5_dim
        self.n_tokens = n_tokens
        self.n_id_ctx = n_id_ctx
        self.use_prompt_ctx = use_prompt_ctx
        self.delta_max_norm = (
            float(delta_max_norm) if delta_max_norm is not None else None
        )
        if out_norm_match not in ("none", "anchor", "base"):
            raise ValueError(
                f"out_norm_match must be 'none', 'anchor', or 'base', got '{out_norm_match}'"
            )
        self.out_norm_match = out_norm_match
        if residual_base not in ("anchor", "orig"):
            raise ValueError(
                f"residual_base must be 'anchor' or 'orig', got '{residual_base}'"
            )
        self.residual_base = residual_base

        # ID(512) → n_id_ctx 個 t5_dim context token
        self.id_norm = nn.LayerNorm(id_dim)
        self.id_proj = nn.Linear(id_dim, t5_dim * n_id_ctx)

        # learnable queries
        self.queries = nn.Parameter(torch.randn(n_tokens, t5_dim) * 0.02)

        # resampler stack
        self.blocks = nn.ModuleList(
            [
                ResamplerBlock(t5_dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(t5_dim)

        # 最後 delta 投影，init 小：未訓練時 out ≈ anchor
        self.delta_proj = nn.Linear(t5_dim, t5_dim)
        nn.init.normal_(self.delta_proj.weight, std=delta_init_std)
        nn.init.zeros_(self.delta_proj.bias)

        # anchor (per-token) — 初始化用 'person' 之類的 T5 output embedding
        if anchor_emb is None:
            anchor_emb = torch.zeros(t5_dim)
        if anchor_emb.shape != (t5_dim,):
            raise ValueError(
                f"anchor_emb must be shape ({t5_dim},), got {tuple(anchor_emb.shape)}"
            )
        self.anchor = nn.Parameter(
            anchor_emb.detach().clone()
            .unsqueeze(0)
            .expand(n_tokens, t5_dim)
            .contiguous()
        )

    @torch.no_grad()
    def set_anchor(self, anchor_emb: torch.Tensor) -> None:
        """Re-set anchor from a freshly computed T5 output embedding."""
        if anchor_emb.shape != (self.t5_dim,):
            raise ValueError(
                f"anchor_emb must be shape ({self.t5_dim},), "
                f"got {tuple(anchor_emb.shape)}"
            )
        self.anchor.data.copy_(
            anchor_emb.to(self.anchor.dtype)
            .to(self.anchor.device)
            .unsqueeze(0)
            .expand(self.n_tokens, self.t5_dim)
            .contiguous()
        )

    def forward(
        self,
        id_feat: torch.Tensor,
        prompt_ctx: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        base_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ── 形狀正規化 ──
        if id_feat.dim() == 1:
            id_feat = id_feat.unsqueeze(0)
        if id_feat.shape[-1] != self.id_dim:
            raise ValueError(
                f"id_feat last dim must be {self.id_dim}, "
                f"got shape {tuple(id_feat.shape)}"
            )
        B = id_feat.size(0)
        device = id_feat.device

        # ID(B, 512) → n_id_ctx 個 t5_dim context token
        id_h = self.id_norm(id_feat)                                            # (B, id_dim)
        id_ctx = self.id_proj(id_h).view(B, self.n_id_ctx, self.t5_dim)         # (B, n_id_ctx, t5_dim)

        # 與 prompt T5 output 串成完整 context
        if self.use_prompt_ctx and prompt_ctx is not None:
            if prompt_ctx.dim() == 2:
                prompt_ctx = prompt_ctx.unsqueeze(0)
            if prompt_ctx.shape[-1] != self.t5_dim:
                raise ValueError(
                    f"prompt_ctx last dim must be {self.t5_dim}, "
                    f"got shape {tuple(prompt_ctx.shape)}"
                )
            if prompt_ctx.size(0) == 1 and B > 1:
                prompt_ctx = prompt_ctx.expand(B, -1, -1)
            ptx = prompt_ctx.to(id_ctx.dtype).to(device)
            ctx = torch.cat([ptx, id_ctx], dim=1)                               # (B, L_p+n_id_ctx, t5_dim)

            if prompt_mask is not None:
                if prompt_mask.dim() == 1:
                    prompt_mask = prompt_mask.unsqueeze(0)
                if prompt_mask.size(0) == 1 and B > 1:
                    prompt_mask = prompt_mask.expand(B, -1)
                pm = prompt_mask.to(device).bool()
                id_pm = torch.ones(B, self.n_id_ctx, dtype=torch.bool, device=device)
                ctx_mask = torch.cat([pm, id_pm], dim=1)
            else:
                ctx_mask = None
        else:
            ctx = id_ctx
            ctx_mask = None

        # learnable queries 跑 resampler stack
        q = (
            self.queries.unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
            .to(device)
        )                                                                       # (B, n_tokens, t5_dim)
        for blk in self.blocks:
            q = blk(q, ctx, ctx_mask=ctx_mask)
        q = self.norm_out(q)
        delta = self.delta_proj(q)                                              # (B, n_tokens, t5_dim)

        # 限制 delta 的 per-token L2 norm（manifold guard）
        if self.delta_max_norm is not None:
            cur = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            scale = (self.delta_max_norm / cur).clamp_max(1.0)
            delta = delta * scale

        # 決定殘差基底：舊 anchor 模式，或 prompt 當下 sks 的原始 T5 output。
        if self.residual_base == "anchor":
            base = self.anchor.unsqueeze(0).expand(B, -1, -1).to(device)
        elif self.residual_base == "orig":
            if base_emb is None:
                raise ValueError(
                    "residual_base='orig' requires base_emb (shape (n_tokens, t5_dim) "
                    "or (B, n_tokens, t5_dim)) at forward time"
                )
            base = base_emb.to(device).to(self.anchor.dtype)
            if base.dim() == 2:
                if base.shape != (self.n_tokens, self.t5_dim):
                    raise ValueError(
                        f"base_emb (2-d) shape must be ({self.n_tokens}, {self.t5_dim}), "
                        f"got {tuple(base.shape)}"
                    )
                base = base.unsqueeze(0).expand(B, -1, -1)
            elif base.dim() == 3:
                if base.shape[1:] != (self.n_tokens, self.t5_dim):
                    raise ValueError(
                        f"base_emb (3-d) last two dims must be ({self.n_tokens}, "
                        f"{self.t5_dim}), got {tuple(base.shape)}"
                    )
                if base.size(0) == 1 and B > 1:
                    base = base.expand(B, -1, -1)
                elif base.size(0) != B:
                    raise ValueError(
                        f"base_emb batch must be 1 or {B}, got {base.size(0)}"
                    )
            else:
                raise ValueError(f"base_emb must be 2-d or 3-d, got {base.dim()}-d")
        else:
            raise RuntimeError(f"unreachable residual_base={self.residual_base}")

        out = base + delta                                                      # (B, n_tokens, t5_dim)

        # 強制 out 的 per-token norm 與 anchor 對齊
        if self.out_norm_match == "anchor":
            target_n = self.anchor.norm(p=2, dim=-1, keepdim=True).unsqueeze(0)  # (1, n_tokens, 1)
            cur_n = out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            out = out / cur_n * target_n.to(out.device)
        elif self.out_norm_match == "base":
            target_n = base.norm(p=2, dim=-1, keepdim=True)                     # (B, n_tokens, 1)
            cur_n = out.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            out = out / cur_n * target_n

        return out


def extract_orig_sks_from_text_features(
    text_features: torch.Tensor,
    sks_indices: List[int],
    n_tokens: int,
) -> torch.Tensor:
    """從 T5 last_hidden_state 中抽出 sks 對應位置的 token 當 residual base.

    Returns:
        (n_tokens, t5_dim) tensor，同 device/dtype as text_features。
    """
    if not sks_indices:
        raise ValueError("sks_indices must be non-empty")
    if text_features.dim() != 3 or text_features.size(0) != 1:
        raise ValueError(
            f"text_features must be (1, L, D), got {tuple(text_features.shape)}"
        )

    feats = text_features[0, sks_indices, :].detach().clone()
    k = len(sks_indices)
    if n_tokens == 1:
        return feats.mean(dim=0, keepdim=True)
    if n_tokens == k:
        return feats
    raise ValueError(
        f"n_tokens={n_tokens} 必須是 1（廣播）或 len(sks_indices)={k}（一對一）"
    )


# ============================================================
# Anchor helper：從 T5 拿一個 word 的 last_hidden_state 當 anchor
# ============================================================

@torch.no_grad()
def get_anchor_t5_embedding(
    text_tokenizer,
    text_encoder,
    anchor_phrase: str = "a person",
    target_word: str = "person",
) -> torch.Tensor:
    """跑 T5 forward 拿 `anchor_phrase` 的 last_hidden_state，找出 `target_word`
    對應的 token 位置，回傳該位置的 embedding (t5_dim,) on CPU。

    用作 IDResampler.anchor 的初始化值——這樣 Resampler 在沒訓練前的輸出
    會落在 'person' 這類「合理人臉位置」附近，訓練早期不會炸掉 prompt 嵌入。
    """
    device = next(text_encoder.parameters()).device
    tokens = text_tokenizer(text=[anchor_phrase], return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    feat = text_encoder(
        input_ids=input_ids, attention_mask=mask
    )["last_hidden_state"].float()                   # (1, L, t5_dim)

    target_lc = target_word.lower().strip()
    found = -1
    decoded_tokens = []
    for i, tid in enumerate(input_ids[0].tolist()):
        tok = text_tokenizer.convert_ids_to_tokens([tid])[0]
        clean = tok.replace("▁", "").replace(" ", "").lower()
        decoded_tokens.append((i, tok, clean))
        if clean and target_lc.startswith(clean):
            # 命中（單一 token 完全 / 前綴匹配 target_word）
            found = i
            break

    if found < 0:
        raise ValueError(
            f"target word '{target_word}' not found in tokens of "
            f"'{anchor_phrase}': {decoded_tokens}"
        )
    return feat[0, found, :].detach().cpu()
