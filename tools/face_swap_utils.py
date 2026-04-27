#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_swap_utils.py — Face-Swap pipeline 共用工具

包含：
  • AdaFaceClient：呼叫 docs/adaface_server.md 描述的 HTTP server
    取得 512-d 人臉 embedding（已 L2-normalized）
  • project_512_to_2048_repeat4：將 512-d 沿通道維度重複 4 次補成 2048-d
  • scale_to_target_norm：把 2048-d 向量縮放成目標 L2 norm
  • manipulate_text_features：在 T5 last_hidden_state 中對指定 token 做
    減去 / 取代 操作（先 repeat4 → 再 norm-scale 到原 token norm）
  • encode_prompt_with_face_op：encode_prompt 的延伸版本，可在
    回傳 kv_compact 之前對 subject token 注入人臉 embedding 操作
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F


# ============================================================
# AdaFace HTTP client
# ============================================================

class AdaFaceClient:
    """簡易 HTTP client，配合 server/adaface_server.py。

    server 假設正在 0.0.0.0:8000 上跑（見 docs/adaface_server.md）。
    回傳的 embedding 是 512-d L2-normalized np.float32 array。
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000",
        timeout: float = 60.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ── liveness / metadata ──
    def health(self) -> dict:
        resp = requests.get(f"{self.url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── embedding 取得 ──
    def embed_files(
        self,
        paths: Iterable[str],
        align_face: bool = True,
        return_norm: bool = True,
    ) -> List[np.ndarray]:
        """把多張本地圖片送到 server 取 embedding。

        Returns:
            list of 512-d np.float32 arrays，順序與輸入 paths 對齊。
            任何 per-image failure 都會 raise RuntimeError，由呼叫端決定處置。
        """
        path_list = [str(p) for p in paths]
        if not path_list:
            return []

        files = []
        opened = []
        try:
            for p in path_list:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"AdaFace input not found: {p}")
                fh = open(p, "rb")
                opened.append(fh)
                files.append(("files", (os.path.basename(p), fh, "image/jpeg")))

            params = {
                "align_face": "true" if align_face else "false",
                "return_norm": "true" if return_norm else "false",
            }
            resp = requests.post(
                f"{self.url}/embed",
                files=files,
                params=params,
                timeout=self.timeout,
            )
        finally:
            for fh in opened:
                try:
                    fh.close()
                except Exception:
                    pass

        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if len(results) != len(path_list):
            raise RuntimeError(
                f"AdaFace returned {len(results)} results for {len(path_list)} inputs"
            )

        out: List[np.ndarray] = []
        for item in results:
            if not item.get("success"):
                raise RuntimeError(
                    f"AdaFace failed on {item.get('filename')}: {item.get('error')}"
                )
            emb = np.asarray(item["embedding"], dtype=np.float32)
            if emb.shape != (512,):
                raise RuntimeError(
                    f"Unexpected embedding shape {emb.shape} (want (512,))"
                )
            out.append(emb)
        return out


# ============================================================
# 維度對齊與 norm 校正
# ============================================================

def project_512_to_2048_repeat4(e512: torch.Tensor) -> torch.Tensor:
    """把最後一維 512 沿最後一維 tile 4 次成 2048。

    例：1D [a,b,...,z] (512) → [a,b,...,z, a,b,...,z, a,b,...,z, a,b,...,z] (2048)
    支援任意前置維度（最後一維必須 = 512）。
    """
    if e512.shape[-1] != 512:
        raise ValueError(f"expected last dim=512, got shape {tuple(e512.shape)}")
    repeat_args = (1,) * (e512.dim() - 1) + (4,)
    return e512.repeat(*repeat_args)


def scale_to_target_norm(vec: torch.Tensor, target_norm: float, eps: float = 1e-8) -> torch.Tensor:
    """沿最後一維重新縮放，使 L2 norm 等於 target_norm。"""
    cur = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return vec * (target_norm / cur)


def manipulate_text_features(
    text_features: torch.Tensor,    # [1, L, 2048]
    token_indices: List[int],
    face_emb_512: torch.Tensor,     # [512]
    mode: str,                      # "subtract" / "replace" / "linear"
    lam1: float = 0.0,
    lam2: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """對 text_features 中指定 token 做 face-embedding 操作。

    流程：
      1. proj = repeat 4× (face_emb_512) → 2048-d
      2. proj_scaled = scale_to_target_norm(proj, ||original token embedding||)
      3. 依 mode 對該 token 做：
         • 'subtract' : new = orig - proj_scaled
         • 'replace'  : new = proj_scaled                 （等同 lam1=0, lam2=1 的 linear）
         • 'linear'   : new = lam1 * orig + lam2 * proj_scaled

    回傳新的 tensor（input 不會被原地修改）。
    """
    valid_modes = ("subtract", "replace", "linear")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
    if not token_indices:
        return text_features.clone()
    if face_emb_512.shape != (512,):
        raise ValueError(f"face_emb_512 must be (512,), got {tuple(face_emb_512.shape)}")

    out = text_features.clone()
    proj_2048_unit = project_512_to_2048_repeat4(
        face_emb_512.to(out.dtype).to(out.device)
    )

    for idx in token_indices:
        orig = out[0, idx, :]
        target_norm = float(orig.norm(p=2).item())
        proj_scaled = scale_to_target_norm(proj_2048_unit, target_norm)
        if mode == "subtract":
            new_vec = orig - proj_scaled
        elif mode == "replace":
            new_vec = proj_scaled
        else:  # linear
            new_vec = lam1 * orig + lam2 * proj_scaled
        if verbose:
            extra = f" lam1={lam1:.3f} lam2={lam2:.3f}" if mode == "linear" else ""
            print(
                f"    [face-op] token[{idx}] mode={mode}{extra} "
                f"orig_norm={target_norm:.3f} "
                f"proj_norm={float(proj_scaled.norm(p=2).item()):.3f} "
                f"new_norm={float(new_vec.norm(p=2).item()):.3f}"
            )
        out[0, idx, :] = new_vec
    return out


# ============================================================
# Encode prompt with face manipulation
# ============================================================

def apply_learned_v_A_to_text_features(
    text_features: torch.Tensor,    # [1, L, 2048]
    token_indices: List[int],
    v_A: torch.Tensor,              # [k, 2048] (k == len(token_indices)) 或 [2048]（單一 token 廣播）
    verbose: bool = False,
) -> torch.Tensor:
    """直接把學好的 v_A 寫入 subject token 位置（不做 repeat-4 / norm-scale）。

    v_A 已經在 Infinity 自身的 embedding 空間裡了，不需要任何投影或 norm 校正。
    回傳 cloned tensor。
    """
    if not token_indices:
        return text_features.clone()
    out = text_features.clone()
    v_A_dev = v_A.to(out.dtype).to(out.device)

    if v_A_dev.dim() == 1:
        # 單一 2048 向量 → 廣播到所有 sub-token
        if v_A_dev.shape[0] != out.shape[-1]:
            raise ValueError(
                f"v_A last dim {v_A_dev.shape[0]} != text_features last dim {out.shape[-1]}"
            )
        for idx in token_indices:
            if verbose:
                orig_norm = float(out[0, idx, :].norm(p=2).item())
                new_norm = float(v_A_dev.norm(p=2).item())
                print(f"    [learned-v_A] token[{idx}] orig_norm={orig_norm:.3f} v_A_norm={new_norm:.3f}")
            out[0, idx, :] = v_A_dev
    else:
        if v_A_dev.shape[0] != len(token_indices):
            raise ValueError(
                f"v_A.shape[0]={v_A_dev.shape[0]} != len(token_indices)={len(token_indices)}"
            )
        for k, idx in enumerate(token_indices):
            if verbose:
                orig_norm = float(out[0, idx, :].norm(p=2).item())
                new_norm = float(v_A_dev[k].norm(p=2).item())
                print(f"    [learned-v_A] token[{idx}] orig_norm={orig_norm:.3f} v_A_norm={new_norm:.3f}")
            out[0, idx, :] = v_A_dev[k]
    return out


def encode_prompt_with_face_op(
    text_tokenizer,
    text_encoder,
    prompt: str,
    face_emb_512: Optional[torch.Tensor] = None,
    op_mode: Optional[str] = None,        # None / "subtract" / "replace" / "linear" / "learned"
    subject_token_indices: Optional[List[int]] = None,
    lam1: float = 0.0,
    lam2: float = 1.0,
    learned_v_A: Optional[torch.Tensor] = None,  # 僅當 op_mode="learned" 時使用
    verbose: bool = False,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
    """encode_prompt 的延伸版：在 trim padding 前對 subject token 做面部 embedding 操作。

    Args:
        prompt:   原始 prompt 字串
        face_emb_512: 512-d AdaFace embedding；op_mode in {subtract, replace, linear} 時使用
        op_mode:  "subtract" / "replace" / "linear" / "learned" / None
        subject_token_indices: 透過 find_focus_token_indices 預先算好的 token 索引
        lam1, lam2: 僅當 op_mode="linear" 時使用，組合成 lam1*orig + lam2*proj_scaled
        learned_v_A: 形狀 [k, 2048] 或 [2048]，op_mode="learned" 時使用；
                     直接寫入 subject token 位置（不做 repeat-4 / norm-scale）

    Returns:
        (kv_compact, lens, cu_seqlens_k, Ltext) — 與 run_p2p_edit.encode_prompt 相同 shape
    """
    captions = [prompt]
    tokens = text_tokenizer(
        text=captions,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)

    text_features = text_encoder(
        input_ids=input_ids, attention_mask=mask
    )["last_hidden_state"].float()                     # [1, 512, 2048]

    if op_mode == "learned":
        if learned_v_A is None:
            raise ValueError("op_mode='learned' requires learned_v_A")
        if not subject_token_indices:
            raise ValueError("subject_token_indices must be non-empty for op_mode='learned'")
        text_features = apply_learned_v_A_to_text_features(
            text_features=text_features,
            token_indices=subject_token_indices,
            v_A=learned_v_A,
            verbose=verbose,
        )
    elif face_emb_512 is not None and op_mode is not None:
        if not subject_token_indices:
            raise ValueError(
                "subject_token_indices must be non-empty when applying face op"
            )
        text_features = manipulate_text_features(
            text_features=text_features,
            token_indices=subject_token_indices,
            face_emb_512=face_emb_512,
            mode=op_mode,
            lam1=lam1,
            lam2=lam2,
            verbose=verbose,
        )

    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(
        mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0)
    )
    Ltext = max(lens)

    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    return (kv_compact, lens, cu_seqlens_k, Ltext)


# ============================================================
# Mean / aggregate face embeddings
# ============================================================

def average_embeddings(
    embeddings: List[np.ndarray],
    renormalize: bool = True,
) -> np.ndarray:
    """對多張人臉 embedding 取平均。

    AdaFace 回傳已 L2-normalized；多張平均後 norm < 1。
    renormalize=True 時把結果再次 L2-normalize，確保 e_A 和 e_B 在同一個球面上。
    """
    if not embeddings:
        raise ValueError("no embeddings to average")
    arr = np.stack(embeddings, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    if renormalize:
        n = float(np.linalg.norm(mean))
        if n > 1e-8:
            mean = mean / n
    return mean
