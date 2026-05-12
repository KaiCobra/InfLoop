#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_id_resampler.py — 訓練 IDResampler（VGGFace2HQ + Gemma4 caption 版）

資料規格
--------
caption JSON 範例（一個 identity 對應一個 JSON）：
  [
    {
      "image_path": ".../vgg_face_parts_extracted/1/1/n000002/0001_01.jpg",
      "description": "The sks woman is looking directly at the camera ..."
    },
    ...
  ]

  • 每個 JSON 內所有 entry 屬於 **同一個人**
  • description 內的 "sks" 是 identity placeholder → subject token
  • 我們對每張圖計算 AdaFace e_A，丟進 Resampler 產出 token，覆蓋 prompt 中
    "sks" 對應的 T5 output 位置；teacher-force VAE bitwise tokens 算 CE loss

Pipeline
--------
1. Walk json_root，找所有 `*_gemma4_captions.json`（可加 --include_batch_sample 把 sample 版也納入）
2. 把所有 (image_path, caption, identity) 攤平成 entries；
   tokenize caption 找 "sks" 位置；找不到的 entry 直接丟掉。
3. Lazy 在 disk 上 cache：
     • <bsc_cache_dir>/pn=<pn>_patchify=<0|1>/<identity>/<basename>.pt   {x_BLC, gt_BL}
     • <adaface_cache_dir>/<identity>/<basename>.npy                    e_A (512,)
4. 訓練 loop：每 step 隨機抽一個 entry → T5(caption) → resampler(e_A, prompt_ctx)
   → inject @ sks 位置 → infinity teacher-force → bit CE + L2-anchor reg → backward。
5. 凍結 Infinity / VAE / T5；只更新 Resampler。

執行
----
  bash scripts/train_id_resampler.sh
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PImage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (  # noqa: E402
    add_common_arguments,
    find_focus_token_indices,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from tools.optimize_face_token import (  # noqa: E402
    bit_ce_with_reweight,
    build_bsc,
    load_and_preprocess_image,
)
from tools.face_swap_utils import (  # noqa: E402
    AdaFaceClient,
    apply_resampler_to_text_features,
)
from tools.id_resampler import (  # noqa: E402
    IDResampler,
    extract_orig_sks_from_text_features,
    get_anchor_t5_embedding,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


# ============================================================
# JSON 探索 + entry 攤平
# ============================================================

def _identity_from_json_path(json_path: str) -> str:
    """從 .../<id>_gemma4_captions[_batch_sample].json 取 <id>"""
    name = os.path.basename(json_path)
    for suf in ("_gemma4_captions_batch_sample.json", "_gemma4_captions.json"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return os.path.splitext(name)[0]


def discover_caption_jsons(
    json_root: str,
    include_batch_sample: bool = False,
) -> List[str]:
    """walk json_root 找 *_gemma4_captions.json（與可選 _batch_sample.json）。"""
    if not os.path.isdir(json_root):
        raise FileNotFoundError(f"json_root not found: {json_root}")
    pat_full = os.path.join(json_root, "**", "*_gemma4_captions.json")
    files = sorted(glob.glob(pat_full, recursive=True))
    if include_batch_sample:
        pat_sample = os.path.join(json_root, "**", "*_gemma4_captions_batch_sample.json")
        files += sorted(glob.glob(pat_sample, recursive=True))
        files = sorted(set(files))
    return files


# ============================================================
# Dataset
# ============================================================

class VGGFaceCaptionDataset(torch.utils.data.Dataset):
    """每個 item: 一張人臉 + 對應 caption + 該 identity 的 AdaFace e_A。

    __getitem__ 回傳 dict（CPU tensors，訓練 loop 自己 .to(device)）：
      image_path : str
      identity   : str
      caption    : str
      sks_indices: List[int]
      x_BLC      : (1, L_x, codebook_dim*?)  bsc 切過的 input feat
      gt_BL      : (1, L_gt, codebook_dim)   bitwise GT
      e_A        : (512,)  AdaFace 已 L2-normalized

    bsc / adaface 都走「lazy disk cache」：第一次取會算並存檔，之後直接讀。
    """

    def __init__(
        self,
        json_paths: List[str],
        text_tokenizer,
        subject_token: str,
        scale_schedule: list,
        apply_spatial_patchify: bool,
        vae,
        bsc,
        adaface_client: Optional[AdaFaceClient],
        bsc_cache_dir: str,
        adaface_cache_dir: str,
        device: torch.device,
        verbose: bool = False,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.subject_token = subject_token
        self.scale_schedule = scale_schedule
        self.apply_spatial_patchify = bool(apply_spatial_patchify)
        self.vae = vae
        self.bsc = bsc
        self.adaface_client = adaface_client
        self.device = device
        self.verbose = verbose

        # --- 推 cache 子目錄（依 pn / patchify 分槽，避免 schedule 改了仍讀舊 cache） ---
        pn_tag = "x".join(f"{pt}-{ph}-{pw}" for (pt, ph, pw) in scale_schedule)
        cache_subkey = f"sched={pn_tag}__patchify={int(self.apply_spatial_patchify)}"
        self.bsc_cache_dir = os.path.join(bsc_cache_dir, cache_subkey)
        self.adaface_cache_dir = adaface_cache_dir
        os.makedirs(self.bsc_cache_dir, exist_ok=True)
        os.makedirs(self.adaface_cache_dir, exist_ok=True)

        # --- 推 image 解析度（同 prepare_image_cache 邏輯）---
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        self.vae_scale_schedule = vae_scale_schedule
        _, h_final, w_final = vae_scale_schedule[-1]
        patch_size = 8 if self.apply_spatial_patchify else 16
        self.h_img = h_final * patch_size
        self.w_img = w_final * patch_size
        self.training_seq_len = int(np.array(scale_schedule).prod(axis=1).sum())
        self.first_scale_len = int(np.array(scale_schedule[0]).prod())

        # --- 攤平 entries + 預先 tokenize 找 sks 位置 ---
        entries: List[Dict] = []
        n_total = 0
        n_no_sks = 0
        n_missing = 0
        for jp in json_paths:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as exc:
                print(f"[Dataset] ⚠ cannot read {jp}: {exc}")
                continue
            identity = _identity_from_json_path(jp)
            for it in items:
                n_total += 1
                ip = it.get("image_path", "")
                cap = it.get("description", "") or it.get("caption", "")
                if not ip or not cap:
                    continue
                if not os.path.exists(ip):
                    n_missing += 1
                    continue
                sks_idx = find_focus_token_indices(
                    text_tokenizer, cap, [subject_token], verbose=False
                )
                if not sks_idx:
                    n_no_sks += 1
                    continue
                entries.append({
                    "image_path": ip,
                    "caption": cap,
                    "identity": identity,
                    "sks_indices": sks_idx,
                })
        self.entries = entries
        print(f"[Dataset] scanned {n_total} entries from {len(json_paths)} JSON file(s)")
        print(f"[Dataset]   kept    : {len(entries)}")
        print(f"[Dataset]   missing img: {n_missing}")
        print(f"[Dataset]   no '{subject_token}' token: {n_no_sks}")

    # ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    # ---------------------------------------------------------
    def _bsc_cache_path(self, identity: str, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0]
        sub = os.path.join(self.bsc_cache_dir, identity)
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{base}.pt")

    def _adaface_cache_path(self, identity: str, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0]
        sub = os.path.join(self.adaface_cache_dir, identity)
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{base}.npy")

    # ---------------------------------------------------------
    def _compute_bsc(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = load_and_preprocess_image(image_path, self.h_img, self.w_img, self.device)
        with torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                raw_features, _, _ = self.vae.encode_for_raw_features(
                    inp, scale_schedule=self.vae_scale_schedule
                )
            x_BLC, gt_ms = self.bsc.flip_requant(
                self.vae_scale_schedule, inp, raw_features, self.device
            )
        x_BLC = x_BLC[:, : self.training_seq_len - self.first_scale_len, :].detach().cpu()
        gt_BL = (
            torch.cat(gt_ms, dim=1)[:, : self.training_seq_len]
            .contiguous().long().detach().cpu()
        )
        return x_BLC, gt_BL

    def _compute_adaface(self, image_path: str) -> np.ndarray:
        if self.adaface_client is None:
            raise RuntimeError("AdaFace client is None but cache miss; cannot compute.")
        return self.adaface_client.embed_files([image_path])[0].astype(np.float32)

    # ---------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        e = self.entries[idx]
        ip = e["image_path"]
        identity = e["identity"]

        # ── BSC cache（pn-dependent）──
        bsc_p = self._bsc_cache_path(identity, ip)
        if os.path.exists(bsc_p) and os.path.getsize(bsc_p) > 0:
            try:
                blob = torch.load(bsc_p, map_location="cpu", weights_only=False)
                x_BLC, gt_BL = blob["x_BLC"], blob["gt_BL"]
            except Exception as exc:
                if self.verbose:
                    print(f"  [data] bsc cache corrupt {bsc_p}: {exc}; recompute")
                x_BLC, gt_BL = self._compute_bsc(ip)
                torch.save({"x_BLC": x_BLC, "gt_BL": gt_BL}, bsc_p)
        else:
            x_BLC, gt_BL = self._compute_bsc(ip)
            torch.save({"x_BLC": x_BLC, "gt_BL": gt_BL}, bsc_p)

        # ── AdaFace cache ──
        ada_p = self._adaface_cache_path(identity, ip)
        if os.path.exists(ada_p):
            arr = np.load(ada_p)
            if arr.shape != (512,):
                arr = self._compute_adaface(ip)
                np.save(ada_p, arr)
        else:
            arr = self._compute_adaface(ip)
            np.save(ada_p, arr)
        e_A = torch.from_numpy(arr.astype(np.float32))

        return {
            "image_path": ip,
            "identity": identity,
            "caption": e["caption"],
            "sks_indices": list(e["sks_indices"]),
            "x_BLC": x_BLC,
            "gt_BL": gt_BL,
            "e_A": e_A,
        }


# ============================================================
# 預先 warm cache（單進程跑過一遍 dataset，把 bsc / adaface 都算好寫到磁碟）
# ============================================================

def warm_cache(dataset: VGGFaceCaptionDataset, log_every: int = 100) -> None:
    n = len(dataset)
    print(f"[WarmCache] iterating {n} entries to fill bsc / adaface cache...")
    t0 = time.time()
    n_err = 0
    for i in range(n):
        try:
            _ = dataset[i]
        except Exception as exc:
            n_err += 1
            if n_err <= 5:
                print(f"  [warm] ⚠ idx={i} {dataset.entries[i]['image_path']}: {exc}")
        if (i + 1) % log_every == 0 or i == n - 1:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"  [warm] {i + 1}/{n}  rate={rate:.1f} it/s  eta={eta/60:.1f}min  err={n_err}")
    print(f"[WarmCache] done in {(time.time()-t0)/60:.1f}min  errors={n_err}")


# ============================================================
# Per-step utility: 把 (text_features, mask) trim padding 變 kv_compact
# ============================================================

def text_features_to_kv_compact(
    text_features: torch.Tensor,              # (1, L, 2048)
    mask: torch.Tensor,                       # (1, L)
) -> Tuple[torch.Tensor, list, torch.Tensor, int]:
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(
        mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0)
    )
    Ltext = max(lens)
    parts = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        parts.append(feat_i[:len_i])
    kv_compact = torch.cat(parts, dim=0)
    return (kv_compact, lens, cu_seqlens_k, Ltext)


def _scale_token_offsets(scale_schedule: list) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    cur = 0
    for pn in scale_schedule:
        n_tok = int(np.array(pn).prod())
        offsets.append((cur, cur + n_tok))
        cur += n_tok
    return offsets


def bit_ce_for_scale_range(
    logits_BLV: torch.Tensor,
    gt_BL: torch.Tensor,
    scale_schedule: list,
    vae,
    scale_start: int,
    scale_end: int,
    reweight_by_scale: bool = True,
) -> torch.Tensor:
    """Bitwise CE restricted to a contiguous 0-based scale range."""
    B, L, C2 = logits_BLV.shape
    assert C2 == vae.codebook_dim * 2, (
        f"logits last dim {C2} != codebook_dim*2 ({vae.codebook_dim*2})"
    )
    offsets = _scale_token_offsets(scale_schedule)
    if scale_start < 0 or scale_end >= len(offsets) or scale_start > scale_end:
        raise ValueError(
            f"invalid scale range [{scale_start}, {scale_end}] for {len(offsets)} scales"
        )
    start = offsets[scale_start][0]
    end = offsets[scale_end][1]

    ce = F.cross_entropy(
        logits_BLV.reshape(B, L, vae.codebook_dim, 2).permute(0, 3, 1, 2),
        gt_BL,
        reduction="none",
    ).mean(dim=-1)                                                            # [B, L]
    ce_sel = ce[:, start:end]

    if reweight_by_scale:
        lw = []
        last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
        for si, (pt, ph, pw) in enumerate(scale_schedule):
            if scale_start <= si <= scale_end:
                this_scale_area = np.sqrt(pt * ph * pw)
                lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
        lw = torch.tensor(lw, device=ce.device, dtype=ce.dtype)
        lw = lw / lw.sum()
    else:
        lw = torch.full((end - start,), 1.0 / (end - start), device=ce.device, dtype=ce.dtype)

    return ce_sel.mul(lw[None, :]).sum(dim=-1).mean()


# ============================================================
# Training loop
# ============================================================

def train_resampler(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    resampler: IDResampler,
    dataset: VGGFaceCaptionDataset,
    scale_schedule: list,
    args,
    device: torch.device,
    out_ckpt: str,
) -> Dict:
    if len(dataset) == 0:
        raise RuntimeError("empty dataset")

    # 凍結 Infinity / VAE / T5
    infinity.eval()
    vae.eval()
    for p in infinity.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    if hasattr(text_encoder, "parameters"):
        for p in text_encoder.parameters():
            p.requires_grad_(False)
    saved_cond_drop_rate = getattr(infinity, "cond_drop_rate", 0.0)
    infinity.cond_drop_rate = 0.0

    # Resampler trainable
    resampler = resampler.to(device).train()
    for p in resampler.parameters():
        p.requires_grad_(True)
    n_train_params = sum(p.numel() for p in resampler.parameters() if p.requires_grad)
    print(f"[Train] resampler trainable params = {n_train_params/1e6:.3f} M")

    opt = torch.optim.AdamW(
        [p for p in resampler.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(args.steps), eta_min=float(args.lr) * 0.05
        )
    else:
        sched = None

    anchor_init = resampler.anchor.detach().clone()                          # (n_tokens, t5_dim)
    prefix_total_len = int(np.array(scale_schedule).prod(axis=1).sum())
    prefix_input_len = prefix_total_len - int(np.array(scale_schedule[0]).prod())

    rng = random.Random(int(args.seed))
    losses: List[float] = []
    init_loss: Optional[float] = None
    t_start = time.time()
    grad_accum = max(1, int(args.grad_accum))
    opt.zero_grad(set_to_none=True)

    n = len(dataset)
    for step in range(int(args.steps)):
        # ── 抽 sample（重抽直到 __getitem__ 不丟 exception）──
        for _retry in range(8):
            idx = rng.randrange(n)
            try:
                item = dataset[idx]
                break
            except Exception as exc:
                if _retry == 7:
                    raise
                if step < 10:
                    print(f"  [data] retry idx={idx}: {exc}")

        caption = item["caption"]
        sks_idx = item["sks_indices"]
        x_BLC = item["x_BLC"][:, :prefix_input_len, :].to(device, non_blocking=True)
        gt_BL = item["gt_BL"][:, :prefix_total_len, :].to(device, non_blocking=True)
        e_A = item["e_A"].to(device, non_blocking=True)

        # ── T5 on-the-fly（freeze 過的，no_grad）──
        tokens = text_tokenizer(
            text=[caption],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        with torch.no_grad():
            text_features = text_encoder(
                input_ids=input_ids, attention_mask=mask
            )["last_hidden_state"].float()                                       # (1, 512, 2048)

        # ── Resampler forward（grad on）──
        orig_sks_emb = extract_orig_sks_from_text_features(
            text_features, sks_idx, resampler.n_tokens
        ).to(device)
        base_emb = orig_sks_emb if resampler.residual_base == "orig" else None
        resampler_out = resampler(
            id_feat=e_A.unsqueeze(0),
            prompt_ctx=text_features if resampler.use_prompt_ctx else None,
            prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
            base_emb=base_emb,
        )                                                                       # (1, n_tokens, 2048)

        modified_tf = apply_resampler_to_text_features(
            text_features=text_features,
            token_indices=sks_idx,
            resampler_output=resampler_out,
            verbose=False,
        )
        kv = text_features_to_kv_compact(modified_tf, mask)

        # ── Infinity teacher-forced forward ──
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            logits_BLV = infinity(
                kv,
                x_BLC,
                scale_schedule=scale_schedule,
            )
            loss = bit_ce_for_scale_range(
                logits_BLV.float(),
                gt_BL,
                scale_schedule=scale_schedule,
                vae=vae,
                scale_start=int(args.train_scale_start),
                scale_end=int(args.train_scale_end),
                reweight_by_scale=True,
            )

        # ── direction / norm regularizers（對抗 manifold drift）──
        if resampler.residual_base == "orig":
            reg_target = orig_sks_emb.unsqueeze(0).to(resampler_out.device)     # (1, n, t5)
        else:
            reg_target = anchor_init.to(resampler_out.device).unsqueeze(0)      # (1, n, t5)

        if float(args.l2_anchor) > 0:
            reg_l2 = (resampler_out - reg_target).pow(2).mean()
            loss = loss + float(args.l2_anchor) * reg_l2

        # cosine-to-target：直接抑制方向漂移（RMSNorm 抹平 norm 後留下來的核心訊號）
        cos_sim_val = None
        if float(args.cos_anchor) > 0:
            cs = F.cosine_similarity(resampler_out, reg_target, dim=-1)         # (1, n)
            cos_sim_val = float(cs.mean().item())
            loss = loss + float(args.cos_anchor) * (1.0 - cs.mean())

        # output norm penalty：避免 ||out|| 漸漸炸大，target 依 residual_base 切換
        out_norm_val = None
        if float(args.norm_penalty) > 0:
            target_norm = reg_target.norm(p=2, dim=-1).mean()
            out_norm = resampler_out.norm(p=2, dim=-1).mean()
            out_norm_val = float(out_norm.item())
            loss = loss + float(args.norm_penalty) * (out_norm - target_norm).pow(2)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in resampler.parameters() if p.requires_grad],
                    max_norm=float(args.grad_clip),
                )
            opt.step()
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        if init_loss is None:
            init_loss = loss_val

        if (step % max(1, int(args.log_every)) == 0) or step == int(args.steps) - 1:
            window = max(1, int(args.log_every))
            avg = float(np.mean(losses[-window:]))
            anchor_drift = float(
                (resampler.anchor.detach() - anchor_init.to(resampler.anchor.device))
                .norm(p=2).item()
            )
            cur_lr = opt.param_groups[0]["lr"]
            extra = ""
            if cos_sim_val is not None:
                extra += f"  cos={cos_sim_val:+.3f}"
            if out_norm_val is not None:
                tgt = float(reg_target.norm(p=2, dim=-1).mean().item())
                extra += f"  out_n={out_norm_val:.3f}/tgt={tgt:.3f}"
            if resampler.residual_base == "orig":
                base_drift = float(
                    (resampler_out - reg_target).norm(p=2, dim=-1).mean().item()
                )
                extra += f"  base_drift={base_drift:.3f}"
            print(f"  step {step:5d}/{int(args.steps)}  loss={loss_val:.4f}  "
                  f"avg{window}={avg:.4f}  anchor_drift={anchor_drift:.4f}  lr={cur_lr:.2e}{extra}  "
                  f"id={item['identity']}  cap=\"{caption[:60]}...\"")

        if args.save_every > 0 and (step + 1) % int(args.save_every) == 0:
            _save_ckpt(
                resampler, out_ckpt, step + 1, args, dataset, init_loss, losses,
                scale_schedule=scale_schedule,
            )

    elapsed = time.time() - t_start
    final_loss = float(np.mean(losses[-min(50, len(losses)):])) if losses else float("nan")

    infinity.cond_drop_rate = saved_cond_drop_rate

    print(f"[Done] elapsed={elapsed:.1f}s  init_loss={init_loss:.4f}  final_loss={final_loss:.4f}")
    _save_ckpt(
        resampler, out_ckpt, int(args.steps), args, dataset, init_loss, losses,
        scale_schedule=scale_schedule,
    )
    return {
        "out_ckpt": out_ckpt,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "elapsed_sec": round(elapsed, 2),
        "n_entries": len(dataset),
    }


def _save_ckpt(
    resampler: IDResampler,
    out_ckpt: str,
    step: int,
    args,
    dataset: VGGFaceCaptionDataset,
    init_loss,
    losses,
    scale_schedule: list,
) -> None:
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    payload = {
        "state_dict": resampler.state_dict(),
        "config": {
            "id_dim": resampler.id_dim,
            "t5_dim": resampler.t5_dim,
            "n_tokens": resampler.n_tokens,
            "n_id_ctx": resampler.n_id_ctx,
            "use_prompt_ctx": resampler.use_prompt_ctx,
            "delta_max_norm": resampler.delta_max_norm,
            "out_norm_match": resampler.out_norm_match,
            "residual_base": resampler.residual_base,
            "train_scale_start": int(args.train_scale_start),
            "train_scale_end": int(args.train_scale_end),
            "scale_schedule": [list(x) for x in scale_schedule],
        },
        "meta": {
            "step": step,
            "n_entries": len(dataset),
            "subject_token": args.subject_token,
            "anchor_word": args.resampler_anchor_word,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "l2_anchor": float(args.l2_anchor),
            "cos_anchor": float(args.cos_anchor),
            "norm_penalty": float(args.norm_penalty),
            "residual_base": resampler.residual_base,
            "train_scale_start": int(args.train_scale_start),
            "train_scale_end": int(args.train_scale_end),
            "grad_accum": int(args.grad_accum),
            "grad_clip": float(args.grad_clip),
            "lr_schedule": args.lr_schedule,
            "init_loss": init_loss,
            "final_loss": float(np.mean(losses[-min(50, len(losses)):])) if losses else None,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    }
    root, ext = os.path.splitext(out_ckpt)
    ext = ext or ".pt"
    step_path = f"{root}_step{int(step):06d}{ext}"
    torch.save(payload, step_path)
    torch.save(payload, out_ckpt)
    print(f"  ✓ saved ckpt @ step {step} → {step_path}  (latest: {out_ckpt})")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IDResampler on VGGFace2HQ + Gemma4 captions"
    )
    add_common_arguments(parser)

    # Dataset
    parser.add_argument("--json_root", type=str, required=True,
                        help="walk 找 *_gemma4_captions.json；遞迴搜尋")
    parser.add_argument("--include_batch_sample", type=int, default=0, choices=[0, 1],
                        help="是否也納入 *_gemma4_captions_batch_sample.json")
    parser.add_argument("--max_jsons", type=int, default=-1,
                        help="只讀前 N 個 JSON（隨機洗牌後）；-1=全部")
    parser.add_argument("--max_entries", type=int, default=-1,
                        help="dataset attach 後再隨機抽 M 個 entry；-1=全部")
    parser.add_argument("--subject_token", type=str, default="sks",
                        help="caption 中要被 Resampler 替換的 placeholder（VGG/Gemma4 用 sks）")
    parser.add_argument("--bsc_cache_dir", type=str, default="weights/bsc_cache",
                        help="x_BLC / gt_BL 的 disk cache 根目錄（依 pn / patchify 分槽）")
    parser.add_argument("--adaface_cache_dir", type=str, default="weights/adaface_cache",
                        help="AdaFace embedding 的 disk cache 根目錄")
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--warm_cache_only", type=int, default=0, choices=[0, 1],
                        help="1=只跑一遍 dataset 把 cache 塞滿就結束，不訓練")

    # Resampler 結構
    parser.add_argument("--resampler_n_tokens", type=int, default=1)
    parser.add_argument("--resampler_n_id_ctx", type=int, default=4)
    parser.add_argument("--resampler_n_layers", type=int, default=2)
    parser.add_argument("--resampler_n_heads", type=int, default=8)
    parser.add_argument("--resampler_use_prompt_ctx", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resampler_anchor_word", type=str, default="person")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="若給了，從這個 ckpt 載入 state_dict 繼續訓")

    # Optim
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--l2_anchor", type=float, default=1e-2,
                        help="L2(out − reg_target).mean() 係數（per-element MSE）")
    parser.add_argument("--cos_anchor", type=float, default=0.1,
                        help="(1 − cos(out, reg_target)) 係數；direction-drift 防護的核心")
    parser.add_argument("--norm_penalty", type=float, default=1e-2,
                        help="(||out|| − ||reg_target||)^2 係數")
    parser.add_argument("--resampler_delta_max_norm", type=float, default=-1.0,
                        help=">0 時 hard-clamp Resampler 的 delta per-token L2；-1=不限")
    parser.add_argument("--resampler_out_norm_match", type=str, default="none",
                        choices=["none", "anchor", "base"],
                        help="forward 時把 out 的 per-token norm 強制對齊 anchor 或 residual base")
    parser.add_argument("--resampler_residual_base", type=str, default="orig",
                        choices=["anchor", "orig"],
                        help="Resampler 殘差基底；orig=用 prompt 當下 sks T5 output，anchor=舊行為")
    parser.add_argument("--train_scale_start", type=int, default=4,
                        help="0-based 第一個計 loss 的 scale；預設 4 = 人類說的第 5 個 scale")
    parser.add_argument("--train_scale_end", type=int, default=6,
                        help="0-based 最後一個計 loss 的 scale；預設 6 = 人類說的第 7 個 scale")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--out_ckpt", type=str, default="weights/id_resampler/resampler.pt")

    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Train IDResampler (VGGFace2HQ + Gemma4 captions)")
    print("=" * 80)
    print(f"json_root          : {args.json_root}")
    print(f"include_batch_sample: {bool(args.include_batch_sample)}")
    print(f"max_jsons / entries: {args.max_jsons} / {args.max_entries}")
    print(f"subject_token      : '{args.subject_token}'")
    print(f"bsc_cache_dir      : {args.bsc_cache_dir}")
    print(f"adaface_cache_dir  : {args.adaface_cache_dir}")
    print(f"adaface_url        : {args.adaface_url}")
    print(f"steps/lr/wd        : {args.steps} / {args.lr} / {args.weight_decay}")
    print(f"grad_accum/clip    : {args.grad_accum} / {args.grad_clip}")
    print(f"lr_schedule        : {args.lr_schedule}")
    print(f"l2_anchor          : {args.l2_anchor}")
    print(f"cos_anchor         : {args.cos_anchor}")
    print(f"norm_penalty       : {args.norm_penalty}")
    print(f"delta_max_norm     : {args.resampler_delta_max_norm}  "
          f"out_norm_match={args.resampler_out_norm_match}")
    print(f"residual_base      : {args.resampler_residual_base}")
    print(f"train_scale_range  : {args.train_scale_start}..{args.train_scale_end} (0-based)")
    print(f"resampler          : n_tokens={args.resampler_n_tokens} "
          f"n_id_ctx={args.resampler_n_id_ctx} n_layers={args.resampler_n_layers} "
          f"n_heads={args.resampler_n_heads} use_prompt_ctx={args.resampler_use_prompt_ctx}")
    print(f"anchor_word        : {args.resampler_anchor_word}")
    print(f"resume_ckpt        : {args.resume_ckpt or '(none)'}")
    print(f"warm_cache_only    : {bool(args.warm_cache_only)}")
    print(f"out_ckpt           : {args.out_ckpt}")
    print("=" * 80 + "\n")

    # Load models
    print("[Init] Loading models once...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    bsc = build_bsc(vae, args)
    print("[Init] Model load complete.\n")

    full_scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    full_scale_schedule = [(1, h, w) for (_, h, w) in full_scale_schedule]
    if int(args.train_scale_start) < 0 or int(args.train_scale_end) < int(args.train_scale_start):
        raise ValueError(
            f"invalid train scale range: {args.train_scale_start}..{args.train_scale_end}"
        )
    if int(args.train_scale_end) >= len(full_scale_schedule):
        raise ValueError(
            f"train_scale_end={args.train_scale_end} exceeds total scales {len(full_scale_schedule)}"
        )
    train_scale_schedule = full_scale_schedule[: int(args.train_scale_end) + 1]
    print(f"[Schedule] full={len(full_scale_schedule)} scales, training prefix={len(train_scale_schedule)} "
          f"scales, loss on {args.train_scale_start}..{args.train_scale_end}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AdaFace client
    adaface_client = AdaFaceClient(url=args.adaface_url)
    try:
        adaface_client.health()
        print(f"[AdaFace] OK at {args.adaface_url}")
    except Exception as exc:
        print(f"[AdaFace] ⚠ unreachable: {exc}")
        sys.exit(1)

    # Discover JSONs
    json_files = discover_caption_jsons(args.json_root, bool(args.include_batch_sample))
    print(f"[Discover] found {len(json_files)} caption JSON files under {args.json_root}")
    if not json_files:
        print("[Error] no caption JSON files found")
        sys.exit(1)
    if args.max_jsons > 0 and len(json_files) > args.max_jsons:
        rnd = random.Random(int(args.seed))
        rnd.shuffle(json_files)
        json_files = json_files[: int(args.max_jsons)]
        print(f"[Discover] capped to first {len(json_files)} JSONs")

    # Build dataset
    dataset = VGGFaceCaptionDataset(
        json_paths=json_files,
        text_tokenizer=text_tokenizer,
        subject_token=args.subject_token,
        scale_schedule=full_scale_schedule,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
        vae=vae,
        bsc=bsc,
        adaface_client=adaface_client,
        bsc_cache_dir=args.bsc_cache_dir,
        adaface_cache_dir=args.adaface_cache_dir,
        device=device,
        verbose=False,
    )
    if args.max_entries > 0 and len(dataset.entries) > args.max_entries:
        rnd = random.Random(int(args.seed))
        rnd.shuffle(dataset.entries)
        dataset.entries = dataset.entries[: int(args.max_entries)]
        print(f"[Dataset] capped entries to {len(dataset.entries)}")

    if len(dataset) == 0:
        print("[Error] dataset is empty after filtering")
        sys.exit(1)

    # Optional warm cache
    if args.warm_cache_only:
        warm_cache(dataset)
        print("[WarmCache] requested; exit before training")
        return

    # Resampler
    try:
        anchor_emb = get_anchor_t5_embedding(
            text_tokenizer, text_encoder,
            anchor_phrase=f"a {args.resampler_anchor_word}",
            target_word=args.resampler_anchor_word,
        )
        print(f"[Resampler] anchor='{args.resampler_anchor_word}'  "
              f"emb_norm={float(anchor_emb.norm()):.3f}")
    except Exception as exc:
        print(f"[Resampler] ⚠ anchor derive failed ({exc}); use zero anchor")
        anchor_emb = torch.zeros(int(args.text_channels))

    delta_max_norm = float(args.resampler_delta_max_norm)
    delta_max_norm = delta_max_norm if delta_max_norm > 0 else None
    resampler = IDResampler(
        id_dim=512,
        t5_dim=int(args.text_channels),
        n_tokens=int(args.resampler_n_tokens),
        n_id_ctx=int(args.resampler_n_id_ctx),
        n_layers=int(args.resampler_n_layers),
        n_heads=int(args.resampler_n_heads),
        use_prompt_ctx=bool(args.resampler_use_prompt_ctx),
        anchor_emb=anchor_emb,
        delta_max_norm=delta_max_norm,
        out_norm_match=args.resampler_out_norm_match,
        residual_base=args.resampler_residual_base,
    )
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        state = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = resampler.load_state_dict(state, strict=False)
        print(f"[Resampler] resumed from {args.resume_ckpt}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    # Train
    result = train_resampler(
        infinity=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        resampler=resampler,
        dataset=dataset,
        scale_schedule=train_scale_schedule,
        args=args,
        device=device,
        out_ckpt=args.out_ckpt,
    )
    print("\n" + "=" * 80)
    print("Training complete:")
    print(json.dumps(result, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
