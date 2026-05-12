#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_id_resampler.py — 純 text-to-image inference，把 prompt 中 sks 對應位置的
T5 output embedding 換成 IDResampler(AdaFace(face_image))。

不走 P2P-Edit、不需要 base image B，只是普通的 Infinity 生圖：
    prompt = "The sks woman is looking at the camera ..."
    face   = some_face.jpg

    e_A     = AdaFace(face)                             # (512,)
    feats   = T5(prompt)                                # (1, 512, 2048)
    feats[sks_idx] = IDResampler(e_A, prompt_ctx=feats) # output 端替換
    image   = Infinity.autoregressive_infer_cfg(feats)

執行：
  bash scripts/infer_id_resampler.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import cv2
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
from tools.face_swap_utils import (  # noqa: E402
    AdaFaceClient,
)
from tools.id_resampler import (  # noqa: E402
    IDResampler,
    extract_orig_sks_from_text_features,
    get_anchor_t5_embedding,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


# ============================================================
# Helpers
# ============================================================

def _t5_forward(text_tokenizer, text_encoder, prompt, device):
    """跑 T5 拿 last_hidden_state，回傳 (text_features (1,L,2048), mask (1,L))。"""
    tokens = text_tokenizer(
        text=[prompt],
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
        )["last_hidden_state"].float()
    return text_features, mask


def _text_features_to_kv_compact(text_features, mask):
    """trim padding → (kv_compact, lens, cu_seqlens_k, Ltext)"""
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


def _print_diagnostics(orig_sks, anchor, replacement_vec, sks_idx,
                       label="resampler", base_emb=None):
    """印 norm + cosine 健康度指標。
    orig_sks: (k, 2048)  / anchor: (n_tokens, 2048)  / replacement_vec: (n_tokens, 2048)
    """
    print(f"[diag] sks_idx={sks_idx}  label={label}")
    # average over sub-tokens / output tokens
    o = orig_sks.mean(dim=0)
    a = anchor.mean(dim=0)
    r = replacement_vec.mean(dim=0)

    def _cos(u, v):
        return float(F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item())

    print(f"[diag]   ||orig_sks||  = {float(orig_sks.norm(dim=-1).mean()):.4f}")
    print(f"[diag]   ||anchor||    = {float(anchor.norm(dim=-1).mean()):.4f}")
    print(f"[diag]   ||{label:<10}||= {float(replacement_vec.norm(dim=-1).mean()):.4f}")
    print(f"[diag]   cos({label}, anchor)   = {_cos(r, a):+.4f}")
    print(f"[diag]   cos({label}, orig_sks) = {_cos(r, o):+.4f}")
    print(f"[diag]   cos(anchor,    orig_sks) = {_cos(a, o):+.4f}")
    if base_emb is not None:
        b = base_emb.mean(dim=0)
        drift = float((replacement_vec - base_emb).norm(p=2, dim=-1).mean())
        print(f"[diag]   ||base||      = {float(base_emb.norm(p=2, dim=-1).mean()):.4f}")
        print(f"[diag]   cos({label}, base)     = {_cos(r, b):+.4f}")
        print(f"[diag]   ||{label}-base|| = {drift:.4f}")


# ============================================================
# Build IDResampler，可從 ckpt 載入；ckpt 內若有 config 會覆寫 CLI
# ============================================================

def build_resampler(args, text_tokenizer, text_encoder, device) -> IDResampler:
    # anchor 從 T5 拿
    try:
        anchor_emb = get_anchor_t5_embedding(
            text_tokenizer, text_encoder,
            anchor_phrase=f"a {args.resampler_anchor_word}",
            target_word=args.resampler_anchor_word,
        )
    except Exception as exc:
        print(f"[Resampler] ⚠ anchor derive failed ({exc}); use zero anchor")
        anchor_emb = torch.zeros(int(args.text_channels))

    # 預設用 CLI 的結構參數
    n_tokens = int(args.resampler_n_tokens)
    n_id_ctx = int(args.resampler_n_id_ctx)
    n_layers = int(args.resampler_n_layers)
    n_heads = int(args.resampler_n_heads)
    use_prompt_ctx = bool(args.resampler_use_prompt_ctx)
    delta_max_norm_arg = float(args.resampler_delta_max_norm)
    delta_max_norm = delta_max_norm_arg if delta_max_norm_arg > 0 else None
    out_norm_match = args.resampler_out_norm_match
    residual_base = args.resampler_residual_base

    ckpt_path = (args.resampler_ckpt or "").strip()
    state_dict = None
    if ckpt_path and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "config" in blob:
            cfg = blob["config"]
            n_tokens = int(cfg.get("n_tokens", n_tokens))
            n_id_ctx = int(cfg.get("n_id_ctx", n_id_ctx))
            use_prompt_ctx = bool(cfg.get("use_prompt_ctx", use_prompt_ctx))
            if "delta_max_norm" in cfg:
                delta_max_norm = cfg["delta_max_norm"]
            if "out_norm_match" in cfg:
                out_norm_match = cfg["out_norm_match"]
            if "residual_base" in cfg:
                residual_base = cfg["residual_base"]
            print(f"[Resampler] ckpt config overrides: n_tokens={n_tokens} "
                  f"n_id_ctx={n_id_ctx} use_prompt_ctx={use_prompt_ctx} "
                  f"delta_max_norm={delta_max_norm} out_norm_match={out_norm_match} "
                  f"residual_base={residual_base}")
        state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    elif ckpt_path:
        print(f"[Resampler] ⚠ ckpt path '{ckpt_path}' 不存在；用初始權重")

    resampler = IDResampler(
        id_dim=512,
        t5_dim=int(args.text_channels),
        n_tokens=n_tokens,
        n_id_ctx=n_id_ctx,
        n_layers=n_layers,
        n_heads=n_heads,
        use_prompt_ctx=use_prompt_ctx,
        anchor_emb=anchor_emb,
        delta_max_norm=delta_max_norm,
        out_norm_match=out_norm_match,
        residual_base=residual_base,
    )
    if state_dict is not None:
        missing, unexpected = resampler.load_state_dict(state_dict, strict=False)
        print(f"[Resampler] loaded {ckpt_path}  missing={len(missing)} unexpected={len(unexpected)}")
    resampler = resampler.to(device).eval()
    n_params = sum(p.numel() for p in resampler.parameters())
    print(f"[Resampler] ready  params={n_params/1e6:.2f}M  device={device}  "
          f"n_tokens={resampler.n_tokens}  use_prompt_ctx={resampler.use_prompt_ctx}")
    return resampler


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plain T2I with sks → IDResampler(AdaFace(face)) injection"
    )
    add_common_arguments(parser)

    # 輸入
    parser.add_argument("--prompt", type=str, required=True,
                        help="caption；裡面要含 subject_token（預設 'sks'）")
    parser.add_argument("--face_image", type=str, required=True,
                        help="人臉圖片路徑（給 AdaFace 算 e_A）")
    parser.add_argument("--subject_token", type=str, default="sks",
                        help="prompt 中要被替換的 token（VGG/Gemma4 caption 用 sks）")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="同一 prompt + face 生幾張（差別只在 seed 遞增）")

    # 輸出
    parser.add_argument("--out_dir", type=str, default="outputs/id_resampler_infer")
    parser.add_argument("--out_prefix", type=str, default="",
                        help="輸出檔名前綴；空=用 face image basename")

    # AdaFace
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")

    # IDResampler 結構（ckpt config 會覆寫）
    parser.add_argument("--resampler_ckpt", type=str, default="")
    parser.add_argument("--resampler_n_tokens", type=int, default=1)
    parser.add_argument("--resampler_n_id_ctx", type=int, default=4)
    parser.add_argument("--resampler_n_layers", type=int, default=2)
    parser.add_argument("--resampler_n_heads", type=int, default=8)
    parser.add_argument("--resampler_use_prompt_ctx", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resampler_anchor_word", type=str, default="person")
    parser.add_argument("--resampler_delta_max_norm", type=float, default=-1.0,
                        help=">0 時 hard-clamp Resampler delta 的 per-token L2；-1=不限。"
                             "ckpt config 內的設定會覆寫此項")
    parser.add_argument("--resampler_out_norm_match", type=str, default="none",
                        choices=["none", "anchor", "base"],
                        help="forward 把 out 的 per-token norm 強制對齊 anchor 或 residual base。ckpt 會覆寫")
    parser.add_argument("--resampler_residual_base", type=str, default="anchor",
                        choices=["anchor", "orig"],
                        help="Resampler 殘差基底；通常由 ckpt config 覆寫，CLI 是舊 ckpt fallback")

    # Inference 行為控制
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="resampler",
        choices=["no_inject", "anchor_only", "resampler"],
        help=(
            "no_inject  = 完全不替換 sks，純 prompt → 預期出 prompt 對應圖；"
            "anchor_only = sks 位置改成 Resampler 的 anchor（純 'person' T5 emb），"
            "跳過 Resampler forward → 預期出泛 person；"
            "resampler  = 用訓練好的 Resampler 輸出替換（現行行為）"
        ),
    )
    parser.add_argument(
        "--inject_alpha",
        type=float,
        default=1.0,
        help="lambda mixing：out[sks] = (1-α)*orig + α*new；α=0 等同 no_inject、α=1 全替換",
    )
    parser.add_argument("--resampler_match_orig_norm", type=int, default=0, choices=[0, 1],
                        help="apply 時是否把 src 先 rescale 到 ||orig_sks||")

    # Infinity 推論（cfg_insertion_layer 已在 add_common_arguments）
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    args.cfg = list(map(float, str(args.cfg).split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Infer with sks → IDResampler(AdaFace) replacement")
    print("=" * 80)
    print(f"prompt        : {args.prompt}")
    print(f"face_image    : {args.face_image}")
    print(f"subject_token : '{args.subject_token}'")
    print(f"resampler_ckpt: {args.resampler_ckpt or '(uninitialized)'}")
    print(f"baseline_mode : {args.baseline_mode}  inject_alpha={args.inject_alpha:.3f}  "
          f"match_orig_norm={bool(args.resampler_match_orig_norm)}")
    print(f"out_dir       : {args.out_dir}")
    print(f"n_samples     : {args.n_samples}  seed_start={args.seed}")
    print(f"pn / cfg / tau: {args.pn} / {args.cfg} / {args.tau}")
    print("=" * 80 + "\n")

    if not os.path.exists(args.face_image):
        print(f"[Error] face_image not found: {args.face_image}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load models ──
    print("[Init] Loading models...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    print("[Init] Done.\n")

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── AdaFace（baseline_mode=no_inject 也會跑：只是不會用 e_A）──
    client = AdaFaceClient(url=args.adaface_url)
    try:
        client.health()
    except Exception as exc:
        print(f"[AdaFace] ⚠ unreachable: {exc}")
        sys.exit(1)
    e_A = client.embed_files([args.face_image])[0]
    e_A_t = torch.from_numpy(e_A.astype(np.float32))
    print(f"[AdaFace] e_A.norm = {float(np.linalg.norm(e_A)):.4f}")

    # ── Subject token indices in this prompt ──
    sks_idx = find_focus_token_indices(
        text_tokenizer, args.prompt, [args.subject_token], verbose=False
    )
    if not sks_idx:
        print(f"[Error] subject_token '{args.subject_token}' not found in prompt:")
        print(f"        '{args.prompt}'")
        sys.exit(1)
    print(f"[Tokens] subject_token '{args.subject_token}' positions: {sks_idx}")

    # ── Resampler（即使 no_inject 也建構，但不會 forward；anchor_only 只用其 .anchor）──
    resampler = build_resampler(args, text_tokenizer, text_encoder, device)

    # ── 跑 T5 拿 last_hidden_state ──
    text_features, mask = _t5_forward(text_tokenizer, text_encoder, args.prompt, device)
    print(f"[T5] text_features shape = {tuple(text_features.shape)}  "
          f"valid_len = {int(mask.sum())}")

    # ── 取 sks 對應位置原本的 token 當 reference（diagnostic 用）──
    orig_sks = text_features[0, sks_idx, :].detach().clone()        # (k, 2048)
    orig_sks_emb = None

    # ── 三種 baseline mode 分支 ──
    alpha = float(args.inject_alpha)
    mode = args.baseline_mode

    if mode == "no_inject":
        # 不替換，純 prompt → 預期出 prompt 對應圖
        modified_tf = text_features
        print(f"[Inject] mode=no_inject  (sks 位置維持 T5 原輸出)")
    else:
        # 計算要寫入的向量
        if mode == "anchor_only":
            # 直接拿 Resampler 的 anchor，不跑 forward
            replacement = resampler.anchor.detach().to(text_features.device)  # (n_tokens, 2048)
            label = "anchor"
        else:  # "resampler"
            orig_sks_emb = extract_orig_sks_from_text_features(
                text_features, sks_idx, resampler.n_tokens
            ).to(device)                                            # (n_tokens, 2048)
            base_emb_arg = orig_sks_emb if resampler.residual_base == "orig" else None
            with torch.no_grad():
                r_out = resampler(
                    id_feat=e_A_t.to(device).unsqueeze(0),
                    prompt_ctx=text_features if resampler.use_prompt_ctx else None,
                    prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
                    base_emb=base_emb_arg,
                )                                                            # (1, n_tokens, 2048)
            replacement = r_out[0]                                          # (n_tokens, 2048)
            label = "resampler"

        # diagnostic
        _print_diagnostics(
            orig_sks=orig_sks,
            anchor=resampler.anchor.detach().to(text_features.device),
            replacement_vec=replacement,
            sks_idx=sks_idx,
            label=label,
            base_emb=orig_sks_emb if mode == "resampler" else None,
        )

        # 對應地寫入 sks 位置（n_tokens=1 廣播；n_tokens=k 一對一）
        modified_tf = text_features.clone()
        n_tokens = replacement.shape[0]
        n_sub = len(sks_idx)
        if n_tokens == 1:
            for idx in sks_idx:
                src = replacement[0]
                if bool(args.resampler_match_orig_norm):
                    eps = 1e-8
                    orig_norm = text_features[0, idx, :].norm(p=2).clamp_min(eps)
                    src = src / src.norm(p=2).clamp_min(eps) * orig_norm
                modified_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
        elif n_tokens == n_sub:
            for k, idx in enumerate(sks_idx):
                src = replacement[k]
                if bool(args.resampler_match_orig_norm):
                    eps = 1e-8
                    orig_norm = text_features[0, idx, :].norm(p=2).clamp_min(eps)
                    src = src / src.norm(p=2).clamp_min(eps) * orig_norm
                modified_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
        else:
            raise RuntimeError(
                f"replacement n_tokens={n_tokens} 必須是 1 或 len(sks_idx)={n_sub}"
            )
        print(f"[Inject] mode={mode}  alpha={alpha:.3f}  "
              f"replaced {n_sub} sks position(s) with {n_tokens} {label} token(s)")

    # ── trim padding → kv_compact ──
    text_cond_tuple = _text_features_to_kv_compact(modified_tf, mask)
    print(f"[Encode] kv_compact shape = {tuple(text_cond_tuple[0].shape)}  "
          f"lens = {text_cond_tuple[1]}")

    # ── 推論 ──
    cfg_list = args.cfg if isinstance(args.cfg, list) else [float(args.cfg)] * len(scale_schedule)
    tau_list = [float(args.tau)] * len(scale_schedule)
    if len(cfg_list) < len(scale_schedule):
        cfg_list = cfg_list + [cfg_list[-1]] * (len(scale_schedule) - len(cfg_list))

    base_name = args.out_prefix.strip()
    if not base_name:
        base_name = os.path.splitext(os.path.basename(args.face_image))[0]

    for i in range(int(args.n_samples)):
        seed = int(args.seed) + i
        out_name = f"{base_name}__seed{seed}.jpg"
        out_path = os.path.join(args.out_dir, out_name)

        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            _, _, img_list = infinity.autoregressive_infer_cfg(
                vae=vae,
                scale_schedule=scale_schedule,
                label_B_or_BLT=text_cond_tuple,
                B=1,
                negative_label_B_or_BLT=None,
                force_gt_Bhw=None,
                g_seed=seed,
                cfg_list=cfg_list,
                tau_list=tau_list,
                cfg_sc=3,
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                returns_vemb=1,
                gumbel=0,
                norm_cfg=False,
                cfg_exp_k=0.0,
                cfg_insertion_layer=[int(args.cfg_insertion_layer)],
                vae_type=int(args.vae_type),
                softmax_merge_topk=-1,
                ret_img=True,
                trunk_scale=1000,
                gt_leak=0,
                gt_ls_Bl=None,
                inference_mode=True,
                sampling_per_bits=int(args.sampling_per_bits),
            )
        elapsed = time.time() - t0

        img = img_list[0].cpu().numpy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, img)
        print(f"  [{i + 1}/{args.n_samples}] seed={seed}  saved {out_path}  ({elapsed:.1f}s)")

    print("\n[Done]")


if __name__ == "__main__":
    main()
