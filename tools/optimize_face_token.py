#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_face_token.py — Textual Inversion for Face-Swap

對每個 identity（face/<id>/*），凍結 model + VAE，
只把 prompt T_t 中 subject token (例如 "boy") 那條 2048-d slice 設成可學習的 v_A，
用 teacher-forced cross-entropy loss（與 trainer.py / validation_loss.py 同款）
訓練 v_A，使 model 在條件 v_A 之下對 A_n 圖片的 likelihood 最大化。

學完的 v_A 存到 weights/identities/<id>/v_A.pt，phase 2 推論時直接寫入
T5 boy-token 位置（face_swap_utils.encode_prompt_with_face_op(op_mode='learned')）。

執行：
  bash scripts/optimize_face_token.sh
"""

from __future__ import annotations

import argparse
import datetime
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
    encode_prompt,
    find_focus_token_indices,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection  # noqa: E402
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_face_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTS:
            out.append(os.path.join(folder, name))
    return out


def load_and_preprocess_image(
    path: str,
    h_img: int,
    w_img: int,
    device: torch.device,
) -> torch.Tensor:
    """模仿 encode_image_to_raw_features 的前處理：resize + [-1,1]，回傳 [1, 3, H, W]。"""
    img_rgb = PImage.open(path).convert("RGB").resize((w_img, h_img), resample=PImage.LANCZOS)
    img_t = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0
    return img_t.unsqueeze(0).to(device)


def build_bsc(vae, args) -> BitwiseSelfCorrection:
    """建立 BitwiseSelfCorrection；TI 期間 noise 全關 (deterministic teacher forcing)。"""
    bsc_args = argparse.Namespace(
        noise_apply_layers=0,
        noise_apply_requant=0,
        noise_apply_strength=0.0,
        apply_spatial_patchify=int(getattr(args, "apply_spatial_patchify", 0)),
        debug_bsc=0,
    )
    return BitwiseSelfCorrection(vae, bsc_args)


def bit_ce_with_reweight(
    logits_BLV: torch.Tensor,           # [B, L, codebook_dim*2]
    gt_BL: torch.Tensor,                # [B, L, codebook_dim] 0/1
    scale_schedule: list,
    vae,
    reweight_by_scale: bool = True,
) -> torch.Tensor:
    """模仿 validation_loss.py:138-151 的 bitwise CE + scale 重新加權。"""
    B, L, C2 = logits_BLV.shape
    assert C2 == vae.codebook_dim * 2, f"logits last dim {C2} != codebook_dim*2 ({vae.codebook_dim*2})"
    # CE per (B, L, codebook_dim)
    ce = F.cross_entropy(
        logits_BLV.reshape(B, L, vae.codebook_dim, 2).permute(0, 3, 1, 2),
        gt_BL,
        reduction="none",
    )                                    # [B, L, codebook_dim]
    ce = ce.mean(dim=-1)                 # [B, L] -- mean over bits

    if reweight_by_scale:
        lw = []
        last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
        for (pt, ph, pw) in scale_schedule:
            this_scale_area = np.sqrt(pt * ph * pw)
            lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
        lw = torch.tensor(lw[:L], device=ce.device, dtype=ce.dtype)
        lw = lw / lw.sum()
    else:
        lw = torch.full((L,), 1.0 / L, device=ce.device, dtype=ce.dtype)

    return ce.mul(lw[None, :]).sum(dim=-1).mean()


def prepare_image_cache(
    paths: List[str],
    vae,
    bsc: BitwiseSelfCorrection,
    scale_schedule: list,
    apply_spatial_patchify: bool,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """對每張圖預先算 (x_BLC_wo_prefix, gt_BL) 並 cache 在 GPU。"""
    if apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    _, h_final, w_final = vae_scale_schedule[-1]
    patch_size = 8 if apply_spatial_patchify else 16
    h_img = h_final * patch_size
    w_img = w_final * patch_size

    training_seq_len = int(np.array(scale_schedule).prod(axis=1).sum())
    first_scale_len = int(np.array(scale_schedule[0]).prod())

    cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for path in paths:
        inp = load_and_preprocess_image(path, h_img, w_img, device)
        with torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                raw_features, _, _ = vae.encode_for_raw_features(
                    inp, scale_schedule=vae_scale_schedule
                )
            x_BLC, gt_ms = bsc.flip_requant(vae_scale_schedule, inp, raw_features, device)
        x_BLC = x_BLC[:, : training_seq_len - first_scale_len, :].detach()
        gt_BL = torch.cat(gt_ms, dim=1)[:, :training_seq_len].contiguous().long().detach()
        cache.append((x_BLC, gt_BL))
        print(f"  [cache] {os.path.basename(path)}  x_BLC={tuple(x_BLC.shape)}  gt_BL={tuple(gt_BL.shape)}")
    return cache, training_seq_len


def optimize_v_A(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    bsc: BitwiseSelfCorrection,
    identity_dir: str,
    prompt_t: str,
    subject_word: str,
    scale_schedule: list,
    args,
    device: torch.device,
) -> Optional[dict]:
    """單一 identity 的 textual inversion 訓練 loop。"""
    paths = list_face_images(identity_dir)
    if not paths:
        print(f"  [warn] no images under {identity_dir}, skip.")
        return None

    # 1) Encode prompt 一次 → 取 boy token 原始 embedding 當 v_A 初值
    kv_compact_full, lens, cu_seqlens_k, Lmax = encode_prompt(
        text_tokenizer, text_encoder, prompt_t
    )
    # NOTE: encode_prompt return 的 kv_compact 已經 trim padding，
    #       且 find_focus_token_indices 算出來的 idx 也是 trim 後的 index，二者對齊
    subject_token_indices = find_focus_token_indices(
        text_tokenizer, prompt_t, [subject_word]
    )
    if not subject_token_indices:
        print(f"  [error] subject_word='{subject_word}' not found in prompt; skip.")
        return None

    v_init = kv_compact_full[subject_token_indices, :].detach().clone()  # [k, 2048]
    v_A = torch.nn.Parameter(v_init.clone().to(device))                  # leaf, requires_grad=True
    print(f"  [init] v_init.shape={tuple(v_init.shape)}  norm/token={[round(float(v_init[i].norm()),3) for i in range(v_init.shape[0])]}")

    # 2) 凍結
    infinity.eval()
    vae.eval()
    for p in infinity.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    if hasattr(text_encoder, "parameters"):
        for p in text_encoder.parameters():
            p.requires_grad_(False)

    # 必關 cond_drop_rate（CFG dropout 會 in-place 把 kv 換成 cfg_uncond，斷 grad）
    saved_cond_drop_rate = getattr(infinity, "cond_drop_rate", 0.0)
    infinity.cond_drop_rate = 0.0

    # 3) Cache (x_BLC_wo_prefix, gt_BL) per image
    cache, training_seq_len = prepare_image_cache(
        paths=paths,
        vae=vae,
        bsc=bsc,
        scale_schedule=scale_schedule,
        apply_spatial_patchify=bool(args.apply_spatial_patchify),
        device=device,
    )

    # 4) 訓練
    opt = torch.optim.AdamW([v_A], lr=float(args.lr), weight_decay=0.0)
    rng = random.Random(int(args.seed))
    losses: List[float] = []
    init_loss: Optional[float] = None
    t_start = time.time()

    for step in range(int(args.steps)):
        x_BLC, gt_BL = rng.choice(cache)

        # 把 v_A 注入 boy 位置（kv_compact 其餘部分 detach 不更新）
        kv = kv_compact_full.detach().clone()
        for k, idx in enumerate(subject_token_indices):
            kv[idx, :] = v_A[k]

        # trainer 模式：bf16 autocast 開啟跑 forward；只有早期 VAE encode 才會局部關掉
        # （infinity.forward 內部會自己 with autocast(enabled=False) 處理 prefix building）
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            logits_BLV = infinity(
                (kv, lens, cu_seqlens_k, Lmax),
                x_BLC,
                scale_schedule=scale_schedule,
            )
            # logits 在 bf16；CE loss 在 fp32 比較穩
            loss = bit_ce_with_reweight(
                logits_BLV.float(),
                gt_BL,
                scale_schedule=scale_schedule,
                vae=vae,
                reweight_by_scale=True,
            )
            if float(args.l2_reg) > 0:
                loss = loss + float(args.l2_reg) * (v_A - v_init.to(v_A.device)).pow(2).sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        if init_loss is None:
            init_loss = loss_val
        if (step % max(1, int(args.log_every)) == 0) or step == int(args.steps) - 1:
            avg = float(np.mean(losses[-max(1, int(args.log_every)):]))
            v_drift = float((v_A.detach() - v_init.to(v_A.device)).norm(p=2).item())
            print(f"  step {step:4d}/{int(args.steps)}  loss={loss_val:.4f}  avg={avg:.4f}  ||v_A-v_init||={v_drift:.3f}")

    elapsed = time.time() - t_start
    final_loss = float(np.mean(losses[-min(20, len(losses)):])) if losses else float("nan")

    # 還原
    infinity.cond_drop_rate = saved_cond_drop_rate

    print(f"  [done] elapsed={elapsed:.1f}s  init_loss={init_loss:.4f}  final_loss={final_loss:.4f}")
    return {
        "v_A": v_A.detach().cpu(),
        "v_init": v_init.cpu(),
        "subject_token_indices": list(subject_token_indices),
        "subject_word": subject_word,
        "prompt_t": prompt_t,
        "iters": int(args.steps),
        "lr": float(args.lr),
        "l2_reg": float(args.l2_reg),
        "init_loss": float(init_loss) if init_loss is not None else None,
        "final_loss": final_loss,
        "n_images": len(paths),
        "image_paths": paths,
        "loss_history": losses,
        "elapsed_sec": round(elapsed, 2),
    }


def save_v_A_cache(
    cache_dir: str,
    identity: str,
    result: dict,
    save_loss_curve: bool = True,
) -> Tuple[str, str]:
    """把 result 寫到 weights/identities/<id>/{v_A.pt, meta.json, loss_curve.png}。"""
    out_dir = os.path.join(cache_dir, identity)
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, "v_A.pt")
    meta_path = os.path.join(out_dir, "meta.json")

    torch.save({
        "v_A": result["v_A"],
        "v_init": result["v_init"],
        "subject_token_indices": result["subject_token_indices"],
        "subject_word": result["subject_word"],
        "prompt_t": result["prompt_t"],
    }, pt_path)

    meta = {k: v for k, v in result.items() if k not in ("v_A", "v_init", "loss_history")}
    meta["identity"] = identity
    meta["timestamp"] = datetime.datetime.now().isoformat()
    meta["v_A_path"] = pt_path
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if save_loss_curve:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(result["loss_history"], lw=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel("CE loss")
            ax.set_title(f"v_A optim — {identity}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] cannot save loss_curve.png: {exc}")

    return pt_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Textual Inversion for face-swap (per-identity v_A optimization)")
    add_common_arguments(parser)

    parser.add_argument("--face_root", type=str, default="face")
    parser.add_argument("--identities", type=str, default="",
                        help="csv，空=全部 face_root 子資料夾")
    parser.add_argument("--identity_cache_dir", type=str, default="weights/identities")
    parser.add_argument("--regen", type=int, default=0, choices=[0, 1],
                        help="1=即使 v_A.pt 已存在也重訓")

    parser.add_argument("--prompt_t", type=str,
                        default="a boy turned his head to his left over the shuolder and tilted up")
    parser.add_argument("--subject_word", type=str, default="boy")

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Textual Inversion: Face-Swap v_A optimizer")
    print("=" * 80)
    print(f"face_root   : {args.face_root}")
    print(f"identity_cache_dir : {args.identity_cache_dir}")
    print(f"prompt_t    : {args.prompt_t}")
    print(f"subject_word: {args.subject_word}")
    print(f"steps/lr/l2 : {args.steps} / {args.lr} / {args.l2_reg}")
    print(f"regen       : {bool(args.regen)}")
    print("=" * 80 + "\n")

    print("[Init] Loading models once...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    bsc = build_bsc(vae, args)
    print("[Init] Model load complete.\n")

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.identities.strip():
        identity_names = [s.strip() for s in args.identities.split(",") if s.strip()]
    else:
        if not os.path.isdir(args.face_root):
            print(f"[Error] face_root not found: {args.face_root}")
            sys.exit(1)
        identity_names = sorted(
            d for d in os.listdir(args.face_root)
            if os.path.isdir(os.path.join(args.face_root, d))
        )

    print(f"[Plan] identities = {identity_names}\n")

    n_done = n_skip = n_err = 0
    for i, identity in enumerate(identity_names):
        print(f"\n{'-' * 70}")
        print(f"[{i + 1}/{len(identity_names)}] identity = {identity}")
        print(f"{'-' * 70}")
        out_pt = os.path.join(args.identity_cache_dir, identity, "v_A.pt")
        if (not args.regen) and os.path.exists(out_pt) and os.path.getsize(out_pt) > 0:
            n_skip += 1
            print(f"  ↓ skip (exists): {out_pt}")
            continue
        try:
            result = optimize_v_A(
                infinity=infinity,
                vae=vae,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                bsc=bsc,
                identity_dir=os.path.join(args.face_root, identity),
                prompt_t=args.prompt_t,
                subject_word=args.subject_word,
                scale_schedule=scale_schedule,
                args=args,
                device=device,
            )
            if result is None:
                n_err += 1
                continue
            pt_path, meta_path = save_v_A_cache(args.identity_cache_dir, identity, result)
            print(f"  ✓ saved: {pt_path}")
            print(f"           {meta_path}")
            n_done += 1
        except Exception as exc:
            import traceback
            print(f"  ✗ failed: {exc}")
            traceback.print_exc()
            n_err += 1
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"Textual Inversion complete  done={n_done}  skipped={n_skip}  errors={n_err}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
