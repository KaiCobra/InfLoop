#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_token_replace_t2i.py — 純 T2I + token-embedding 替換 批量測試 pipeline

不使用 P2P-Edit 三階段；只做：
  1. 對每個 prompt 直接做一次 T2I 生成 → no_replace baseline
  2. 把 prompt 中的 subject token (boy / girl / man / woman) 在 T5 hidden state
     上以 proj(e_A) (AdaFace 512 → 2048 repeat-4 + norm-scale) 整段替換，
     再做一次 T2I 生成 → replaced

輸出結構：
  {output_dir}/
    _baseline/{p_idx:03d}.jpg          # 共用，60 張
    {face_id}/
      ref_{filename}                   # 該 identity 的原圖（複製過來方便比對）
      no_replace/{p_idx:03d}.jpg       # 60 張，hardlink 自 _baseline
      replaced/{p_idx:03d}.jpg         # 60 張，token 替換後生成
    manifest.json

10 個 identity 從 face_root 取 deterministic shuffle 後第一批 MTCNN 對得上的圖。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

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
from tools.run_pie_edit_faceSwap import gen_one_img_kv  # noqa: E402
from tools.face_swap_utils import (  # noqa: E402
    AdaFaceClient,
    encode_prompt_with_face_op,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


SUBJECT_WORDS = ("boy", "girl", "man", "woman")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ============================================================
# Helpers
# ============================================================

def find_subject_word(prompt: str, prompt_id: str) -> Optional[str]:
    """根據 jsonl 'id' 直接決定 subject word；fallback 用 regex。"""
    pid = (prompt_id or "").lower().strip()
    if pid in SUBJECT_WORDS:
        return pid
    p_lower = " " + prompt.lower() + " "
    for w in SUBJECT_WORDS:
        if f" {w} " in p_lower or f" {w}." in p_lower or f" {w}," in p_lower:
            return w
    return None


def pick_valid_faces(
    client: AdaFaceClient,
    face_root: str,
    n_target: int,
    seed: int = 0,
) -> Tuple[List[str], List[np.ndarray]]:
    """從 face_root 取出 n_target 個能成功取得 AdaFace embedding 的圖檔。

    Deterministic：用 seed shuffle，從頭往後挑，跳過 MTCNN 對不到臉的。
    """
    all_files = sorted(
        os.path.join(face_root, fn) for fn in os.listdir(face_root)
        if fn.lower().endswith(IMG_EXTS)
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_files))
    rng.shuffle(indices)
    shuffled = [all_files[i] for i in indices]

    selected: List[str] = []
    embs: List[np.ndarray] = []
    n_tried = 0
    for path in shuffled:
        if len(selected) >= n_target:
            break
        n_tried += 1
        try:
            emb = client.embed_files([path], align_face=True, return_norm=True)[0]
        except RuntimeError as exc:
            # MTCNN failed for this image, just skip
            continue
        except Exception as exc:
            print(f"  [pick_faces] ERR {os.path.basename(path)}: {exc}")
            continue
        selected.append(path)
        embs.append(emb)
        print(f"  [pick_faces] {len(selected):2d}/{n_target}  {os.path.basename(path)}  "
              f"(tried={n_tried})")
    return selected, embs


def gen_image_np(
    infinity, vae, kv_tuple, scale_schedule, args, seed: int
) -> np.ndarray:
    img = gen_one_img_kv(
        infinity, vae,
        text_cond_tuple=kv_tuple,
        g_seed=int(seed),
        gt_leak=0, gt_ls_Bl=None,
        cfg_list=args.cfg, tau_list=args.tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        p2p_token_storage=None,
        p2p_token_replace_prob=0.0,
        p2p_use_mask=False,
        p2p_save_tokens=False,
        p2p_attn_full_replace_scales=0,
        inject_image_features=None,
        inject_schedule=None,
    )
    img_np = img.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


def link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)        # hardlink first (zero-cost)
    except OSError:
        shutil.copy2(src, dst)   # different fs → copy


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch token-embedding replacement T2I generation"
    )
    add_common_arguments(parser)
    parser.add_argument("--prompts_file", type=str,
                        default="./posePrompt/t2i_pose_prompts.jsonl")
    parser.add_argument("--face_root", type=str,
                        default="/media/avlab/ee303_4T/faces_dataset_small")
    parser.add_argument("--n_faces", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--gen_seed", type=int, default=42)
    parser.add_argument("--face_pick_seed", type=int, default=0)
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Batch Token-Replacement T2I  (no P2P-Edit, plain text-only)")
    print("=" * 80)
    print(f"prompts_file : {args.prompts_file}")
    print(f"face_root    : {args.face_root}")
    print(f"n_faces      : {args.n_faces}")
    print(f"output_dir   : {args.output_dir}")
    print(f"pn / cfg / tau / seed : {args.pn} / {args.cfg} / {args.tau} / {args.gen_seed}")
    print("=" * 80 + "\n")

    # ── Load prompts ──
    prompts: List[dict] = []
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                prompts.append(json.loads(ln))
    print(f"[prompts] loaded {len(prompts)} entries from {args.prompts_file}")

    # ── AdaFace client ──
    client = AdaFaceClient(url=args.adaface_url)
    health = client.health()
    print(f"[AdaFace] {health}")

    # ── Pick identities ──
    print(f"\n[pick_faces] selecting up to {args.n_faces} valid faces "
          f"(seed={args.face_pick_seed})")
    selected_paths, embs = pick_valid_faces(
        client, args.face_root, args.n_faces, seed=args.face_pick_seed
    )
    if not selected_paths:
        print("[ERROR] no valid faces found, abort")
        sys.exit(1)
    if len(selected_paths) < args.n_faces:
        print(f"[warn] only got {len(selected_paths)} valid faces (target {args.n_faces})")

    # ── Output dir + manifest ──
    os.makedirs(args.output_dir, exist_ok=True)
    baseline_dir = os.path.join(args.output_dir, "_baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    manifest = {
        "prompts_file": os.path.abspath(args.prompts_file),
        "face_root": os.path.abspath(args.face_root),
        "n_prompts": len(prompts),
        "n_faces": len(selected_paths),
        "faces": [
            {
                "rank": i,
                "path": p,
                "filename": os.path.basename(p),
                "id": os.path.splitext(os.path.basename(p))[0],
                "emb_norm": float(np.linalg.norm(emb)),
            }
            for i, (p, emb) in enumerate(zip(selected_paths, embs))
        ],
        "gen_seed": args.gen_seed,
        "face_pick_seed": args.face_pick_seed,
        "pn": args.pn,
        "cfg": args.cfg,
        "tau": args.tau,
        "model_path": args.model_path,
        "vae_path": args.vae_path,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print("[dry_run] manifest written; not loading model. Exit.")
        return

    # ── Load model once ──
    print("\n[Init] Loading models...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    print("[Init] Model load complete.")

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(f"[scale_schedule] {len(scale_schedule)} scales, final={scale_schedule[-1]}")

    # 預先計算每個 prompt 的 subject word + token indices
    prompt_meta: List[dict] = []
    for p_idx, rec in enumerate(prompts):
        prompt = rec["prompt"]
        subject = find_subject_word(prompt, rec.get("id", ""))
        subj_idx: List[int] = []
        if subject is not None:
            subj_idx = find_focus_token_indices(
                text_tokenizer, prompt, [subject], verbose=False
            )
        prompt_meta.append({
            "p_idx": p_idx,
            "id": rec.get("id", ""),
            "prompt": prompt,
            "subject_word": subject,
            "subject_token_indices": subj_idx,
        })
        if not subj_idx:
            print(f"  [warn] prompt {p_idx:03d}: subject='{subject}' not found in tokens")

    # 同時把 prompt_meta 也存進 manifest
    with open(os.path.join(args.output_dir, "prompt_meta.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_meta, f, indent=2, ensure_ascii=False)

    # =====================================================
    # Phase A：生成 baseline (no replace)，一次 60 張共用
    # =====================================================
    print("\n" + "=" * 80)
    print(f"[Phase A] Baseline (no replace) — {len(prompts)} prompts")
    print("=" * 80)
    t_phaseA = time.time()
    for meta in prompt_meta:
        p_idx = meta["p_idx"]
        prompt = meta["prompt"]
        out_path = os.path.join(baseline_dir, f"{p_idx:03d}.jpg")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        kv = encode_prompt(text_tokenizer, text_encoder, prompt)
        with torch.no_grad():
            img_np = gen_image_np(infinity, vae, kv, scale_schedule, args, args.gen_seed)
        cv2.imwrite(out_path, img_np)
        if (p_idx + 1) % 5 == 0 or p_idx == len(prompts) - 1:
            print(f"  [{p_idx+1:2d}/{len(prompts)}] {out_path}  "
                  f"elapsed {time.time()-t_phaseA:.1f}s")
    print(f"[Phase A] done — {time.time()-t_phaseA:.1f}s")

    # =====================================================
    # Phase B：每個 identity，token-replace 後生成 60 張
    # =====================================================
    for face_idx, (face_path, emb) in enumerate(zip(selected_paths, embs)):
        face_id = os.path.splitext(os.path.basename(face_path))[0]
        face_dir = os.path.join(args.output_dir, face_id)
        no_replace_dir = os.path.join(face_dir, "no_replace")
        replaced_dir = os.path.join(face_dir, "replaced")
        os.makedirs(no_replace_dir, exist_ok=True)
        os.makedirs(replaced_dir, exist_ok=True)

        # 把 reference 原圖複製進 face folder（方便看）
        ref_dst = os.path.join(face_dir, "ref_" + os.path.basename(face_path))
        link_or_copy(face_path, ref_dst)

        # no_replace：hardlink 自 baseline_dir
        for p_idx in range(len(prompts)):
            src = os.path.join(baseline_dir, f"{p_idx:03d}.jpg")
            dst = os.path.join(no_replace_dir, f"{p_idx:03d}.jpg")
            if os.path.exists(src):
                link_or_copy(src, dst)

        # replaced
        emb_t = torch.from_numpy(emb.astype(np.float32))
        print("\n" + "-" * 80)
        print(f"[Face {face_idx+1}/{len(selected_paths)}] {face_id}  "
              f"emb_norm={float(np.linalg.norm(emb)):.3f}")
        print("-" * 80)
        t_face = time.time()

        for meta in prompt_meta:
            p_idx = meta["p_idx"]
            prompt = meta["prompt"]
            subj_idx = meta["subject_token_indices"]
            out_path = os.path.join(replaced_dir, f"{p_idx:03d}.jpg")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue
            if not subj_idx:
                # 沒對到 token，把 baseline 直接 link 過來（保留同數量）
                src = os.path.join(baseline_dir, f"{p_idx:03d}.jpg")
                if os.path.exists(src):
                    link_or_copy(src, out_path)
                continue

            kv = encode_prompt_with_face_op(
                text_tokenizer, text_encoder,
                prompt=prompt,
                face_emb_512=emb_t,
                op_mode="replace",
                subject_token_indices=subj_idx,
                verbose=False,
            )
            with torch.no_grad():
                img_np = gen_image_np(infinity, vae, kv, scale_schedule, args, args.gen_seed)
            cv2.imwrite(out_path, img_np)

            done_count = p_idx + 1
            if done_count % 10 == 0 or done_count == len(prompts):
                print(f"  [{done_count:2d}/{len(prompts)}]  elapsed {time.time()-t_face:.1f}s")
        print(f"[Face {face_idx+1}] {face_id} done — {time.time()-t_face:.1f}s")

    print("\n" + "=" * 80)
    print(f"All done. Output at: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
