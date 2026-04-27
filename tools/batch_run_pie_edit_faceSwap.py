#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_pie_edit_faceSwap.py — Face-Swap 批量管線（模型只載入一次）

每個 identity（如 face/smith/）跑一次：
  1. 用固定 prompt T_t 產出 base 圖 B（cache 到 output_dir/<identity>/B.jpg）。
  2. 透過 AdaFace HTTP server 取得：
       e_A = mean(ε_f(A_n)) over face/<identity>/* (再 L2-normalize)
       e_B = ε_f(B)
  3. 用三組 T5 embedding 跑 P2P-Edit pipeline：
       phase 1 : T_t 原樣
       phase 1.7: T_t 中 boy token -= proj(e_B)
       phase 2 : T_t 中 boy token  = proj(e_A)
     （proj：512-d repeat 4× → 2048-d，再 scale 到原 token L2 norm）

用法：
  bash scripts/batch_run_pie_edit_faceSwap.sh
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
import time
import traceback
from typing import List, Optional

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
from tools.run_pie_edit_faceSwap import gen_one_img_kv, run_one_case_faceSwap  # noqa: E402
from tools.face_swap_utils import (  # noqa: E402
    AdaFaceClient,
    average_embeddings,
    encode_prompt_with_face_op,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Tee: 同時輸出到終端與 case.log
# ============================================================

class TeeStream:
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        ret = self.stream.write(data)
        self.file_obj.write(data)
        return ret

    def flush(self):
        self.stream.flush()
        self.file_obj.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ============================================================
# Helpers
# ============================================================

def list_face_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTS:
            out.append(os.path.join(folder, name))
    return out


def write_task_info(save_dir: str, payload: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    info_path = os.path.join(save_dir, "task_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def generate_base_image(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    prompt_t: str,
    args,
    scale_schedule,
    out_path: str,
) -> None:
    """以固定 prompt T_t 跑純 text-to-image 產出 B，存到 out_path。"""
    kv_raw = encode_prompt(text_tokenizer, text_encoder, prompt_t)
    with torch.no_grad():
        img = gen_one_img_kv(
            infinity, vae,
            text_cond_tuple=kv_raw,
            g_seed=args.seed,
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_np)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face-swap 批量 P2P-Edit（模型只載入一次）"
    )
    add_common_arguments(parser)

    # 批量設定
    parser.add_argument("--face_root", type=str, default="face",
                        help="identity 父目錄（每個子資料夾 = 一個 identity）")
    parser.add_argument("--identities", type=str, default="",
                        help="只跑指定 identity（逗號分隔；空白=全部）")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])

    # Face-swap 專屬
    parser.add_argument(
        "--prompt_t", type=str,
        default="a boy turned his head to his left over the shuolder and tilted up",
        help="固定 prompt T_t",
    )
    parser.add_argument("--subject_word", type=str, default="boy",
                        help="prompt 中代表主體的 token 字串")
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--regen_B", type=int, default=0, choices=[0, 1],
                        help="1=強制重新生成 B；0=若 cache 存在就讀取")
    parser.add_argument("--debug_face_op", type=int, default=0, choices=[0, 1],
                        help="1=印出 token 操作前後的 norm 變化")
    parser.add_argument("--lam1", type=float, default=0.0,
                        help="phase 2 線性混合係數：new = lam1*e_I + lam2*proj(e_A)。預設 0=完全 replace")
    parser.add_argument("--lam2", type=float, default=1.0,
                        help="phase 2 線性混合係數，proj(e_A) 的權重。預設 1")
    parser.add_argument("--use_learned_v_A", type=int, default=0, choices=[0, 1],
                        help="1=phase 2 改用 weights/identities/<id>/v_A.pt 中學好的 token 直接寫入 boy 位置；"
                             "0=用 AdaFace linear 路徑")
    parser.add_argument("--identity_cache_dir", type=str, default="weights/identities",
                        help="存放每個 identity 的 v_A.pt（由 tools/optimize_face_token.py 產生）")

    # P2P-Edit 設定（與 batch_run_pie_edit.sh 對齊；移除 PIE-Bench 專屬選項）
    parser.add_argument("--num_full_replace_scales", type=int, default=2)
    parser.add_argument("--attn_threshold_percentile", type=float, default=80.0)
    parser.add_argument("--attn_block_start", type=int, default=2)
    parser.add_argument("--attn_block_end", type=int, default=-1)
    parser.add_argument("--attn_batch_idx", type=int, default=0)
    parser.add_argument("--p2p_token_replace_prob", type=float, default=0.0)
    parser.add_argument("--save_attn_vis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--image_injection_scales", type=int, default=2)
    parser.add_argument("--inject_weights", type=str, default="")
    parser.add_argument("--use_normalized_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_last_scale_mask", type=int, default=0, choices=[0, 1])
    parser.add_argument("--last_scale_majority_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_method", type=int, default=1, choices=list(range(1, 15)))
    parser.add_argument("--absolute_high", type=float, default=0.7)
    parser.add_argument("--absolute_low", type=float, default=0.3)
    parser.add_argument("--debug_mode", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Face-Swap Batch P2P-Edit")
    print("=" * 80)
    print(f"face_root      : {args.face_root}")
    print(f"output_dir     : {args.output_dir}")
    print(f"prompt_t       : {args.prompt_t}")
    print(f"subject_word   : {args.subject_word}")
    print(f"adaface_url    : {args.adaface_url}")
    print(f"lam1, lam2     : {args.lam1}, {args.lam2}  (phase2: lam1*e_I + lam2*proj(e_A))")
    print(f"use_learned_v_A: {bool(args.use_learned_v_A)}  (cache_dir={args.identity_cache_dir})")
    print(f"regen_B        : {bool(args.regen_B)}")
    print(f"skip_existing  : {bool(args.skip_existing)}")
    print(f"full_replace   : {args.num_full_replace_scales}")
    print(f"attn_percentile: {args.attn_threshold_percentile}")
    print("=" * 80 + "\n")

    # ── AdaFace server liveness check ──
    client = AdaFaceClient(url=args.adaface_url)
    try:
        h = client.health()
        print(f"[AdaFace] server OK: {h}")
    except Exception as exc:
        print(f"[AdaFace] ✗ cannot reach server at {args.adaface_url}: {exc}")
        sys.exit(1)

    # ── Models ──
    print("[Init] Loading models once...")
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

    # ── 找出要處理的 identities ──
    if args.identities.strip():
        identity_names = [s.strip() for s in args.identities.split(",") if s.strip()]
    else:
        if not os.path.isdir(args.face_root):
            print(f"[Error] face_root 不存在: {args.face_root}")
            sys.exit(1)
        identity_names = sorted(
            d for d in os.listdir(args.face_root)
            if os.path.isdir(os.path.join(args.face_root, d))
        )

    if not identity_names:
        print(f"[Error] {args.face_root} 下沒有任何 identity 子資料夾")
        sys.exit(1)
    print(f"[Plan] Identities to run: {identity_names}\n")

    # ── 預先找出 subject token 在 prompt 中的位置 ──
    subject_token_indices = find_focus_token_indices(
        text_tokenizer, args.prompt_t, [args.subject_word]
    )
    if not subject_token_indices:
        print(
            f"[Fatal] 在 prompt_t 中找不到 subject_word='{args.subject_word}'。"
            "請改用其他 subject_word，否則 face-swap 操作不會有效果。"
        )
        sys.exit(1)
    print(f"[Plan] subject_token_indices = {subject_token_indices}")

    total_done = 0
    total_skip = 0
    total_err = 0

    for idx, identity in enumerate(identity_names):
        identity_dir = os.path.join(args.face_root, identity)
        save_dir = os.path.join(args.output_dir, identity)

        print(f"\n{'-' * 70}")
        print(f"[{idx + 1}/{len(identity_names)}] identity: {identity}")
        print(f"{'-' * 70}")

        target_out = os.path.join(save_dir, "target.jpg")
        if args.skip_existing and os.path.exists(target_out) and os.path.getsize(target_out) > 0:
            total_skip += 1
            print(f"  ↓ skip existing: {target_out}")
            continue

        a_paths = list_face_images(identity_dir)
        if not a_paths:
            total_err += 1
            print(f"  ✗ no source face images under {identity_dir}")
            continue
        print(f"  source faces: {len(a_paths)}")

        os.makedirs(save_dir, exist_ok=True)
        case_log_path = os.path.join(save_dir, "case.log")
        b_path = os.path.join(save_dir, "B.jpg")
        t_start = time.time()

        try:
            with open(case_log_path, "a", encoding="utf-8") as case_log:
                case_log.write("\n" + "=" * 80 + "\n")
                case_log.write(f"timestamp: {datetime.datetime.now().isoformat()}\n")
                case_log.write(f"identity: {identity}\n")
                case_log.write(f"a_paths: {a_paths}\n")
                case_log.write(f"prompt_t: {args.prompt_t}\n")
                case_log.write(f"subject_word: {args.subject_word}\n")
                case_log.write(f"subject_token_indices: {subject_token_indices}\n")
                case_log.write("-" * 80 + "\n")

                tee_out = TeeStream(sys.stdout, case_log)
                tee_err = TeeStream(sys.stderr, case_log)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    # 1. Generate or reuse cached B
                    if args.regen_B or (not os.path.exists(b_path)) or os.path.getsize(b_path) == 0:
                        print(f"  [B] generating base image with T_t...")
                        generate_base_image(
                            infinity, vae, text_tokenizer, text_encoder,
                            prompt_t=args.prompt_t,
                            args=args,
                            scale_schedule=scale_schedule,
                            out_path=b_path,
                        )
                    else:
                        print(f"  [B] using cached: {b_path}")

                    # 2. AdaFace embeddings
                    print(f"  [AdaFace] embedding {len(a_paths)} source images...")
                    e_a_list = client.embed_files(a_paths)
                    e_a_np = average_embeddings(e_a_list, renormalize=True)
                    print(f"  [AdaFace] embedding B...")
                    e_b_np = client.embed_files([b_path])[0]

                    e_a_norm = float(np.linalg.norm(e_a_np))
                    e_b_norm = float(np.linalg.norm(e_b_np))
                    print(f"  [AdaFace] |e_A|={e_a_norm:.4f}  |e_B|={e_b_norm:.4f}")

                    e_a = torch.from_numpy(e_a_np.astype(np.float32))
                    e_b = torch.from_numpy(e_b_np.astype(np.float32))

                    # 3. Build three kv tuples
                    verbose = bool(args.debug_face_op)
                    kv_phase1 = encode_prompt_with_face_op(
                        text_tokenizer, text_encoder,
                        prompt=args.prompt_t,
                        face_emb_512=None, op_mode=None,
                        subject_token_indices=subject_token_indices,
                        verbose=verbose,
                    )
                    kv_phase17 = encode_prompt_with_face_op(
                        text_tokenizer, text_encoder,
                        prompt=args.prompt_t,
                        face_emb_512=e_b, op_mode="subtract",
                        subject_token_indices=subject_token_indices,
                        verbose=verbose,
                    )
                    # Phase 2 路徑：learned v_A 優先（若啟用且 cache 存在），否則 linear blend
                    using_learned_v_A = False
                    v_A_path = None
                    v_A_meta = None
                    if int(args.use_learned_v_A):
                        v_A_path_candidate = os.path.join(args.identity_cache_dir, identity, "v_A.pt")
                        if os.path.exists(v_A_path_candidate):
                            blob = torch.load(v_A_path_candidate, map_location="cpu", weights_only=False)
                            v_A_tensor = blob["v_A"].float()           # [k, 2048] or [2048]
                            cached_indices = blob.get("subject_token_indices", subject_token_indices)
                            if list(cached_indices) != list(subject_token_indices):
                                print(
                                    f"  [warn] v_A.pt subject_token_indices {cached_indices} != "
                                    f"current {subject_token_indices}; using current and broadcasting"
                                )
                            kv_phase2 = encode_prompt_with_face_op(
                                text_tokenizer, text_encoder,
                                prompt=args.prompt_t,
                                op_mode="learned",
                                subject_token_indices=subject_token_indices,
                                learned_v_A=v_A_tensor,
                                verbose=verbose,
                            )
                            using_learned_v_A = True
                            v_A_path = v_A_path_candidate
                            v_A_meta = {
                                k: blob.get(k) for k in
                                ("subject_token_indices", "subject_word", "prompt_t")
                            }
                            print(f"  [phase2] using learned v_A from {v_A_path_candidate}")
                        else:
                            print(
                                f"  [warn] use_learned_v_A=1 but cache missing: {v_A_path_candidate}; "
                                f"fallback to AdaFace linear (lam1={args.lam1}, lam2={args.lam2})"
                            )
                    if not using_learned_v_A:
                        kv_phase2 = encode_prompt_with_face_op(
                            text_tokenizer, text_encoder,
                            prompt=args.prompt_t,
                            face_emb_512=e_a, op_mode="linear",
                            subject_token_indices=subject_token_indices,
                            lam1=float(args.lam1),
                            lam2=float(args.lam2),
                            verbose=verbose,
                        )

                    # 4. Run face-swap pipeline
                    run_one_case_faceSwap(
                        infinity=infinity,
                        vae=vae,
                        text_tokenizer=text_tokenizer,
                        text_encoder=text_encoder,
                        source_image_path=b_path,
                        prompt_text=args.prompt_t,
                        subject_word=args.subject_word,
                        kv_phase1=kv_phase1,
                        kv_phase17=kv_phase17,
                        kv_phase2=kv_phase2,
                        save_dir=save_dir,
                        args=args,
                        scale_schedule=scale_schedule,
                        attn_block_indices=attn_block_indices,
                        total_scales=total_scales,
                        device_cuda=device_cuda,
                    )

            elapsed = time.time() - t_start
            with open(os.path.join(save_dir, "timing.json"), "w", encoding="utf-8") as tf:
                json.dump({"inference_sec": round(elapsed, 3)}, tf)
            write_task_info(save_dir, {
                "identity": identity,
                "identity_dir": identity_dir,
                "a_paths": a_paths,
                "b_path": b_path,
                "target_path": target_out,
                "prompt_t": args.prompt_t,
                "subject_word": args.subject_word,
                "subject_token_indices": subject_token_indices,
                "lam1": float(args.lam1),
                "lam2": float(args.lam2),
                "use_learned_v_A": bool(int(args.use_learned_v_A)),
                "using_learned_v_A": bool(using_learned_v_A),
                "v_A_path": v_A_path,
                "v_A_meta": v_A_meta,
                "e_A_norm": e_a_norm,
                "e_B_norm": e_b_norm,
                "status": "success",
                "elapsed_sec": round(elapsed, 3),
                "timestamp": datetime.datetime.now().isoformat(),
                "log_path": case_log_path,
                "timing_path": os.path.join(save_dir, "timing.json"),
            })
            total_done += 1
            print(f"  ✓ done ({elapsed:.1f}s) -> {target_out}")

        except Exception as exc:
            total_err += 1
            elapsed = time.time() - t_start
            print(f"  ✗ failed: {exc}")
            traceback.print_exc()
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "timing.json"), "w", encoding="utf-8") as tf:
                json.dump({"inference_sec": round(elapsed, 3)}, tf)
            write_task_info(save_dir, {
                "identity": identity,
                "identity_dir": identity_dir,
                "a_paths": a_paths,
                "b_path": b_path,
                "target_path": target_out,
                "prompt_t": args.prompt_t,
                "subject_word": args.subject_word,
                "subject_token_indices": subject_token_indices,
                "lam1": float(args.lam1),
                "lam2": float(args.lam2),
                "status": "failed",
                "error": str(exc),
                "elapsed_sec": round(elapsed, 3),
                "timestamp": datetime.datetime.now().isoformat(),
                "log_path": case_log_path,
            })
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Face-Swap Batch P2P-Edit complete")
    print(f"  success : {total_done}")
    print(f"  skipped : {total_skip}")
    print(f"  errors  : {total_err}")
    print(f"  output  : {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
