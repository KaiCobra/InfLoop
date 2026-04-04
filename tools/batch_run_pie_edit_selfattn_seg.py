#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_pie_edit_selfattn_seg.py — PIE-Bench 批量 DiffSegmenter-style 物件分割

與 batch_run_pie_edit.py 的差異：
  • Source gen 時同時掛載 CrossAttentionExtractor + SelfAttentionExtractor
  • 使用 DiffSegmenter 迭代精煉（SA × CA）產生物件 mask
  • 不做 target gen — 只輸出 source.jpg + 物件 mask 視覺化
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
import traceback
import contextlib
import datetime
from typing import List, Tuple

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (
    add_common_arguments,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from tools.run_pie_edit_selfattn_seg import run_one_case
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


# ============================================================
# 工具函式（與 batch_run_pie_edit.py 共用邏輯）
# ============================================================

def remove_prompt_brackets(prompt: str) -> str:
    """移除 []，保留括號內文字。"""
    if not prompt:
        return ""
    return re.sub(r"\[([^\]]*)\]", r"\1", prompt).strip()


def tokenize_words(prompt: str) -> List[str]:
    """將 prompt 轉成詞序列（小寫，去標點）。"""
    return re.findall(r"[\w']+", (prompt or "").lower(), flags=re.UNICODE)


def _unique_keep_order(words: List[str]) -> List[str]:
    """去重但保留原順序。"""
    seen = set()
    out: List[str] = []
    for w in words:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out


def derive_focus_words_from_prompt_diff(
    source_prompt: str,
    target_prompt: str,
) -> Tuple[str, str, List[str], List[str]]:
    """
    比對 source/target prompt 詞級差異，回傳 focus words。
    """
    src_words = tokenize_words(source_prompt)
    tgt_words = tokenize_words(target_prompt)
    matcher = difflib.SequenceMatcher(None, src_words, tgt_words)

    src_diff: List[str] = []
    tgt_diff: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete") and i2 > i1:
            src_diff.extend(src_words[i1:i2])
        if tag in ("replace", "insert") and j2 > j1:
            tgt_diff.extend(tgt_words[j1:j2])

    src_focus_words = _unique_keep_order(src_diff)
    tgt_focus_words = _unique_keep_order(tgt_diff)
    src_focus_str = " ".join(src_focus_words)
    tgt_focus_str = " ".join(tgt_focus_words)
    return src_focus_str, tgt_focus_str, src_focus_words, tgt_focus_words


class TeeStream:
    """同時寫入終端與檔案。"""

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


def write_task_info(save_dir: str, payload: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    info_path = os.path.join(save_dir, "task_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIE-Bench 批量 DiffSegmenter-style 物件分割（模型只載入一次）"
    )
    add_common_arguments(parser)

    # 批量設定
    parser.add_argument("--bench_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--max_per_cat", type=int, default=-1)
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])

    # P2P-Edit 基礎參數（仍需要供 gen_one_img / run_one_case 讀取）
    parser.add_argument("--num_full_replace_scales", type=int, default=2)
    parser.add_argument("--attn_threshold_percentile", type=float, default=80.0)
    parser.add_argument("--attn_block_start", type=int, default=2)
    parser.add_argument("--attn_block_end", type=int, default=-1)
    parser.add_argument("--attn_batch_idx", type=int, default=0)
    parser.add_argument("--p2p_token_replace_prob", type=float, default=0.0)
    parser.add_argument("--save_attn_vis", type=int, default=1, choices=[0, 1])

    # Source image injection
    parser.add_argument("--image_injection_scales", type=int, default=2)
    parser.add_argument("--inject_weights", type=str, default="")

    # Self-Attention Seg（DiffSegmenter）專用參數
    parser.add_argument("--sa_refine_iterations", type=int, default=3,
                        help="DiffSegmenter SA×CA 迭代次數（預設：3）")
    parser.add_argument("--sa_block_start", type=int, default=2,
                        help="Self-attention 擷取起始 block（預設：2）")
    parser.add_argument("--sa_block_end", type=int, default=5,
                        help="Self-attention 擷取結束 block（-1=最後，預設：5）")
    parser.add_argument("--sa_max_scale", type=int, default=9,
                        help="Self-attention 擷取的最大 scale index（超過此值改用上採樣）。"
                             "-1=全部（預設：9）")

    # 以下參數僅作相容用途（本 pipeline 不使用）
    parser.add_argument("--use_pie_mask", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--pie_mask_attn_fallback", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mask_expand_percent", type=float, default=0.0)
    parser.add_argument("--use_attn_cache", type=int, default=0, choices=[0, 1])
    parser.add_argument("--attn_cache_phase", type=str, default="both")
    parser.add_argument("--attn_cache_max_scale", type=int, default=6)
    parser.add_argument("--attn_cache_align_mode", type=str, default="full_p2p")
    parser.add_argument("--phase17_fallback_replace_scales", type=int, default=0)
    parser.add_argument("--use_normalized_attn", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--use_cumulative_prob_mask", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    # cfg 處理
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Batch PIE SelfattnSeg (DiffSegmenter-style)")
    print("=" * 80)
    print(f"bench_dir           : {args.bench_dir}")
    print(f"output_dir          : {args.output_dir}")
    print(f"skip_existing       : {bool(args.skip_existing)}")
    print(f"max_per_cat         : {args.max_per_cat if args.max_per_cat > 0 else 'all'}")
    print(f"sa_refine_iters     : {args.sa_refine_iterations}")
    print(f"sa_block_range      : {args.sa_block_start}~{args.sa_block_end}")
    print(f"sa_max_scale        : {args.sa_max_scale}")
    print(f"attn_percentile     : {args.attn_threshold_percentile}")
    print("=" * 80 + "\n")

    # 模型只載入一次
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

    if args.categories.strip():
        cat_names = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        cat_names = None  # will be determined from data

    # ── Detect dataset format ──
    mapping_file = os.path.join(args.bench_dir, "mapping_file.json")
    annotation_dir = os.path.join(args.bench_dir, "annotation_images")
    use_v1_format = os.path.exists(mapping_file) and os.path.isdir(annotation_dir)

    if use_v1_format:
        print(f"[Dataset] PIE-Bench_v1 format detected (mapping_file.json)")
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        from collections import defaultdict
        tasks_by_cat: dict[str, list[dict]] = defaultdict(list)
        for task_id, info in mapping.items():
            img_rel = info.get("image_path", "")
            category = img_rel.split("/")[0] if "/" in img_rel else "unknown"
            img_abs = os.path.join(annotation_dir, img_rel)
            tasks_by_cat[category].append({
                "task_id": task_id,
                "image_path": img_abs,
                "original_prompt": info.get("original_prompt", ""),
                "editing_prompt": info.get("editing_prompt", ""),
                "editing_instruction": info.get("editing_instruction", ""),
                "editing_type_id": info.get("editing_type_id", ""),
                "blended_word": info.get("blended_word", ""),
                "mask": info.get("mask", []),
            })
        for cat in tasks_by_cat:
            tasks_by_cat[cat].sort(key=lambda t: t["task_id"])

        if cat_names is None:
            cat_names = sorted(tasks_by_cat.keys())
        else:
            cat_names = [c for c in cat_names if c in tasks_by_cat]
    else:
        print(f"[Dataset] extracted_pie_bench format (per-task meta.json)")
        tasks_by_cat = None
        if cat_names is None:
            cat_names = sorted(
                d for d in os.listdir(args.bench_dir)
                if os.path.isdir(os.path.join(args.bench_dir, d))
            )

    total_done = 0
    total_skip = 0
    total_err = 0

    for cat_name in cat_names:
        # ── Collect cases for this category ──
        if use_v1_format:
            raw_cases = tasks_by_cat.get(cat_name, [])
            if args.max_per_cat > 0:
                raw_cases = raw_cases[:args.max_per_cat]
            cases = []
            for tc in raw_cases:
                cases.append({
                    "case_id": tc["task_id"],
                    "img_path": tc["image_path"],
                    "mask_path": None,
                    "source_prompt_raw": tc["original_prompt"],
                    "target_prompt_raw": tc["editing_prompt"],
                    "case_dir": os.path.dirname(tc["image_path"]),
                    "meta_raw": tc,
                })
        else:
            cat_dir = os.path.join(args.bench_dir, cat_name)
            if not os.path.isdir(cat_dir):
                print(f"[Warn] Missing category dir: {cat_dir}")
                continue
            case_ids = sorted(
                d for d in os.listdir(cat_dir)
                if os.path.isdir(os.path.join(cat_dir, d))
            )
            if args.max_per_cat > 0:
                case_ids = case_ids[:args.max_per_cat]
            cases = []
            for cid in case_ids:
                cdir = os.path.join(cat_dir, cid)
                meta_path = os.path.join(cdir, "meta.json")
                img_path = os.path.join(cdir, "image.jpg")
                mask_path = os.path.join(cdir, "mask.png")
                if not os.path.exists(meta_path) or not os.path.exists(img_path):
                    cases.append({"case_id": cid, "error": "missing meta.json or image.jpg"})
                    continue
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cases.append({
                    "case_id": cid,
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "source_prompt_raw": meta.get("source_prompt", ""),
                    "target_prompt_raw": meta.get("target_prompt", ""),
                    "case_dir": cdir,
                    "meta_raw": meta,
                })

        print(f"\n{'-' * 70}")
        print(f"[Category] {cat_name} ({len(cases)} cases)")
        print(f"{'-' * 70}")

        for idx, case in enumerate(cases):
            case_id = case["case_id"]

            if "error" in case:
                print(f"  [{idx + 1}/{len(cases)}] {case_id}  ✗ {case['error']}")
                total_err += 1
                continue

            img_path = case["img_path"]
            mask_path = case.get("mask_path")
            case_dir = case["case_dir"]

            if not os.path.exists(img_path):
                print(f"  [{idx + 1}/{len(cases)}] {case_id}  ✗ missing image: {img_path}")
                total_err += 1
                continue

            save_dir = os.path.join(args.output_dir, cat_name, case_id)
            seg_meta = os.path.join(save_dir, "selfattn_seg", "meta.json")
            if args.skip_existing and os.path.exists(seg_meta):
                total_skip += 1
                print(f"  [{idx + 1}/{len(cases)}] {case_id}  ↓ skip existing")
                continue

            source_prompt = remove_prompt_brackets(case["source_prompt_raw"])
            target_prompt = remove_prompt_brackets(case["target_prompt_raw"])

            src_focus_str, tgt_focus_str, src_focus_words, tgt_focus_words = (
                derive_focus_words_from_prompt_diff(source_prompt, target_prompt)
            )

            print(f"\n  [{idx + 1}/{len(cases)}] {case_id}")
            print(f"    src_prompt : {source_prompt}")
            print(f"    tgt_prompt : {target_prompt}")
            print(f"    src_focus  : {src_focus_str}")
            print(f"    tgt_focus  : {tgt_focus_str}")

            t_start = time.time()
            os.makedirs(save_dir, exist_ok=True)
            case_log_path = os.path.join(save_dir, "case.log")
            try:
                with open(case_log_path, "a", encoding="utf-8") as case_log:
                    case_log.write("\n" + "=" * 80 + "\n")
                    case_log.write(f"timestamp: {datetime.datetime.now().isoformat()}\n")
                    case_log.write(f"category: {cat_name}, case_id: {case_id}\n")
                    case_log.write(f"source_prompt: {source_prompt}\n")
                    case_log.write(f"target_prompt: {target_prompt}\n")
                    case_log.write(f"source_focus: {src_focus_str}\n")
                    case_log.write(f"target_focus: {tgt_focus_str}\n")
                    case_log.write("-" * 80 + "\n")

                    tee_out = TeeStream(sys.stdout, case_log)
                    tee_err = TeeStream(sys.stderr, case_log)
                    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                        run_one_case(
                            infinity=infinity,
                            vae=vae,
                            text_tokenizer=text_tokenizer,
                            text_encoder=text_encoder,
                            source_image_path=img_path,
                            source_prompt=source_prompt,
                            target_prompt=target_prompt,
                            source_focus_words=src_focus_words,
                            target_focus_words=tgt_focus_words,
                            save_dir=save_dir,
                            args=args,
                            scale_schedule=scale_schedule,
                            attn_block_indices=attn_block_indices,
                            total_scales=total_scales,
                            device_cuda=device_cuda,
                            mask_path=mask_path,
                            blended_words=None,
                            case_source_dir=case_dir,
                        )
                elapsed = time.time() - t_start
                write_task_info(save_dir, {
                    "category": cat_name,
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "success",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_sec": round(elapsed, 3),
                    "source_prompt_clean": source_prompt,
                    "target_prompt_clean": target_prompt,
                    "source_focus_words": src_focus_words,
                    "target_focus_words": tgt_focus_words,
                })
                total_done += 1
                print(f"    ✓ done ({elapsed:.1f}s)")
            except Exception as exc:
                total_err += 1
                print(f"    ✗ failed: {exc}")
                traceback.print_exc()
                elapsed = time.time() - t_start
                write_task_info(save_dir, {
                    "category": cat_name,
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "failed",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_sec": round(elapsed, 3),
                    "error": str(exc),
                })
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Batch PIE SelfattnSeg complete")
    print(f"  success : {total_done}")
    print(f"  skipped : {total_skip}")
    print(f"  errors  : {total_err}")
    print(f"  output  : {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
