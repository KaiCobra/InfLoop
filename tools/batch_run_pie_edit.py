#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_pie_edit.py — PIE-Bench 批量 P2P-Edit（模型只載入一次）

支援兩種資料集格式：
A) PIE-Bench_v1 官方格式（--bench_dir 指向含 mapping_file.json 的目錄）：
   - mapping_file.json 內有所有 task 的 prompt / editing info
   - annotation_images/<image_path> 存放 source 圖片
B) extracted_pie_bench 扁平格式（舊版，向下相容）：
   - <bench_dir>/<category>/<task_id>/meta.json + image.jpg

流程：
1. 讀取 original_prompt / editing_prompt，清除 []。
2. 從差異詞彙建立 focus words。
3. 每個 task 使用 source image + prompt 跑 run_one_case。
4. 模型僅初始化一次。
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

from tools.run_p2p_edit import (  # noqa: E402
    add_common_arguments,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from tools.run_pie_edit import run_one_case  # noqa: E402
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


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

    Returns:
        source_focus_str, target_focus_str, source_focus_words, target_focus_words
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


def collect_mask_stats(mask_path: str) -> dict:
    """讀取 mask.png（若存在）並回傳白/黑比例。"""
    stats = {
        "mask_path": mask_path,
        "mask_exists": False,
        "white_ratio": None,
        "black_ratio": None,
        "white_percent": None,
        "black_percent": None,
    }
    if not os.path.exists(mask_path):
        return stats

    try:
        from PIL import Image
        import numpy as np

        mask = np.array(Image.open(mask_path).convert("L"))
        white_ratio = float((mask >= 128).mean())
        black_ratio = float(1.0 - white_ratio)
        stats.update({
            "mask_exists": True,
            "white_ratio": round(white_ratio, 6),
            "black_ratio": round(black_ratio, 6),
            "white_percent": round(white_ratio * 100.0, 3),
            "black_percent": round(black_ratio * 100.0, 3),
        })
    except Exception as exc:
        stats["mask_error"] = str(exc)
    return stats


class TeeStream:
    """同時寫入終端與檔案，方便保存每個 case 的 print log。"""

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
        # torch._dynamo / fx 會檢查 sys.stdout.isatty()
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        # 其餘屬性（例如 encoding）透過原始 stream 轉發
        return getattr(self.stream, name)


def write_task_info(save_dir: str, payload: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    info_path = os.path.join(save_dir, "task_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIE-Bench 批量 P2P-Edit（模型只載入一次）"
    )
    add_common_arguments(parser)

    # 批量設定
    parser.add_argument("--bench_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--max_per_cat", type=int, default=-1)
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])

    # P2P-Edit 設定（維持 infer_p2p_edit.sh 常用值）
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

    # 以下參數需存在，供 run_one_case 直接讀取
    parser.add_argument("--use_pie_mask", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--pie_mask_attn_fallback", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mask_expand_percent", type=float, default=0.0)
    parser.add_argument("--use_attn_cache", type=int, default=0, choices=[0, 1])
    parser.add_argument("--attn_cache_phase", type=str, default="both", choices=["phase17", "phase2", "both"])
    parser.add_argument("--attn_cache_max_scale", type=int, default=6)
    parser.add_argument("--attn_cache_align_mode", type=str, default="full_p2p", choices=["blended", "full_p2p"])
    parser.add_argument("--phase17_fallback_replace_scales", type=int, default=4,
                        help="Single-focus fallback時，Phase 1.7 以 source gen token 替換前幾個 scale（0=停用）。預設：4")
    parser.add_argument("--use_normalized_attn", type=int, default=0, choices=[0, 1],
                        help="使用 z-score normalized threshold 取代固定 percentile（0=停用，1=啟用）")
    parser.add_argument("--use_last_scale_mask", type=int, default=0, choices=[0, 1],
                        help="僅從最後一個 scale 提取 attention mask，再向前逐步推導各 scale（0=停用，1=啟用）")
    parser.add_argument("--last_scale_majority_threshold", type=float, default=0.5,
                        help="Last-scale mask 向前推導時的多數投票閾值（預設 0.5 = 50%%）")
    parser.add_argument("--use_dynamic_threshold", type=int, default=0, choices=[0, 1],
                        help="使用 reference mask 引導的二分法搜尋 attention threshold（0=停用，1=啟用）")
    parser.add_argument("--dynamic_threshold_iters", type=int, default=20,
                        help="二分法最大迭代次數（預設 20）")
    parser.add_argument("--threshold_method", type=int, default=1, choices=list(range(1, 14)),
                        help='閾值方法：1=固定percentile 2=dynamic ternary 3=Otsu 4=FFT+Otsu '
                             '5=SpectralEnergy 6=EdgeCoherence 7=GMM 8=Composite '
                             '9=IPR 10=Entropy 11=BlockConsensus 12=Kneedle '
                             '13=MetaAdaptive。預設：1')

    args = parser.parse_args()

    # add_common_arguments 內 cfg 為字串，這裡保持與 run_p2p_edit 一致
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Batch PIE P2P-Edit")
    print("=" * 80)
    print(f"bench_dir         : {args.bench_dir}")
    print(f"output_dir        : {args.output_dir}")
    print(f"skip_existing     : {bool(args.skip_existing)}")
    print(f"max_per_cat       : {args.max_per_cat if args.max_per_cat > 0 else 'all'}")
    print(f"full_replace      : {args.num_full_replace_scales}")
    print(f"attn_percentile   : {args.attn_threshold_percentile}")
    print(f"save_attn_vis     : {bool(args.save_attn_vis)}")
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

        # Group tasks by top-level category from image_path
        # image_path examples: "0_random_140/000000000000.jpg"
        #                      "1_change_object_80/1_artificial/1_animal/111000000000.jpg"
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
        # Sort tasks within each category
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
                    "mask_path": None,  # mask encoded in mapping, not a file
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
                case_dir = os.path.join(cat_dir, cid)
                meta_path = os.path.join(case_dir, "meta.json")
                img_path = os.path.join(case_dir, "image.jpg")
                mask_path = os.path.join(case_dir, "mask.png")
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
                    "case_dir": case_dir,
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
            source_prompt_raw = case["source_prompt_raw"]
            target_prompt_raw = case["target_prompt_raw"]
            case_dir = case["case_dir"]

            if not os.path.exists(img_path):
                print(f"  [{idx + 1}/{len(cases)}] {case_id}  ✗ missing image: {img_path}")
                total_err += 1
                continue

            save_dir = os.path.join(args.output_dir, cat_name, case_id)
            target_out = os.path.join(save_dir, "target.jpg")
            if args.skip_existing and os.path.exists(target_out) and os.path.getsize(target_out) > 0:
                total_skip += 1
                print(f"  [{idx + 1}/{len(cases)}] {case_id}  ↓ skip existing")
                mask_stats = collect_mask_stats(mask_path) if mask_path else {"mask_exists": False}
                write_task_info(save_dir, {
                    "category": cat_name,
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "skipped_existing",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "image_path": img_path,
                    "target_path": target_out,
                    "log_path": os.path.join(save_dir, "case.log"),
                    "mask_stats": mask_stats,
                })
                continue

            source_prompt_raw = case["source_prompt_raw"]
            target_prompt_raw = case["target_prompt_raw"]

            # Step 1: 清理兩側 prompt 的 []
            source_prompt = remove_prompt_brackets(source_prompt_raw)
            target_prompt = remove_prompt_brackets(target_prompt_raw)

            # Step 2: 從差異詞生成 focus words（join with space）
            src_focus_str, tgt_focus_str, src_focus_words, tgt_focus_words = derive_focus_words_from_prompt_diff(
                source_prompt, target_prompt
            )
            mask_stats = collect_mask_stats(mask_path) if mask_path else {"mask_exists": False}

            print(f"\n  [{idx + 1}/{len(cases)}] {case_id}")
            print(f"    src_prompt : {source_prompt}")
            print(f"    tgt_prompt : {target_prompt}")
            print(f"    src_focus  : {src_focus_str}")
            print(f"    tgt_focus  : {tgt_focus_str}")
            if mask_stats.get("mask_exists"):
                print(
                    f"    mask_ratio : white={mask_stats['white_percent']}% "
                    f"black={mask_stats['black_percent']}%"
                )

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
                    case_log.write(f"mask_stats: {json.dumps(mask_stats, ensure_ascii=False)}\n")
                    case_log.write("-" * 80 + "\n")

                    tee_out = TeeStream(sys.stdout, case_log)
                    tee_err = TeeStream(sys.stderr, case_log)
                    # Extract ref_mask_rle from mapping_file.json for dynamic threshold
                    _ref_mask_rle = None
                    if int(args.use_dynamic_threshold) and use_v1_format:
                        _ref_mask_rle = case.get("meta_raw", {}).get("mask", None)

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
                            mask_path=mask_path if mask_path and int(args.use_pie_mask) in (1, 2) else None,
                            blended_words=None,
                            case_source_dir=case_dir,
                            ref_mask_rle=_ref_mask_rle,
                        )
                elapsed = time.time() - t_start
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "timing.json"), "w", encoding="utf-8") as tf:
                    json.dump({"inference_sec": round(elapsed, 3)}, tf)
                write_task_info(save_dir, {
                    "category": cat_name,
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "success",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_sec": round(elapsed, 3),
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "source_prompt_raw": source_prompt_raw,
                    "target_prompt_raw": target_prompt_raw,
                    "source_prompt_clean": source_prompt,
                    "target_prompt_clean": target_prompt,
                    "source_focus_words": src_focus_words,
                    "target_focus_words": tgt_focus_words,
                    "source_focus_string": src_focus_str,
                    "target_focus_string": tgt_focus_str,
                    "mask_stats": mask_stats,
                    "target_path": target_out,
                    "timing_path": os.path.join(save_dir, "timing.json"),
                    "log_path": case_log_path,
                })
                total_done += 1
                print(f"    ✓ done ({elapsed:.1f}s) -> {target_out}")
            except Exception as exc:
                total_err += 1
                print(f"    ✗ failed: {exc}")
                traceback.print_exc()
                elapsed = time.time() - t_start
                with open(os.path.join(save_dir, "timing.json"), "w", encoding="utf-8") as tf:
                    json.dump({"inference_sec": round(elapsed, 3)}, tf)
                write_task_info(save_dir, {
                    "category": cat_name,
                    "case_id": case_id,
                    "case_dir": case_dir,
                    "status": "failed",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_sec": round(elapsed, 3),
                    "error": str(exc),
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "source_prompt_raw": source_prompt_raw,
                    "target_prompt_raw": target_prompt_raw,
                    "source_prompt_clean": source_prompt,
                    "target_prompt_clean": target_prompt,
                    "source_focus_words": src_focus_words,
                    "target_focus_words": tgt_focus_words,
                    "source_focus_string": src_focus_str,
                    "target_focus_string": tgt_focus_str,
                    "mask_stats": mask_stats,
                    "target_path": target_out,
                    "timing_path": os.path.join(save_dir, "timing.json"),
                    "log_path": case_log_path,
                })
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Batch PIE P2P-Edit complete")
    print(f"  success : {total_done}")
    print(f"  skipped : {total_skip}")
    print(f"  errors  : {total_err}")
    print(f"  output  : {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
