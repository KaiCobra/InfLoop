#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pie_results.py — PIE-Bench 結果定量評估腳本（官方 PnPInversion 指標）

使用 evaluation/matrics_calculator.py (MetricsCalculator) 計算全部 8 個官方指標：

  背景保留（Background Preservation）：
    • psnr_unedit_part           ↑  torchmetrics PSNR (data_range=1.0)
    • ssim_unedit_part           ↑  torchmetrics SSIM (data_range=1.0)
    • lpips_unedit_part          ↓  torchmetrics LPIPS (SqueezeNet)
    • mse_unedit_part            ↓  torchmetrics MSE

  結構保留（Structure Preservation）：
    • structure_distance         ↓  DINO ViT-B/8 key self-similarity MSE

  編輯品質（Edit Quality）：
    • clip_similarity_source_image       CLIP(source_img, source_prompt)
    • clip_similarity_target_image     ↑ CLIP(target_img, target_prompt)     — CLIPw
    • clip_similarity_target_image_edit_part ↑ CLIP(target_img*mask, target_prompt) — CLIPe

  Mask 使用 mapping_file.json 的 RLE 格式，與官方一致。

使用方式：
  bash scripts/eval_pie.sh
  或
  python3 tools/eval_pie_results.py --bench_dir <path> --result_dir <path> [options]
"""

import os
import re
import sys
import csv
import json
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# 靜音 FutureWarning
warnings.filterwarnings('ignore')

# ── 確保工作目錄在 sys.path 中 ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.matrics_calculator import MetricsCalculator
from evaluation.evaluate import mask_decode, calculate_metric


# ============================================================
# HPS v2 / ImageReward 載入
# ============================================================

def _load_hps():
    """載入 HPSv2 module（lazy，首次使用時呼叫）。"""
    import hpsv2
    return hpsv2


def _load_image_reward(device: str = 'cuda'):
    """載入 ImageReward model（lazy，首次使用時呼叫）。"""
    import ImageReward as RM
    model = RM.load('ImageReward-v1.0', device=device)
    model.eval()
    return model


# ============================================================
# 官方 8 項指標
# ============================================================

OFFICIAL_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]

# 額外偏好指標（Human Preference Score v2、ImageReward）
EXTRA_METRICS = [
    "hps_v2",
    "image_reward",
]


# ============================================================
# mapping_file.json 載入
# ============================================================

def _load_mapping_file(bench_dir: str) -> Dict[str, Dict]:
    """
    讀取 mapping_file.json，建立 {case_id: entry} 索引。
    """
    mapping_path = os.path.join(bench_dir, 'mapping_file.json')
    if not os.path.isfile(mapping_path):
        print(f'[Error] mapping_file.json not found: {mapping_path}')
        sys.exit(1)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _group_by_category(mapping: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    將 mapping_file.json 按 category 分組。
    回傳 {category_name: {case_id: entry}}。
    """
    grouped: Dict[str, Dict[str, Dict]] = {}
    for case_id, entry in mapping.items():
        cat = entry['image_path'].split('/')[0]
        grouped.setdefault(cat, {})[case_id] = entry
    return grouped


# ============================================================
# 主評估迴圈
# ============================================================

def evaluate_all(
    bench_dir: str,
    result_dir: str,
    output_csv: str,
    summary_json: str,
    categories: List[str],
    max_per_cat: int,
    skip_missing: bool,
    no_structure_dist: bool,
    device_str: str,
    no_hps: bool = False,
    no_image_reward: bool = False,
) -> None:

    # ── 載入 mapping_file.json ──
    print('\n[Init] 載入 mapping_file.json...')
    mapping = _load_mapping_file(bench_dir)
    grouped = _group_by_category(mapping)
    src_image_dir = os.path.join(bench_dir, 'annotation_images')

    # ── 決定要計算的 metrics ──
    metrics = [m for m in OFFICIAL_METRICS]
    if no_structure_dist:
        metrics = [m for m in metrics if m != "structure_distance"]

    extra = []
    if not no_hps:
        extra.append("hps_v2")
    if not no_image_reward:
        extra.append("image_reward")

    # ── 初始化官方 MetricsCalculator ──
    print(f'[Init] 載入官方 MetricsCalculator (device={device_str})...')
    metrics_calc = MetricsCalculator(device_str)
    print('[Init] 所有模型載入完成')

    # ── 初始化額外偏好指標模型 ──
    hps_module = None
    ir_model = None
    if "hps_v2" in extra:
        print('[Init] 載入 HPSv2...')
        hps_module = _load_hps()
        print('[Init] HPSv2 載入完成')
    if "image_reward" in extra:
        print('[Init] 載入 ImageReward-v1.0...')
        ir_model = _load_image_reward(device_str)
        print('[Init] ImageReward 載入完成')

    # ── CSV 標頭 ──
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    csv_fields = ['category', 'case_id', 'editing_type_id'] + metrics + extra

    all_rows: List[Dict] = []
    cat_stats: Dict[str, List[Dict]] = {}

    print(f'\n[Eval] 開始評估  bench={bench_dir}  result={result_dir}\n')

    for cat_name in categories:
        if cat_name not in grouped:
            print(f'[Warning] category 不在 mapping_file.json 中，跳過：{cat_name}')
            continue

        cat_mapping = grouped[cat_name]
        cat_result = os.path.join(result_dir, cat_name)

        if not os.path.isdir(cat_result):
            print(f'[Warning] result category 不存在，跳過：{cat_result}')
            continue

        # 取得 result_dir 中實際有的 case_id，與 mapping 取交集
        result_case_ids = sorted(
            d for d in os.listdir(cat_result)
            if os.path.isdir(os.path.join(cat_result, d)) and d in cat_mapping
        )
        if max_per_cat > 0:
            result_case_ids = result_case_ids[:max_per_cat]

        print(f'{"─" * 60}')
        print(f'[Category] {cat_name}  ({len(result_case_ids)} 個案例)')
        print(f'{"─" * 60}')

        cat_rows: List[Dict] = []

        for idx, case_id in enumerate(tqdm(result_case_ids, desc=f'  {cat_name}')):
            entry = cat_mapping[case_id]
            result_case = os.path.join(cat_result, case_id)

            # ── 路徑 ──
            src_image_path = os.path.join(src_image_dir, entry['image_path'])
            target_path = os.path.join(result_case, 'target.jpg')

            # ── 檢查檔案 ──
            if not os.path.exists(src_image_path):
                print(f'  [{idx+1}] {case_id}  ⚠ source 不存在：{src_image_path}，跳過')
                continue
            if not os.path.exists(target_path):
                if skip_missing:
                    continue
                else:
                    print(f'  [{idx+1}] {case_id}  ✗ target.jpg 不存在')
                    continue
            if os.path.getsize(target_path) == 0:
                print(f'  [{idx+1}] {case_id}  ⚠ target.jpg 為空檔，跳過')
                continue

            # ── 讀取影像（PIL Image，與官方一致）──
            src_image = Image.open(src_image_path).convert('RGB')
            tgt_image = Image.open(target_path).convert('RGB')

            # 官方 size 處理邏輯
            if tgt_image.size[0] != tgt_image.size[1]:
                tgt_image = tgt_image.crop((
                    tgt_image.size[0] - 512, tgt_image.size[1] - 512,
                    tgt_image.size[0], tgt_image.size[1]
                ))
            if tgt_image.size != src_image.size:
                tgt_image = tgt_image.resize(src_image.size, Image.LANCZOS)

            # ── 解碼 mask（官方做法，含邊界修正）──
            mask = mask_decode(entry['mask'])  # [512, 512] float64
            mask_3ch = mask[:, :, np.newaxis].repeat([3], axis=2)  # [512, 512, 3]

            # ── Prompts（去除方括號）──
            original_prompt = entry.get('original_prompt', '').replace('[', '').replace(']', '')
            editing_prompt = entry.get('editing_prompt', '').replace('[', '').replace(']', '')

            # ── 計算所有指標 ──
            row = {
                'category': cat_name,
                'case_id': case_id,
                'editing_type_id': entry.get('editing_type_id', ''),
            }

            for metric_name in metrics:
                value = calculate_metric(
                    metrics_calc, metric_name,
                    src_image, tgt_image,
                    mask_3ch, mask_3ch,
                    original_prompt, editing_prompt,
                )
                # numpy array → float
                if hasattr(value, 'item'):
                    value = float(value.item())
                elif isinstance(value, np.ndarray):
                    value = float(value)
                elif value == "nan":
                    value = float('nan')

                row[metric_name] = value

            # ── 額外偏好指標 ──
            if hps_module is not None:
                scores = hps_module.score(tgt_image, editing_prompt, hps_version='v2.1')
                row['hps_v2'] = float(scores[0]) if isinstance(scores, list) else float(scores)

            if ir_model is not None:
                with __import__('torch').no_grad():
                    reward = ir_model.score(editing_prompt, tgt_image)
                row['image_reward'] = float(reward)

            all_rows.append(row)
            cat_rows.append(row)

        cat_stats[cat_name] = cat_rows
        _print_category_summary(cat_name, cat_rows, metrics, extra)

    # ── 儲存 CSV ──
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            write_row = {}
            for field in csv_fields:
                v = row.get(field, '')
                if isinstance(v, float) and np.isnan(v):
                    write_row[field] = 'nan'
                elif isinstance(v, float):
                    write_row[field] = round(v, 6)
                else:
                    write_row[field] = v
            writer.writerow(write_row)
    print(f'\n[Output] Per-case CSV → {output_csv}')

    # ── 儲存 Summary JSON ──
    summary = _build_summary(cat_stats, metrics, extra)
    os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[Output] Summary JSON → {summary_json}')

    # ── 顯示全域摘要 ──
    _print_global_summary(summary)


# ============================================================
# 摘要計算 / 顯示
# ============================================================

def _nanmean(values: list) -> Optional[float]:
    valid = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    return float(np.mean(valid)) if valid else None


def _build_summary(cat_stats: Dict[str, List[Dict]], metrics: List[str], extra: List[str] = []) -> Dict:
    all_metrics = metrics + extra
    summary = {}
    all_rows_flat = []

    for cat_name, rows in cat_stats.items():
        all_rows_flat.extend(rows)

        entry = {
            'n_cases': len(rows),
        }

        for m in all_metrics:
            entry[m + '_mean'] = _nanmean([r.get(m, float('nan')) for r in rows])

        # 相容 write_eval_overall_txt.py 的欄位名稱
        entry['psnr_mean'] = entry.get('psnr_unedit_part_mean')
        entry['ssim_mean'] = entry.get('ssim_unedit_part_mean')
        entry['lpips_mean'] = entry.get('lpips_unedit_part_mean')
        entry['mse_mean'] = entry.get('mse_unedit_part_mean')
        entry['structure_dist_mean'] = entry.get('structure_distance_mean')
        entry['clip_sim_whole_mean'] = entry.get('clip_similarity_target_image_mean')
        entry['clip_sim_edited_mean'] = entry.get('clip_similarity_target_image_edit_part_mean')
        # 偏好指標相容欄位
        entry['hps_v2_mean'] = entry.get('hps_v2_mean')
        entry['image_reward_mean'] = entry.get('image_reward_mean')

        summary[cat_name] = entry

    # 全域平均
    overall = {'n_cases': len(all_rows_flat)}
    for m in all_metrics:
        overall[m + '_mean'] = _nanmean([r.get(m, float('nan')) for r in all_rows_flat])

    overall['psnr_mean'] = overall.get('psnr_unedit_part_mean')
    overall['ssim_mean'] = overall.get('ssim_unedit_part_mean')
    overall['lpips_mean'] = overall.get('lpips_unedit_part_mean')
    overall['mse_mean'] = overall.get('mse_unedit_part_mean')
    overall['structure_dist_mean'] = overall.get('structure_distance_mean')
    overall['clip_sim_whole_mean'] = overall.get('clip_similarity_target_image_mean')
    overall['clip_sim_edited_mean'] = overall.get('clip_similarity_target_image_edit_part_mean')

    summary['__overall__'] = overall
    return summary


def _print_category_summary(cat_name: str, rows: List[Dict], metrics: List[str], extra: List[str] = []) -> None:
    if not rows:
        return
    all_m = metrics + extra
    values = {}
    for m in all_m:
        values[m] = _nanmean([r.get(m, float('nan')) for r in rows])

    def fmt(v, d=4): return f'{v:.{d}f}' if v is not None else '  N/A  '

    print(f'\n  ┌── {cat_name} 小計 ({len(rows)} cases) ──')
    print(f'  │  PSNR={fmt(values.get("psnr_unedit_part"))}  '
          f'SSIM={fmt(values.get("ssim_unedit_part"))}  '
          f'LPIPS={fmt(values.get("lpips_unedit_part"))}  '
          f'MSE={fmt(values.get("mse_unedit_part"))}')
    print(f'  │  StructDist={fmt(values.get("structure_distance"), 6)}  '
          f'CLIPsrc={fmt(values.get("clip_similarity_source_image"))}  '
          f'CLIPw={fmt(values.get("clip_similarity_target_image"))}  '
          f'CLIPe={fmt(values.get("clip_similarity_target_image_edit_part"))}')
    extra_parts = []
    if 'hps_v2' in values and values['hps_v2'] is not None:
        extra_parts.append(f'HPSv2={fmt(values["hps_v2"])}')
    if 'image_reward' in values and values['image_reward'] is not None:
        extra_parts.append(f'ImageReward={fmt(values["image_reward"])}')
    if extra_parts:
        print(f'  │  {"  ".join(extra_parts)}')
    print(f'  └{"─" * 60}')


def _nanmean_key(d: Dict, key: str) -> Optional[float]:
    v = d.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def _print_global_summary(summary: Dict) -> None:
    ov = summary.get('__overall__', {})
    def fmt(v, d=4): return f'{v:.{d}f}' if v is not None else ' N/A '

    has_hps = ov.get('hps_v2_mean') is not None
    has_ir = ov.get('image_reward_mean') is not None

    W = 110
    if has_hps:
        W += 9
    if has_ir:
        W += 13

    print('\n' + '=' * W)
    print(f'{"全域評估摘要（官方 PnPInversion 指標 + 偏好指標）":^{W}}')
    print('=' * W)
    header = (f'{"Category":<38} {"PSNR":>7} {"SSIM":>7} {"LPIPS":>7} {"MSE":>8} '
              f'{"StructDist":>11} {"CLIPsrc":>8} {"CLIPw":>7} {"CLIPe":>7}')
    if has_hps:
        header += f' {"HPSv2":>8}'
    if has_ir:
        header += f' {"ImgReward":>12}'
    print(header)
    print('─' * W)

    for cat, s in summary.items():
        if cat == '__overall__':
            continue
        clip_src = _nanmean_key(s, 'clip_similarity_source_image_mean')
        line = (f'{cat:<38} '
                f'{fmt(s.get("psnr_mean")):>7} '
                f'{fmt(s.get("ssim_mean")):>7} '
                f'{fmt(s.get("lpips_mean")):>7} '
                f'{fmt(s.get("mse_mean")):>8} '
                f'{fmt(s.get("structure_dist_mean"), 6):>11} '
                f'{fmt(clip_src):>8} '
                f'{fmt(s.get("clip_sim_whole_mean")):>7} '
                f'{fmt(s.get("clip_sim_edited_mean")):>7}')
        if has_hps:
            line += f' {fmt(s.get("hps_v2_mean")):>8}'
        if has_ir:
            line += f' {fmt(s.get("image_reward_mean")):>12}'
        print(line)

    print('─' * W)
    ov_clip_src = _nanmean_key(ov, 'clip_similarity_source_image_mean')
    ov_line = (f'{"Overall":<38} '
               f'{fmt(ov.get("psnr_mean")):>7} '
               f'{fmt(ov.get("ssim_mean")):>7} '
               f'{fmt(ov.get("lpips_mean")):>7} '
               f'{fmt(ov.get("mse_mean")):>8} '
               f'{fmt(ov.get("structure_dist_mean"), 6):>11} '
               f'{fmt(ov_clip_src):>8} '
               f'{fmt(ov.get("clip_sim_whole_mean")):>7} '
               f'{fmt(ov.get("clip_sim_edited_mean")):>7}')
    if has_hps:
        ov_line += f' {fmt(ov.get("hps_v2_mean")):>8}'
    if has_ir:
        ov_line += f' {fmt(ov.get("image_reward_mean")):>12}'
    print(ov_line)
    print('=' * W)
    print(f'  總案例數：{ov.get("n_cases", 0)}')
    print(f'  指標說明：PSNR/SSIM ↑（背景保留），LPIPS/MSE ↓（背景），StructDist ↓（結構保留）')
    print(f'  CLIP 分數：CLIPsrc=CLIP(source,src_prompt)  CLIPw=CLIP(target,tgt_prompt)  CLIPe=CLIP(target*mask,tgt_prompt)')
    print(f'  偏好指標：HPSv2 ↑（Human Preference Score v2.1）  ImageReward ↑（BLIP-based reward）')
    print(f'  注意：CLIP 分數範圍 [0,100]（torchmetrics CLIPScore）')
    print('=' * W + '\n')


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PIE-Bench 結果定量評估（官方 PnPInversion 8 項指標）'
    )

    # ── 路徑設定 ──
    parser.add_argument('--bench_dir', type=str, required=True,
                        help='PIE-Bench 根目錄（含 mapping_file.json 與 annotation_images/）')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='batch_run_pie_edit.py 輸出根目錄')
    parser.add_argument('--output_csv', type=str,
                        default='./outputs/eval_pie/per_case.csv',
                        help='Per-case 結果 CSV 路徑')
    parser.add_argument('--summary_json', type=str,
                        default='./outputs/eval_pie/summary.json',
                        help='Per-category 摘要 JSON 路徑')

    # ── 過濾設定 ──
    parser.add_argument('--categories', type=str, default='',
                        help='只評估指定 category（逗號分隔），預設全部')
    parser.add_argument('--max_per_cat', type=int, default=-1,
                        help='每個 category 最多幾個案例（-1 = 全部）')
    parser.add_argument('--skip_missing', type=int, default=1, choices=[0, 1],
                        help='若 target.jpg 不存在則跳過（預設：1）')

    # ── 模型設定 ──
    parser.add_argument('--no_structure_dist', action='store_true',
                        help='跳過 Structure Distance 計算')
    parser.add_argument('--no_hps', action='store_true',
                        help='跳過 HPSv2 計算')
    parser.add_argument('--no_image_reward', action='store_true',
                        help='跳過 ImageReward 計算')

    # ── 裝置 ──
    parser.add_argument('--device', type=str, default='',
                        help='cuda / cpu（預設：自動偵測）')

    args = parser.parse_args()

    import torch
    device_str = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Config] device={device_str}')

    # ── 決定要評估的 categories ──
    bench_dir = args.bench_dir
    mapping = _load_mapping_file(bench_dir)
    grouped = _group_by_category(mapping)
    all_categories = sorted(grouped.keys())

    if args.categories.strip():
        categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    else:
        categories = all_categories

    print(f'[Config] {len(categories)} 個 category，max_per_cat={args.max_per_cat}')
    print(f'[Config] 使用官方 PnPInversion MetricsCalculator')
    print(f'[Config] LPIPS=SqueezeNet  CLIP=openai/clip-vit-large-patch14  DINO=ViT-B/8')
    print(f'[Config] StructureDist={"OFF" if args.no_structure_dist else "ON"}')
    print(f'[Config] HPSv2={"OFF" if args.no_hps else "ON"}  ImageReward={"OFF" if args.no_image_reward else "ON"}')

    # ── 建立輸出目錄 ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    evaluate_all(
        bench_dir         = bench_dir,
        result_dir        = args.result_dir,
        output_csv        = args.output_csv,
        summary_json      = args.summary_json,
        categories        = categories,
        max_per_cat       = args.max_per_cat,
        skip_missing      = bool(args.skip_missing),
        no_structure_dist = args.no_structure_dist,
        device_str        = device_str,
        no_hps            = args.no_hps,
        no_image_reward   = args.no_image_reward,
    )


if __name__ == '__main__':
    main()
