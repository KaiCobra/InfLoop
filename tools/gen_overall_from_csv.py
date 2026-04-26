#!/usr/bin/env python3
"""從 outputs/eval_pnp_official 下每個方法的 result.csv 生成 overall.txt（按類別平均）。"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TITLE = "全域評估摘要"

CATEGORY_NAMES = {
    0: "0_random_140",
    1: "1_change_object_80",
    2: "2_add_object_80",
    3: "3_delete_object_80",
    4: "4_change_attribute_content_40",
    5: "5_change_attribute_pose_40",
    6: "6_change_attribute_color_40",
    7: "7_change_attribute_material_40",
    8: "8_change_background_80",
    9: "9_change_style_80",
}

# (display, csv_suffix, decimals)
METRICS = [
    ("PSNR",       "psnr_unedit_part",                      4),
    ("SSIM",       "ssim_unedit_part",                      4),
    ("LPIPS",      "lpips_unedit_part",                     4),
    ("StructDist", "structure_distance",                     6),
    ("CLIPw",      "clip_similarity_source_image",           4),
    ("CLIPe",      "clip_similarity_target_image_edit_part", 4),
]


def file_id_to_category(fid: str) -> int:
    return int(str(fid).zfill(12)[0])


def find_col(headers: List[str], suffix: str) -> Optional[str]:
    for h in headers:
        if h.endswith(suffix):
            return h
    return None


def safe_float(v: str) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return None


def process_csv(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    # 找各 metric 欄位名
    metric_cols: Dict[str, Optional[str]] = {}
    for display, suffix, _ in METRICS:
        metric_cols[display] = find_col(headers, suffix)

    # 依類別分組收集數值
    cat_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_values: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        fid = row.get("file_id", "")
        cat = file_id_to_category(fid)
        for display, _, _ in METRICS:
            col = metric_cols[display]
            if col is None:
                continue
            val = safe_float(row.get(col, ""))
            if val is not None:
                cat_values[cat][display].append(val)
                all_values[display].append(val)

    # 計算平均
    def mean_or_none(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    cat_stats: Dict[int, Dict[str, Optional[float]]] = {}
    for cat_id in sorted(CATEGORY_NAMES.keys()):
        cat_stats[cat_id] = {
            display: mean_or_none(cat_values[cat_id][display])
            for display, _, _ in METRICS
        }

    # Overall = 10 個類別平均的平均（每個類別權重相同）
    overall = {}
    for display, _, _ in METRICS:
        cat_means = [
            cat_stats[cat_id][display]
            for cat_id in sorted(CATEGORY_NAMES.keys())
            if cat_stats[cat_id][display] is not None
        ]
        overall[display] = mean_or_none(cat_means)

    # 每個 metric 各類別的有效筆數
    cat_counts: Dict[str, List[int]] = {}
    for display, _, _ in METRICS:
        cat_counts[display] = [
            len(cat_values[cat_id][display])
            for cat_id in sorted(CATEGORY_NAMES.keys())
        ]

    return render_table(cat_stats, overall, cat_counts)


def fmt(val: Optional[float], decimals: int) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def render_table(
    cat_stats: Dict[int, Dict[str, Optional[float]]],
    overall: Dict[str, Optional[float]],
    cat_counts: Optional[Dict[str, List[int]]] = None,
) -> str:
    cat_width = max(len("Category"), len("Overall"),
                    *(len(CATEGORY_NAMES[k]) for k in cat_stats)) + 2

    header = (
        f"{'Category':<{cat_width}}"
        f"{'PSNR':>8}  "
        f"{'SSIM':>7}  "
        f"{'LPIPS':>7}  "
        f"{'StructDist':>11}  "
        f"{'CLIPw':>6}  "
        f"{'CLIPe':>6}"
    )
    border = "=" * len(header)
    sep = "─" * len(header)

    lines = [border, TITLE.center(len(header)), border, header, sep]

    for cat_id in sorted(cat_stats.keys()):
        name = CATEGORY_NAMES[cat_id]
        s = cat_stats[cat_id]
        row = (
            f"{name:<{cat_width}}"
            f"{fmt(s.get('PSNR'), 4):>8}  "
            f"{fmt(s.get('SSIM'), 4):>7}  "
            f"{fmt(s.get('LPIPS'), 4):>7}  "
            f"{fmt(s.get('StructDist'), 6):>11}  "
            f"{fmt(s.get('CLIPw'), 4):>6}  "
            f"{fmt(s.get('CLIPe'), 4):>6}"
        )
        lines.append(row)

    lines.append(sep)
    overall_row = (
        f"{'Overall':<{cat_width}}"
        f"{fmt(overall.get('PSNR'), 4):>8}  "
        f"{fmt(overall.get('SSIM'), 4):>7}  "
        f"{fmt(overall.get('LPIPS'), 4):>7}  "
        f"{fmt(overall.get('StructDist'), 6):>11}  "
        f"{fmt(overall.get('CLIPw'), 4):>6}  "
        f"{fmt(overall.get('CLIPe'), 4):>6}"
    )
    lines.append(overall_row)
    lines.append(border)

    # 附加各 metric 分數向量與有效筆數
    if cat_counts:
        sorted_cats = sorted(cat_stats.keys())
        lines.append("")
        lines.append("各類別分數向量:")
        for display, _, decimals in METRICS:
            vals = [cat_stats[c].get(display) for c in sorted_cats]
            val_str = "[" + " ".join(fmt(v, decimals) for v in vals) + "]"
            lines.append(f"  {display:<12} {val_str}")

        lines.append("")
        lines.append("有效筆數 (各類別):")
        for display, _, _ in METRICS:
            counts = cat_counts.get(display, [])
            total = sum(counts)
            count_str = "[" + "  ".join(f"{c:>3}" for c in counts) + "]"
            lines.append(f"  {display:<12} {count_str}  total={total}")

    return "\n".join(lines) + "\n"


def main():
    root = Path("outputs/eval_pnp_official")
    csv_files = sorted(root.rglob("result.csv"))
    if not csv_files:
        print("找不到任何 result.csv", file=sys.stderr)
        return 1

    for csv_path in csv_files:
        print(f"處理: {csv_path}")
        table = process_csv(csv_path)
        out_path = csv_path.with_name("overall.txt")
        out_path.write_text(table, encoding="utf-8")
        print(f"  → {out_path}")
        print(table)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
