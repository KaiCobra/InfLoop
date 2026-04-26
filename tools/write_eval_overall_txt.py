#!/usr/bin/env python3
"""Generate overall.txt beside every summary.json under outputs/evals."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TITLE = "全域評估摘要"
OVERALL_KEY = "__overall__"

METRICS: List[Tuple[str, str, int, str, str]] = [
    ("PSNR", "psnr_mean", 4, "n_with_bg", ""),
    ("SSIM", "ssim_mean", 4, "n_with_bg", ""),
    ("LPIPS", "lpips_mean", 4, "n_with_bg", ""),
    ("StructDist", "structure_dist_mean", 6, "n_cases", ""),
    ("CLIPw", "clip_sim_whole_mean", 4, "n_cases", ""),
    ("CLIPe", "clip_sim_edited_mean", 4, "n_cases", ""),
    ("HPSv2", "hps_v2_mean", 4, "n_cases", ""),
    ("ImgReward", "image_reward_mean", 4, "n_cases", ""),
    ("Speed", "inference_sec_mean", 1, "n_cases", "s"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write overall.txt beside every summary.json under an eval root."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/outputs_loop_exp/evals"),
        help="Directory to scan for summary.json files.",
    )
    parser.add_argument(
        "--pattern",
        default="summary.json",
        help="Filename pattern to scan for. Default: summary.json",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_categories(summary: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
    return [key for key in summary.keys() if key != OVERALL_KEY]


def is_missing(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def weighted_mean(
    summary: Dict[str, Dict[str, Optional[float]]],
    categories: Iterable[str],
    field: str,
    weight_field: str,
) -> Optional[float]:
    total = 0.0
    total_weight = 0.0
    for category in categories:
        item = summary[category]
        value = item.get(field)
        if is_missing(value):
            continue
        weight = item.get(weight_field)
        if weight is None:
            weight = item.get("n_cases", 0)
        if not weight:
            continue
        total += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight == 0.0:
        return None
    return total / total_weight


def compute_overall(
    summary: Dict[str, Dict[str, Optional[float]]], categories: List[str]
) -> Dict[str, Optional[float]]:
    overall: Dict[str, Optional[float]] = {"n_cases": 0}
    overall["n_cases"] = sum(int(summary[category].get("n_cases", 0) or 0) for category in categories)
    for _, field, _, weight_field, _ in METRICS:
        overall[field] = weighted_mean(summary, categories, field, weight_field)
    return overall


def format_value(value: Optional[float], decimals: int, suffix: str = "") -> str:
    if is_missing(value):
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{decimals}f}{suffix}"


def build_header(category_width: int) -> str:
    return (
        f"{'Category':<{category_width}}"
        f"{'PSNR':>8}  "
        f"{'SSIM':>7}  "
        f"{'LPIPS':>7}  "
        f"{'StructDist':>11}  "
        f"{'CLIPw':>6}  "
        f"{'CLIPe':>6}  "
        f"{'HPSv2':>7}  "
        f"{'ImgReward':>10}  "
        f"{'Speed':>7}"
    )


def build_row(name: str, item: Dict[str, Optional[float]], category_width: int) -> str:
    values = {
        field: format_value(item.get(field), decimals, suffix)
        for _, field, decimals, _, suffix in METRICS
    }
    return (
        f"{name:<{category_width}}"
        f"{values['psnr_mean']:>8}  "
        f"{values['ssim_mean']:>7}  "
        f"{values['lpips_mean']:>7}  "
        f"{values['structure_dist_mean']:>11}  "
        f"{values['clip_sim_whole_mean']:>6}  "
        f"{values['clip_sim_edited_mean']:>6}  "
        f"{values['hps_v2_mean']:>7}  "
        f"{values['image_reward_mean']:>10}  "
        f"{values['inference_sec_mean']:>7}"
    )


def render_table(summary: Dict[str, Dict[str, Optional[float]]]) -> str:
    categories = ordered_categories(summary)
    overall = compute_overall(summary, categories)
    category_width = max(len("Category"), len("Overall"), *(len(name) for name in categories)) + 2
    header = build_header(category_width)
    border = "=" * len(header)
    separator = "\u2500" * len(header)

    lines = [
        border,
        TITLE.center(len(header)),
        border,
        header,
        separator,
    ]
    for category in categories:
        lines.append(build_row(category, summary[category], category_width))
    lines.extend(
        [
            separator,
            build_row("Overall", overall, category_width),
            border,
        ]
    )
    return "\n".join(lines) + "\n"


def write_overall(summary_path: Path) -> Path:
    summary = load_summary(summary_path)
    output_path = summary_path.with_name("overall.txt")
    output_path.write_text(render_table(summary), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    summary_paths = sorted(args.root.rglob(args.pattern))
    if not summary_paths:
        raise SystemExit(f"No {args.pattern} found under {args.root}")

    for summary_path in summary_paths:
        output_path = write_overall(summary_path)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
