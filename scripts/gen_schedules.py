#!/usr/bin/env python3
"""
Generate rollback schedule experiments for batch testing.

Outputs a JSON file compatible with run_loop_batch.py / infer_loop_batch.sh.

Simple Usage:
    # 1. 先看看會產幾組
    python scripts/gen_schedules.py --scale_min 3 --scale_max 8 --times 1 2 3 --multi_rule --dry_run

    # 2. 確認數量 OK，產出 JSON
    python scripts/gen_schedules.py --scale_min 3 --scale_max 8 --times 1 2 3 --multi_rule -o scripts/schedules_sweep.json

    # 3. 跑批次推論（模型只 load 一次）
    SCHEDULE_FILE=scripts/schedules_sweep.json bash scripts/infer_loop_batch.sh


Usage:
    # 產生預設的全掃描實驗 (會很多!)
    python scripts/gen_schedules.py -o scripts/schedules_sweep.json

    # 只掃 scale 3~8, rollback gap 1~2, times 1~3
    python scripts/gen_schedules.py \
        --scale_min 3 --scale_max 8 \
        --gap_min 1 --gap_max 2 \
        --times 1 2 3 \
        -o scripts/schedules_custom.json

    # 加入雙規則組合 (scale pairs)
    python scripts/gen_schedules.py \
        --scale_min 3 --scale_max 8 \
        --gap_min 1 --gap_max 2 \
        --times 1 2 3 \
        --multi_rule \
        -o scripts/schedules_with_multi.json

    # 看有幾個實驗但不寫檔
    python scripts/gen_schedules.py --dry_run

Scale schedule reference (1M, ratio=1.0, 13 scales, index 0~12):
    idx  t   h×w        resolution (×16)
    ───  ──  ─────      ────────────────
     0    1   1× 1        16×  16
     1    2   2× 2        32×  32
     2    3   4× 4        64×  64
     3    4   6× 6        96×  96
     4    5   8× 8       128× 128
     5    6  12×12       192× 192
     6    7  16×16       256× 256
     7    9  20×20       320× 320
     8   11  24×24       384× 384
     9   13  32×32       512× 512
    10   15  40×40       640× 640
    11   17  48×48       768× 768
    12   21  64×64      1024×1024
"""

import argparse
import json
import itertools
from typing import List, Dict, Tuple

# ── scale info (1M, ratio 1.0) for reference in naming ──────────────────
SCALE_HW_1M = [
    (1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16),
    (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64),
]
NUM_SCALES = len(SCALE_HW_1M)  # 13


def gen_single_rule_experiments(
    scale_min: int,
    scale_max: int,
    gap_min: int,
    gap_max: int,
    times_list: List[int],
) -> List[dict]:
    """Generate all single-rule experiments within the given ranges.

    Args:
        scale_min/max: trigger scale index range (inclusive)
        gap_min/max:   rollback distance range (scale - rollback_to)
        times_list:    list of retry counts to try
    """
    exps = []
    for scale in range(scale_min, min(scale_max, NUM_SCALES) + 1):
        for gap in range(gap_min, gap_max + 1):
            rb_to = scale - gap
            if rb_to < 0:
                continue
            for times in times_list:
                rule = {"scale": scale, "rollback_to": rb_to, "times": times}
                name = f"s{scale}rb{rb_to}x{times}"
                exps.append({"name": name, "rules": [rule]})
    return exps


def gen_dual_rule_experiments(
    scale_min: int,
    scale_max: int,
    gap_min: int,
    gap_max: int,
    times_list: List[int],
) -> List[dict]:
    """Generate dual-rule experiments: two non-overlapping rollback rules.

    Rules are non-overlapping if rule_A.scale <= rule_B.rollback_to
    (i.e. the first rollback finishes before the second one starts).
    """
    # First build all valid single rules
    single_rules = []
    for scale in range(scale_min, min(scale_max, NUM_SCALES) + 1):
        for gap in range(gap_min, gap_max + 1):
            rb_to = scale - gap
            if rb_to < 0:
                continue
            single_rules.append((scale, rb_to))

    exps = []
    for (s1, rb1), (s2, rb2) in itertools.combinations(single_rules, 2):
        # Ensure non-overlapping: first rule's scale <= second rule's rollback_to
        a_s, a_rb, b_s, b_rb = (s1, rb1, s2, rb2) if s1 < s2 else (s2, rb2, s1, rb1)
        if a_s > b_rb:
            continue  # overlapping ranges
        # Use same times for both rules for simplicity; sweep over times_list
        for times in times_list:
            rules = [
                {"scale": a_s, "rollback_to": a_rb, "times": times},
                {"scale": b_s, "rollback_to": b_rb, "times": times},
            ]
            name = f"s{a_s}rb{a_rb}_s{b_s}rb{b_rb}_x{times}"
            exps.append({"name": name, "rules": rules})
    return exps


def gen_triple_rule_experiments(
    scale_min: int,
    scale_max: int,
    gap_min: int,
    gap_max: int,
    times_list: List[int],
) -> List[dict]:
    """Generate triple-rule experiments: three non-overlapping rollback rules."""
    single_rules = []
    for scale in range(scale_min, min(scale_max, NUM_SCALES) + 1):
        for gap in range(gap_min, gap_max + 1):
            rb_to = scale - gap
            if rb_to < 0:
                continue
            single_rules.append((scale, rb_to))

    exps = []
    for combo in itertools.combinations(single_rules, 3):
        # Sort by scale
        sorted_combo = sorted(combo, key=lambda x: x[0])
        # Check non-overlapping: each rule's scale <= next rule's rollback_to
        valid = True
        for i in range(len(sorted_combo) - 1):
            if sorted_combo[i][0] > sorted_combo[i + 1][1]:
                valid = False
                break
        if not valid:
            continue
        for times in times_list:
            rules = [
                {"scale": s, "rollback_to": rb, "times": times}
                for s, rb in sorted_combo
            ]
            name = '_'.join(f"s{s}rb{rb}" for s, rb in sorted_combo) + f"_x{times}"
            exps.append({"name": name, "rules": rules})
    return exps


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollback schedule experiments for batch testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── scale range ──────────────────────────────────────────────────────
    parser.add_argument('--scale_min', type=int, default=2,
                        help='Minimum trigger scale index (default: 2)')
    parser.add_argument('--scale_max', type=int, default=10,
                        help='Maximum trigger scale index (default: 10). '
                             'Total scales for 1M is 13 (index 0~12).')

    # ── rollback gap ─────────────────────────────────────────────────────
    parser.add_argument('--gap_min', type=int, default=1,
                        help='Minimum rollback gap: scale - rollback_to (default: 1)')
    parser.add_argument('--gap_max', type=int, default=3,
                        help='Maximum rollback gap (default: 3)')

    # ── times ────────────────────────────────────────────────────────────
    parser.add_argument('--times', type=int, nargs='+', default=[1, 2, 3],
                        help='List of retry counts to sweep (default: 1 2 3)')

    # ── multi-rule ───────────────────────────────────────────────────────
    parser.add_argument('--multi_rule', action='store_true',
                        help='Also generate dual-rule combination experiments.')
    parser.add_argument('--triple_rule', action='store_true',
                        help='Also generate triple-rule combination experiments '
                             '(can be very large!).')

    # ── output ───────────────────────────────────────────────────────────
    parser.add_argument('-o', '--output', type=str,
                        default='scripts/schedules_sweep.json',
                        help='Output JSON file path (default: scripts/schedules_sweep.json)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print experiment count; do not write file.')

    args = parser.parse_args()

    # ── generate ─────────────────────────────────────────────────────────
    all_exps = []

    # Single-rule sweep
    single = gen_single_rule_experiments(
        args.scale_min, args.scale_max,
        args.gap_min, args.gap_max,
        args.times,
    )
    all_exps.extend(single)
    print(f'Single-rule experiments: {len(single)}')

    # Dual-rule combos
    if args.multi_rule or args.triple_rule:
        dual = gen_dual_rule_experiments(
            args.scale_min, args.scale_max,
            args.gap_min, args.gap_max,
            args.times,
        )
        all_exps.extend(dual)
        print(f'Dual-rule experiments:   {len(dual)}')

    # Triple-rule combos
    if args.triple_rule:
        triple = gen_triple_rule_experiments(
            args.scale_min, args.scale_max,
            args.gap_min, args.gap_max,
            args.times,
        )
        all_exps.extend(triple)
        print(f'Triple-rule experiments: {len(triple)}')

    print(f'{"─"*40}')
    print(f'Total experiments:       {len(all_exps)}')

    # ── summary table ────────────────────────────────────────────────────
    print(f'\nSample experiments (first 10):')
    for i, exp in enumerate(all_exps[:10]):
        rules_str = '  '.join(
            f's{r["scale"]}→{r["rollback_to"]}×{r["times"]}'
            for r in exp['rules']
        )
        print(f'  [{i+1:3d}] {exp["name"]:<30s}  {rules_str}')
    if len(all_exps) > 10:
        print(f'  ... and {len(all_exps)-10} more')

    # ── write ────────────────────────────────────────────────────────────
    if args.dry_run:
        print('\n[dry_run] No file written.')
        return

    with open(args.output, 'w') as f:
        json.dump(all_exps, f, indent=2, ensure_ascii=False)
    print(f'\n[✓] Saved to {args.output}')
    print(f'    Run with: SCHEDULE_FILE={args.output} bash scripts/infer_loop_batch.sh')


if __name__ == '__main__':
    main()
