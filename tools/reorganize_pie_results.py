#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reorganize_pie_results.py — 將我們的輸出格式重組為 PnPInversion evaluate.py 所需格式

我們的輸出：
  result_dir/<category>/<case_id>/target.jpg
  e.g.  result_dir/0_random_140/000000000000/target.jpg

PnPInversion evaluate.py 期望：
  output_dir/<method_name>/annotation_images/<image_path>
  e.g.  output_dir/infinity_p2p_edit/annotation_images/0_random_140/000000000000.jpg

（image_path 來自 mapping_file.json 的 "image_path" 欄位）

使用方式：
  python3 tools/reorganize_pie_results.py \
    --mapping_file  <bench_dir>/mapping_file.json \
    --result_dir    <our_result_dir> \
    --output_dir    <organized_output_dir> \
    --method_name   infinity_p2p_edit
"""

import argparse
import json
import os
import shutil
import sys


def reorganize(mapping_file: str, result_dir: str, output_dir: str,
               method_name: str, use_symlink: bool = True) -> None:
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    ann_dir = os.path.join(output_dir, method_name, 'annotation_images')

    ok = 0
    missing = 0

    for case_id, entry in mapping.items():
        image_path = entry['image_path']          # e.g. "0_random_140/000000000000.jpg"
        #                                         # or "1_change_object_80/1_artificial/1_animal/111000000000.jpg"
        top_cat = image_path.split('/')[0]         # e.g. "0_random_140" (頂層 category，與 batch_run 的 save_dir 一致)

        # batch_run_pie_edit.py 儲存路徑：output_dir/<top_cat>/<case_id>/target.jpg
        src = os.path.join(result_dir, top_cat, case_id, 'target.jpg')
        dst = os.path.join(ann_dir, image_path)

        if not os.path.exists(src):
            missing += 1
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)

        if use_symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)

        ok += 1

    print(f'[Reorganize] 完成：{ok} 張，缺少：{missing} 張（共 {len(mapping)} 筆）')
    print(f'[Reorganize] 輸出目錄：{ann_dir}')

    if missing > 0:
        print(f'[Reorganize] ⚠ 有 {missing} 張圖片缺失，評估時這些案例將被跳過')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='將 P2P-Edit 輸出重組為 PnPInversion 評估格式')
    parser.add_argument('--mapping_file', required=True,
                        help='PIE-Bench mapping_file.json 路徑')
    parser.add_argument('--result_dir', required=True,
                        help='我們的批量輸出目錄（含 <category>/<case_id>/target.jpg）')
    parser.add_argument('--output_dir', required=True,
                        help='重組後的輸出目錄')
    parser.add_argument('--method_name', default='infinity_p2p_edit',
                        help='方法名稱（預設：infinity_p2p_edit）')
    parser.add_argument('--copy', action='store_true',
                        help='使用複製而非 symlink（預設：symlink）')
    args = parser.parse_args()

    reorganize(
        mapping_file=args.mapping_file,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        method_name=args.method_name,
        use_symlink=not args.copy,
    )
