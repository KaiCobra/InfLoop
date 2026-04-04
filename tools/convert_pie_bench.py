#!/usr/bin/env python3
"""
convert_pie_bench.py — 將 PIE-Bench 原始資料轉換為 eval_pie_results.py 所需格式

輸入結構（PIE-Bench 原始）：
  pie_bench_data/PIE-Bench_v1/
    mapping_file.json
    annotation_images/{category}/{case_id}.jpg

輸出結構（eval 所需）：
  extracted_pie_bench/{category}/{case_id}/
    image.jpg        — source 圖片
    mask.png         — 編輯遮罩（白=編輯區, 黑=背景）
    meta.json        — source_prompt, target_prompt, blended_words, edit_action

使用方式：
  python tools/convert_pie_bench.py \
    --input_dir  pie_bench_data/PIE-Bench_v1 \
    --output_dir outputs/outputs_loop_exp/extracted_pie_bench
"""

import os
import re
import json
import shutil
import argparse

import numpy as np
from PIL import Image


def decode_rle_mask(rle_list, h=512, w=512):
    """解碼 PIE-Bench 的 RLE mask（start, length pairs, column-major）"""
    mask = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(rle_list), 2):
        start = rle_list[i]
        length = rle_list[i + 1]
        mask[start:start + length] = 255
    return mask.reshape((h, w), order='F')


def clean_prompt(prompt):
    """移除 prompt 中的 [] 括號標記，保留內容"""
    return re.sub(r'\[([^\]]*)\]', r'\1', prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='pie_bench_data/PIE-Bench_v1')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/outputs_loop_exp/extracted_pie_bench')
    args = parser.parse_args()

    mapping_path = os.path.join(args.input_dir, 'mapping_file.json')
    img_dir = os.path.join(args.input_dir, 'annotation_images')

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 建立 filename → actual path 的索引（處理平放 vs 子資料夾的差異）
    file_index = {}
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.endswith('.jpg'):
                file_index[fname] = os.path.join(root, fname)

    total = 0
    skipped = 0

    for case_id, entry in sorted(mapping.items()):
        img_rel = entry['image_path']                    # e.g. "0_random_140/000000000000.jpg"
        category = img_rel.split('/')[0]                 # e.g. "0_random_140"
        img_filename = os.path.basename(img_rel)
        src_img_path = file_index.get(img_filename,
                                       os.path.join(img_dir, img_rel))

        if not os.path.exists(src_img_path):
            print(f'  [SKIP] {img_rel} not found')
            skipped += 1
            continue

        # 建立輸出目錄
        case_dir = os.path.join(args.output_dir, category, case_id)
        os.makedirs(case_dir, exist_ok=True)

        # 複製 source image
        shutil.copy2(src_img_path, os.path.join(case_dir, 'image.jpg'))

        # 解碼並儲存 mask
        img = Image.open(src_img_path)
        w, h = img.size
        rle = entry['mask']
        mask = decode_rle_mask(rle, h=h, w=w)
        Image.fromarray(mask).save(os.path.join(case_dir, 'mask.png'))

        # 產生 meta.json
        blended = entry.get('blended_word', '').split()
        source_words = blended[:len(blended) // 2] if blended else []
        target_words = blended[len(blended) // 2:] if blended else []

        meta = {
            'source_prompt': clean_prompt(entry.get('original_prompt', '')),
            'target_prompt': entry.get('editing_prompt', ''),
            'blended_words': blended,
            'source_words': source_words,
            'target_words': target_words,
            'editing_instruction': entry.get('editing_instruction', ''),
            'editing_type_id': entry.get('editing_type_id', ''),
            'edit_action': {
                entry.get('editing_instruction', 'edit'): True
            },
        }
        with open(os.path.join(case_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        total += 1

    print(f'\nDone! Converted {total} cases, skipped {skipped}')
    print(f'Output: {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
