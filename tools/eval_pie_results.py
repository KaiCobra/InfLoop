#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pie_results.py — PIE-Bench 結果定量評估腳本

評估指標：
  背景保留（Background Preservation）：
    • PSNR            — ↑，比較 source vs edited 的背景區域
    • SSIM            — ↑，比較 source vs edited 的背景區域
    • LPIPS           — ↓，比較 source vs edited 的背景區域

  結構保留（Structure Preservation）：
    • Structure Dist  — ↓，DINO ViT-S/8 key-feature MSE（source vs edited 全圖）

  編輯品質（Edit Quality）：
    • CLIP sim（image-text）— ↑，edited image vs target_prompt

  遮罩規則：
    mask.png = 255（白色）→ 編輯區域（排除在背景指標之外）
    mask.png = 0  （黑色）→ 背景區域（用於 PSNR/SSIM/LPIPS）

  若整張遮罩皆為 255（無背景），背景指標記為 NaN/跳過。

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

import cv2
import numpy as np
import torch
from PIL import Image

# 靜音 FutureWarning
warnings.filterwarnings('ignore')

# ── 確保工作目錄在 sys.path 中 ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ============================================================
# 套件懶載入（避免在不需要時卡在 import）
# ============================================================

def _require_lpips():
    try:
        import lpips
        return lpips
    except ImportError:
        print('[Error] lpips 尚未安裝。請執行：pip install lpips')
        sys.exit(1)


def _require_open_clip():
    try:
        import open_clip
        return open_clip
    except ImportError:
        print('[Error] open_clip 尚未安裝。請執行：pip install open-clip-torch')
        sys.exit(1)


def _require_torchvision_transforms():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    return transforms, InterpolationMode


# ============================================================
# 圖像工具函式
# ============================================================

def load_image_rgb(path: str, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    讀取 RGB uint8 影像 [H, W, 3]。
    若 target_hw=(H, W) 則縮小/放大至目標尺寸。
    """
    img = np.array(Image.open(path).convert('RGB'))  # [H, W, 3] uint8
    if target_hw is not None and img.shape[:2] != target_hw:
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return img


def load_mask(path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    讀取遮罩，回傳 bool 陣列 [H, W]。
    True = 編輯區域（mask=255），False = 背景（mask=0）。
    """
    arr = np.array(Image.open(path).convert('L'))  # [H, W] uint8
    if arr.shape != target_hw:
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 128  # True = edited region


# ============================================================
# 背景保留指標
# ============================================================

def psnr_masked(img_a: np.ndarray, img_b: np.ndarray, bg_mask: np.ndarray) -> float:
    """
    只在背景像素（bg_mask=True）上計算 PSNR。
    img_a, img_b: [H, W, 3] uint8
    bg_mask: [H, W] bool，True = 背景像素
    """
    if bg_mask.sum() == 0:
        return float('nan')
    a = img_a[bg_mask].astype(np.float64)
    b = img_b[bg_mask].astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0.0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_masked(img_a: np.ndarray, img_b: np.ndarray, bg_mask: np.ndarray) -> float:
    """
    計算全圖 SSIM map，只對背景像素取平均。
    img_a, img_b: [H, W, 3] uint8
    bg_mask: [H, W] bool，True = 背景像素
    """
    from skimage.metrics import structural_similarity
    if bg_mask.sum() == 0:
        return float('nan')
    # 以 channel_axis=-1 計算多通道 SSIM，回傳每像素 SSIM map
    ssim_val, ssim_map = structural_similarity(
        img_a, img_b,
        channel_axis=-1,
        data_range=255.0,
        full=True,
        win_size=11,
    )
    # ssim_map: [H, W, 3]，取三通道平均後在背景區域取均值
    ssim_map_avg = ssim_map.mean(axis=-1)  # [H, W]
    return float(ssim_map_avg[bg_mask].mean())


class LPIPSCalculator:
    """
    包裝 LPIPS 模型，只計算一次 forward pass。
    masked版本：將編輯區域設定為兩張圖片同樣的中性色（128 灰色），
    使編輯區域對 LPIPS 的貢獻趨近於零。
    """

    def __init__(self, net: str = 'alex', device: torch.device = torch.device('cpu')):
        lpips = _require_lpips()
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def compute(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        bg_mask: np.ndarray,
    ) -> float:
        """
        img_a, img_b: [H, W, 3] uint8
        bg_mask: [H, W] bool，True = 背景（保留），False = 編輯區域（設成一樣）
        """
        if bg_mask.sum() == 0:
            return float('nan')

        # 編輯區域設為相同中性色，消除其對 LPIPS 的貢獻
        edited_mask = ~bg_mask  # [H, W]
        a = img_a.copy().astype(np.float32)
        b = img_b.copy().astype(np.float32)
        a[edited_mask] = 128.0
        b[edited_mask] = 128.0

        # [H, W, 3] → [1, 3, H, W], 歸一化至 [-1, 1]
        def to_tensor(arr):
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return t / 127.5 - 1.0

        dist = self.model(to_tensor(a), to_tensor(b))
        return float(dist.item())


# ============================================================
# CLIP 相似度
# ============================================================

class CLIPCalculator:
    """
    計算 edited image 與 target_prompt 的 CLIP cosine similarity。
    """

    def __init__(self, model_name: str = 'ViT-L-14', pretrained: str = 'openai',
                 device: torch.device = torch.device('cpu')):
        open_clip = _require_open_clip()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenize = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()
        self.device = device
        print(f'  [CLIP] 模型載入：{model_name} ({pretrained})')

    @torch.no_grad()
    def image_embedding(self, img_rgb: np.ndarray) -> torch.Tensor:
        pil_img = Image.fromarray(img_rgb)
        t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(t)
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenize([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        return feat / feat.norm(dim=-1, keepdim=True)

    def similarity(self, img_rgb: np.ndarray, text: str) -> float:
        img_emb  = self.image_embedding(img_rgb)
        txt_emb  = self.text_embedding(text)
        return float((img_emb * txt_emb).sum().item())


# ============================================================
# Structure Distance（DINO ViT-S/8 Key MSE）
# ============================================================

class DINOStructureCalculator:
    """
    Structure Distance（PIE-Bench paper 採用的指標）。
    使用 DINO ViT-S/8 最後一個 block 的 key features，
    計算 source image 與 edited image 之間的 MSE。
    越低表示結構保留越好。
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    _INPUT_SIZE    = 224

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        print('  [DINO] 載入 DINO ViT-S/8 (facebookresearch/dino)...')
        self.model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vits8', verbose=False
        ).to(device).eval()

        transforms, InterpolationMode = _require_torchvision_transforms()
        self.transform = transforms.Compose([
            transforms.Resize(self._INPUT_SIZE,
                              interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self._INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN,
                                 std=self._IMAGENET_STD),
        ])
        print('  [DINO] 載入完成')

    def _extract_keys(self, img_rgb: np.ndarray) -> torch.Tensor:
        """
        Hook DINO 最後一個 block 的 qkv 線性層，取出 key features。
        回傳 Tensor [num_heads * N * head_dim]，已 flatten。
        """
        saved: Dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            # output: [1, N, 3*embed_dim]  (N = num_patches + 1)
            B, N, three_d = output.shape
            num_heads = self.model.blocks[-1].attn.num_heads
            head_dim  = three_d // (3 * num_heads)
            # reshape → [3, B, num_heads, N, head_dim]
            qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            saved['k'] = qkv[1].flatten().detach()  # keys

        handle = self.model.blocks[-1].attn.qkv.register_forward_hook(_hook)
        pil_img = Image.fromarray(img_rgb)
        t = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(t)
        handle.remove()
        return saved['k']

    @torch.no_grad()
    def compute(self, img_src: np.ndarray, img_tgt: np.ndarray) -> float:
        """
        img_src, img_tgt: [H, W, 3] uint8
        回傳 structure distance（MSE of DINO keys）↓ 越低越好
        """
        k_src = self._extract_keys(img_src)
        k_tgt = self._extract_keys(img_tgt)
        return float(((k_src - k_tgt) ** 2).mean().item())


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
    lpips_net: str,
    clip_model: str,
    clip_pretrained: str,
    no_structure_dist: bool,
    device: torch.device,
) -> None:

    # ── 初始化模型 ──
    print('\n[Init] 載入 LPIPS 模型...')
    lpips_calc = LPIPSCalculator(net=lpips_net, device=device)

    print('[Init] 載入 CLIP 模型...')
    clip_calc = CLIPCalculator(model_name=clip_model, pretrained=clip_pretrained, device=device)

    dino_calc: Optional[DINOStructureCalculator] = None
    if not no_structure_dist:
        print('[Init] 載入 DINO ViT-S/8 模型...')
        dino_calc = DINOStructureCalculator(device=device)

    # ── CSV 標頭 ──
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    csv_fields = [
        'category', 'case_id', 'edit_action',
        'psnr', 'ssim', 'lpips',
        'structure_dist',
        'clip_sim',
        'bg_pixel_pct',   # 背景像素佔比（%）
    ]

    all_rows: List[Dict] = []
    cat_stats: Dict[str, List[Dict]] = {}

    print(f'\n[Eval] 開始評估  bench={bench_dir}  result={result_dir}\n')

    for cat_name in categories:
        cat_bench  = os.path.join(bench_dir, cat_name)
        cat_result = os.path.join(result_dir, cat_name)

        if not os.path.isdir(cat_bench):
            print(f'[Warning] bench category 不存在，跳過：{cat_bench}')
            continue
        if not os.path.isdir(cat_result):
            print(f'[Warning] result category 不存在，跳過：{cat_result}')
            continue

        case_ids = sorted(d for d in os.listdir(cat_bench)
                          if os.path.isdir(os.path.join(cat_bench, d)))
        if max_per_cat > 0:
            case_ids = case_ids[:max_per_cat]

        print(f'{"─" * 60}')
        print(f'[Category] {cat_name}  ({len(case_ids)} 個案例)')
        print(f'{"─" * 60}')

        cat_rows: List[Dict] = []

        for idx, case_id in enumerate(case_ids):
            bench_case  = os.path.join(cat_bench, case_id)
            result_case = os.path.join(cat_result, case_id)

            # ── 路徑檢查 ──
            src_path    = os.path.join(bench_case, 'image.jpg')
            mask_path   = os.path.join(bench_case, 'mask.png')
            meta_path   = os.path.join(bench_case, 'meta.json')
            target_path = os.path.join(result_case, 'target.jpg')

            missing = [p for p in [src_path, mask_path, meta_path] if not os.path.exists(p)]
            if missing:
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ bench 檔案缺失：{missing}，跳過')
                continue
            if not os.path.exists(target_path):
                if skip_missing:
                    print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ↓ target.jpg 不存在，跳過')
                    continue
                else:
                    print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ✗ target.jpg 不存在')
                    continue
            if os.path.getsize(target_path) == 0:
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ target.jpg 為空檔（寫入中斷），跳過')
                continue

            # ── 讀取 meta ──
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            target_prompt = re.sub(r'\[([^\]]*)\]', r'\1', meta.get('target_prompt', '')).strip()
            edit_action   = ','.join(meta.get('edit_action', {}).keys())

            # ── 讀取影像 ──
            try:
                src_img    = load_image_rgb(src_path)
                ref_hw     = src_img.shape[:2]
                target_img = load_image_rgb(target_path, target_hw=ref_hw)
            except Exception as e:
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ 影像讀取失敗（{e}），跳過')
                continue

            # ── 讀取遮罩（True=編輯，False=背景）──
            edit_mask = load_mask(mask_path, ref_hw)    # [H, W] bool, True=edited
            bg_mask   = ~edit_mask                      # True = background
            bg_pct    = 100.0 * bg_mask.mean()

            # ── 背景保留指標 ──
            psnr_val  = psnr_masked(src_img, target_img, bg_mask)
            ssim_val  = ssim_masked(src_img, target_img, bg_mask)
            lpips_val = lpips_calc.compute(src_img, target_img, bg_mask)

            # ── Structure Distance（DINO keys MSE，整張圖 source vs target）──
            struct_val = dino_calc.compute(src_img, target_img) if dino_calc else float('nan')

            # ── CLIP 相似度（整張 target 圖 vs target_prompt）──
            clip_val  = clip_calc.similarity(target_img, target_prompt)

            row = {
                'category'      : cat_name,
                'case_id'       : case_id,
                'edit_action'   : edit_action,
                'psnr'          : round(psnr_val,    4) if not np.isnan(psnr_val)    else 'nan',
                'ssim'          : round(ssim_val,    4) if not np.isnan(ssim_val)    else 'nan',
                'lpips'         : round(lpips_val,   4) if not np.isnan(lpips_val)   else 'nan',
                'structure_dist': round(struct_val,  6) if not np.isnan(struct_val)  else 'nan',
                'clip_sim'      : round(clip_val,    4),
                'bg_pixel_pct'  : round(bg_pct, 2),
            }
            all_rows.append(row)
            cat_rows.append(row)

            psnr_str   = f'{psnr_val:.2f}'   if not np.isnan(psnr_val)   else '  NaN  '
            ssim_str   = f'{ssim_val:.4f}'   if not np.isnan(ssim_val)   else '  NaN  '
            lpips_str  = f'{lpips_val:.4f}'  if not np.isnan(lpips_val)  else '  NaN  '
            struct_str = f'{struct_val:.6f}' if not np.isnan(struct_val) else '  NaN  '
            print(f'  [{idx+1:3d}/{len(case_ids)}] {case_id}'
                  f'  PSNR={psnr_str}  SSIM={ssim_str}'
                  f'  LPIPS={lpips_str}  Sdist={struct_str}  CLIP={clip_val:.4f}'
                  f'  bg={bg_pct:.0f}%')

        cat_stats[cat_name] = cat_rows
        _print_category_summary(cat_name, cat_rows)

    # ── 儲存 CSV ──
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\n[Output] Per-case CSV → {output_csv}')

    # ── 儲存 Summary JSON ──
    summary = _build_summary(cat_stats)
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[Output] Summary JSON → {summary_json}')

    # ── 顯示全域摘要 ──
    _print_global_summary(summary)


# ============================================================
# 摘要計算 / 顯示
# ============================================================

def _nanmean(values: list) -> Optional[float]:
    """排除 NaN 後取平均。若全為 NaN 回傳 None。"""
    valid = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    return float(np.mean(valid)) if valid else None


def _build_summary(cat_stats: Dict[str, List[Dict]]) -> Dict:
    summary = {}
    all_rows_flat = []

    for cat_name, rows in cat_stats.items():
        all_rows_flat.extend(rows)
        has_bg = [r for r in rows if r['bg_pixel_pct'] > 0]
        summary[cat_name] = {
            'n_cases'           : len(rows),
            'n_with_bg'         : len(has_bg),
            'psnr_mean'         : _nanmean([r['psnr']           for r in rows]),
            'ssim_mean'         : _nanmean([r['ssim']           for r in rows]),
            'lpips_mean'        : _nanmean([r['lpips']          for r in rows]),
            'structure_dist_mean': _nanmean([r['structure_dist'] for r in rows]),
            'clip_sim_mean'     : _nanmean([r['clip_sim']       for r in rows]),
        }

    # 全域平均
    summary['__overall__'] = {
        'n_cases'           : len(all_rows_flat),
        'psnr_mean'         : _nanmean([r['psnr']           for r in all_rows_flat]),
        'ssim_mean'         : _nanmean([r['ssim']           for r in all_rows_flat]),
        'lpips_mean'        : _nanmean([r['lpips']          for r in all_rows_flat]),
        'structure_dist_mean': _nanmean([r['structure_dist'] for r in all_rows_flat]),
        'clip_sim_mean'     : _nanmean([r['clip_sim']       for r in all_rows_flat]),
    }
    return summary


def _print_category_summary(cat_name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    psnr_m   = _nanmean([r['psnr']           for r in rows])
    ssim_m   = _nanmean([r['ssim']           for r in rows])
    lpips_m  = _nanmean([r['lpips']          for r in rows])
    struct_m = _nanmean([r['structure_dist'] for r in rows])
    clip_m   = _nanmean([r['clip_sim']       for r in rows])
    n_bg     = sum(1 for r in rows if r['bg_pixel_pct'] > 0)

    def fmt4(v): return f'{v:.4f}' if v is not None else '  N/A  '
    def fmt6(v): return f'{v:.6f}' if v is not None else '    N/A   '
    print(f'\n  ┌── {cat_name} 小計 ({len(rows)} cases, {n_bg} w/ background) ──')
    print(f'  │  PSNR={fmt4(psnr_m)}  SSIM={fmt4(ssim_m)}  LPIPS={fmt4(lpips_m)}')
    print(f'  │  StructDist={fmt6(struct_m)}  CLIP={fmt4(clip_m)}')
    print(f'  └{"─" * 60}')


def _print_global_summary(summary: Dict) -> None:
    ov = summary.get('__overall__', {})
    def fmt4(v): return f'{v:.4f}' if v is not None else '  N/A '
    def fmt6(v): return f'{v:.6f}' if v is not None else '   N/A  '

    W = 88
    print('\n' + '=' * W)
    print(f'{"全域評估摘要":^{W}}')
    print('=' * W)
    print(f'{"Category":<38} {"PSNR":>7} {"SSIM":>7} {"LPIPS":>7} {"StructDist":>11} {"CLIP":>7}')
    print('─' * W)
    for cat, s in summary.items():
        if cat == '__overall__':
            continue
        p  = fmt4(s.get('psnr_mean'))
        q  = fmt4(s.get('ssim_mean'))
        r  = fmt4(s.get('lpips_mean'))
        sd = fmt6(s.get('structure_dist_mean'))
        c  = fmt4(s.get('clip_sim_mean'))
        print(f'{cat:<38} {p:>7} {q:>7} {r:>7} {sd:>11} {c:>7}')
    print('─' * W)
    print(f'{"Overall":<38} {fmt4(ov.get("psnr_mean")):>7} '
          f'{fmt4(ov.get("ssim_mean")):>7} '
          f'{fmt4(ov.get("lpips_mean")):>7} '
          f'{fmt6(ov.get("structure_dist_mean")):>11} '
          f'{fmt4(ov.get("clip_sim_mean")):>7}')
    print('=' * W)
    print(f'  總案例數：{ov.get("n_cases", 0)}')
    print(f'  指標說明：PSNR/SSIM ↑（背景保留），LPIPS ↓（背景），StructDist ↓（結構保留），CLIP ↑（編輯對齊）')
    print('=' * W + '\n')


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PIE-Bench 結果定量評估（PSNR / SSIM / LPIPS / Structure Dist / CLIP）'
    )

    # ── 路徑設定 ──
    parser.add_argument('--bench_dir', type=str, required=True,
                        help='extracted_pie_bench 根目錄')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='run_pie_edit.py 輸出根目錄')
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
    parser.add_argument('--lpips_net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS backbone（預設：alex）')
    parser.add_argument('--clip_model', type=str, default='ViT-L-14',
                        help='CLIP model name（open_clip 格式，預設：ViT-L-14）')
    parser.add_argument('--clip_pretrained', type=str, default='openai',
                        help='CLIP pretrained weight（預設：openai）')
    parser.add_argument('--no_structure_dist', action='store_true',
                        help='跳過 Structure Distance 計算（較慢，需要 DINO）')
    parser.add_argument('--device', type=str, default='',
                        help='cuda / cpu（預設：自動偵測）')

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'[Config] device={device}')

    # ── 決定要評估的 categories ──
    bench_dir = args.bench_dir
    if args.categories.strip():
        categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    else:
        categories = sorted(
            d for d in os.listdir(bench_dir)
            if os.path.isdir(os.path.join(bench_dir, d))
        )

    print(f'[Config] {len(categories)} 個 category，max_per_cat={args.max_per_cat}')
    print(f'[Config] LPIPS={args.lpips_net}  CLIP={args.clip_model}/{args.clip_pretrained}')
    print(f'[Config] StructureDist={"OFF" if args.no_structure_dist else "ON (DINO ViT-S/8)"}')

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
        lpips_net         = args.lpips_net,
        clip_model        = args.clip_model,
        clip_pretrained   = args.clip_pretrained,
        no_structure_dist = args.no_structure_dist,
        device            = device,
    )


if __name__ == '__main__':
    main()
