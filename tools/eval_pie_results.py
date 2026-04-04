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
    • Structure Dist  — ↓，DINO ViT-B/8 key self-similarity MSE（source vs edited 全圖）

  編輯品質（Edit Quality）：
    • CLIP sim whole  — ↑，edited image 全圖 vs target_prompt
    • CLIP sim edited — ↑，edited image 編輯區域 vs target_prompt

  推論速度（Inference Speed）：
    • inference_sec   — 每張圖推論秒數（從 timing.json 讀取）

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


def decode_rle_mask(rle: List[int], hw: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    解碼 PIE-Bench mapping_file.json 中的 mask RLE。
    格式：(start, count) 交替排列，foreground=True。
    回傳 bool 陣列 [H, W]，True = 編輯區域。
    """
    h, w = hw
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(rle), 2):
        start = rle[i]
        count = rle[i + 1]
        end = min(start + count, h * w)
        flat[start:end] = 1
    return flat.reshape(h, w).astype(bool)


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

    def similarity_edited(self, img_rgb: np.ndarray, text: str,
                           edit_mask: np.ndarray) -> float:
        """
        只保留編輯區域（edit_mask=True），背景設為黑色，再計算 CLIP 相似度。
        與 PnPInversion 官方做法一致：img * mask → CLIP score。
        """
        masked_img = img_rgb.copy()
        masked_img[~edit_mask] = 0  # 背景設為黑色
        return self.similarity(masked_img, text)


# ============================================================
# Structure Distance（DINO ViT-B/8 Key Self-Similarity MSE）
# ============================================================

class DINOStructureCalculator:
    """
    Structure Distance（PIE-Bench / PnPInversion 官方指標）。
    使用 DINO ViT-B/8 第 11 層（最後一個 block）的 key features，
    計算 cosine self-similarity matrix，再對 source 與 edited 的
    self-similarity matrix 取 MSE。越低表示結構保留越好。

    參考：https://github.com/cure-lab/PnPInversion/blob/main/evaluation/matrics_calculator.py
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    _INPUT_SIZE    = 224
    _LAYER_NUM     = 11  # 官方使用 layer 11（ViT-B 共 12 層，0-indexed）

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        print('  [DINO] 載入 DINO ViT-B/8 (facebookresearch/dino)...')
        self.model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vitb8', verbose=False
        ).to(device).eval()
        self.num_heads = self.model.blocks[self._LAYER_NUM].attn.num_heads

        transforms, _ = _require_torchvision_transforms()
        self.transform = transforms.Compose([
            transforms.Resize(self._INPUT_SIZE, max_size=480),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN,
                                 std=self._IMAGENET_STD),
        ])
        print(f'  [DINO] 載入完成 (num_heads={self.num_heads})')

    def _extract_keys(self, img_rgb: np.ndarray) -> torch.Tensor:
        """
        Hook DINO 第 11 層 block 的 qkv 線性層，取出 key features。
        回傳 Tensor [1, num_heads, N, head_dim]。
        """
        saved: Dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            # output: [B, N, 3*embed_dim]  (N = num_patches + 1)
            B, N, three_d = output.shape
            num_heads = self.num_heads
            head_dim  = three_d // (3 * num_heads)
            # reshape → [3, B, num_heads, N, head_dim]
            qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            saved['k'] = qkv[1].detach()  # keys: [B, num_heads, N, head_dim]

        handle = self.model.blocks[self._LAYER_NUM].attn.qkv.register_forward_hook(_hook)
        pil_img = Image.fromarray(img_rgb)
        t = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(t)
        handle.remove()
        return saved['k']

    @staticmethod
    def _keys_self_similarity(keys: torch.Tensor) -> torch.Tensor:
        """
        計算 key features 的 cosine self-similarity matrix。
        keys: [B, num_heads, N, head_dim]
        回傳: [B, N, N] cosine similarity matrix
        """
        B, h, N, d = keys.shape
        # 合併所有 head 的 key：[B, N, h*d]
        concat = keys.permute(0, 2, 1, 3).reshape(B, N, h * d)
        # Cosine self-similarity
        norm = concat.norm(dim=2, keepdim=True)          # [B, N, 1]
        factor = torch.clamp(norm @ norm.transpose(1, 2), min=1e-8)  # [B, N, N]
        sim = (concat @ concat.transpose(1, 2)) / factor  # [B, N, N]
        return sim

    @torch.no_grad()
    def compute(self, img_src: np.ndarray, img_tgt: np.ndarray) -> float:
        """
        img_src, img_tgt: [H, W, 3] uint8
        回傳 structure distance（MSE of DINO key self-similarity）↓ 越低越好
        """
        k_src = self._extract_keys(img_src)
        k_tgt = self._extract_keys(img_tgt)
        sim_src = self._keys_self_similarity(k_src)
        sim_tgt = self._keys_self_similarity(k_tgt)
        return float(torch.nn.functional.mse_loss(sim_tgt, sim_src).item())


# ============================================================
# 主評估迴圈
# ============================================================

def _load_mapping_file(bench_dir: str) -> Dict:
    """
    載入 PIE-Bench_v1 的 mapping_file.json，
    並建立 {category: {case_id: entry}} 的索引。
    """
    mapping_path = os.path.join(bench_dir, 'mapping_file.json')
    if not os.path.isfile(mapping_path):
        print(f'[Error] mapping_file.json 不存在：{mapping_path}')
        sys.exit(1)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # 按 category 分組
    grouped: Dict[str, Dict] = {}
    for case_id, entry in raw.items():
        cat = entry['image_path'].split('/')[0]  # e.g. "2_add_object_80"
        grouped.setdefault(cat, {})[case_id] = entry
    return grouped


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

    # ── 載入 mapping_file.json ──
    print('\n[Init] 載入 mapping_file.json...')
    cat_mapping = _load_mapping_file(bench_dir)
    annotation_dir = os.path.join(bench_dir, 'annotation_images')

    # ── 初始化模型 ──
    print('[Init] 載入 LPIPS 模型...')
    lpips_calc = LPIPSCalculator(net=lpips_net, device=device)

    print('[Init] 載入 CLIP 模型...')
    clip_calc = CLIPCalculator(model_name=clip_model, pretrained=clip_pretrained, device=device)

    dino_calc: Optional[DINOStructureCalculator] = None
    if not no_structure_dist:
        print('[Init] 載入 DINO ViT-B/8 模型...')
        dino_calc = DINOStructureCalculator(device=device)

    # ── CSV 標頭 ──
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    csv_fields = [
        'category', 'case_id', 'edit_action',
        'psnr', 'ssim', 'lpips',
        'structure_dist',
        'clip_sim_whole', 'clip_sim_edited',
        'inference_sec',
        'bg_pixel_pct',   # 背景像素佔比（%）
    ]

    all_rows: List[Dict] = []
    cat_stats: Dict[str, List[Dict]] = {}

    print(f'\n[Eval] 開始評估  bench={bench_dir}  result={result_dir}\n')

    for cat_name in categories:
        if cat_name not in cat_mapping:
            print(f'[Warning] bench category 不存在於 mapping_file.json，跳過：{cat_name}')
            continue
        cat_result = os.path.join(result_dir, cat_name)
        if not os.path.isdir(cat_result):
            print(f'[Warning] result category 不存在，跳過：{cat_result}')
            continue

        cat_entries = cat_mapping[cat_name]
        case_ids = sorted(cat_entries.keys())
        if max_per_cat > 0:
            case_ids = case_ids[:max_per_cat]

        print(f'{"─" * 60}')
        print(f'[Category] {cat_name}  ({len(case_ids)} 個案例)')
        print(f'{"─" * 60}')

        cat_rows: List[Dict] = []

        for idx, case_id in enumerate(case_ids):
            entry       = cat_entries[case_id]
            result_case = os.path.join(cat_result, case_id)

            # ── 路徑檢查 ──
            src_path    = os.path.join(annotation_dir, entry['image_path'])
            target_path = os.path.join(result_case, 'target.jpg')

            if not os.path.exists(src_path):
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ source 影像缺失：{src_path}，跳過')
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

            # ── 讀取 meta（from mapping_file.json entry）──
            target_prompt = re.sub(r'\[([^\]]*)\]', r'\1', entry.get('editing_prompt', '')).strip()
            edit_action   = entry.get('editing_instruction', '')

            # ── 讀取影像 ──
            try:
                src_img    = load_image_rgb(src_path)
                ref_hw     = src_img.shape[:2]
                target_img = load_image_rgb(target_path, target_hw=ref_hw)
            except Exception as e:
                print(f'  [{idx+1}/{len(case_ids)}] {case_id}  ⚠ 影像讀取失敗（{e}），跳過')
                continue

            # ── 解碼遮罩（True=編輯，False=背景）──
            edit_mask = decode_rle_mask(entry['mask'], hw=ref_hw)  # [H, W] bool, True=edited
            bg_mask   = ~edit_mask                                 # True = background
            bg_pct    = 100.0 * bg_mask.mean()

            # ── 背景保留指標 ──
            psnr_val  = psnr_masked(src_img, target_img, bg_mask)
            ssim_val  = ssim_masked(src_img, target_img, bg_mask)
            lpips_val = lpips_calc.compute(src_img, target_img, bg_mask)

            # ── Structure Distance（DINO keys self-similarity MSE，整張圖 source vs target）──
            struct_val = dino_calc.compute(src_img, target_img) if dino_calc else float('nan')

            # ── CLIP 相似度 ──
            clip_whole_val  = clip_calc.similarity(target_img, target_prompt)
            clip_edited_val = clip_calc.similarity_edited(target_img, target_prompt, edit_mask)

            # ── Inference Speed（從 timing.json 讀取）──
            timing_path = os.path.join(result_case, 'timing.json')
            if os.path.exists(timing_path):
                try:
                    with open(timing_path, 'r', encoding='utf-8') as f_t:
                        timing_data = json.load(f_t)
                    inference_sec = float(timing_data.get('inference_sec', float('nan')))
                except Exception:
                    inference_sec = float('nan')
            else:
                inference_sec = float('nan')

            row = {
                'category'       : cat_name,
                'case_id'        : case_id,
                'edit_action'    : edit_action,
                'psnr'           : round(psnr_val,         4) if not np.isnan(psnr_val)        else 'nan',
                'ssim'           : round(ssim_val,         4) if not np.isnan(ssim_val)        else 'nan',
                'lpips'          : round(lpips_val,        4) if not np.isnan(lpips_val)       else 'nan',
                'structure_dist' : round(struct_val,       6) if not np.isnan(struct_val)      else 'nan',
                'clip_sim_whole' : round(clip_whole_val,   4),
                'clip_sim_edited': round(clip_edited_val,  4),
                'inference_sec'  : round(inference_sec,    3) if not np.isnan(inference_sec)   else 'nan',
                'bg_pixel_pct'   : round(bg_pct, 2),
            }
            all_rows.append(row)
            cat_rows.append(row)

            psnr_str   = f'{psnr_val:.2f}'   if not np.isnan(psnr_val)   else '  NaN  '
            ssim_str   = f'{ssim_val:.4f}'   if not np.isnan(ssim_val)   else '  NaN  '
            lpips_str  = f'{lpips_val:.4f}'  if not np.isnan(lpips_val)  else '  NaN  '
            struct_str = f'{struct_val:.6f}' if not np.isnan(struct_val) else '  NaN  '
            infer_str  = f'{inference_sec:.1f}s' if not np.isnan(inference_sec) else ' N/A '
            print(f'  [{idx+1:3d}/{len(case_ids)}] {case_id}'
                  f'  PSNR={psnr_str}  SSIM={ssim_str}'
                  f'  LPIPS={lpips_str}  Sdist={struct_str}'
                  f'  CLIPw={clip_whole_val:.4f}  CLIPe={clip_edited_val:.4f}'
                  f'  t={infer_str}  bg={bg_pct:.0f}%')

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
            'n_cases'              : len(rows),
            'n_with_bg'            : len(has_bg),
            'psnr_mean'            : _nanmean([r['psnr']            for r in rows]),
            'ssim_mean'            : _nanmean([r['ssim']            for r in rows]),
            'lpips_mean'           : _nanmean([r['lpips']           for r in rows]),
            'structure_dist_mean'  : _nanmean([r['structure_dist']  for r in rows]),
            'clip_sim_whole_mean'  : _nanmean([r['clip_sim_whole']  for r in rows]),
            'clip_sim_edited_mean' : _nanmean([r['clip_sim_edited'] for r in rows]),
            'inference_sec_mean'   : _nanmean([r['inference_sec']   for r in rows]),
        }

    # 全域平均
    summary['__overall__'] = {
        'n_cases'              : len(all_rows_flat),
        'psnr_mean'            : _nanmean([r['psnr']            for r in all_rows_flat]),
        'ssim_mean'            : _nanmean([r['ssim']            for r in all_rows_flat]),
        'lpips_mean'           : _nanmean([r['lpips']           for r in all_rows_flat]),
        'structure_dist_mean'  : _nanmean([r['structure_dist']  for r in all_rows_flat]),
        'clip_sim_whole_mean'  : _nanmean([r['clip_sim_whole']  for r in all_rows_flat]),
        'clip_sim_edited_mean' : _nanmean([r['clip_sim_edited'] for r in all_rows_flat]),
        'inference_sec_mean'   : _nanmean([r['inference_sec']   for r in all_rows_flat]),
    }
    return summary


def _print_category_summary(cat_name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    psnr_m    = _nanmean([r['psnr']            for r in rows])
    ssim_m    = _nanmean([r['ssim']            for r in rows])
    lpips_m   = _nanmean([r['lpips']           for r in rows])
    struct_m  = _nanmean([r['structure_dist']  for r in rows])
    clipw_m   = _nanmean([r['clip_sim_whole']  for r in rows])
    clipe_m   = _nanmean([r['clip_sim_edited'] for r in rows])
    infer_m   = _nanmean([r['inference_sec']   for r in rows])
    n_bg      = sum(1 for r in rows if r['bg_pixel_pct'] > 0)

    def fmt4(v): return f'{v:.4f}' if v is not None else '  N/A  '
    def fmt6(v): return f'{v:.6f}' if v is not None else '    N/A   '
    def fmts(v): return f'{v:.1f}s' if v is not None else '  N/A '
    print(f'\n  ┌── {cat_name} 小計 ({len(rows)} cases, {n_bg} w/ background) ──')
    print(f'  │  PSNR={fmt4(psnr_m)}  SSIM={fmt4(ssim_m)}  LPIPS={fmt4(lpips_m)}')
    print(f'  │  StructDist={fmt6(struct_m)}  CLIPwhole={fmt4(clipw_m)}  CLIPedited={fmt4(clipe_m)}')
    print(f'  │  InferSpeed={fmts(infer_m)}')
    print(f'  └{"─" * 60}')


def _print_global_summary(summary: Dict) -> None:
    ov = summary.get('__overall__', {})
    def fmt4(v): return f'{v:.4f}' if v is not None else '  N/A '
    def fmt6(v): return f'{v:.6f}' if v is not None else '   N/A  '
    def fmts(v): return f'{v:.1f}s' if v is not None else ' N/A '

    W = 108
    print('\n' + '=' * W)
    print(f'{"全域評估摘要":^{W}}')
    print('=' * W)
    print(f'{"Category":<38} {"PSNR":>7} {"SSIM":>7} {"LPIPS":>7} {"StructDist":>11} {"CLIPw":>7} {"CLIPe":>7} {"Speed":>7}')
    print('─' * W)
    for cat, s in summary.items():
        if cat == '__overall__':
            continue
        p  = fmt4(s.get('psnr_mean'))
        q  = fmt4(s.get('ssim_mean'))
        r  = fmt4(s.get('lpips_mean'))
        sd = fmt6(s.get('structure_dist_mean'))
        cw = fmt4(s.get('clip_sim_whole_mean'))
        ce = fmt4(s.get('clip_sim_edited_mean'))
        sp = fmts(s.get('inference_sec_mean'))
        print(f'{cat:<38} {p:>7} {q:>7} {r:>7} {sd:>11} {cw:>7} {ce:>7} {sp:>7}')
    print('─' * W)
    print(f'{"Overall":<38} {fmt4(ov.get("psnr_mean")):>7} '
          f'{fmt4(ov.get("ssim_mean")):>7} '
          f'{fmt4(ov.get("lpips_mean")):>7} '
          f'{fmt6(ov.get("structure_dist_mean")):>11} '
          f'{fmt4(ov.get("clip_sim_whole_mean")):>7} '
          f'{fmt4(ov.get("clip_sim_edited_mean")):>7} '
          f'{fmts(ov.get("inference_sec_mean")):>7}')
    print('=' * W)
    print(f'  總案例數：{ov.get("n_cases", 0)}')
    print(f'  指標說明：PSNR/SSIM ↑（背景保留），LPIPS ↓（背景），StructDist ↓（結構保留），'
          f'CLIPw/CLIPe ↑（編輯對齊 whole/edited），Speed = 推論時間')
    print('=' * W + '\n')


# ============================================================
# 主程式
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PIE-Bench 結果定量評估（PSNR / SSIM / LPIPS / Structure Dist / CLIP whole+edited / Speed）'
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
        # 從 mapping_file.json 自動列出所有 category
        mapping_path = os.path.join(bench_dir, 'mapping_file.json')
        if os.path.isfile(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            categories = sorted({v['image_path'].split('/')[0] for v in raw.values()})
        else:
            categories = sorted(
                d for d in os.listdir(bench_dir)
                if os.path.isdir(os.path.join(bench_dir, d))
            )

    print(f'[Config] {len(categories)} 個 category，max_per_cat={args.max_per_cat}')
    print(f'[Config] LPIPS={args.lpips_net}  CLIP={args.clip_model}/{args.clip_pretrained}')
    print(f'[Config] StructureDist={"OFF" if args.no_structure_dist else "ON (DINO ViT-B/8)"}')

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