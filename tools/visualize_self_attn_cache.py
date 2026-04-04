#!/usr/bin/env python3
"""
將 self-attn cache（scale_XX.pt）轉成每個 block 的 head heatmap 圖。

輸入支援：
1) 單一檔案：  --input /path/to/scale_03.pt
2) 目錄批次：  --input /path/to/self_attn_cache

輸出結構（預設 output 在 input 同層）：
  <output>/
    scale_03/
      block_000/
        head_00.png
        head_01.png
        ...
        head_grid.png
        mean.png
      block_001/
      ...
      summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize self-attn cache per block/head")
    parser.add_argument("--input", type=str, required=True, help="scale_XX.pt 或 self_attn_cache 目錄")
    parser.add_argument("--output", type=str, default="", help="輸出目錄（預設：<input>_viz）")
    parser.add_argument("--q-reduce", type=str, default="mean", choices=["mean", "max"],
                        help="將 query 維度聚合成 key heatmap 的方式")
    parser.add_argument("--normalize", type=str, default="per_head", choices=["per_head", "per_block"],
                        help="熱圖正規化範圍")
    parser.add_argument("--colormap", type=str, default="jet", choices=["jet", "hot", "viridis", "magma"],
                        help="色盤")
    parser.add_argument("--head-size", type=int, default=192, help="單一 head 視覺化圖邊長")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="張量運算裝置。auto=有 CUDA 就用 CUDA")
    return parser.parse_args()


def get_colormap(name: str) -> int:
    mapping = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
    }
    return mapping[name]


def list_scale_pt_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".pt":
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("scale_*.pt"))
        if files:
            return files
        nested = input_path / "self_attn_cache"
        if nested.is_dir():
            nested_files = sorted(nested.glob("scale_*.pt"))
            if nested_files:
                return nested_files
        recursive_files = sorted(input_path.glob("**/scale_*.pt"))
        if recursive_files:
            return recursive_files
    raise FileNotFoundError(f"找不到可用的 scale_*.pt：{input_path}")


def safe_reshape_to_hw(vec: np.ndarray, h: int, w: int) -> np.ndarray:
    """將 1D key attention reshape 成 (h, w)，若長度不符則 fallback 最近鄰縮放。"""
    n = vec.shape[0]
    if n == h * w:
        return vec.reshape(h, w)

    side = int(math.sqrt(n))
    if side * side == n:
        square = vec.reshape(side, side)
    else:
        # 近似成 1 x n 再縮放
        square = vec.reshape(1, n)

    return cv2.resize(square.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


def normalize_map(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def colorize_heatmap(gray01: np.ndarray, cmap: int) -> np.ndarray:
    u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def make_head_grid(images_bgr: List[np.ndarray], cell_size: int) -> np.ndarray:
    if not images_bgr:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    n = len(images_bgr)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for i, img in enumerate(images_bgr):
        r = i // cols
        c = i % cols
        y0, y1 = r * cell_size, (r + 1) * cell_size
        x0, x1 = c * cell_size, (c + 1) * cell_size
        canvas[y0:y1, x0:x1] = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
    return canvas


def reduce_query(attn_hqk: torch.Tensor, mode: str) -> torch.Tensor:
    # attn_hqk: [H, Q, K] -> [H, K]
    if mode == "max":
        return attn_hqk.max(dim=1).values
    return attn_hqk.mean(dim=1)


def process_scale_file(
    pt_path: Path,
    out_root: Path,
    q_reduce: str,
    normalize: str,
    cmap: int,
    head_size: int,
    device: torch.device,
) -> Dict:
    data = torch.load(pt_path, map_location="cpu")
    scale_idx = int(data.get("scale_idx", -1))
    scale_info = data.get("scale_info", None)
    block_attn: Dict[int, torch.Tensor] = data.get("block_attn", {})

    if scale_info is not None and len(scale_info) == 3:
        _, h, w = int(scale_info[0]), int(scale_info[1]), int(scale_info[2])
    else:
        # fallback: 無尺寸資訊時用 1xK 形式
        h, w = 1, -1

    scale_dir = out_root / f"scale_{scale_idx:02d}"
    scale_dir.mkdir(parents=True, exist_ok=True)

    scale_summary = {
        "scale_idx": scale_idx,
        "scale_info": list(scale_info) if scale_info is not None else None,
        "num_blocks": len(block_attn),
        "blocks": {},
    }

    for block_idx in sorted(block_attn.keys()):
        attn = block_attn[block_idx]  # [1, H, Q, K]
        if not torch.is_tensor(attn) or attn.ndim != 4:
            continue

        _, n_heads, q_len, k_len = attn.shape
        if h <= 0 or w <= 0:
            h_eff, w_eff = 1, k_len
        else:
            h_eff, w_eff = h, w

        q_vis = min(h_eff * w_eff, q_len)
        k_vis = min(h_eff * w_eff, k_len)
        q_start = q_len - q_vis
        k_start = k_len - k_vis

        attn_hqk = attn[0, :, q_start:, k_start:].to(device=device, dtype=torch.float32)  # [H, q_vis, k_vis]
        attn_hk = reduce_query(attn_hqk, q_reduce)         # [H, k_vis]

        block_dir = scale_dir / f"block_{block_idx:03d}"
        block_dir.mkdir(parents=True, exist_ok=True)

        per_head_maps: List[np.ndarray] = []
        for head_id in range(n_heads):
            key_vec = attn_hk[head_id].detach().cpu().numpy().astype(np.float32)
            heat_2d = safe_reshape_to_hw(key_vec, h_eff, w_eff)
            per_head_maps.append(heat_2d)

        del attn_hqk, attn_hk
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if normalize == "per_block":
            stacked = np.stack(per_head_maps, axis=0)
            stacked = normalize_map(stacked)
            normed_maps = [stacked[i] for i in range(stacked.shape[0])]
        else:
            normed_maps = [normalize_map(m) for m in per_head_maps]

        head_imgs: List[np.ndarray] = []
        for head_id, heat01 in enumerate(normed_maps):
            color = colorize_heatmap(heat01, cmap)
            out_path = block_dir / f"head_{head_id:02d}.png"
            cv2.imwrite(str(out_path), color)
            head_imgs.append(color)

        mean_map = normalize_map(np.mean(np.stack(per_head_maps, axis=0), axis=0))
        mean_img = colorize_heatmap(mean_map, cmap)
        cv2.imwrite(str(block_dir / "mean.png"), mean_img)

        grid = make_head_grid(head_imgs, cell_size=head_size)
        cv2.imwrite(str(block_dir / "head_grid.png"), grid)

        scale_summary["blocks"][str(block_idx)] = {
            "shape": [int(x) for x in attn.shape],
            "q_len": int(q_len),
            "k_len": int(k_len),
            "q_vis": int(q_vis),
            "k_vis": int(k_vis),
            "num_heads": int(n_heads),
        }

    with open(scale_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(scale_summary, f, ensure_ascii=False, indent=2)

    return scale_summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    pt_files = list_scale_pt_files(input_path)

    if args.output:
        out_root = Path(args.output).expanduser().resolve()
    else:
        base = input_path if input_path.is_dir() else input_path.parent
        out_root = (base.parent / f"{base.name}_viz").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cmap = get_colormap(args.colormap)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定 --device cuda，但目前環境沒有可用 CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global_summary = {
        "input": str(input_path),
        "num_scales": len(pt_files),
        "device": str(device),
        "scales": [],
    }

    print(f"[SelfAttn Viz] input={input_path}")
    print(f"[SelfAttn Viz] output={out_root}")
    print(f"[SelfAttn Viz] device={device}")

    for pt_path in pt_files:
        print(f"[SelfAttn Viz] processing {pt_path.name} ...")
        info = process_scale_file(
            pt_path=pt_path,
            out_root=out_root,
            q_reduce=args.q_reduce,
            normalize=args.normalize,
            cmap=cmap,
            head_size=args.head_size,
            device=device,
        )
        global_summary["scales"].append(info)

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(f"✓ 完成，共轉換 {len(pt_files)} 個 scale 檔案")


if __name__ == "__main__":
    main()
