"""
P2P-Edit (Prompt-to-Prompt + Source Image Injection) 圖像編輯管線

功能說明
========
在原有 P2P 管線（run_p2p.py）的基礎上加入 **attention-based 空間遮罩**，
專為「同一場景、局部文字內容不同」的 prompt pair 設計。

設計理念
--------
$P_s$: A train platform sign that reads "PLEASE STAND BEHIND LINE" as a train approaches.
$P_t$: A train platform sign that reads "DESTINATION: LONDON" as a train approaches.

問題：直接全部替換 → 文字區域也被固定，target 無法渲染新文字。
解決：
    1. 前 N 個 scale（粗略結構）：100% source token 替換（保留整體佈局）
    2. 第 N+1 個 scale 之後：
       - 從 source 生成過程中擷取 cross-attention 圖
       - 對 focus_words（如 "PLEASE STAND BEHIND LINE"）的 token 求平均 attention
       - 高 attention 區域 = 文字位置 → 不替換（讓 target 自由生成新文字）
       - 低 attention 區域 = 背景 → 替換為 source token（保留背景結構）

工作流程
--------
    [Phase 1] Source 生成
        ├─ 掛載 CrossAttentionExtractor（捕捉各 scale 的 cross-attention）
        ├─ 生成 source image
        ├─ 儲存所有 scale 的 bitwise token → BitwiseTokenStorage
        ├─ 卸除 CrossAttentionExtractor
        └─ 計算 attention-based 空間遮罩 → 存入 storage.masks

    [Phase 2] Target 生成
        ├─ scale < num_full_replace_scales：100% token 替換
        ├─ scale >= num_full_replace_scales：使用 attention 遮罩
        │   ├─ 高 attention 區（文字）→ 保留 target 自由生成
        │   └─ 低 attention 區（背景）→ 替換 source token
        └─ 輸出 target image

執行方式
--------
    python3 tools/run_p2p_attn.py \\
        --source_prompt "A train platform sign that reads \\"PLEASE STAND BEHIND LINE\\" as a train approaches." \\
        --target_prompt "A train platform sign that reads \\"DESTINATION: LONDON\\" as a train approaches." \\
        --focus_words "PLEASE STAND BEHIND LINE" \\
        --num_full_replace_scales 4 \\
        --attn_threshold_percentile 75 \\
        ...（其餘參數請見 infer_p2p_attn.sh）

    或直接：
        bash scripts/infer_p2p_attn.sh
"""

import os
import sys

# 將專案根目錄加入 Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os.path as osp
import re
import math
import time
import hashlib
import argparse
import shutil
import json
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch

torch._dynamo.config.cache_size_limit = 64

from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import PIL.Image as PImage
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_tensor

# 使用 selfAttn-Edit 版本的模型
from infinity.models.infinity_selfAttn_edit import Infinity
from infinity.models.basic import *
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

# Attention Extractors
from attention_map.extractor import CrossAttentionExtractor, SelfAttentionExtractor


# ============================================================
# Tokenizer 工具函式
# ============================================================

def find_focus_token_indices(
    tokenizer,
    prompt: str,
    focus_words: List[str],
    verbose: bool = True,
) -> List[int]:
    """
    在 source prompt 中尋找 focus_words 對應的 T5 token indices。

    T5 使用 SentencePiece 分詞，可能將一個詞拆分成多個子詞（sub-token）。
    此函式會找出所有對應子詞的 index，確保不遺漏。

    Args:
        tokenizer: T5 tokenizer 實例
        prompt: source prompt 完整文字
        focus_words: 欲關注的詞彙清單（如 ["PLEASE", "STAND", "BEHIND", "LINE"]）
        verbose: 是否印出匹配詳情

    Returns:
        List[int]: 在 tokenized（未 padding）序列中，對應 focus_words 的 index 清單
    """
    # 對整段 prompt 進行 tokenize
    tokens = tokenizer(
        text=[prompt],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = tokens.input_ids[0].tolist()          # [512]
    attn_mask = tokens.attention_mask[0].tolist()     # [512]
    seq_len = sum(attn_mask)  # 實際非 padding 長度

    if verbose:
        print(f"\n[Focus Token Search] prompt 長度（token）= {seq_len}")

    focus_words_lower = [w.lower().strip() for w in focus_words]
    focus_indices: List[int] = []

    # 解碼所有 token（去除 SentencePiece 前綴後的小寫文字）
    decoded = []
    for i in range(seq_len):
        token_text = tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
        token_clean = token_text.replace('▁', '').replace(' ', '').lower()
        decoded.append((i, token_text, token_clean))

    # 對每個 focus word 進行「連續子序列拼接匹配」
    # 這樣可以正確處理 SentencePiece 將單詞拆成多個 sub-token 的情況
    # 例：'Frog' → ['▁F', 'rog']，'frog'.startswith('f') 且 'f'+'rog'=='frog' → 匹配兩個 token
    for fw in focus_words_lower:
        fw_nospace = fw.replace(' ', '')
        matched_for_fw = False
        for start_i in range(len(decoded)):
            accumulated = ''
            span_indices = []
            for j in range(start_i, len(decoded)):
                _, tok_text, tok_clean = decoded[j]
                if not tok_clean:
                    continue
                accumulated += tok_clean
                span_indices.append(j)
                if accumulated == fw_nospace:
                    # 找到完整匹配的 token span
                    for idx in span_indices:
                        if idx not in focus_indices:
                            focus_indices.append(idx)
                    if verbose:
                        span_strs = [f"'{decoded[k][1]}'" for k in span_indices]
                        print(
                            f"  ✓ token[{span_indices[0]:3d}"
                            + (f"~{span_indices[-1]}" if len(span_indices) > 1 else "")
                            + f"] = {'+'.join(span_strs)} (clean='{accumulated}') "
                            f"← matched focus word '{fw}'"
                        )
                    matched_for_fw = True
                    break
                elif not fw_nospace.startswith(accumulated):
                    break  # 這條路無法湊出 focus word，提前結束
            if matched_for_fw:
                break  # 找到第一個匹配位置即可

    if not focus_indices:
        print(
            f"⚠️  警告：未找到任何 focus_words {focus_words} 的對應 token！"
            "將使用 fallback 機率替換。"
        )
    else:
        print(f"[Focus Token Search] 共找到 {len(focus_indices)} 個 focus token indices: {focus_indices}")

    return focus_indices


# ============================================================
# Attention 遮罩計算
# ============================================================

def compute_attention_mask_for_scale(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_idx: int,
    spatial_h: int,
    spatial_w: int,
    block_indices: List[int],
    threshold_percentile: float = 75.0,
    low_attn: bool = False,
) -> Optional[np.ndarray]:
    """
    計算指定 scale 的 attention-based 空間遮罩。

    流程：
        1. 收集 block_indices 中各 block 在 scale_idx 的 attention map
        2. 對 focus_token_indices 求平均（每個 focus token 取平均 attention map）
        3. 對所有 block 做 IQR 過濾後平均（去除離群的 block）
        4. 使用百分位數閾值二值化

    Args:
        extractor: 已記錄 source 生成 attention 的 CrossAttentionExtractor
        focus_token_indices: focus words 在 source prompt 中的 T5 token indices
        scale_idx: 要計算遮罩的 scale 索引（對應生成順序）
        spatial_h: 該 scale 的空間高度
        spatial_w: 該 scale 的空間寬度
        block_indices: 要從哪些 transformer block 擷取 attention
        threshold_percentile: 二值化閾值的百分位數（預設 75 = 前 25% 為高 attention）
        low_attn: True = 取最低 (100-threshold_percentile)% 作為 preserve 區域；
                  False（預設）= 取最高 (100-threshold_percentile)% 作為 focus 區域

    Returns:
        mask: (H, W) bool ndarray
            low_attn=False: True = 高 attention（focus 區域，不替換）
            low_attn=True:  True = 低 attention（preserve 區域，Phase 1.7 強制使用 source token）
        None 若無可用 attention map
    """
    all_block_attn_maps = []

    for block_idx in block_indices:
        attn_map = extractor.extract_word_attention(
            block_idx=block_idx,
            scale_idx=scale_idx,
            token_indices=focus_token_indices,
            spatial_size=(spatial_h, spatial_w),
        )
        if attn_map is not None:
            all_block_attn_maps.append(attn_map)

    if not all_block_attn_maps:
        print(f"  ⚠️  Scale {scale_idx}：無有效 attention map，跳過遮罩計算。")
        return None

    # 使用 IQR 過濾離群 block，增加 attention map 穩健性
    attn_stack = torch.tensor(np.stack(all_block_attn_maps), dtype=torch.float32)
    filtered_attn, num_outliers, num_used = _iqr_filtered_mean(attn_stack)

    # 百分位數閾值二值化
    if low_attn:
        # 取最低 (100 - threshold_percentile)% = 與 focus word 最不相關的背景區域
        low_threshold = np.percentile(filtered_attn, 100.0 - threshold_percentile)
        region = filtered_attn < low_threshold  # True = 極低 attention = preserve
        coverage_pct = region.mean() * 100
        print(
            f"  Scale {scale_idx}: "
            f"使用 {num_used}/{num_used + num_outliers} 個 block，"
            f"low threshold = {low_threshold:.4f}（{100.0 - threshold_percentile:.0f} pct），"
            f"preserve 區域佔比 = {coverage_pct:.1f}%"
        )
    else:
        threshold = np.percentile(filtered_attn, threshold_percentile)
        region = filtered_attn >= threshold  # True = 文字區域（高 attention）
        coverage_pct = region.mean() * 100
        print(
            f"  Scale {scale_idx}: "
            f"使用 {num_used}/{num_used + num_outliers} 個 block，"
            f"閾值 = {threshold:.4f}（{threshold_percentile:.0f} pct），"
            f"文字區域佔比 = {coverage_pct:.1f}%"
        )

    return region.astype(bool)


def compute_attention_heatmap_for_scale(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_idx: int,
    spatial_h: int,
    spatial_w: int,
    block_indices: List[int],
) -> Optional[np.ndarray]:
    """
    計算指定 scale 的 cross-attention heatmap（IQR 過濾後平均，不做二值化）。

    Returns:
        (H, W) float32 ndarray；若該 scale 無可用 attention map 則回傳 None。
    """
    all_block_attn_maps = []
    for block_idx in block_indices:
        attn_map = extractor.extract_word_attention(
            block_idx=block_idx,
            scale_idx=scale_idx,
            token_indices=focus_token_indices,
            spatial_size=(spatial_h, spatial_w),
        )
        if attn_map is not None:
            all_block_attn_maps.append(attn_map)

    if not all_block_attn_maps:
        return None

    attn_stack = torch.tensor(np.stack(all_block_attn_maps), dtype=torch.float32)
    filtered_attn, _, _ = _iqr_filtered_mean(attn_stack)
    return filtered_attn.astype(np.float32)


def _iqr_filtered_mean(
    attn_stack: torch.Tensor,
) -> Tuple[np.ndarray, int, int]:
    """
    IQR 過濾後的平均 attention（去除離群 block）。

    Args:
        attn_stack: (num_blocks, H, W)

    Returns:
        filtered_attn: (H, W) numpy float32
        num_outliers: 被移除的 block 數
        num_used: 保留的 block 數
    """
    num_blocks = attn_stack.shape[0]
    if num_blocks == 1:
        return attn_stack[0].numpy(), 0, 1

    attn_mean = attn_stack.mean(dim=0)  # (H, W)
    # 每個 block 與平均的 MSE，用來偵測離群 block
    mse = torch.sum((attn_stack - attn_mean.unsqueeze(0)) ** 2, dim=[1, 2])

    q1 = torch.quantile(mse, 0.25)
    q3 = torch.quantile(mse, 0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    keep = mse <= threshold
    num_outliers = int((~keep).sum().item())
    num_used = int(keep.sum().item())

    if num_used > 0:
        filtered_attn = attn_stack[keep].mean(dim=0).numpy()
    else:
        # 極端情況：全部被過濾，改用整體平均
        filtered_attn = attn_mean.numpy()
        num_used = num_blocks
        num_outliers = 0

    return filtered_attn, num_outliers, num_used


def build_and_store_attention_masks(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    p2p_token_storage: BitwiseTokenStorage,
    num_full_replace_scales: int,
    attn_block_indices: List[int],
    threshold_percentile: float = 75.0,
) -> int:
    """
    在 source 生成完畢後，為 scale >= num_full_replace_scales 的每個 scale
    計算 attention 遮罩並存入 BitwiseTokenStorage。

    Args:
        extractor: 已收集 source 生成 attention 的 CrossAttentionExtractor
        focus_token_indices: focus words 的 T5 token indices（來自 source prompt）
        scale_schedule: 完整的尺度排程列表
        p2p_token_storage: 儲存 token + 遮罩的物件
        num_full_replace_scales: 前幾個 scale 做 100% 替換（此函式只處理之後的）
        attn_block_indices: 用於計算遮罩的 transformer block index 列表
        threshold_percentile: attention 閾值百分位數

    Returns:
        成功存入遮罩的 scale 數量
    """
    print(f"\n[Attention 遮罩計算] 處理 scale {num_full_replace_scales} ~ {len(scale_schedule)-1} ...")
    masks_stored = 0

    for si in range(num_full_replace_scales, len(scale_schedule)):
        _, h, w = scale_schedule[si]

        text_mask = compute_attention_mask_for_scale(
            extractor=extractor,
            focus_token_indices=focus_token_indices,
            scale_idx=si,
            spatial_h=h,
            spatial_w=w,
            block_indices=attn_block_indices,
            threshold_percentile=threshold_percentile,
        )

        if text_mask is None:
            print(f"  Scale {si}: 跳過（無 attention 資料）")
            continue

        # 轉換為 PyTorch bool tensor：[1, 1, h, w, 1]
        # True = 非文字區域（背景） → 替換為 source token
        # False = 文字區域（高 attention）→ 保留 target 自由生成
        replacement_mask = ~text_mask  # 反轉：非文字區域才替換
        mask_tensor = torch.tensor(
            replacement_mask, dtype=torch.bool
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, h, w, 1]

        # 存入 storage（會繞過 num_scales 限制，直接寫入 masks dict）
        p2p_token_storage.masks[si] = mask_tensor.cpu()

        # 視覺化統計
        text_pct = text_mask.mean() * 100
        bg_pct = replacement_mask.mean() * 100
        print(
            f"  ✓ Scale {si} ({h}×{w}): "
            f"文字區域 {text_pct:.1f}%（不替換），"
            f"背景區域 {bg_pct:.1f}%（替換 source）"
        )
        masks_stored += 1

    print(f"[Attention 遮罩計算] 共儲存 {masks_stored} 個遮罩。\n")
    return masks_stored


def collect_attention_text_masks(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    num_full_replace_scales: int,
    attn_block_indices: List[int],
    threshold_percentile: float = 75.0,
    label: str = "source",
    low_attn: bool = False,
) -> Dict[int, np.ndarray]:
    """
    收集各 scale 的 attention mask，不存入 storage。

    Args:
        low_attn: False（預設）= 高 attention focus mask（True=focus 不替換）
                  True = 低 attention preserve mask（True=極背景 強制保留 source token）

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]
        low_attn=False: True = focus 區域 → Phase 2 不被 source token 覆蓋
        low_attn=True:  True = preserve 區域 → Phase 1.7 錨定為 source image token
    """
    text_masks: Dict[int, np.ndarray] = {}
    mode_label = "低 attention preserve" if low_attn else "高 attention focus"
    print(f"\n[Attention 遮罩計算 – {label} ({mode_label})] scale {num_full_replace_scales} ~ {len(scale_schedule)-1}")

    for si in range(num_full_replace_scales, len(scale_schedule)):
        _, h, w = scale_schedule[si]

        text_mask = compute_attention_mask_for_scale(
            extractor=extractor,
            focus_token_indices=focus_token_indices,
            scale_idx=si,
            spatial_h=h,
            spatial_w=w,
            block_indices=attn_block_indices,
            threshold_percentile=threshold_percentile,
            low_attn=low_attn,
        )

        if text_mask is not None:
            text_masks[si] = text_mask

    print(f"[Attention 遮罩計算 – {label}] 收集到 {len(text_masks)} 個 scale 的 mask。")
    return text_masks


def combine_and_store_masks(
    source_text_masks: Dict[int, np.ndarray],
    target_text_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    p2p_token_storage: BitwiseTokenStorage,
    num_full_replace_scales: int,
    external_filter_masks: Optional[Dict[int, np.ndarray]] = None,
) -> int:
    """
    合併 source 與 target focus mask，存入 storage 作為 replacement mask。

    合併邏輯（Union）：
        combined_focus    = source_focus | target_focus
        replacement_mask  = ~combined_focus
        → 只有在 source 和 target 兩者都沒有標記為 focus 的區域，才做 source token 替換。
        → 確保 target 生成的物件即使超出 source 的範圍，也不會被 source token 破壞。

    Returns:
        成功存入的 scale 數量
    """
    print(f"\n[合併遮罩] 合併 source + target focus mask → replacement mask ...")
    masks_stored = 0

    for si in range(num_full_replace_scales, len(scale_schedule)):
        src_mask = source_text_masks.get(si)
        tgt_mask = target_text_masks.get(si)
        ext_mask = external_filter_masks.get(si) if external_filter_masks else None

        if src_mask is None and tgt_mask is None and ext_mask is None:
            continue

        _, h, w = scale_schedule[si]

        # 缺少其中一方時，視為全 False（無 focus 區域）
        if src_mask is None:
            src_mask = np.zeros((h, w), dtype=bool)
        if tgt_mask is None:
            tgt_mask = np.zeros((h, w), dtype=bool)
        if ext_mask is not None and ext_mask.shape != (h, w):
            ext_mask_u8 = ext_mask.astype(np.uint8) * 255
            ext_mask = cv2.resize(ext_mask_u8, (w, h), interpolation=cv2.INTER_NEAREST) >= 128

        # Union：source 或 target 任一認為是 focus → 不替換
        combined_focus   = src_mask | tgt_mask
        replacement_mask = ~combined_focus  # True = 替換（背景）
        if ext_mask is not None:
            # 外部 mask 規則：白色=篩掉（不替換），黑色=保留（可替換）
            replacement_mask = replacement_mask & (~ext_mask)

        mask_tensor = torch.tensor(
            replacement_mask, dtype=torch.bool
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, h, w, 1]

        p2p_token_storage.masks[si] = mask_tensor.cpu()

        src_pct   = src_mask.mean() * 100
        tgt_pct   = tgt_mask.mean() * 100
        union_pct = combined_focus.mean() * 100
        bg_pct    = replacement_mask.mean() * 100
        filtered_pct = (ext_mask.mean() * 100) if ext_mask is not None else 0.0
        print(
            f"  ✓ Scale {si} ({h}×{w}): "
            f"src focus={src_pct:.1f}%, tgt focus={tgt_pct:.1f}%, "
            f"union（不替換）={union_pct:.1f}%, "
            + (f"外部白色篩掉={filtered_pct:.1f}%, " if ext_mask is not None else "")
            + f"替換={bg_pct:.1f}%"
        )
        masks_stored += 1

    print(f"[合併遮罩] 共儲存 {masks_stored} 個合併 replacement mask。\n")
    return masks_stored


def _save_masks_to_dir(
    bool_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    vis_dir: str,
    file_prefix: str,
    invert: bool = False,
) -> List[np.ndarray]:
    """
    將一組 bool mask 存為 PNG，並生成疊加灰階 overlay。

    Args:
        bool_masks : Dict[scale_idx -> (H, W) bool ndarray]
        invert     : True = 存 ~mask（白色代表 True 後反轉，方便顯示 replacement mask）

    Returns:
        collected  : 各 scale 已放大的 ndarray 列表
    """
    os.makedirs(vis_dir, exist_ok=True)
    collected: List[np.ndarray] = []

    for si, bool_mask in sorted(bool_masks.items()):
        _, h, w = scale_schedule[si]
        display_mask = (~bool_mask if invert else bool_mask).astype(np.uint8) * 255
        vis_size = max(256, h * 4, w * 4)
        mask_vis = cv2.resize(display_mask, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        vis_path = os.path.join(vis_dir, f"{file_prefix}_scale{si:02d}_{h}x{w}.png")
        cv2.imwrite(vis_path, mask_vis)
        collected.append(mask_vis)

    if collected:
        max_size = max(m.shape[0] for m in collected)
        stacked = np.zeros((max_size, max_size), dtype=np.float32)
        for m in collected:
            if m.shape[0] != max_size:
                m = cv2.resize(m, (max_size, max_size), interpolation=cv2.INTER_NEAREST)
            stacked += m.astype(np.float32)
        if stacked.max() > 0:
            stacked = stacked / stacked.max() * 255.0
        cv2.imwrite(os.path.join(vis_dir, "overlay.png"), stacked.astype(np.uint8))
        print(f"✓ [{file_prefix}] 遮罩視覺化已儲存至 {vis_dir}/ （{len(collected)} 個 scale + overlay）")

    return collected


def _dump_self_attn_cache_per_scale(
    extractor: SelfAttentionExtractor,
    scale_schedule: List[Tuple[int, int, int]],
    out_dir: str,
    scale_start: int = 0,
    scale_end: int = -1,
) -> int:
    """
    將 self-attention cache 依照「每個 scale 一個檔案」落盤。

    輸出格式：
      out_dir/
        scale_00.pt  # {scale_idx, block_attn: {block_idx: Tensor[1,H,Lq,Lk]}}
        scale_01.pt
        ...
        summary.json
    """
    os.makedirs(out_dir, exist_ok=True)

    if not extractor.attention_maps:
        print("⚠️  無 self-attention cache 可落盤。")
        return 0

    block_ids = sorted(extractor.attention_maps.keys())
    max_scales = max((len(extractor.attention_maps[b]) for b in block_ids), default=0)

    if max_scales == 0:
        print("⚠️  Self-attention cache 為空。")
        return 0

    start_idx = max(0, int(scale_start))
    end_idx = (max_scales - 1) if int(scale_end) < 0 else min(int(scale_end), max_scales - 1)
    if start_idx > end_idx:
        print(f"⚠️  scale 範圍無效：start={start_idx}, end={end_idx}")
        return 0
    saved_scales = 0

    summary = {
        "num_blocks": len(block_ids),
        "num_scales_detected": int(max_scales),
        "dump_scale_start": int(start_idx),
        "dump_scale_end": int(end_idx),
        "scale_schedule": [{"scale": i, "t": int(t), "h": int(h), "w": int(w)}
                           for i, (t, h, w) in enumerate(scale_schedule)],
        "blocks": {},
    }

    for bidx in block_ids:
        shapes = [list(attn.shape) for attn in extractor.attention_maps[bidx]]
        summary["blocks"][str(bidx)] = {
            "num_scales": len(shapes),
            "shapes": shapes,
        }

    for si in range(max_scales):
        block_attn: Dict[int, torch.Tensor] = {}
        for bidx in block_ids:
            if extractor.attention_maps[bidx]:
                # 依序 pop，可在落盤過程中釋放 RAM
                attn = extractor.attention_maps[bidx].pop(0)
                if start_idx <= si <= end_idx:
                    block_attn[bidx] = attn

        if not block_attn:
            continue

        pt_path = os.path.join(out_dir, f"scale_{si:02d}.pt")
        torch.save(
            {
                "scale_idx": si,
                "scale_info": scale_schedule[si] if si < len(scale_schedule) else None,
                "block_attn": block_attn,
            },
            pt_path,
        )
        saved_scales += 1

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"✓ Self-attention cache 已落盤：{out_dir}/ "
        f"（{saved_scales} 個 scale, 範圍 {start_idx}~{end_idx}）"
    )
    return saved_scales


def _load_binary_mask_white_fg(mask_path: str, threshold: float = 0.5) -> np.ndarray:
    """
    載入遮罩並二值化。

    規則：
    - 白色(255) = 篩掉（不替換）
    - 黑色(0)   = 保留（可替換）
    """
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"無法讀取 mask：{mask_path}")
    thr = int(max(0.0, min(1.0, threshold)) * 255)
    mask_bool = mask_gray >= thr
    if mask_bool.all():
        print("⚠️  提供的 mask 為全白。")
    elif (~mask_bool).all():
        print("⚠️  提供的 mask 為全黑。")
    return mask_bool


def _save_aligned_mask_per_scale(
    mask_bool: np.ndarray,
    scale_schedule: List[Tuple[int, int, int]],
    out_dir: str,
) -> int:
    """
    將原始 mask 以 nearest 對齊縮放到每個 scale，存出 exact + 視覺化版本。
    白色=篩掉（不替換），黑色=保留（可替換）。
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    src_h, src_w = mask_bool.shape
    print(f"[Mask 對齊] 原始 mask 尺寸：{src_h}x{src_w}（白=篩掉，黑=保留）")

    for si, (_, h, w) in enumerate(scale_schedule):
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        aligned = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

        exact_path = os.path.join(out_dir, f"scale_{si:02d}_{h}x{w}.png")
        cv2.imwrite(exact_path, aligned)

        vis_size = max(256, h * 8, w * 8)
        vis = cv2.resize(aligned, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
        vis_path = os.path.join(out_dir, f"scale_{si:02d}_{h}x{w}_vis.png")
        cv2.imwrite(vis_path, vis)
        saved += 1

    print(f"✓ Mask 對齊視覺化已儲存：{out_dir}/ （{saved} 個 scale）")
    return saved


def _build_external_filter_masks_per_scale(
    mask_bool: np.ndarray,
    scale_schedule: List[Tuple[int, int, int]],
    scale_start: int = 0,
) -> Dict[int, np.ndarray]:
    """
    將外部 mask 對齊到各 scale。

    回傳值語意：
    - True = 外部白色（篩掉，不替換）
    - False = 外部黑色（保留，可替換）
    """
    aligned: Dict[int, np.ndarray] = {}
    start_idx = max(0, int(scale_start))
    for si, (_, h, w) in enumerate(scale_schedule):
        if si < start_idx:
            continue
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        aligned[si] = resized >= 128
    return aligned


def _save_cross_attention_aligned_with_external_mask(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    attn_block_indices: List[int],
    num_full_replace_scales: int,
    out_dir: str,
    label: str,
    external_filter_masks: Optional[Dict[int, np.ndarray]] = None,
) -> int:
    """
    輸出 cross-attention 熱圖，並對齊外部 mask（白=篩掉，黑=保留）。
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for si in range(num_full_replace_scales, len(scale_schedule)):
        _, h, w = scale_schedule[si]
        heatmap = compute_attention_heatmap_for_scale(
            extractor=extractor,
            focus_token_indices=focus_token_indices,
            scale_idx=si,
            spatial_h=h,
            spatial_w=w,
            block_indices=attn_block_indices,
        )
        if heatmap is None:
            continue

        attn_min = float(heatmap.min())
        attn_max = float(heatmap.max())
        denom = max(attn_max - attn_min, 1e-8)
        attn_u8 = ((heatmap - attn_min) / denom * 255.0).clip(0, 255).astype(np.uint8)

        vis_size = max(256, h * 8, w * 8)
        attn_vis = cv2.resize(attn_u8, (vis_size, vis_size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_dir, f"{label}_cross_attn_scale_{si:02d}_{h}x{w}.png"), attn_vis)

        if external_filter_masks and si in external_filter_masks:
            ext = external_filter_masks[si].astype(np.uint8) * 255
            ext_vis = cv2.resize(ext, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
            heat_bgr = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)
            # 白色篩掉區域（不可替換）標白，便於檢查外部 mask 的阻擋範圍
            heat_bgr[ext_vis > 127] = np.array([255, 255, 255], dtype=np.uint8)
            cv2.imwrite(
                os.path.join(out_dir, f"{label}_cross_attn_scale_{si:02d}_{h}x{w}_masked.png"),
                heat_bgr,
            )

        saved += 1

    print(f"✓ [{label}] cross-attention 對齊圖已儲存：{out_dir}/ （{saved} 個 scale）")
    return saved


def _build_foreground_indices_from_external_mask(
    external_filter_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    scale_start: int = 0,
) -> Dict[int, List[int]]:
    """
    Build foreground token indices per scale for KV-Edit injection.

    External mask semantics:
      - True  (white) = background = frozen, use source K/V
      - False (black) = foreground = editing area, use target K/V

    Returns:
        Dict[scale_idx -> List[int]] of foreground (editing) flat token indices.
    """
    fg_indices: Dict[int, List[int]] = {}
    for si, ext_mask in external_filter_masks.items():
        if si < int(scale_start):
            continue
        # ext_mask: True=white=bg, False=black=fg
        fg_np = ~ext_mask   # foreground = NOT background
        idx = np.flatnonzero(fg_np.reshape(-1)).astype(np.int64).tolist()
        fg_indices[int(si)] = idx
    return fg_indices


# ============================================================
# Prompt 編碼工具
# ============================================================

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    """將 prompt 文字編碼為 T5 特徵（與 run_p2p.py 完全相同）"""
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(
        text=captions, max_length=512, padding='max_length',
        truncation=True, return_tensors='pt',
    )
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(
        input_ids=input_ids, attention_mask=mask
    )['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


def aug_with_positive_prompt(prompt):
    for key in [
        'man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person',
        'human', 'adult', 'teenager', 'employee', 'employer', 'worker',
        'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather',
        'son', 'daughter',
    ]:
        if key in prompt:
            prompt += '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def enhance_image(image):
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)
    return color_image


# ============================================================
# 圖像生成函式（P2P-Attn 版本）
# ============================================================

def gen_one_img(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    # P2P 基本參數
    p2p_token_storage=None,
    p2p_token_replace_prob=0.5,
    p2p_use_mask=False,
    p2p_save_tokens=True,
    # P2P-Attn 新增參數
    p2p_attn_full_replace_scales=0,
    # P2P-Edit 新增參數：連續 VAE feature 注入
    inject_image_features=None,
    # Tensor[1,d,1,H_full,W_full]，來自 encode_image_to_raw_features()
    inject_schedule=None,
    # List[float]，每個 scale 的生成權重
    # 0.0 = 100% source image；1.0 = 100% 自由生成
):
    """
    生成一張圖片（P2P-Edit 版本）。

    新增參數（相比 run_p2p_attn.py）：
        p2p_attn_full_replace_scales (int):
            前幾個 scale 做 100% source token 替換（離散層級）。
            0 = 禁用（image 模式下不需要，injection 已處理）。
        inject_image_features (Tensor[1,d,1,H,W]):
            encode_image_to_raw_features() 回傳的 raw VAE encoder 輸出。
        inject_schedule (List[float]):
            每個 scale 的生成權重。0.0 = 100% image，1.0 = 自由生成。
    """
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=g_seed,
            B=1,
            negative_label_B_or_BLT=negative_label_B_or_BLT,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=gumbel,
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            # P2P 參數
            p2p_token_storage=p2p_token_storage,
            p2p_token_replace_prob=p2p_token_replace_prob,
            p2p_use_mask=p2p_use_mask,
            p2p_save_tokens=p2p_save_tokens,
            # P2P-Attn 參數
            p2p_attn_full_replace_scales=p2p_attn_full_replace_scales,
            # P2P-Edit 參數：連續 VAE feature 注入
            inject_image_features=inject_image_features,
            inject_schedule=inject_schedule,
        )
    print(f"cost: {time.time() - sstt:.2f}s, infinity cost={time.time() - stt:.2f}s")
    img = img_list[0]
    del img_list
    return img


# ============================================================
# 模型載入工具
# ============================================================

def load_tokenizer(t5_path=''):
    """載入 T5 tokenizer 與 text encoder"""
    print(f'[載入 tokenizer 與 text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
        t5_path, revision=None, legacy=True
    )
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
        t5_path, torch_dtype=torch.float16
    )
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


def load_infinity(
    rope2d_each_sa_layer,
    rope2d_normalized_by_hw,
    use_scale_schedule_embedding,
    pn,
    use_bit_label,
    add_lvl_embeding_only_first_block,
    model_path='',
    scale_schedule=None,
    vae=None,
    device='cuda',
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
):
    """載入 P2P-Edit 版本的 Infinity 模型"""
    print(f'[載入 Infinity P2P-Edit 模型]')
    text_maxlen = 512
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae,
            text_channels=text_channels,
            text_maxlen=text_maxlen,
            shared_aln=True,
            raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(
            f'[Infinity P2P-Edit 模型大小: '
            f'{sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, '
            f'bf16={bf16}]'
        )

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[載入 Infinity 模型權重]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def encode_image_to_raw_features(
    vae,
    pil_img: PImage.Image,
    scale_schedule: List[Tuple[int, int, int]],
    device: torch.device,
    apply_spatial_patchify: bool = False,
) -> torch.Tensor:
    """
    將 source image 編碼為 VAE encoder 的連續特徵（不經量化）。

    與 gen.py / MaskFeatureProcessor.set_mask() 完全相同的做法：
      vae.encode_for_raw_features() → raw_features → unsqueeze(2)
    傳回 shape: [1, d, 1, H_full, W_full]，其中 H_full/W_full 為最大 latent 尺寸。

    注意：這是「無損」的連續特徵，不做量化 round-trip，
    因此在 summed_codes 層級做混合時效果遠優於先量化再解碼的方式。

    Args:
        vae:                   BSQ VAE 模型（已載入到 device）
        pil_img:               PIL.Image（RGB）
        scale_schedule:        List[(T, h, w)]，與推論使用的 scale_schedule 相同
        device:                torch.device
        apply_spatial_patchify: 是否使用 spatial patchify（與模型參數一致）

    Returns:
        Tensor[1, d, 1, H_full, W_full]  — 連續 VAE encoder 輸出（bfloat16）
    """
    _, h_final, w_final = scale_schedule[-1]
    if apply_spatial_patchify:
        h_final, w_final = h_final * 2, w_final * 2

    patch_size = 8 if apply_spatial_patchify else 16
    h_img = h_final * patch_size
    w_img = w_final * patch_size

    # Center-crop resize（與 MaskFeatureProcessor 使用 1024px 的邏輯相同）
    img_rgb = pil_img.convert('RGB')
    img_rgb = img_rgb.resize((w_img, h_img), resample=PImage.LANCZOS)

    img_t = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0          # [0,1] → [-1,1]
    img_t = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]

    if apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            # 直接取 VAE encoder 的連續輸出，不做量化
            raw_features, _, _ = vae.encode_for_raw_features(
                img_t, scale_schedule=vae_scale_schedule
            )  # [1, d, H_full, W_full]

    # 加入 temporal 維度以符合模型內部 5D 格式 [B, d, 1, H, W]
    f = raw_features.unsqueeze(2).detach()  # [1, d, 1, H_full, W_full]
    print(
        f"[encode_image_to_raw_features] shape={tuple(f.shape)}, "
        f"dtype={f.dtype}, "
        f"range=[{f.min():.3f}, {f.max():.3f}]"
    )
    return f


def encode_image_to_scale_tokens(
    vae,
    pil_img: PImage.Image,
    scale_schedule: List[Tuple[int, int, int]],
    device: torch.device,
    apply_spatial_patchify: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    將 source image 直接量化為各 scale 的離散 BSQ bit token（透過 vae.encode()）。

    與 encode_image_to_raw_features() 的差異：
      ── raw_features：連續特徵，用於 summed_codes 混合（影響 source gen 的解碼路徑）
      ── scale_tokens：離散 bit 索引，直接用於 P2P token 替換（target gen 的 idx_Bld）

    設計目的：
      原本 P2P 替換使用的是 source prompt 生成過程的採樣 token，
      這些 token 受到 source prompt 語義影響，不能精確反映 source image 的原始內容。
      此函式改用 source image → VAE encoder → 量化器 的路徑直接取得離散 token，
      讓 target gen 的 P2P 替換能真正參考 source image 的像素內容。

    Returns:
        Dict[si -> Tensor[1, 1, h_si, w_si, d_vae]] — 各 scale 的 bit indices (long, CPU)
        （apply_spatial_patchify=True 時 h, w 為 vae_scale_schedule 的尺寸）
    """
    if apply_spatial_patchify:
        vae_scale_schedule = [(t, h * 2, w * 2) for (t, h, w) in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    _, h_final_vae, w_final_vae = vae_scale_schedule[-1]
    patch_size = 8 if apply_spatial_patchify else 16
    h_img = h_final_vae * patch_size
    w_img = w_final_vae * patch_size

    img_rgb = pil_img.convert('RGB')
    img_rgb = img_rgb.resize((w_img, h_img), resample=PImage.LANCZOS)

    img_t = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            # vae.encode() 回傳: (h, z, all_indices, all_bit_indices, residual_norm, var_input)
            _, _, _, all_bit_indices, _, _ = vae.encode(
                img_t, scale_schedule=vae_scale_schedule
            )

    scale_tokens: Dict[int, torch.Tensor] = {}
    for si, bit_idx in enumerate(all_bit_indices):
        # bit_idx shape: [1, 1, h_si, w_si, d_vae]（與 idx_Bld 相同格式）
        scale_tokens[si] = bit_idx.long().cpu()

    print(
        f"[encode_image_to_scale_tokens] 共編碼 {len(scale_tokens)} 個 scale，"
        f"範例 scale 0: shape={tuple(scale_tokens[0].shape)}, "
        f"dtype={scale_tokens[0].dtype}"
    )
    return scale_tokens


def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.vae_type in [14, 16, 18, 20, 24, 32, 64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2 ** codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult = [1, 2, 4, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4, 4]
        vae = vae_model(
            args.vae_path, schedule_mode, codebook_dim, codebook_size,
            patch_size=patch_size, encoder_ch_mult=encoder_ch_mult,
            decoder_ch_mult=decoder_ch_mult, test_mode=True,
        ).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae


def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch':
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        shutil.copyfile(model_path, local_model_path)
                    # save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    else:
        raise ValueError(f'未知模型類型: {args.model_type}')

    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=slim_model_path,
        scale_schedule=None,
        vae=vae,
        device=device,
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )
    return infinity


def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1, 2, 4, 8, 16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0, 1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0, 1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0, 1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0, 1])


# ============================================================
# 主程式 ── KV-Edit on Infinity
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'KV-Edit 圖像編輯管線\n'
            '\n'
            '核心原理（KV-Edit paper 2502.17363）：\n'
            '  Phase 1 - Source forward：擷取 source image 每個 transformer block、\n'
            '            每個 scale 的 self-attention K/V，作為 frozen background memory。\n'
            '  Phase 2 - Target forward：\n'
            '    - Background tokens（mask 白色）：使用 source K/V（frozen）\n'
            '    - Foreground tokens（mask 黑色）：使用 target 當前 K/V（自由生成）\n'
            '    - 所有 token 的 Q 均 attend 到此合併後的 K/V\n'
            '    → 無 bitwise token 替換，純粹 K/V memory 重用。\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)

    # ── 基本參數 ──
    parser.add_argument('--source_prompt', type=str, required=True,
                        help='描述 source 圖像內容的 prompt')
    parser.add_argument('--target_prompt', type=str, required=True,
                        help='描述 target 圖像（編輯後）的 prompt')
    parser.add_argument('--save_file', type=str,
                        default='./outputs/outputs_loop_exp/kv_edit',
                        help='輸出目錄')

    # ── Mask（背景/前景分割）──
    parser.add_argument('--mask_image', type=str, default='',
                        help=(
                            '前景/背景 mask 路徑（PNG/JPG）。\n'
                            '  白色（255）= background（frozen，使用 source K/V）\n'
                            '  黑色（  0）= foreground（editing，使用 target K/V）\n'
                            '若未提供，全部視為 foreground → 等同純 target 生成。'
                        ))
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='mask 二值化閾值（0~1，預設 0.5）')

    # ── Source Image（可選：VAE feature 注入）──
    parser.add_argument('--source_image', type=str, default=None,
                        help=(
                            '可選 source image 路徑（PNG/JPG）。\n'
                            '若提供，Phase 1 會用 VAE encoder 連續特徵輔助 source gen。'
                        ))
    parser.add_argument('--image_injection_scales', type=int, default=2,
                        help='前幾個 scale 使用 source image 注入（weight=0）。預設：2')
    parser.add_argument('--inject_weights', type=str, default='',
                        help='各 scale 注入強度（空格分隔，0.0=100%% source，1.0=自由生成）')

    # ── KV 注入範圍 ──
    parser.add_argument('--kv_scale_start', type=int, default=0,
                        help='KV 擷取/注入起始 scale（含）。預設：0')
    parser.add_argument('--kv_scale_end', type=int, default=-1,
                        help='KV 擷取/注入結束 scale（含，-1 = 最後一個）。預設：-1')
    parser.add_argument('--attn_batch_idx', type=int, default=0,
                        help='CFG batch 中哪個 batch 捕捉 K/V（0=conditioned）')

    # ── 診斷：僅跑 source（不做 target）──
    parser.add_argument('--source_only', type=int, default=0, choices=[0, 1],
                        help='1 = 只生成 source image，不做 target 編輯（除錯用）')

    args = parser.parse_args()

    # 解析 cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    save_dir = os.path.abspath(args.save_file)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("KV-Edit 圖像編輯管線（self-attention K/V memory reuse）")
    print("=" * 80)
    print(f"Source prompt  : {args.source_prompt}")
    print(f"Target prompt  : {args.target_prompt}")
    print(f"Mask image     : {args.mask_image or '（未提供 → 全部視為 foreground）'}")
    print(f"Source image   : {args.source_image or '（未提供）'}")
    print(f"KV scale range : {args.kv_scale_start} ~ {'last' if args.kv_scale_end < 0 else args.kv_scale_end}")
    print(f"Output dir     : {save_dir}")
    print("=" * 80 + "\n")

    # ── 載入模型 ──
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    # ── Scale Schedule ──
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)
    depth = len(infinity.unregistered_blocks)
    all_block_indices = list(range(depth))
    print(f"[Scale Schedule] 共 {total_scales} 個 scale")
    print(f"[Model depth]    {depth} blocks")

    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 載入外部 mask ──
    external_filter_masks: Dict[int, np.ndarray] = {}
    if args.mask_image:
        external_mask_bool = _load_binary_mask_white_fg(args.mask_image,
                                                         threshold=args.mask_threshold)
        is_all_white = bool(external_mask_bool.all())
        is_all_black = bool((~external_mask_bool).all())
        if is_all_white or is_all_black:
            print("[Mask] 全白或全黑，視為無效，不做 KV-Edit 分割")
        else:
            external_filter_masks = _build_external_filter_masks_per_scale(
                mask_bool=external_mask_bool,
                scale_schedule=scale_schedule,
                scale_start=args.kv_scale_start,
            )
            _save_aligned_mask_per_scale(
                mask_bool=external_mask_bool,
                scale_schedule=scale_schedule,
                out_dir=os.path.join(save_dir, "mask_aligned"),
            )
            fg_coverage = (~external_mask_bool).mean() * 100
            bg_coverage = external_mask_bool.mean() * 100
            print(f"[Mask] 已載入：foreground（黑）= {fg_coverage:.1f}%, "
                  f"background（白）= {bg_coverage:.1f}%")
    else:
        print("[Mask] 未提供 --mask_image：所有 token 視為 foreground（等同純 target 生成）")

    # ── 建立 foreground token 索引 ──
    # 有 mask → fg from black pixels；無 mask → 空 dict（inject_kv 不改 K/V，等同原始生成）
    fg_indices_per_scale: Dict[int, List[int]] = {}
    if external_filter_masks:
        fg_indices_per_scale = _build_foreground_indices_from_external_mask(
            external_filter_masks=external_filter_masks,
            scale_schedule=scale_schedule,
            scale_start=args.kv_scale_start,
        )
        total_fg_scales = len(fg_indices_per_scale)
        print(f"[FG Indices] 共 {total_fg_scales} 個 scale 有 foreground 索引")
        for si, idxs in sorted(fg_indices_per_scale.items())[:3]:
            _, h, w = scale_schedule[si]
            pct = len(idxs) / (h * w) * 100 if h * w > 0 else 0
            print(f"  Scale {si} ({h}x{w}): {len(idxs)} fg tokens ({pct:.1f}%)")

    # ─────────────────────────────────────────────────────
    # Phase 0：載入並編碼 Source Image（若有提供）
    # ─────────────────────────────────────────────────────
    image_raw_features: Optional[torch.Tensor] = None
    inject_schedule: Optional[list] = None

    if args.source_image is not None:
        print("\n" + "=" * 60)
        print("[Phase 0] 載入 Source Image 並編碼")
        print("=" * 60)
        source_pil_img = Image.open(args.source_image).convert('RGB')
        print(f"Source image: {args.source_image}  (原始尺寸={source_pil_img.size})")

        image_raw_features = encode_image_to_raw_features(
            vae=vae,
            pil_img=source_pil_img,
            scale_schedule=scale_schedule,
            device=device_cuda,
            apply_spatial_patchify=bool(args.apply_spatial_patchify),
        )
        print(f"✓ Source image raw features: {tuple(image_raw_features.shape)}")

        if args.inject_weights.strip():
            parsed_w = [float(w) for w in args.inject_weights.strip().split()]
            if len(parsed_w) != total_scales:
                raise ValueError(
                    f"--inject_weights 長度 ({len(parsed_w)}) 與 scale 總數 ({total_scales}) 不符。"
                )
            inject_schedule = parsed_w
        else:
            inject_schedule = [
                0.0 if si < args.image_injection_scales else 1.0
                for si in range(total_scales)
            ]
        inj_str = '  '.join([f"s{si}={w:.2f}" for si, w in enumerate(inject_schedule)])
        print(f"[Injection Schedule] {inj_str}")
    else:
        print("\n[Phase 0] 無 source image → source gen 使用純 source prompt")

    # ─────────────────────────────────────────────────────
    # Phase 1：Source 生成 + Self-Attention K/V 擷取
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1] Source 圖像生成 + K/V 擷取")
    print(f"  KV 擷取範圍：scale {args.kv_scale_start} ~ "
          f"{'last' if args.kv_scale_end < 0 else args.kv_scale_end}")
    print("=" * 60)

    source_kv_extractor = SelfAttentionExtractor(
        model=infinity,
        block_indices=all_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
        capture_attention=False,   # don't need to store attention maps
        capture_kv=True,
        capture_scale_start=args.kv_scale_start,
        capture_scale_end=args.kv_scale_end,
    )
    source_kv_extractor.register_patches()

    # Phase 1 生成：不做任何 token 替換，僅擷取 K/V
    dummy_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                args.source_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                p2p_token_storage=dummy_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    source_kv_extractor.remove_patches()
    source_kv_maps = source_kv_extractor.kv_maps

    total_kv_entries = sum(len(v) for v in source_kv_maps.values())
    captured_scales = (
        len(next(iter(source_kv_maps.values()))) if source_kv_maps else 0
    )
    print(f"✓ K/V 已擷取：{len(source_kv_maps)} blocks × {captured_scales} scales "
          f"= {total_kv_entries} 筆")

    source_save_path = os.path.join(save_dir, "source.jpg")
    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(source_save_path, img_np)
    print(f"✓ Source 圖像已儲存：{source_save_path}")

    del source_image, img_np, dummy_storage
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if args.source_only:
        print("\n[source_only=1] 完成 source 生成，跳過 target 生成。")
        exit(0)

    # ─────────────────────────────────────────────────────
    # Phase 2：Target 生成 + KV-Edit 注入
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 2] Target 圖像生成（KV-Edit 注入）")
    print("  Background（mask 白色）→ 使用 source K/V（frozen memory）")
    print("  Foreground（mask 黑色）→ 使用 target 當前 K/V（自由生成）")
    print("  所有 Q → attend to merged K/V（KV-Edit 核心機制）")
    print("=" * 60)

    if not source_kv_maps:
        print("⚠️  source_kv_maps 為空，Phase 2 將執行純 target 生成（無 KV 注入）")

    kv_injector = SelfAttentionExtractor(
        model=infinity,
        block_indices=all_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
        capture_attention=False,
        capture_kv=False,
        inject_kv=True,
        source_kv_maps=source_kv_maps,
        foreground_indices_per_scale=fg_indices_per_scale,
        capture_scale_start=args.kv_scale_start,
        capture_scale_end=args.kv_scale_end,
    )
    kv_injector.register_patches()

    dummy_storage2 = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                args.target_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                # 不做 bitwise token 替換
                p2p_token_storage=dummy_storage2,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=0,
                inject_image_features=None,
                inject_schedule=None,
            )

    kv_injector.remove_patches()

    target_save_path = os.path.join(save_dir, "target.jpg")
    img_np = target_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(target_save_path, img_np)
    print(f"✓ Target 圖像已儲存：{target_save_path}")

    del target_image, img_np, dummy_storage2
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("KV-Edit 管線完成")
    print("=" * 80)
    print(f"結果儲存於: {save_dir}/")
    print(f"  - source.jpg    Source 重建圖像")
    print(f"  - target.jpg    KV-Edit 編輯後圖像")
    if external_filter_masks:
        print(f"  - mask_aligned/ 各 scale 對齊 mask")
    print("=" * 80 + "\n")

