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
import difflib
import argparse
import shutil
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

# 使用 P2P-Edit 版本的模型
from infinity.models.infinity_p2p_edit import Infinity
from infinity.models.basic import *
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

# CrossAttentionExtractor：從 source 生成過程捕捉 attention map
from attention_map.extractor import CrossAttentionExtractor


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


def _tokenize_prompt_words(prompt: str) -> List[str]:
    """將 prompt 轉為詞級序列（小寫，去除標點）。"""
    return re.findall(r"[\w']+", prompt.lower(), flags=re.UNICODE)


def derive_focus_terms_from_prompt_diff(
    source_prompt: str,
    target_prompt: str,
) -> Tuple[List[str], List[str]]:
    """
    由 source/target prompt 的詞級差異，提取各自應關注的片段。

    Returns:
        (source_terms, target_terms)
    """
    src_words = _tokenize_prompt_words(source_prompt)
    tgt_words = _tokenize_prompt_words(target_prompt)
    matcher = difflib.SequenceMatcher(None, src_words, tgt_words)

    src_terms: List[str] = []
    tgt_terms: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete') and i2 > i1:
            term = ' '.join(src_words[i1:i2]).strip()
            if term:
                src_terms.append(term)
        if tag in ('replace', 'insert') and j2 > j1:
            term = ' '.join(tgt_words[j1:j2]).strip()
            if term:
                tgt_terms.append(term)
    return src_terms, tgt_terms


def parse_focus_words_arg(text: str) -> List[str]:
    """
    解析 focus words 文字：
      - 含逗號：視為多個 phrase
      - 否則：保留整段 phrase，並額外拆詞作為 fallback（提高匹配率）
    """
    raw = (text or '').strip()
    if not raw:
        return []

    if ',' in raw:
        terms = [seg.strip() for seg in raw.split(',') if seg.strip()]
    else:
        terms = [raw]
        terms.extend([w for w in raw.split() if w.strip()])

    dedup: List[str] = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key not in seen:
            dedup.append(term)
            seen.add(key)
    return dedup


def merge_focus_terms(base_terms: List[str], extra_terms: List[str]) -> List[str]:
    """合併兩組 focus terms，忽略大小寫去重。"""
    merged = list(base_terms)
    seen = {t.lower() for t in merged}
    for term in extra_terms:
        k = term.lower()
        if k not in seen:
            merged.append(term)
            seen.add(k)
    return merged


# ============================================================
# Attention 遮罩計算
# ============================================================

from infinity.utils.adaptiveThreshold import compute_threshold as _compute_adaptive_threshold
from infinity.utils.adaptiveThreshold import METHOD_NAMES as _THRESHOLD_METHOD_NAMES

def compute_attention_mask_for_scale(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_idx: int,
    spatial_h: int,
    spatial_w: int,
    block_indices: List[int],
    threshold_percentile: float = 75.0,
    low_attn: bool = False,
    use_normalized_attn: bool = False,
    threshold_method: int = 1,
    source_image_np: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
    dynamic_max_iters: int = 20,
) -> Optional[np.ndarray]:
    """
    計算指定 scale 的 attention-based 空間遮罩。

    流程：
        1. 收集 block_indices 中各 block 在 scale_idx 的 attention map
        2. 對 focus_token_indices 求平均（每個 focus token 取平均 attention map）
        3. 對所有 block 做 IQR 過濾後平均（去除離群的 block）
        4. 呼叫 compute_threshold() 依 threshold_method 二值化

    Args:
        extractor: 已記錄 source 生成 attention 的 CrossAttentionExtractor
        focus_token_indices: focus words 在 source prompt 中的 T5 token indices
        scale_idx: 要計算遮罩的 scale 索引（對應生成順序）
        spatial_h: 該 scale 的空間高度
        spatial_w: 該 scale 的空間寬度
        block_indices: 要從哪些 transformer block 擷取 attention
        threshold_percentile: 二值化閾值的百分位數（方法 1/5 使用）
        low_attn: True = 取低 attention 作為 preserve 區域
        use_normalized_attn: True = 使用 z-score normalized threshold（僅方法 1 時生效）
        threshold_method: 1~8，對應 adaptiveThreshold 中的 8 種策略
        source_image_np: (H, W, 3) uint8 source image（方法 6/8 需要）
        ref_mask: [H_ref, W_ref] bool reference mask（方法 2 需要）
        dynamic_max_iters: 方法 2 ternary search 迭代次數

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

    # ── 使用 adaptiveThreshold 模組計算閾值 ──
    # 特殊情況：use_normalized_attn + method==1 時沿用舊的 z-score 邏輯
    if use_normalized_attn and threshold_method == 1:
        mu = filtered_attn.mean()
        sigma = filtered_attn.std()
        if sigma < 1e-8:
            print(
                f"  Scale {scale_idx}: "
                f"使用 {num_used}/{num_used + num_outliers} 個 block，"
                f"normalized attn: std≈0，跳過（全 False mask）"
            )
            return np.zeros((spatial_h, spatial_w), dtype=bool)
        normalized = (filtered_attn - mu) / sigma
        if low_attn:
            region = normalized < filtered_attn
        else:
            region = normalized > filtered_attn
        coverage_pct = region.mean() * 100
        mode_str = "preserve" if low_attn else "focus"
        print(
            f"  Scale {scale_idx}: "
            f"使用 {num_used}/{num_used + num_outliers} 個 block，"
            f"normalized attn ({mode_str}): μ={mu:.4f} σ={sigma:.4f}，"
            f"coverage = {coverage_pct:.1f}%"
        )
        return region.astype(bool)

    # 一般路徑：呼叫統一的 compute_threshold
    thr, processed_attn, info = _compute_adaptive_threshold(
        filtered_attn=filtered_attn,
        method=threshold_method,
        low_attn=low_attn,
        percentile=threshold_percentile,
        ref_mask=ref_mask,
        max_iters=dynamic_max_iters,
        fallback_percentile=threshold_percentile,
        source_image=source_image_np,
    )

    # 二值化
    if low_attn:
        region = processed_attn < thr
    else:
        region = processed_attn >= thr
    coverage_pct = region.mean() * 100

    print(
        f"  Scale {scale_idx}: "
        f"使用 {num_used}/{num_used + num_outliers} 個 block，"
        f"[method {threshold_method}] {info}，coverage = {coverage_pct:.1f}%"
    )

    return region.astype(bool)


def compute_attention_mask_dynamic_threshold(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_idx: int,
    spatial_h: int,
    spatial_w: int,
    block_indices: List[int],
    ref_mask: np.ndarray,
    max_iters: int = 20,
    low_attn: bool = False,
    fallback_percentile: float = 80.0,
    threshold_method: int = 2,
    source_image_np: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    使用 reference mask 引導或其他自適應閾值方法計算 attention mask。

    當 threshold_method == 2 時使用 ternary search + ref_mask（原行為）。
    其他 method 值會透過 compute_attention_mask_for_scale 處理。

    Args:
        ref_mask: [H_ref, W_ref] bool, True = 編輯區域
        max_iters: 三分搜尋最大迭代次數（T in paper）
        low_attn: True = preserve mask（低 attention 背景區域，比對 ~ref_mask）
        fallback_percentile: 當 ref_mask 全空或搜尋失敗時的 fallback
        threshold_method: 1~8，閾值策略
        source_image_np: (H, W, 3) uint8（方法 6/8 需要）

    Returns:
        mask: (spatial_h, spatial_w) bool ndarray
        None 若無可用 attention map
    """
    # 非 method 2 → 直接委派給 compute_attention_mask_for_scale
    if threshold_method != 2:
        return compute_attention_mask_for_scale(
            extractor=extractor,
            focus_token_indices=focus_token_indices,
            scale_idx=scale_idx,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            block_indices=block_indices,
            threshold_percentile=fallback_percentile,
            low_attn=low_attn,
            use_normalized_attn=False,
            threshold_method=threshold_method,
            source_image_np=source_image_np,
            ref_mask=ref_mask,
            dynamic_max_iters=max_iters,
        )

    # ── method 2: 原始 ternary search 路徑 ──
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
    filtered_attn, num_outliers, num_used = _iqr_filtered_mean(attn_stack)

    # 呼叫 adaptiveThreshold 的 method 2
    thr, processed_attn, info = _compute_adaptive_threshold(
        filtered_attn=filtered_attn,
        method=2,
        low_attn=low_attn,
        ref_mask=ref_mask,
        max_iters=max_iters,
        fallback_percentile=fallback_percentile,
    )

    # 二值化
    if low_attn:
        region = processed_attn < thr
    else:
        region = processed_attn >= thr
    coverage_pct = region.mean() * 100

    print(
        f"  Scale {scale_idx}: "
        f"{num_used}/{num_used + num_outliers} blocks，"
        f"[method 2] {info}，coverage={coverage_pct:.1f}%"
    )
    return region.astype(bool)


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
    use_normalized_attn: bool = False,
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
            use_normalized_attn=use_normalized_attn,
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
    use_normalized_attn: bool = False,
    threshold_method: int = 1,
    source_image_np: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
    dynamic_max_iters: int = 20,
) -> Dict[int, np.ndarray]:
    """
    收集各 scale 的 attention mask，不存入 storage。

    Args:
        low_attn: False（預設）= 高 attention focus mask（True=focus 不替換）
                  True = 低 attention preserve mask（True=極背景 強制保留 source token）
        threshold_method: 1~8，閾值策略
        source_image_np: (H, W, 3) uint8（方法 6/8 需要）
        ref_mask: [H_ref, W_ref] bool（方法 2 需要）
        dynamic_max_iters: 方法 2 迭代次數

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]
        low_attn=False: True = focus 區域 → Phase 2 不被 source token 覆蓋
        low_attn=True:  True = preserve 區域 → Phase 1.7 錨定為 source image token
    """
    text_masks: Dict[int, np.ndarray] = {}
    method_name = _THRESHOLD_METHOD_NAMES.get(threshold_method, f"method{threshold_method}")
    mode_label = "低 attention preserve" if low_attn else "高 attention focus"
    print(f"\n[Attention 遮罩計算 – {label} ({mode_label}, {method_name})] scale {num_full_replace_scales} ~ {len(scale_schedule)-1}")

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
            use_normalized_attn=use_normalized_attn,
            threshold_method=threshold_method,
            source_image_np=source_image_np,
            ref_mask=ref_mask,
            dynamic_max_iters=dynamic_max_iters,
        )

        if text_mask is not None:
            text_masks[si] = text_mask

    print(f"[Attention 遮罩計算 – {label}] 收集到 {len(text_masks)} 個 scale 的 mask。")
    return text_masks


def collect_attention_text_masks_dynamic(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    num_full_replace_scales: int,
    attn_block_indices: List[int],
    ref_mask: np.ndarray,
    max_iters: int = 20,
    fallback_percentile: float = 80.0,
    label: str = "source",
    low_attn: bool = False,
    threshold_method: int = 2,
    source_image_np: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    收集各 scale 的 attention mask（支援 dynamic threshold + 其他方法）。

    Args:
        ref_mask: [H_ref, W_ref] bool, True = 編輯區域（PIE-Bench GT mask）
        max_iters: ternary search 最大迭代次數
        fallback_percentile: ref_mask 不可用時的 fallback percentile
        threshold_method: 1~8，閾值策略
        source_image_np: (H, W, 3) uint8（方法 6/8 需要）

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]
    """
    text_masks: Dict[int, np.ndarray] = {}
    method_name = _THRESHOLD_METHOD_NAMES.get(threshold_method, f"method{threshold_method}")
    mode_label = "低 attention preserve" if low_attn else "高 attention focus"
    print(f"\n[Threshold – {label} ({mode_label}, {method_name})] scale {num_full_replace_scales} ~ {len(scale_schedule)-1}")

    for si in range(num_full_replace_scales, len(scale_schedule)):
        _, h, w = scale_schedule[si]

        text_mask = compute_attention_mask_dynamic_threshold(
            extractor=extractor,
            focus_token_indices=focus_token_indices,
            scale_idx=si,
            spatial_h=h,
            spatial_w=w,
            block_indices=attn_block_indices,
            ref_mask=ref_mask,
            max_iters=max_iters,
            low_attn=low_attn,
            fallback_percentile=fallback_percentile,
            threshold_method=threshold_method,
            source_image_np=source_image_np,
        )

        if text_mask is not None:
            text_masks[si] = text_mask

    print(f"[Threshold – {label}] 收集到 {len(text_masks)} 個 scale 的 mask。")
    return text_masks


def downsample_mask_majority_vote(
    mask_fine: np.ndarray,
    h_coarse: int,
    w_coarse: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    將高解析度 bool mask 降採樣到較低解析度，使用多數投票法。

    對於目標 (h_coarse, w_coarse) 中的每個 patch，計算其在 mask_fine 中
    對應區域的 True 比例。若比例 > threshold，則該 patch 為 True。

    Args:
        mask_fine:  (H_fine, W_fine) bool ndarray
        h_coarse:   目標高度
        w_coarse:   目標寬度
        threshold:  多數投票閾值（預設 0.5 = 50%）

    Returns:
        (h_coarse, w_coarse) bool ndarray
    """
    h_fine, w_fine = mask_fine.shape
    if h_fine == h_coarse and w_fine == w_coarse:
        return mask_fine.copy()

    result = np.zeros((h_coarse, w_coarse), dtype=bool)
    for i in range(h_coarse):
        # 計算該 patch 在 fine mask 中對應的行範圍
        r_start = int(round(i * h_fine / h_coarse))
        r_end   = int(round((i + 1) * h_fine / h_coarse))
        r_end   = max(r_end, r_start + 1)  # 至少涵蓋一行
        r_end   = min(r_end, h_fine)
        for j in range(w_coarse):
            c_start = int(round(j * w_fine / w_coarse))
            c_end   = int(round((j + 1) * w_fine / w_coarse))
            c_end   = max(c_end, c_start + 1)
            c_end   = min(c_end, w_fine)
            patch = mask_fine[r_start:r_end, c_start:c_end]
            ratio = patch.mean()
            result[i, j] = ratio > threshold
    return result


def propagate_mask_backward(
    last_scale_mask: np.ndarray,
    scale_schedule: List[Tuple[int, int, int]],
    start_scale: int,
    majority_threshold: float = 0.5,
) -> Dict[int, np.ndarray]:
    """
    從最後一個 scale 的 attention mask 向前（粗）逐步推導各 scale 的 mask。

    流程：
        last_scale → second-to-last → ... → start_scale
        每一步使用 downsample_mask_majority_vote 將前一個（較細）scale 的
        mask 降採樣到當前 scale 的解析度。

    Args:
        last_scale_mask:    最後一個 scale 的 bool mask (H_last, W_last)
        scale_schedule:     完整尺度排程 [(t, h, w), ...]
        start_scale:        往回推的終止 scale index（通常 = image_injection_scales）
        majority_threshold: 多數投票閾值

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]，包含 start_scale 到 last_scale 的所有 mask
    """
    total = len(scale_schedule)
    last_idx = total - 1
    masks: Dict[int, np.ndarray] = {last_idx: last_scale_mask}

    # 從倒數第二個 scale 往回推到 start_scale
    current_mask = last_scale_mask
    for si in range(last_idx - 1, start_scale - 1, -1):
        _, h_coarse, w_coarse = scale_schedule[si]
        current_mask = downsample_mask_majority_vote(
            mask_fine=current_mask,
            h_coarse=h_coarse,
            w_coarse=w_coarse,
            threshold=majority_threshold,
        )
        masks[si] = current_mask
        _, h_fine, w_fine = scale_schedule[si + 1] if si + 1 < total else scale_schedule[si]
        print(
            f"  Scale {si} ({h_coarse}×{w_coarse}): "
            f"從 scale {si + 1} 推導，focus 區域佔比 = {current_mask.mean() * 100:.1f}%"
        )

    return masks


def collect_last_scale_attention_mask(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    start_scale: int,
    attn_block_indices: List[int],
    threshold_percentile: float = 75.0,
    label: str = "source",
    low_attn: bool = False,
    use_normalized_attn: bool = False,
    majority_threshold: float = 0.5,
    threshold_method: int = 1,
    source_image_np: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
    dynamic_max_iters: int = 20,
) -> Dict[int, np.ndarray]:
    """
    僅從最後一個 scale 提取 cross-attention mask，再向前逐步推導各 scale 的 mask。

    Args:
        extractor:             已記錄 attention 的 CrossAttentionExtractor
        focus_token_indices:   focus words 的 T5 token indices
        scale_schedule:        完整尺度排程
        start_scale:           往回推的終止 scale（通常 = num_full_replace_scales）
        attn_block_indices:    用於計算遮罩的 block indices
        threshold_percentile:  二值化閾值百分位數
        label:                 標籤（用於 print）
        low_attn:              True = 取低 attention preserve mask
        use_normalized_attn:   True = 使用 z-score normalized threshold
        majority_threshold:    多數投票降採樣閾值（預設 0.5）
        threshold_method:      1~8，閾值策略
        source_image_np:       (H, W, 3) uint8（方法 6/8 需要）
        ref_mask:              [H_ref, W_ref] bool（方法 2 需要）
        dynamic_max_iters:     方法 2 迭代次數

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]
    """
    last_idx = len(scale_schedule) - 1
    _, h_last, w_last = scale_schedule[last_idx]

    mode_label = "低 attention preserve" if low_attn else "高 attention focus"
    print(f"\n[Last-Scale Mask – {label} ({mode_label})] "
          f"從 scale {last_idx} ({h_last}×{w_last}) 提取，向前推導至 scale {start_scale}")

    # 僅提取最後一個 scale 的 attention mask
    last_mask = compute_attention_mask_for_scale(
        extractor=extractor,
        focus_token_indices=focus_token_indices,
        scale_idx=last_idx,
        spatial_h=h_last,
        spatial_w=w_last,
        block_indices=attn_block_indices,
        threshold_percentile=threshold_percentile,
        low_attn=low_attn,
        use_normalized_attn=use_normalized_attn,
        threshold_method=threshold_method,
        source_image_np=source_image_np,
        ref_mask=ref_mask,
        dynamic_max_iters=dynamic_max_iters,
    )

    if last_mask is None:
        print(f"  ⚠️ Last scale {last_idx}：無有效 attention map，跳過。")
        return {}

    print(f"  ✓ Last scale {last_idx} ({h_last}×{w_last}): "
          f"focus 區域佔比 = {last_mask.mean() * 100:.1f}%")

    # 向前逐步推導
    masks = propagate_mask_backward(
        last_scale_mask=last_mask,
        scale_schedule=scale_schedule,
        start_scale=start_scale,
        majority_threshold=majority_threshold,
    )

    print(f"[Last-Scale Mask – {label}] 共推導 {len(masks)} 個 scale 的 mask。")
    return masks


def combine_and_store_masks(
    source_text_masks: Dict[int, np.ndarray],
    target_text_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    p2p_token_storage: BitwiseTokenStorage,
    num_full_replace_scales: int,
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

        if src_mask is None and tgt_mask is None:
            continue

        _, h, w = scale_schedule[si]

        # 缺少其中一方時，視為全 False（無 focus 區域）
        if src_mask is None:
            src_mask = np.zeros((h, w), dtype=bool)
        if tgt_mask is None:
            tgt_mask = np.zeros((h, w), dtype=bool)

        # Union：source 或 target 任一認為是 focus → 不替換
        combined_focus   = src_mask | tgt_mask
        replacement_mask = ~combined_focus  # True = 替換（背景）

        mask_tensor = torch.tensor(
            replacement_mask, dtype=torch.bool
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, h, w, 1]

        p2p_token_storage.masks[si] = mask_tensor.cpu()

        src_pct   = src_mask.mean() * 100
        tgt_pct   = tgt_mask.mean() * 100
        union_pct = combined_focus.mean() * 100
        bg_pct    = replacement_mask.mean() * 100
        print(
            f"  ✓ Scale {si} ({h}×{w}): "
            f"src focus={src_pct:.1f}%, tgt focus={tgt_pct:.1f}%, "
            f"union（不替換）={union_pct:.1f}%, 替換={bg_pct:.1f}%"
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


def _mask_tensor_to_prob_map(mask_t: torch.Tensor) -> np.ndarray:
    """將 storage mask tensor 轉成 [h, w] 的機率圖（0~1）。"""
    mask_2d = mask_t.squeeze().detach().cpu()
    if mask_2d.dtype == torch.bool:
        prob = mask_2d.float().numpy()
    else:
        prob = mask_2d.float().clamp(0.0, 1.0).numpy()
    return prob


def _save_prob_masks_to_dir(
    prob_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    vis_dir: str,
    file_prefix: str,
) -> List[np.ndarray]:
    """
    將機率遮罩（0~1）存為灰階 PNG，並生成平均 overlay。

    Returns:
        collected: 各 scale 已放大的灰階圖列表（uint8）
    """
    os.makedirs(vis_dir, exist_ok=True)
    collected: List[np.ndarray] = []

    for si, prob_map in sorted(prob_masks.items()):
        _, h, w = scale_schedule[si]
        prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
        prob_u8 = (prob * 255.0).round().astype(np.uint8)
        vis_size = max(256, h * 4, w * 4)
        mask_vis = cv2.resize(prob_u8, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
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
        overlay = np.clip(stacked / float(len(collected)), 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, "overlay.png"), overlay)
        print(f"✓ [{file_prefix}] 機率遮罩已儲存至 {vis_dir}/ （{len(collected)} 個 scale + overlay）")

    return collected


def build_cumulative_replacement_prob_masks(
    masks: Dict[int, torch.Tensor],
    scale_schedule: List[Tuple[int, int, int]],
    num_full_replace_scales: int,
) -> Dict[int, torch.Tensor]:
    """
    將 replacement mask 做跨尺度累積：
      當前 scale 的 replace_prob = 平均(從 num_full_replace_scales 到當前 scale 的所有 mask，resize 後)

    輸出格式：Dict[si -> Tensor[1,1,h,w,1] float32 in [0,1]]
    """
    if not masks:
        return {}

    cumulative: Dict[int, torch.Tensor] = {}
    max_si = len(scale_schedule)

    for si in range(num_full_replace_scales, max_si):
        _, h_cur, w_cur = scale_schedule[si]
        resized_hist = []
        for sj in range(num_full_replace_scales, si + 1):
            mask_t = masks.get(sj)
            if mask_t is None:
                continue
            m = mask_t.detach().cpu()
            if m.dtype == torch.bool:
                m = m.float()
            else:
                m = m.float().clamp(0.0, 1.0)
            m_2d = m.squeeze(0).squeeze(0).squeeze(-1).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
            m_resized = F.interpolate(m_2d, size=(h_cur, w_cur), mode='nearest')
            resized_hist.append(m_resized)

        if not resized_hist:
            continue

        prob = torch.stack(resized_hist, dim=0).mean(dim=0)  # [1,1,h,w]
        prob = prob.clamp(0.0, 1.0).squeeze(1).unsqueeze(-1).contiguous()  # [1,1,h,w,1]
        cumulative[si] = prob.cpu()

    return cumulative


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
    parser.add_argument('--use_cumulative_prob_mask', type=int, default=0, choices=[0, 1],
                        help='1=每個 scale 使用前面所有 scale mask 的累積平均機率（跨尺度 overlay）')


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P2P-Edit（Source Image Injection + Attention-Guided P2P）圖像編輯管線')
    add_common_arguments(parser)

    # ── P2P 基本參數 ──
    parser.add_argument('--source_prompt', type=str, required=True)
    parser.add_argument('--target_prompt', type=str, required=True)
    parser.add_argument('--save_file', type=str, default='./outputs/p2p_edit')
    parser.add_argument('--p2p_token_file', type=str, default='./tokens_p2p_edit.pkl')

    # ── Focus Words ──
    parser.add_argument('--source_focus_words', type=str, default='',
                        help='Source prompt focus 詞彙（可用逗號分隔 phrase；空白時可自動從 prompt 差異推導）')
    parser.add_argument('--target_focus_words', type=str, default='',
                        help='Target prompt focus 詞彙（可用逗號分隔 phrase；空白時可自動從 prompt 差異推導）')
    parser.add_argument('--source_keep_words', type=str, default='',
                        help='Source prompt 中欲保留的詞彙（高 attention 區域 → Phase 1.7 錨定 source gen token）')
    parser.add_argument('--auto_focus_from_prompt_diff', type=int, default=1, choices=[0, 1],
                        help='1=自動把 source/target prompt 差異片段加入各自 focus words')

    # ── P2P Token 替換參數 ──
    parser.add_argument('--num_full_replace_scales', type=int, default=2,
                        help='前幾個 scale 做 100%% source token 替換。預設：2')
    parser.add_argument('--attn_threshold_percentile', type=float, default=75.0,
                        help='Attention 閾值百分位數。預設：75')
    parser.add_argument('--attn_block_start', type=int, default=-1,
                        help='起始 block index（-1 = 自動，後 1/2 block）')
    parser.add_argument('--attn_block_end', type=int, default=-1,
                        help='結束 block index（-1 = 自動，最後一個 block）')
    parser.add_argument('--attn_batch_idx', type=int, default=0,
                        help='0 = conditioned batch（對應 prompt 文字）')
    parser.add_argument('--p2p_token_replace_prob', type=float, default=0.0,
                        help='Fallback 機率替換（無遮罩時使用）。預設：0.0')
    parser.add_argument('--save_attn_vis', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_normalized_attn', type=int, default=0, choices=[0, 1],
                        help='使用 z-score normalized threshold 取代固定 percentile（0=停用，1=啟用）')
    parser.add_argument('--threshold_method', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='閾值方法：1=固定percentile 2=dynamic ternary 3=Otsu 4=FFT+Otsu '
                             '5=SpectralEnergy 6=EdgeCoherence 7=GMM 8=Composite。預設：1')
    parser.add_argument('--phase17_fallback_replace_scales', type=int, default=4,
                        help='Single-focus fallback（只有 target focus）時，'
                             'Phase 1.7 以 source gen token 替換前幾個 scale（0=停用）。預設：4')
    parser.add_argument('--debug_mode', type=int, default=0, choices=[0, 1],
                        help='1=儲存所有中間過程圖片（Phase 1.7 guided gen、fallback gen）。預設：0')

    # ── Source Image Injection 參數 ──
    parser.add_argument('--source_image', type=str, default=None,
                        help='Source image 路徑（PNG/JPG）。留空 → 純 P2P-Attn 模式')
    parser.add_argument('--image_injection_scales', type=int, default=2,
                        help='前幾個 scale 使用 source image 注入（weight=0 → 100%% image）')
    parser.add_argument('--inject_weights', type=str, default='',
                        help='各 scale 注入強度（空格分隔，長度=scale 總數）。'
                             '0.0=100%% source，1.0=自由生成。'
                             '若不指定，由 image_injection_scales 自動生成')

    args = parser.parse_args()

    # 解析 cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    source_focus_words_list = parse_focus_words_arg(args.source_focus_words)
    target_focus_words_list = parse_focus_words_arg(args.target_focus_words)
    source_keep_words_list = parse_focus_words_arg(args.source_keep_words)

    if args.auto_focus_from_prompt_diff:
        auto_src_terms, auto_tgt_terms = derive_focus_terms_from_prompt_diff(
            args.source_prompt, args.target_prompt
        )
        source_focus_words_list = merge_focus_terms(source_focus_words_list, auto_src_terms)
        target_focus_words_list = merge_focus_terms(target_focus_words_list, auto_tgt_terms)
    else:
        auto_src_terms, auto_tgt_terms = [], []

    save_dir = os.path.abspath(args.save_file)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("P2P-EDIT 圖像編輯管線（Source Image Injection + Attention-Guided P2P）")
    print("=" * 80)
    print(f"Source image       : {args.source_image or '（未提供，使用 P2P-Attn 模式）'}")
    print(f"Source prompt      : {args.source_prompt}")
    print(f"Target prompt      : {args.target_prompt}")
    print(f"Auto focus diff    : {bool(args.auto_focus_from_prompt_diff)}")
    if args.auto_focus_from_prompt_diff:
        print(f"  Auto source diff : {auto_src_terms}")
        print(f"  Auto target diff : {auto_tgt_terms}")
    print(f"Source focus words : {source_focus_words_list}")
    print(f"Target focus words : {target_focus_words_list}")
    print(f"Source keep words  : {source_keep_words_list}")
    print(f"Full replace scales: {args.num_full_replace_scales}")
    print(f"Attn 閾值          : {args.attn_threshold_percentile}th percentile")
    print(f"Threshold method   : {args.threshold_method} ({_THRESHOLD_METHOD_NAMES.get(args.threshold_method, '?')})")
    print(f"Cumulative prob mask: {bool(args.use_cumulative_prob_mask)}")
    print(f"輸出目錄           : {save_dir}")
    print("=" * 80 + "\n")

    # ── 載入模型 ──
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    # ── Scale Schedule ──
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)
    print(f"[Scale Schedule] 共 {total_scales} 個 scale，前 {args.num_full_replace_scales} 個 100% 替換")

    # ── 決定 attention block 範圍 ──
    depth = len(infinity.unregistered_blocks)
    attn_block_start = (depth // 2) if args.attn_block_start < 0 else min(args.attn_block_start, depth - 1)
    attn_block_end   = (depth - 1)  if args.attn_block_end   < 0 else min(args.attn_block_end,   depth - 1)
    attn_block_indices = list(range(attn_block_start, attn_block_end + 1))
    print(f"[Attention Block 範圍] block {attn_block_start} ~ {attn_block_end} （共 {len(attn_block_indices)} 個）")

    # ── Focus Token Indices ──
    print("\n[Phase 0.5] 分析 Focus Token ...")
    source_focus_token_indices = find_focus_token_indices(
        text_tokenizer, args.source_prompt, source_focus_words_list, verbose=True,
    ) if source_focus_words_list else []
    target_focus_token_indices = find_focus_token_indices(
        text_tokenizer, args.target_prompt, target_focus_words_list, verbose=True,
    ) if target_focus_words_list else []
    source_keep_token_indices = find_focus_token_indices(
        text_tokenizer, args.source_prompt, source_keep_words_list, verbose=True,
    ) if source_keep_words_list else []

    # ─────────────────────────────────────────────────────
    # Phase 0：載入並編碼 Source Image（若有提供）
    # ─────────────────────────────────────────────────────
    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_raw_features: Optional[torch.Tensor] = None
    image_scale_tokens: Dict[int, torch.Tensor] = {}
    inject_schedule: Optional[list] = None

    if args.source_image is not None:
        print("\n" + "=" * 60)
        print("[Phase 0] 載入 Source Image 並編碼")
        print("=" * 60)
        source_pil_img = Image.open(args.source_image).convert('RGB')
        print(f"Source image: {args.source_image}  (原始尺寸={source_pil_img.size})")
        source_image_np_for_threshold = np.array(source_pil_img)  # (H, W, 3) uint8

        # (A) 連續特徵：影響 source gen 的 summed_codes
        image_raw_features = encode_image_to_raw_features(
            vae=vae,
            pil_img=source_pil_img,
            scale_schedule=scale_schedule,
            device=device_cuda,
            apply_spatial_patchify=bool(args.apply_spatial_patchify),
        )
        print(f"✓ Source image raw features: {tuple(image_raw_features.shape)}")

        # (B) 離散 bit token：供 Phase 2 P2P 替換使用
        image_scale_tokens = encode_image_to_scale_tokens(
            vae=vae,
            pil_img=source_pil_img,
            scale_schedule=scale_schedule,
            device=device_cuda,
            apply_spatial_patchify=bool(args.apply_spatial_patchify),
        )
        print(f"✓ Source image scale tokens: {len(image_scale_tokens)} 個 scale")

        # 建立 injection schedule
        if args.inject_weights.strip():
            parsed_w = [float(w) for w in args.inject_weights.strip().split()]
            if len(parsed_w) != total_scales:
                raise ValueError(
                    f"--inject_weights 長度 ({len(parsed_w)}) 與 scale 總數 ({total_scales}) 不符。"
                )
            inject_schedule = parsed_w
            args.image_injection_scales = next(
                (i for i, w in enumerate(inject_schedule) if w >= 1.0), total_scales,
            )
        else:
            inject_schedule = [
                0.0 if si < args.image_injection_scales else 1.0
                for si in range(total_scales)
            ]
        inj_str = '  '.join([f"s{si}={w:.2f}" for si, w in enumerate(inject_schedule)])
        print(f"[Injection Schedule] {inj_str}")
    else:
        print("\n[Phase 0] 無 source image → 純 P2P-Attn 模式")
        source_image_np_for_threshold = None

    # ─────────────────────────────────────────────────────
    # Phase 1：Source 生成 + Attention 擷取
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1] Source 圖像生成 + Attention 擷取")
    print("=" * 60)

    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    source_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
    )
    if source_focus_token_indices or source_keep_token_indices:
        source_extractor.register_patches()
        print(f"✓ Source CrossAttentionExtractor 已掛載（{len(attn_block_indices)} 個 block）")
    else:
        print("跳過 Source CrossAttentionExtractor（無 source focus/keep tokens）")

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
                # 存 token，不做替換
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                # 注入 source image VAE codes
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    if source_focus_token_indices or source_keep_token_indices:
        source_extractor.remove_patches()
        source_extractor.get_summary()

    source_save_path = os.path.join(save_dir, "source.jpg")
    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(source_save_path, img_np)
    print(f"✓ Source 圖像已儲存：{source_save_path}")
    print(f"✓ 已提取 {p2p_token_storage.get_num_stored_scales()}/{total_scales} 個 scale 的 token")

    # ─────────────────────────────────────────────────────
    # Phase 1.5：Source Focus Mask
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1.5] 計算 Source Attention Focus Mask")
    print("=" * 60)

    source_text_masks: Dict[int, np.ndarray] = {}
    source_low_attn_masks: Dict[int, np.ndarray] = {}
    source_keep_masks: Dict[int, np.ndarray] = {}
    if source_focus_token_indices and len(source_extractor.attention_maps) > 0:
        source_text_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source",
            low_attn=False,
            threshold_method=args.threshold_method,
            source_image_np=source_image_np_for_threshold,
        )
        print(f"✓ Source focus mask：{len(source_text_masks)} 個 scale")

        # 同一組 source attention，取最低 (100-percentile)% 作為 Phase 1.7 的 preserve 區域
        source_low_attn_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source_preserve",
            low_attn=True,
            threshold_method=args.threshold_method,
            source_image_np=source_image_np_for_threshold,
        )
        print(f"✓ Source low-attention preserve mask：{len(source_low_attn_masks)} 個 scale")
    else:
        print("⚠️  無 source attention map 可用。")

    # source_keep_words：高 attention 區域 → 保留 source gen token（與 focus 邏輯相反）
    if source_keep_token_indices and len(source_extractor.attention_maps) > 0:
        source_keep_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_keep_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source_keep",
            low_attn=False,  # 高 attention = keep 區域
            threshold_method=args.threshold_method,
            source_image_np=source_image_np_for_threshold,
        )
        print(f"✓ Source keep mask：{len(source_keep_masks)} 個 scale")
    else:
        if source_keep_words_list:
            print("⚠️  有 source_keep_words 但無 attention map 可用。")

    # ─────────────────────────────────────────────────────
    # Phase 1.6：建立 Phase 1.7 用的 preserve storage
    # 在低 attention（最背景）區域強制使用 source image token 錨定
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1.6] 建立 Phase 1.7 Preserve Storage（低 attention 錨定 + keep words 錨定）")
    print("=" * 60)

    phase17_storage: Optional[BitwiseTokenStorage] = None
    has_low_attn_preserve = bool(source_low_attn_masks and image_scale_tokens)
    has_keep_preserve = bool(source_keep_masks and p2p_token_storage.tokens)

    if has_low_attn_preserve or has_keep_preserve:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
        count_loaded = 0

        # (A) 來自 source_low_attn_masks + image_scale_tokens（原有邏輯）
        if has_low_attn_preserve:
            for si, low_mask in source_low_attn_masks.items():
                if si not in image_scale_tokens:
                    continue
                phase17_storage.tokens[si] = image_scale_tokens[si].clone()
                mask_tensor = torch.tensor(
                    low_mask, dtype=torch.bool
                ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, h, w, 1]
                phase17_storage.masks[si] = mask_tensor.cpu()
                _, h, w = scale_schedule[si]
                preserve_pct = low_mask.mean() * 100
                print(f"  ✓ Scale {si} ({h}×{w}): low-attn preserve = {preserve_pct:.1f}%")
                count_loaded += 1

        # (B) 來自 source_keep_masks + source gen tokens（新增邏輯）
        # keep mask = 高 attention 區域（True = keep），使用 source gen token 錨定
        if has_keep_preserve:
            for si, keep_mask in source_keep_masks.items():
                if si not in p2p_token_storage.tokens:
                    continue
                _, h, w = scale_schedule[si]
                keep_pct = keep_mask.mean() * 100

                if si in phase17_storage.masks:
                    # 已有 low-attn preserve → union 兩者
                    existing_mask = phase17_storage.masks[si].squeeze().numpy().astype(bool)
                    combined_preserve = existing_mask | keep_mask
                    phase17_storage.masks[si] = torch.tensor(
                        combined_preserve, dtype=torch.bool
                    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()
                    combined_pct = combined_preserve.mean() * 100
                    print(f"  ✓ Scale {si} ({h}×{w}): keep preserve = {keep_pct:.1f}% → union = {combined_pct:.1f}%")
                else:
                    # 僅 keep mask，使用 source gen token
                    phase17_storage.tokens[si] = p2p_token_storage.tokens[si].clone()
                    phase17_storage.masks[si] = torch.tensor(
                        keep_mask, dtype=torch.bool
                    ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cpu()
                    print(f"  ✓ Scale {si} ({h}×{w}): keep preserve (source gen token) = {keep_pct:.1f}%")
                    count_loaded += 1

        print(f"✓ Phase 1.7 preserve storage 完成：{count_loaded} 個 scale")
    else:
        print("⚠️  無 source preserve mask 或無 source image token → Phase 1.7 維持純 free-gen")

    # ─────────────────────────────────────────────────────
    # Phase 1.7：Target 生成 → Target Focus Mask
    # （有 preserve mask → 低 attention 背景區域錨定為 source image token）
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1.7] Target 生成 → Attention Focus Mask")
    if phase17_storage is not None:
        print("  • 低 attention 背景區域：錨定為 source image token（消除 source/model 衝突）")
        print("  • 高 attention focus 區域：target prompt 自由生成")
    else:
        print("  • 純 free-gen（無 source preserve mask 可用）")
    print("=" * 60)

    target_text_masks: Dict[int, np.ndarray] = {}
    if target_focus_token_indices:
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
        )
        target_extractor.register_patches()
        print(f"✓ Target CrossAttentionExtractor 已掛載（{len(attn_block_indices)} 個 block）")

        # Single-focus fallback：無 source focus → 改用 source gen token 錨定前 N scale
        _phase17_storage = phase17_storage
        _phase17_use_mask = (phase17_storage is not None)
        _phase17_full_replace = args.num_full_replace_scales
        if _phase17_storage is None and args.phase17_fallback_replace_scales > 0 and p2p_token_storage.tokens:
            _phase17_storage = p2p_token_storage
            _phase17_use_mask = False
            _phase17_full_replace = args.phase17_fallback_replace_scales
            print(f"  ℹ Phase 1.7 fallback: source gen token 替換前 {_phase17_full_replace} 個 scale")

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _phase17_img = gen_one_img(
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
                    # Phase 1.7：低 attention 背景區域錨定為 source image token
                    # 高 attention focus 區域仍由 target prompt 自由生成
                    p2p_token_storage=_phase17_storage,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=_phase17_use_mask,
                    p2p_save_tokens=False,  # 不覆寫 preserve tokens
                    p2p_attn_full_replace_scales=_phase17_full_replace,
                    inject_image_features=None,
                    inject_schedule=None,
                )
        if args.debug_mode:
            _dbg_np = _phase17_img.cpu().numpy()
            if _dbg_np.dtype != np.uint8:
                _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, 'phase17_target.jpg'), _dbg_np)
            del _dbg_np
        del _phase17_img

        target_extractor.remove_patches()
        target_extractor.get_summary()

        target_text_masks = collect_attention_text_masks(
            extractor=target_extractor,
            focus_token_indices=target_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="target",
            threshold_method=args.threshold_method,
            source_image_np=source_image_np_for_threshold,
        )
        print(f"✓ Target focus mask：{len(target_text_masks)} 個 scale")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("⚠️  無 target focus token，Phase 1.7 跳過。")

    # ─────────────────────────────────────────────────────
    # Phase 1.9：合併 Mask + 覆寫 storage.tokens
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1.9] 合併 Mask + 覆寫 storage.tokens")
    print("=" * 60)

    # 合併 source ∪ target focus mask → replacement mask → 存入 storage.masks
    if source_text_masks or target_text_masks:
        masks_stored = combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=args.num_full_replace_scales,
        )
        print(f"✓ Combined replacement mask 已存入 storage：{masks_stored} 個 scale")
    else:
        print("⚠️  無任何 focus mask，Phase 2 將使用 fallback 機率替換。")

    if args.use_cumulative_prob_mask and p2p_token_storage.masks:
        p2p_token_storage.masks = build_cumulative_replacement_prob_masks(
            masks=p2p_token_storage.masks,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
        )
        print(f"✓ 已啟用跨尺度累積機率遮罩（{len(p2p_token_storage.masks)} 個 scale）")

    # P2P-Edit 特有：用 source image 離散 token 覆寫 storage.tokens
    if image_scale_tokens:
        for si_tok, tok in image_scale_tokens.items():
            p2p_token_storage.tokens[si_tok] = tok
        print(f"✓ storage.tokens 已覆寫為 source image 離散 token（{len(image_scale_tokens)} 個 scale）")
    else:
        print("⚠ storage.tokens 來自 source gen 採樣（純 P2P-Attn 模式）")

    # 遮罩視覺化
    if args.save_attn_vis:
        attn_vis_dir = os.path.join(save_dir, "attn_masks")
        if source_text_masks:
            _save_masks_to_dir(
                bool_masks=source_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "source"),
                file_prefix="source_focus",
                invert=True,   # 黑色 = focus（不替換），白色 = 背景（替換）
            )
        # Phase 1.7 preserve mask（黑色 = 低 attention 錨定區域 = source image token 強制保留）
        if source_low_attn_masks:
            _save_masks_to_dir(
                bool_masks=source_low_attn_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "phase17_preserve"),
                file_prefix="preserve",
                invert=False,  # 白色 = preserve 區域（source token 錨定），黑色 = focus 自由区
            )
        # source_keep_words preserve mask（白色 = 高 attention 保留區域）
        if source_keep_masks:
            _save_masks_to_dir(
                bool_masks=source_keep_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "phase17_keep"),
                file_prefix="keep",
                invert=False,  # 白色 = keep 區域（source gen token 錨定），黑色 = 自由區
            )
        if target_text_masks:
            _save_masks_to_dir(
                bool_masks=target_text_masks,
                scale_schedule=scale_schedule,
                vis_dir=os.path.join(attn_vis_dir, "target"),
                file_prefix="target_focus",
                invert=True,   # 黑色 = focus（不替換），白色 = 背景（替換）
            )
        # combined replacement mask（True=替換背景，False=focus 保護區）
        if p2p_token_storage.masks:
            has_prob_mask = any(m.dtype != torch.bool for m in p2p_token_storage.masks.values())
            if has_prob_mask:
                combined_prob_vis = {
                    si: _mask_tensor_to_prob_map(m)
                    for si, m in p2p_token_storage.masks.items()
                }
                _save_prob_masks_to_dir(
                    prob_masks=combined_prob_vis,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "combined"),
                    file_prefix="combined_replace_prob",
                )
            else:
                combined_vis = {si: ~m.squeeze().numpy() for si, m in p2p_token_storage.masks.items()}
                _save_masks_to_dir(
                    bool_masks=combined_vis,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "combined"),
                    file_prefix="combined_focus",
                    invert=True,   # 黑色 = union focus（不替換），白色 = 背景（替換）
                )

    p2p_token_storage.save_to_file(args.p2p_token_file)
    print(f"✓ Token + 遮罩已儲存：{args.p2p_token_file}")

    del source_image, img_np
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ─────────────────────────────────────────────────────
    # Phase 2：Target 生成（P2P Token 替換 + Attention 遮罩）
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 2] Target 圖像生成")
    print("=" * 60)

    tgt_full_replace = args.num_full_replace_scales
    has_mask = len(p2p_token_storage.masks) > 0
    has_prob_mask = has_mask and any(m.dtype != torch.bool for m in p2p_token_storage.masks.values())
    print(f"策略：")
    print(f"  Scale 0 ~ {tgt_full_replace - 1}：100% Source Token 替換（全域結構）")
    if has_mask:
        if has_prob_mask:
            print(f"  Scale {tgt_full_replace} ~ {total_scales - 1}：Combined 累積機率遮罩替換")
            print(f"      灰階值(0~1) = source token 替換機率")
            print(f"      每個 token 依機率隨機決定是否替換")
        else:
            print(f"  Scale {tgt_full_replace} ~ {total_scales - 1}：Combined Attention 遮罩替換")
            print(f"      focus=True（Union）→ 保留 target 自由生成")
            print(f"      focus=False        → 替換 source image token")
    else:
        print(f"  Scale {tgt_full_replace} ~ {total_scales - 1}：Fallback（無遮罩）")

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
                # P2P：使用 storage（tokens=source image token，masks=combined replacement mask）
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=has_mask,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=tgt_full_replace,
                # Phase 2 永不注入 source image（確保 target 可自由改變內容）
                inject_image_features=None,
                inject_schedule=None,
            )

    target_save_path = os.path.join(save_dir, "target.jpg")
    img_np = target_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(target_save_path, img_np)
    print(f"✓ Target 圖像已儲存：{target_save_path}")

    del target_image, img_np, p2p_token_storage
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("P2P-EDIT 管線完成")
    print("=" * 80)
    print(f"結果儲存於: {save_dir}/")
    print(f"  - source.jpg     Source 重建圖像")
    print(f"  - target.jpg     編輯後圖像")
    print(f"  - attn_masks/")
    print(f"      ├─ source/          Source focus 遮罩（黑色=focus 不替換）")
    print(f"      ├─ phase17_preserve/ Phase 1.7 preserve 遮罩（白色=low-attn 錨定區域）")
    print(f"      ├─ phase17_keep/    Phase 1.7 keep 遮罩（白色=keep words 錨定區域）")
    print(f"      ├─ target/          Target focus 遮罩（黑色=focus 自由區）")
    print(f"      └─ combined/         Combined replacement 遮罩（source∪target union）")
    print(f"  - {os.path.basename(args.p2p_token_file)}   Token 資料")
    print("=" * 80 + "\n")
