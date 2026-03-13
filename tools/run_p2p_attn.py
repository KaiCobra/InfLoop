"""
P2P-Attn (Prompt-to-Prompt + Attention-Guided) 圖像編輯管線

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

# 使用 P2P-Attn 版本的模型
from infinity.models.infinity_p2p_attn import Infinity
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
    # 例：'Frog' → ['▁F', 'rog']，'f'+'rog'=='frog' → 匹配兩個 token
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
                    break
            if matched_for_fw:
                break

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

    Returns:
        text_region_mask: (H, W) bool ndarray
            True  = 高 attention（文字區域，不替換）
            False = 低 attention（背景區域，替換為 source token）
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

    # 百分位數閾值：高於此值 = 文字區域
    threshold = np.percentile(filtered_attn, threshold_percentile)
    text_region = filtered_attn >= threshold  # True = 文字區域（高 attention）

    coverage_pct = text_region.mean() * 100
    print(
        f"  Scale {scale_idx}: "
        f"使用 {num_used}/{num_used + num_outliers} 個 block，"
        f"閾值 = {threshold:.4f}（{threshold_percentile:.0f} pct），"
        f"文字區域佔比 = {coverage_pct:.1f}%"
    )

    return text_region.astype(bool)


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
) -> Dict[int, np.ndarray]:
    """
    收集各 scale 的 attention focus mask，不存入 storage。

    Returns:
        Dict[scale_idx -> (H, W) bool ndarray]
        True = 高 attention（focus 區域，例如狗的位置）→ 不應被 source token 覆蓋
    """
    text_masks: Dict[int, np.ndarray] = {}
    print(f"\n[Attention 遮罩計算 – {label}] scale {num_full_replace_scales} ~ {len(scale_schedule)-1}")

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

        if text_mask is not None:
            text_masks[si] = text_mask

    print(f"[Attention 遮罩計算 – {label}] 收集到 {len(text_masks)} 個 scale 的 focus mask。")
    return text_masks


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
):
    """
    生成一張圖片（P2P-Attn 版本）。

    新增參數（相比 run_p2p.py）：
        p2p_attn_full_replace_scales (int):
            前幾個 scale 做 100% source token 替換。
            0 = 禁用（與原始 p2p 行為相同）。
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
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
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
    """載入 P2P-Attn 版本的 Infinity 模型"""
    print(f'[載入 Infinity P2P-Attn 模型]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
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
            f'[Infinity P2P-Attn 模型大小: '
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
# 主程式
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P2P-Attn（Prompt-to-Prompt + Attention-Guided）圖像編輯管線')
    add_common_arguments(parser)

    # ── P2P 基本參數 ──
    parser.add_argument('--source_prompt', type=str, required=True,
                        help='Source prompt（用於生成並提取 token/attention）')
    parser.add_argument('--target_prompt', type=str, required=True,
                        help='Target prompt（用於生成目標圖像）')
    parser.add_argument('--save_file', type=str, default='./outputs/p2p_attn',
                        help='輸出目錄')
    parser.add_argument('--p2p_token_file', type=str, default='./tokens_p2p_attn.pkl',
                        help='儲存/載入 token 的檔案路徑')

    # ── P2P-Attn 新增參數 ──
    parser.add_argument('--source_focus_words', type=str, required=True,
                        help='Source prompt 中欲關注的目標詞彙（以空格分隔）。'
                             '例："dog"')
    parser.add_argument('--target_focus_words', type=str, required=True,
                        help='Target prompt 中欲關注的目標詞彙（以空格分隔）。'
                             '通常與 source_focus_words 相同，但物件名稱不同時可分開設定。'
                             '例："dog"')
    parser.add_argument('--num_full_replace_scales', type=int, default=4,
                        help='前幾個 scale 做 100%% source token 替換（結構保留）。'
                             '之後的 scale 改用 attention 遮罩。預設：4')
    parser.add_argument('--attn_threshold_percentile', type=float, default=75.0,
                        help='Attention 閾值百分位數。高於此值的空間位置為文字區域（不替換）。'
                             '值越小 = 文字區域越大。預設：75（前 25%% 為文字區域）')
    parser.add_argument('--attn_block_start', type=int, default=-1,
                        help='用於計算 attention 遮罩的起始 block index。'
                             '-1 = 自動（使用後 1/2 的 block）')
    parser.add_argument('--attn_block_end', type=int, default=-1,
                        help='用於計算 attention 遮罩的結束 block index（含）。'
                             '-1 = 自動（使用模型最後一個 block）')
    parser.add_argument('--attn_batch_idx', type=int, default=0,
                        help='CFG 設定下擷取哪個 batch 的 attention。'
                             '0 = 條件化（conditioned，對應 source prompt 文字），'
                             '1 = 非條件化（unconditioned）。預設：0')
    parser.add_argument('--p2p_token_replace_prob', type=float, default=0.5,
                        help='Fallback 機率替換（當無 attention 遮罩時使用）。預設：0.5')
    parser.add_argument('--save_attn_vis', type=int, default=1, choices=[0, 1],
                        help='是否儲存 attention 遮罩視覺化圖。預設：1（儲存）')

    args = parser.parse_args()

    # 解析 cfg 參數
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # ── 參數整理 ──
    source_focus_words_list = args.source_focus_words.strip().split()
    target_focus_words_list = args.target_focus_words.strip().split()
    save_dir = os.path.abspath(args.save_file)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("P2P-ATTN 圖像編輯管線（Attention-Guided Prompt-to-Prompt）")
    print("=" * 80)
    print(f"Source prompt      : {args.source_prompt}")
    print(f"Target prompt      : {args.target_prompt}")
    print(f"Source focus words : {source_focus_words_list}")
    print(f"Target focus words : {target_focus_words_list}")
    print(f"Full replace       : 前 {args.num_full_replace_scales} 個 scale（100% 替換）")
    print(f"Attn 閾值          : {args.attn_threshold_percentile}th percentile")
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

    # ── 決定用於計算 attention 的 block 索引範圍 ──
    # 自動：使用模型後 1/2 的 block（這些 block 的 attention 語意更豐富）
    depth = len(infinity.unregistered_blocks)  # 模型深度（block 數量）
    if args.attn_block_start < 0:
        attn_block_start = depth // 2  # 後半段 block
    else:
        attn_block_start = min(args.attn_block_start, depth - 1)
    if args.attn_block_end < 0:
        attn_block_end = depth - 1
    else:
        attn_block_end = min(args.attn_block_end, depth - 1)

    attn_block_indices = list(range(attn_block_start, attn_block_end + 1))
    print(
        f"[Attention Block 範圍] block {attn_block_start} ~ {attn_block_end} "
        f"（共 {len(attn_block_indices)} 個 block）"
    )

    # ── 尋找 Source / Target Focus Token Indices ──
    print("\n[Phase 0] 分析 Focus Token ...")
    print(f"  Source focus: {source_focus_words_list}")
    source_focus_token_indices = find_focus_token_indices(
        tokenizer=text_tokenizer,
        prompt=args.source_prompt,
        focus_words=source_focus_words_list,
        verbose=True,
    )
    print(f"  Target focus: {target_focus_words_list}")
    target_focus_token_indices = find_focus_token_indices(
        tokenizer=text_tokenizer,
        prompt=args.target_prompt,
        focus_words=target_focus_words_list,
        verbose=True,
    )

    if not source_focus_token_indices and not target_focus_token_indices:
        print("⚠️  source / target 均無法找到 focus token，將使用 fallback 機率替換。")

    # ── Phase 1：Source 生成 + Attention 擷取 ──
    print("\n" + "=" * 60)
    print("[Phase 1] Source 圖像生成 + Attention 擷取")
    print("=" * 60)

    # 初始化 BitwiseTokenStorage（儲存所有 scale 的 token）
    p2p_token_storage = BitwiseTokenStorage(
        num_scales=total_scales,  # 儲存全部 scale（非只有前 N 個）
        device='cpu',
    )

    # 設定 Source CrossAttentionExtractor
    # 注意：batch_idx=0 = conditioned batch（對應 source prompt 文字）
    source_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
    )

    if source_focus_token_indices:
        source_extractor.register_patches()
        print(f"✓ Source CrossAttentionExtractor 已掛載（監聽 {len(attn_block_indices)} 個 block）")
    else:
        print("跳過 Source CrossAttentionExtractor 掛載（無 source focus tokens）")

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.source_prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                # Source 生成：儲存 token，不做替換
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
            )

    # 卸除 source attention 擷取器
    if source_focus_token_indices:
        source_extractor.remove_patches()
        source_extractor.get_summary()

    # 儲存 source 圖像
    source_save_path = os.path.join(save_dir, "source.jpg")
    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(source_save_path, img_np)
    print(f"✓ Source 圖像已儲存：{source_save_path}")

    num_stored = p2p_token_storage.get_num_stored_scales()
    print(f"✓ 已提取 {num_stored}/{total_scales} 個 scale 的 token")

    # ── Phase 1.5：從 source attention 收集 focus mask ──
    print("\n" + "=" * 60)
    print("[Phase 1.5] 計算 Source Attention Focus Mask")
    print("=" * 60)

    source_text_masks: Dict[int, np.ndarray] = {}
    if source_focus_token_indices and len(source_extractor.attention_maps) > 0:
        source_text_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
            label="source",
        )
        print(f"✓ Source focus mask：{len(source_text_masks)} 個 scale")
    else:
        print("⚠️  無 source attention map 可用。")

    # ── Phase 1.7：自由生成 Target（不替換），擷取 target attention ──
    print("\n" + "=" * 60)
    print("[Phase 1.7] Target 自由生成（無替換，僅擷取 Attention）")
    print("=" * 60)
    print("說明：此步驟產生的圖像不儲存，僅用於收集 target focus 區域的 attention map。")

    target_text_masks: Dict[int, np.ndarray] = {}
    if target_focus_token_indices:
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
        )
        target_extractor.register_patches()
        print(f"✓ Target CrossAttentionExtractor 已掛載（監聽 {len(attn_block_indices)} 個 block）")

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _ = gen_one_img(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    args.target_prompt,
                    g_seed=args.seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=args.enable_positive_prompt,
                    # 自由生成：不使用任何 P2P 替換
                    p2p_token_storage=None,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=False,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=0,
                )

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
        )
        print(f"✓ Target focus mask：{len(target_text_masks)} 個 scale")
        torch.cuda.empty_cache()
    else:
        print("⚠️  無 target focus token，跳過 target attention 擷取。")

    # ── Phase 1.9：合併 source + target mask，存入 storage ──
    print("\n" + "=" * 60)
    print("[Phase 1.9] 合併 Source + Target Focus Mask → Replacement Mask")
    print("=" * 60)

    masks_stored = 0
    if source_text_masks or target_text_masks:
        masks_stored = combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=args.num_full_replace_scales,
        )
        print(f"✓ 合併並儲存 {masks_stored} 個 replacement mask")

        if args.save_attn_vis and masks_stored > 0:
            attn_vis_dir = os.path.join(save_dir, "attn_masks")
            # Source focus mask（白色 = source 認為是 focus 的區域）
            if source_text_masks:
                _save_masks_to_dir(
                    bool_masks=source_text_masks,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "source"),
                    file_prefix="source_focus",
                    invert=False,
                )
            # Target focus mask（白色 = target 認為是 focus 的區域）
            if target_text_masks:
                _save_masks_to_dir(
                    bool_masks=target_text_masks,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "target"),
                    file_prefix="target_focus",
                    invert=False,
                )
            # Combined replacement mask（白色 = 替換，黑色 = union focus 保護區）
            combined_replacement_masks: Dict[int, np.ndarray] = {}
            for si in range(args.num_full_replace_scales, total_scales):
                if p2p_token_storage.has_mask_for_scale(si):
                    mask_t = p2p_token_storage.load_mask(si, 'cpu')
                    combined_replacement_masks[si] = mask_t[0, 0, :, :, 0].numpy()
            if combined_replacement_masks:
                _save_masks_to_dir(
                    bool_masks=combined_replacement_masks,
                    scale_schedule=scale_schedule,
                    vis_dir=os.path.join(attn_vis_dir, "combined"),
                    file_prefix="replacement",
                    invert=False,
                )
    else:
        print("⚠️  無 attention map 可用，target 生成將使用 fallback 機率替換。")

    # 儲存 token + 遮罩至檔案
    p2p_token_storage.save_to_file(args.p2p_token_file)
    print(f"✓ Token + 遮罩已儲存：{args.p2p_token_file}")

    # 釋放 GPU 記憶體
    del source_image, img_np
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ── Phase 2：Target 生成（使用 Attention 遮罩）──
    print("\n" + "=" * 60)
    print("[Phase 2] Target 圖像生成（Attention-Guided P2P）")
    print("=" * 60)
    print(f"策略：")
    print(f"  Scale 0 ~ {args.num_full_replace_scales - 1}：100% Source Token 替換（全域結構保留）")
    if masks_stored > 0:
        print(f"  Scale {args.num_full_replace_scales} ~ {total_scales - 1}：")
        print(f"      非文字區域（低 attention）→ 替換為 source token（保留背景）")
        print(f"      文字區域（高 attention）   → 保留 target 自由生成（渲染新文字）")
    else:
        print(f"  Scale {args.num_full_replace_scales} ~ {total_scales - 1}：")
        print(f"      Fallback：機率替換（prob={args.p2p_token_replace_prob}）")

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.target_prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                # Target 生成：使用 storage 中的 token + 遮罩
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=(masks_stored > 0),   # 有遮罩才啟用，否則 fallback
                p2p_save_tokens=False,              # target 生成不需儲存
                p2p_attn_full_replace_scales=args.num_full_replace_scales,
            )

    # 儲存 target 圖像
    target_save_path = os.path.join(save_dir, "target.jpg")
    img_np = target_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(target_save_path, img_np)
    print(f"✓ Target 圖像已儲存：{target_save_path}")

    # 釋放 GPU 記憶體
    del target_image, img_np, p2p_token_storage
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("P2P-ATTN 管線完成")
    print("=" * 80)
    print(f"結果儲存於:{save_dir}/")
    print(f"  - source.jpg        原始圖像")
    print(f"  - target.jpg        編輯後圖像（文字已替換）")
    print(f"  - attn_masks/")
    print(f"      ├─ source/    Source focus 遮罩（白色=source focus 區域）")
    print(f"      │   └─ overlay.png")
    print(f"      ├─ target/    Target focus 遮罩（白色=target focus 區域）")
    print(f"      │   └─ overlay.png")
    print(f"      └─ combined/  合併 replacement 遮罩（白色=替換，黑色=union 保護區）")
    print(f"          └─ overlay.png  各 scale 疊加灰階圖（亮=背景，暗=保護）")
    print(f"  - {os.path.basename(args.p2p_token_file)}   Token + 遮罩資料")
    print("=" * 80 + "\n")
