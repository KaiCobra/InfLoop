"""
PIE-P2P：批次 Prompt-to-Prompt 圖像生成管線（PIE-Bench 資料集）

功能說明
========
讀取 PIE-Bench mapping_file.json，對每筆資料的 original_prompt（source）
和 editing_prompt（target）各生成一張圖片，使用 P2P-Attn pipeline。

設計重點
--------
- 模型只載入一次，批次處理所有資料（避免重複載入浪費時間）
- 輸出結構：output_dir/{image_key}/source.jpg + target.jpg
- 支援斷點續跑：已存在 target.jpg 的 key 自動跳過
- Focus words 自動從 blended_word 欄位解析
- editing_prompt 中的 [bracket] 標記自動去除後餵給模型

JSON 資料格式（mapping_file.json）
-----------------------------------
{
  "000000000000": {
    "image_path": "...",
    "original_prompt": "a slanted mountain bicycle ...",
    "editing_prompt": "a slanted [rusty] mountain bicycle ...",
    "blended_word": "bicycle bicycle",
    ...
  },
  ...
}

執行方式
--------
    bash scripts/pie_p2p.sh
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import math
import time
import shutil
import argparse
import os.path as osp
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

from infinity.models.infinity_p2p_attn import Infinity
from infinity.models.basic import *
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

from attention_map.extractor import CrossAttentionExtractor


# ============================================================
# 資料前處理工具
# ============================================================

def clean_editing_prompt(prompt: str) -> str:
    """
    移除 editing_prompt 中的 [bracket] 標記，還原為乾淨的 prompt。

    範例：
        "a slanted [rusty] mountain bicycle" → "a slanted rusty mountain bicycle"
        "a [large] [brown] dog"              → "a large brown dog"
    """
    return re.sub(r'\[([^\]]+)\]', r'\1', prompt).strip()


def parse_blended_word(blended_word: str) -> Tuple[str, str]:
    """
    解析 blended_word 欄位，拆分為 source_focus 與 target_focus。

    格式："{source_word} {target_word}"
    若只有一個詞，source 和 target 共用同一詞。
    若為空，回傳 ("", "")。

    範例：
        "bicycle bicycle" → ("bicycle", "bicycle")
        "cat fox"         → ("cat", "fox")
        ""                → ("", "")
    """
    parts = blended_word.strip().split()
    if len(parts) == 0:
        return "", ""
    elif len(parts) == 1:
        return parts[0], parts[0]
    else:
        # 文件格式：前半是 source，後半是 target
        mid = len(parts) // 2
        source_focus = " ".join(parts[:mid])
        target_focus = " ".join(parts[mid:])
        return source_focus, target_focus


# ============================================================
# Tokenizer 工具函式（與 run_p2p_attn.py 相同）
# ============================================================

def find_focus_token_indices(
    tokenizer,
    prompt: str,
    focus_words: List[str],
    verbose: bool = False,
) -> List[int]:
    if not focus_words or all(w.strip() == "" for w in focus_words):
        return []

    tokens = tokenizer(
        text=[prompt],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = tokens.input_ids[0].tolist()
    attn_mask = tokens.attention_mask[0].tolist()
    seq_len = sum(attn_mask)

    focus_words_lower = [w.lower().strip() for w in focus_words if w.strip()]
    focus_indices: List[int] = []

    decoded = []
    for i in range(seq_len):
        token_text = tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
        token_clean = token_text.replace('▁', '').replace(' ', '').lower()
        decoded.append((i, token_text, token_clean))

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
                            + f"] = {'+'.join(span_strs)} ← '{fw}'"
                        )
                    matched_for_fw = True
                    break
                elif not fw_nospace.startswith(accumulated):
                    break
            if matched_for_fw:
                break

    return focus_indices


# ============================================================
# Attention 遮罩計算（與 run_p2p_attn.py 相同）
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
    threshold = np.percentile(filtered_attn, threshold_percentile)
    text_region = filtered_attn >= threshold
    return text_region.astype(bool)


def _iqr_filtered_mean(attn_stack: torch.Tensor) -> Tuple[np.ndarray, int, int]:
    num_blocks = attn_stack.shape[0]
    if num_blocks == 1:
        return attn_stack[0].numpy(), 0, 1

    attn_mean = attn_stack.mean(dim=0)
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
        filtered_attn = attn_mean.numpy()
        num_used = num_blocks
        num_outliers = 0

    return filtered_attn, num_outliers, num_used


def collect_attention_text_masks(
    extractor: CrossAttentionExtractor,
    focus_token_indices: List[int],
    scale_schedule: List[Tuple[int, int, int]],
    num_full_replace_scales: int,
    attn_block_indices: List[int],
    threshold_percentile: float = 75.0,
) -> Dict[int, np.ndarray]:
    text_masks: Dict[int, np.ndarray] = {}
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
    return text_masks


def combine_and_store_masks(
    source_text_masks: Dict[int, np.ndarray],
    target_text_masks: Dict[int, np.ndarray],
    scale_schedule: List[Tuple[int, int, int]],
    p2p_token_storage: BitwiseTokenStorage,
    num_full_replace_scales: int,
) -> int:
    masks_stored = 0
    for si in range(num_full_replace_scales, len(scale_schedule)):
        src_mask = source_text_masks.get(si)
        tgt_mask = target_text_masks.get(si)
        if src_mask is None and tgt_mask is None:
            continue

        _, h, w = scale_schedule[si]
        if src_mask is None:
            src_mask = np.zeros((h, w), dtype=bool)
        if tgt_mask is None:
            tgt_mask = np.zeros((h, w), dtype=bool)

        combined_focus = src_mask | tgt_mask
        replacement_mask = ~combined_focus
        mask_tensor = torch.tensor(
            replacement_mask, dtype=torch.bool
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        p2p_token_storage.masks[si] = mask_tensor.cpu()
        masks_stored += 1

    return masks_stored


# ============================================================
# Prompt 編碼工具（與 run_p2p_attn.py 相同）
# ============================================================

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
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
    return (kv_compact, lens, cu_seqlens_k, Ltext)


# ============================================================
# 圖像生成函式（與 run_p2p_attn.py 相同）
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
    p2p_token_storage=None,
    p2p_token_replace_prob=0.5,
    p2p_use_mask=False,
    p2p_save_tokens=True,
    p2p_attn_full_replace_scales=0,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    negative_label_B_or_BLT = None

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
            p2p_token_storage=p2p_token_storage,
            p2p_token_replace_prob=p2p_token_replace_prob,
            p2p_use_mask=p2p_use_mask,
            p2p_save_tokens=p2p_save_tokens,
            p2p_attn_full_replace_scales=p2p_attn_full_replace_scales,
        )
    print(f"  cost: {time.time() - sstt:.2f}s (infinity={time.time() - stt:.2f}s)")
    img = img_list[0]
    del img_list
    return img


# ============================================================
# 模型載入工具（與 run_p2p_attn.py 相同）
# ============================================================

def load_tokenizer(t5_path=''):
    print('[載入 T5 tokenizer + text encoder]')
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
    print('[載入 Infinity P2P-Attn 模型]')
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

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        print('[載入模型權重]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test


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
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    model_configs = {
        'infinity_2b':      dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        'infinity_8b':      dict(depth=40, embed_dim=3584, num_heads=28,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
        'infinity_layer12': dict(depth=12, embed_dim=768,  num_heads=8,            drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        'infinity_layer16': dict(depth=16, embed_dim=1152, num_heads=12,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        'infinity_layer24': dict(depth=24, embed_dim=1536, num_heads=16,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        'infinity_layer32': dict(depth=32, embed_dim=2080, num_heads=20,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        'infinity_layer40': dict(depth=40, embed_dim=2688, num_heads=24,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        'infinity_layer48': dict(depth=48, embed_dim=3360, num_heads=28,           drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    }
    if args.model_type not in model_configs:
        raise ValueError(f'未知模型類型: {args.model_type}')
    kwargs_model = model_configs[args.model_type]

    return load_infinity(
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


# ============================================================
# 單筆資料的 P2P-Attn 完整管線
# ============================================================

def process_one_entry(
    entry_key: str,
    source_prompt: str,
    target_prompt: str,
    source_focus_words: List[str],
    target_focus_words: List[str],
    save_dir: str,
    # 模型
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    # 生成參數
    scale_schedule,
    attn_block_indices: List[int],
    args,
):
    """執行單筆資料的完整 P2P-Attn 生成流程（source + target）"""
    os.makedirs(save_dir, exist_ok=True)
    total_scales = len(scale_schedule)

    # ── Phase 1：Source 生成 + Attention 擷取 ──
    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    source_extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=attn_block_indices,
        batch_idx=args.attn_batch_idx,
        aggregate_method="mean",
    )

    source_focus_token_indices = find_focus_token_indices(
        text_tokenizer, source_prompt, source_focus_words
    )
    target_focus_token_indices = find_focus_token_indices(
        text_tokenizer, target_prompt, target_focus_words
    )

    if source_focus_token_indices:
        source_extractor.register_patches()

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
            )

    if source_focus_token_indices:
        source_extractor.remove_patches()

    source_save_path = osp.join(save_dir, "source.jpg")
    img_np = source_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(source_save_path, img_np)
    del source_image, img_np

    # ── Phase 1.5：收集 Source Attention Focus Mask ──
    source_text_masks: Dict[int, np.ndarray] = {}
    if source_focus_token_indices and len(source_extractor.attention_maps) > 0:
        source_text_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
        )

    # ── Phase 1.7：Target 自由生成（僅擷取 Attention）──
    target_text_masks: Dict[int, np.ndarray] = {}
    if target_focus_token_indices:
        target_extractor = CrossAttentionExtractor(
            model=infinity,
            block_indices=attn_block_indices,
            batch_idx=args.attn_batch_idx,
            aggregate_method="mean",
        )
        target_extractor.register_patches()

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _ = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder,
                    target_prompt,
                    g_seed=args.seed,
                    gt_leak=0, gt_ls_Bl=None,
                    cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    enable_positive_prompt=args.enable_positive_prompt,
                    p2p_token_storage=None,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=False,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=0,
                )

        target_extractor.remove_patches()

        target_text_masks = collect_attention_text_masks(
            extractor=target_extractor,
            focus_token_indices=target_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=args.num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=args.attn_threshold_percentile,
        )
        torch.cuda.empty_cache()

    # ── Phase 1.9：合併並存入 replacement mask ──
    masks_stored = 0
    if source_text_masks or target_text_masks:
        masks_stored = combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=args.num_full_replace_scales,
        )

    # ── Phase 2：Target 生成（Attention-Guided P2P）──
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=args.seed,
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=(masks_stored > 0),
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=args.num_full_replace_scales,
            )

    target_save_path = osp.join(save_dir, "target.jpg")
    img_np = target_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(target_save_path, img_np)
    del target_image, img_np, p2p_token_storage

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"  ✓ source.jpg  →  {source_save_path}")
    print(f"  ✓ target.jpg  →  {target_save_path}")


# ============================================================
# 主程式
# ============================================================

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIE-P2P 批次 Prompt-to-Prompt 圖像生成（P2P-Attn）')
    add_common_arguments(parser)

    # ── 批次資料集參數 ──
    parser.add_argument('--json_file', type=str, required=True,
                        help='PIE-Bench mapping_file.json 路徑')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出根目錄，每筆資料依 key 建立子目錄')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='從第幾筆開始處理（0-based，用於分段/除錯）')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='處理到第幾筆（-1 = 全部）')
    parser.add_argument('--skip_existing', type=int, default=1, choices=[0, 1],
                        help='跳過已存在 target.jpg 的筆數（斷點續跑）。預設：1（開啟）')

    # ── P2P-Attn 參數 ──
    parser.add_argument('--num_full_replace_scales', type=int, default=0,
                        help='前幾個 scale 做 100%% source token 替換。預設：0')
    parser.add_argument('--attn_threshold_percentile', type=float, default=80.0,
                        help='Attention 閾值百分位數。預設：80')
    parser.add_argument('--attn_block_start', type=int, default=2,
                        help='計算 attention 遮罩的起始 block index。預設：2')
    parser.add_argument('--attn_block_end', type=int, default=-1,
                        help='計算 attention 遮罩的結束 block index。-1 = 自動')
    parser.add_argument('--attn_batch_idx', type=int, default=0,
                        help='CFG 下擷取哪個 batch 的 attention。預設：0（conditioned）')
    parser.add_argument('--p2p_token_replace_prob', type=float, default=0.5,
                        help='Fallback 機率替換。預設：0.5')

    args = parser.parse_args()

    # 解析 cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # ── 讀取 JSON ──
    print(f"\n[載入資料集] {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_keys = sorted(dataset.keys())
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_keys)
    keys_to_process = all_keys[args.start_idx:end_idx]
    print(f"共 {len(dataset)} 筆，處理範圍 [{args.start_idx}, {end_idx})，共 {len(keys_to_process)} 筆")

    # ── 載入模型（只載入一次）──
    print("\n" + "=" * 60)
    print("載入模型（僅執行一次）")
    print("=" * 60)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    # ── Scale Schedule ──
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)

    # ── Attention Block 範圍 ──
    depth = len(infinity.unregistered_blocks)
    attn_block_start = args.attn_block_start if args.attn_block_start >= 0 else depth // 2
    attn_block_end = args.attn_block_end if args.attn_block_end >= 0 else depth - 1
    attn_block_indices = list(range(attn_block_start, min(attn_block_end, depth - 1) + 1))
    print(f"Attention block 範圍：{attn_block_start} ~ {attn_block_end}（共 {len(attn_block_indices)} 個）")

    # ── 批次處理 ──
    os.makedirs(args.output_dir, exist_ok=True)
    success_count = 0
    skip_count = 0
    error_count = 0

    print("\n" + "=" * 60)
    print("開始批次生成")
    print("=" * 60)

    for i, key in enumerate(keys_to_process):
        entry = dataset[key]
        save_dir = osp.join(args.output_dir, key)

        # 斷點續跑：target.jpg 已存在則跳過
        if args.skip_existing and osp.exists(osp.join(save_dir, "target.jpg")):
            skip_count += 1
            print(f"[{i+1}/{len(keys_to_process)}] {key}  跳過（已存在）")
            continue

        source_prompt = entry['original_prompt']
        raw_editing_prompt = entry['editing_prompt']
        target_prompt = clean_editing_prompt(raw_editing_prompt)
        blended_word = entry.get('blended_word', '')
        source_focus_str, target_focus_str = parse_blended_word(blended_word)
        source_focus_words = source_focus_str.split() if source_focus_str else []
        target_focus_words = target_focus_str.split() if target_focus_str else []

        print(f"\n[{i+1}/{len(keys_to_process)}] key={key}")
        print(f"  source : {source_prompt}")
        print(f"  target : {target_prompt}")
        print(f"  focus  : src={source_focus_words}  tgt={target_focus_words}")

        try:
            t0 = time.time()
            process_one_entry(
                entry_key=key,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                source_focus_words=source_focus_words,
                target_focus_words=target_focus_words,
                save_dir=save_dir,
                infinity=infinity,
                vae=vae,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                scale_schedule=scale_schedule,
                attn_block_indices=attn_block_indices,
                args=args,
            )
            elapsed = time.time() - t0
            print(f"  完成，耗時 {elapsed:.1f}s")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 錯誤（{key}）：{e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print("\n" + "=" * 60)
    print("批次生成完成")
    print(f"  成功：{success_count}  跳過：{skip_count}  錯誤：{error_count}")
    print(f"  輸出目錄：{osp.abspath(args.output_dir)}")
    print("=" * 60)
