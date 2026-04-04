#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pie_bench_batch.py — PIE-Bench 五配置批次實驗

配置：
  1. run_p2p  num_source_scales=1
  2. run_p2p  num_source_scales=2
  3. run_p2p  num_source_scales=4
  4. run_p2p  num_source_scales=6
  5. run_p2p_attn  (attention-guided, num_full_replace=4, threshold=75)

使用方式：
  python3 tools/run_pie_bench_batch.py --mode validate   # 每 category 2 cases
  python3 tools/run_pie_bench_batch.py --mode full        # 全部 700 cases
"""

import os
import sys
import re
import json
import time
import argparse
import copy
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image

from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


# ============================================================
# 共用工具函式（從 run_p2p.py 擷取）
# ============================================================

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        prompt = aug_with_positive_prompt(prompt)
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


def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child',
                'person', 'human', 'adult', 'teenager', 'employee',
                'employer', 'worker', 'mother', 'father', 'sister',
                'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def gen_one_img(
    infinity_test, vae, text_tokenizer, text_encoder,
    prompt, cfg_list=[], tau_list=[], negative_prompt='',
    scale_schedule=None, top_k=900, top_p=0.97,
    cfg_sc=3, cfg_exp_k=0.0, cfg_insertion_layer=-5,
    vae_type=0, gumbel=0, softmax_merge_topk=-1,
    gt_leak=-1, gt_ls_Bl=None, g_seed=None,
    sampling_per_bits=1, enable_positive_prompt=0,
    p2p_token_storage=None, p2p_token_replace_prob=0.5,
    p2p_use_mask=False, p2p_save_tokens=True,
    p2p_attn_full_replace_scales=0,
):
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    negative_label_B_or_BLT = None
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)

    kwargs = dict(
        vae=vae, scale_schedule=scale_schedule,
        label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
        B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
        cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list,
        top_k=top_k, top_p=top_p,
        returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
        cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
        vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
        ret_img=True, trunk_scale=1000,
        gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
        sampling_per_bits=sampling_per_bits,
        p2p_token_storage=p2p_token_storage,
        p2p_token_replace_prob=p2p_token_replace_prob,
        p2p_use_mask=p2p_use_mask,
        p2p_save_tokens=p2p_save_tokens,
    )
    # p2p_attn_full_replace_scales only supported by infinity_p2p_attn
    if p2p_attn_full_replace_scales > 0:
        kwargs['p2p_attn_full_replace_scales'] = p2p_attn_full_replace_scales

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        _, _, img_list = infinity_test.autoregressive_infer_cfg(**kwargs)

    img = img_list[0]
    del img_list
    return img


def save_image(img_tensor, path):
    img_np = img_tensor.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_np)


def clean_prompt(prompt):
    return re.sub(r'\[([^\]]*)\]', r'\1', prompt).strip()


# ============================================================
# 模型載入
# ============================================================

def load_tokenizer(t5_path):
    text_tokenizer = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda').eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


def load_vae(vae_type, vae_path, apply_spatial_patchify=0):
    from infinity.models.bsq_vae.vae import vae_model
    codebook_dim = vae_type
    codebook_size = 2 ** codebook_dim
    if apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]
    vae = vae_model(
        vae_path, "dynamic", codebook_dim, codebook_size,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        test_mode=True,
    ).to('cuda')
    return vae


def load_infinity_model(model_cls, model_path, vae, model_kwargs, args):
    """Load an Infinity model (either p2p or p2p_attn variant)."""
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity = model_cls(
            vae_local=vae, text_channels=args.text_channels,
            text_maxlen=512, shared_aln=True,
            raw_scale_schedule=None,
            checkpointing='full-block',
            customized_flash_attn=False, fused_norm=True,
            pad_to_multiplier=128, use_flex_attn=args.use_flex_attn,
            add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
            use_bit_label=args.use_bit_label,
            rope2d_each_sa_layer=args.rope2d_each_sa_layer,
            rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
            pn=args.pn,
            apply_spatial_patchify=args.apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to('cuda')

        if args.bf16:
            for block in infinity.unregistered_blocks:
                block.bfloat16()

        infinity.eval()
        infinity.requires_grad_(False)
        infinity.cuda()
        torch.cuda.empty_cache()

        state_dict = torch.load(model_path, map_location='cuda')
        print(infinity.load_state_dict(state_dict))
        infinity.rng = torch.Generator(device='cuda')
    return infinity


# ============================================================
# 收集 PIE-Bench cases
# ============================================================

def collect_cases(bench_dir, max_per_cat=-1, categories=None):
    """返回 [(category, case_id, meta_dict), ...]"""
    cases = []
    cat_dirs = sorted(
        d for d in os.listdir(bench_dir)
        if os.path.isdir(os.path.join(bench_dir, d))
    )
    if categories:
        cat_dirs = [d for d in cat_dirs if d in categories]

    for cat in cat_dirs:
        cat_path = os.path.join(bench_dir, cat)
        case_ids = sorted(
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        )
        if max_per_cat > 0:
            case_ids = case_ids[:max_per_cat]
        for cid in case_ids:
            meta_path = os.path.join(cat_path, cid, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            cases.append((cat, cid, meta))

    return cases


# ============================================================
# Config 1-4: run_p2p style (token replacement by scale count)
# ============================================================

def run_p2p_config(
    infinity, vae, text_tokenizer, text_encoder,
    cases, num_source_scales, result_dir, args, scale_schedule,
):
    config_name = f"p2p_s{num_source_scales}"
    print(f"\n{'='*80}")
    print(f"[Config: {config_name}] num_source_scales={num_source_scales}")
    print(f"{'='*80}")

    total = len(cases)
    for i, (cat, cid, meta) in enumerate(cases):
        case_dir = os.path.join(result_dir, config_name, cat, cid)
        target_path = os.path.join(case_dir, 'target.jpg')

        # Skip if already done
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"  [{i+1}/{total}] {cat}/{cid} — already done, skip")
            continue

        source_prompt = clean_prompt(meta.get('source_prompt', ''))
        target_prompt = clean_prompt(meta.get('target_prompt', ''))

        print(f"  [{i+1}/{total}] {cat}/{cid}")
        print(f"    src: {source_prompt[:60]}...")
        print(f"    tgt: {target_prompt[:60]}...")

        # Step 1: Generate source, collect tokens
        storage = BitwiseTokenStorage(num_scales=num_source_scales, device='cpu')
        with autocast(dtype=torch.bfloat16), torch.no_grad():
            source_img = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                p2p_token_storage=storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
            )

        save_image(source_img, os.path.join(case_dir, 'source.jpg'))
        del source_img
        torch.cuda.empty_cache()

        # Step 2: Generate target with P2P guidance
        with autocast(dtype=torch.bfloat16), torch.no_grad():
            target_img = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                p2p_token_storage=storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=False,
                p2p_save_tokens=False,
            )

        save_image(target_img, target_path)
        del target_img, storage
        torch.cuda.empty_cache()

    print(f"[{config_name}] Done.")


# ============================================================
# Config 5: run_p2p_attn style (attention-guided)
# ============================================================

def run_p2p_attn_config(
    infinity_attn, vae, text_tokenizer, text_encoder,
    cases, result_dir, args, scale_schedule,
):
    from attention_map.extractor import CrossAttentionExtractor
    # Import attn helper functions
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from run_p2p_attn import (
        find_focus_token_indices,
        collect_attention_text_masks,
        combine_and_store_masks,
    )

    config_name = "p2p_attn"
    num_full_replace = args.attn_num_full_replace_scales
    threshold = args.attn_threshold_percentile
    total_scales = len(scale_schedule)

    # Determine attention block range
    depth = len(infinity_attn.unregistered_blocks)
    attn_block_start = depth // 2
    attn_block_end = depth - 1
    attn_block_indices = list(range(attn_block_start, attn_block_end + 1))

    print(f"\n{'='*80}")
    print(f"[Config: {config_name}] num_full_replace={num_full_replace}, threshold={threshold}")
    print(f"  Attention blocks: {attn_block_start}~{attn_block_end}")
    print(f"{'='*80}")

    total = len(cases)
    for i, (cat, cid, meta) in enumerate(cases):
        case_dir = os.path.join(result_dir, config_name, cat, cid)
        target_path = os.path.join(case_dir, 'target.jpg')

        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"  [{i+1}/{total}] {cat}/{cid} — already done, skip")
            continue

        source_prompt = clean_prompt(meta.get('source_prompt', ''))
        target_prompt = clean_prompt(meta.get('target_prompt', ''))

        # Extract focus words from blended_words
        blended = meta.get('blended_words', [])
        if blended:
            half = len(blended) // 2
            source_focus = blended[:half] if half > 0 else blended[:1]
            target_focus = blended[half:] if half > 0 else blended[:1]
        else:
            source_focus = []
            target_focus = []

        print(f"  [{i+1}/{total}] {cat}/{cid}")
        print(f"    src: {source_prompt[:50]}... focus={source_focus}")
        print(f"    tgt: {target_prompt[:50]}... focus={target_focus}")

        # Find focus token indices
        src_focus_idx = []
        tgt_focus_idx = []
        if source_focus:
            src_focus_idx = find_focus_token_indices(
                text_tokenizer, source_prompt, source_focus, verbose=False,
            )
        if target_focus:
            tgt_focus_idx = find_focus_token_indices(
                text_tokenizer, target_prompt, target_focus, verbose=False,
            )

        # Phase 1: Source generation + attention extraction
        storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

        src_extractor = CrossAttentionExtractor(
            model=infinity_attn,
            block_indices=attn_block_indices,
            batch_idx=0,
            aggregate_method="mean",
        )
        if src_focus_idx:
            src_extractor.register_patches()

        with autocast(dtype=torch.bfloat16), torch.no_grad():
            source_img = gen_one_img(
                infinity_attn, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                p2p_token_storage=storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
            )

        if src_focus_idx:
            src_extractor.remove_patches()
            src_extractor.get_summary()

        save_image(source_img, os.path.join(case_dir, 'source.jpg'))
        del source_img
        torch.cuda.empty_cache()

        # Phase 1.5: Source attention masks
        source_text_masks = {}
        if src_focus_idx and len(src_extractor.attention_maps) > 0:
            source_text_masks = collect_attention_text_masks(
                extractor=src_extractor,
                focus_token_indices=src_focus_idx,
                scale_schedule=scale_schedule,
                num_full_replace_scales=num_full_replace,
                attn_block_indices=attn_block_indices,
                threshold_percentile=threshold,
                label="source",
            )

        # Phase 1.7: Target free generation for attention
        target_text_masks = {}
        if tgt_focus_idx:
            tgt_extractor = CrossAttentionExtractor(
                model=infinity_attn,
                block_indices=attn_block_indices,
                batch_idx=0,
                aggregate_method="mean",
            )
            tgt_extractor.register_patches()

            with autocast(dtype=torch.bfloat16), torch.no_grad():
                _ = gen_one_img(
                    infinity_attn, vae, text_tokenizer, text_encoder,
                    target_prompt,
                    g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                    p2p_token_storage=None,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=False,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=0,
                )

            tgt_extractor.remove_patches()
            tgt_extractor.get_summary()

            target_text_masks = collect_attention_text_masks(
                extractor=tgt_extractor,
                focus_token_indices=tgt_focus_idx,
                scale_schedule=scale_schedule,
                num_full_replace_scales=num_full_replace,
                attn_block_indices=attn_block_indices,
                threshold_percentile=threshold,
                label="target",
            )
            del tgt_extractor
            torch.cuda.empty_cache()

        # Phase 1.9: Combine masks
        masks_stored = 0
        if source_text_masks or target_text_masks:
            masks_stored = combine_and_store_masks(
                source_text_masks=source_text_masks,
                target_text_masks=target_text_masks,
                scale_schedule=scale_schedule,
                p2p_token_storage=storage,
                num_full_replace_scales=num_full_replace,
            )

        del src_extractor
        torch.cuda.empty_cache()

        # Phase 2: Target generation with attention-guided P2P
        with autocast(dtype=torch.bfloat16), torch.no_grad():
            target_img = gen_one_img(
                infinity_attn, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=args.seed, cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                p2p_token_storage=storage,
                p2p_token_replace_prob=args.p2p_token_replace_prob,
                p2p_use_mask=(masks_stored > 0),
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=num_full_replace,
            )

        save_image(target_img, target_path)
        del target_img, storage
        torch.cuda.empty_cache()

    print(f"[{config_name}] Done.")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIE-Bench 五配置批次實驗')

    # Mode
    parser.add_argument('--mode', type=str, default='validate',
                        choices=['validate', 'full'],
                        help='validate=每category 2 cases, full=全部 700')
    parser.add_argument('--max_per_cat', type=int, default=-1,
                        help='覆蓋每 category 最大 case 數（-1=按 mode 決定）')

    # Paths
    parser.add_argument('--bench_dir', type=str,
                        default='outputs/outputs_loop_exp/extracted_pie_bench')
    parser.add_argument('--result_base', type=str,
                        default='outputs/outputs_loop_exp')
    parser.add_argument('--eval_base', type=str,
                        default='outputs/eval_pie')

    # Model paths
    parser.add_argument('--model_path', type=str,
                        default='weights/infinity_2b_reg.pth')
    parser.add_argument('--vae_path', type=str,
                        default='weights/infinity_vae_d32_reg.pth')
    parser.add_argument('--text_encoder_ckpt', type=str,
                        default='weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001')

    # Model config
    parser.add_argument('--pn', type=str, default='1M')
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1)
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2)
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0)
    parser.add_argument('--use_bit_label', type=int, default=1)
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0)
    parser.add_argument('--use_flex_attn', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1)
    parser.add_argument('--h_div_w_template', type=float, default=1.0)

    # Generation config
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--sampling_per_bits', type=int, default=1)
    parser.add_argument('--p2p_token_replace_prob', type=float, default=0.5)

    # Attn config
    parser.add_argument('--attn_num_full_replace_scales', type=int, default=4)
    parser.add_argument('--attn_threshold_percentile', type=float, default=75.0)

    # Which configs to run
    parser.add_argument('--configs', type=str, default='all',
                        help='Comma-separated: p2p_s1,p2p_s2,p2p_s4,p2p_s6,p2p_attn or "all"')

    # Skip eval
    parser.add_argument('--skip_eval', action='store_true',
                        help='只跑生成，不跑評估')

    args = parser.parse_args()

    # Determine max_per_cat
    if args.max_per_cat < 0:
        args.max_per_cat = 2 if args.mode == 'validate' else -1

    # Parse configs
    if args.configs == 'all':
        run_configs = ['p2p_s1', 'p2p_s2', 'p2p_s4', 'p2p_s6', 'p2p_attn']
    else:
        run_configs = [c.strip() for c in args.configs.split(',')]

    p2p_configs = [c for c in run_configs if c.startswith('p2p_s')]
    run_attn = 'p2p_attn' in run_configs

    # Model kwargs
    model_kwargs = dict(
        depth=32, embed_dim=2048, num_heads=2048 // 128,
        drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
    )

    # Collect cases
    cases = collect_cases(args.bench_dir, max_per_cat=args.max_per_cat)
    print(f"\n[PIE-Bench] {len(cases)} cases collected (mode={args.mode})")

    # Scale schedule
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(f"[Scale Schedule] {len(scale_schedule)} scales")

    # Load shared components
    print("\n[Loading] Text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt)
    print("[Loading] VAE...")
    vae = load_vae(args.vae_type, args.vae_path, args.apply_spatial_patchify)

    # ── Run P2P configs (1-4) ──
    if p2p_configs:
        print("\n[Loading] Infinity (P2P version)...")
        from infinity.models.infinity_p2p import Infinity as InfinityP2P
        infinity_p2p = load_infinity_model(
            InfinityP2P, args.model_path, vae, model_kwargs, args,
        )

        for config_name in p2p_configs:
            num_scales = int(config_name.split('_s')[1])
            run_p2p_config(
                infinity_p2p, vae, text_tokenizer, text_encoder,
                cases, num_scales,
                args.result_base, args, scale_schedule,
            )

        # Free P2P model if we also need attn
        if run_attn:
            del infinity_p2p
            torch.cuda.empty_cache()

    # ── Run P2P-Attn config (5) ──
    if run_attn:
        print("\n[Loading] Infinity (P2P-Attn version)...")
        from infinity.models.infinity_p2p_attn import Infinity as InfinityP2PAttn
        infinity_attn = load_infinity_model(
            InfinityP2PAttn, args.model_path, vae, model_kwargs, args,
        )

        run_p2p_attn_config(
            infinity_attn, vae, text_tokenizer, text_encoder,
            cases, args.result_base, args, scale_schedule,
        )

        del infinity_attn
        torch.cuda.empty_cache()

    # ── Eval ──
    if not args.skip_eval:
        print(f"\n{'='*80}")
        print("[Evaluation] Running eval for each config...")
        print(f"{'='*80}")

        for config_name in run_configs:
            result_dir = os.path.join(args.result_base, f"pie_bench_results_{config_name}")
            # Rename output dirs to match eval expectation
            actual_dir = os.path.join(args.result_base, config_name)
            eval_out = os.path.join(args.eval_base, config_name)
            os.makedirs(eval_out, exist_ok=True)

            eval_cmd = (
                f"{sys.executable} tools/eval_pie_results.py"
                f" --bench_dir {args.bench_dir}"
                f" --result_dir {actual_dir}"
                f" --output_csv {eval_out}/per_case.csv"
                f" --summary_json {eval_out}/summary.json"
                f" --skip_missing 1"
                f" --lpips_net alex"
                f" --clip_model ViT-L-14"
                f" --clip_pretrained openai"
                f" --source_from_result 1"
            )
            print(f"\n[Eval] {config_name}")
            print(f"  cmd: {eval_cmd}")
            os.system(eval_cmd)

    print(f"\n{'='*80}")
    print("[ALL DONE]")
    print(f"{'='*80}")
