#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mid-scale IDResampler inference.

Non-ID scales use the prompt condition with the subject token removed.
ID scales use IDResampler(face_id) replacement at the subject token positions.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_  # noqa: E402
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402
from tools.face_swap_utils import AdaFaceClient  # noqa: E402
from tools.id_resampler import extract_orig_sks_from_text_features  # noqa: E402
from tools.infer_id_resampler import (  # noqa: E402
    _print_diagnostics,
    _t5_forward,
    build_resampler,
)
from tools.run_p2p_edit import (  # noqa: E402
    add_common_arguments,
    find_focus_token_indices,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)


def _compact_from_rows(rows: torch.Tensor) -> Tuple[torch.Tensor, list, torch.Tensor, int]:
    rows = rows.contiguous()
    L = int(rows.shape[0])
    cu = torch.tensor([0, L], dtype=torch.int32, device=rows.device)
    return rows, [L], cu, L


def _remove_indices_condition(text_features: torch.Tensor, mask: torch.Tensor, token_indices) -> Tuple:
    valid_len = int(mask.sum().item())
    drop = set(int(x) for x in token_indices)
    keep = [i for i in range(valid_len) if i not in drop]
    if not keep:
        raise ValueError("removing subject tokens would leave an empty text condition")
    return _compact_from_rows(text_features[0, keep, :])


def _compact_from_features(text_features: torch.Tensor, mask: torch.Tensor) -> Tuple:
    valid_len = int(mask.sum().item())
    return _compact_from_rows(text_features[0, :valid_len, :])


def _prepare_condition(infinity, text_cond_tuple, B: int, use_cfg: bool):
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
    if use_cfg:
        kv_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_un[total:total + le] = infinity.cfg_uncond[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
        bs = 2 * B
    else:
        bs = B

    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
    kv_compact = infinity.text_proj_for_ca(kv_compact)
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    with torch.amp.autocast("cuda", enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    sos = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    return {
        "bs": bs,
        "sos": sos,
        "cond_BD": cond_BD,
        "cond_BD_or_gss": cond_BD_or_gss,
        "ca_kv": ca_kv,
    }


@torch.no_grad()
def autoregressive_infer_midscale_conditions(
    infinity,
    vae,
    scale_schedule,
    cond_by_scale: Dict[int, Tuple],
    B: int,
    g_seed: int,
    cfg_list,
    tau_list,
    cfg_insertion_layer,
    vae_type: int,
    top_k: int,
    top_p: float,
    sampling_per_bits: int,
):
    if g_seed is None:
        rng = None
    else:
        infinity.rng.manual_seed(int(g_seed))
        rng = infinity.rng

    if infinity.apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    use_cfg_by_scale = [float(cfg_list[si]) != 1.0 for si in range(len(scale_schedule))]
    cond_cache = {
        si: _prepare_condition(infinity, cond_by_scale[si], B, use_cfg_by_scale[si])
        for si in range(len(scale_schedule))
    }

    leng = len(infinity.unregistered_blocks)
    abs_cfg_insertion_layers = []
    add_cfg_on_logits = False
    for item in cfg_insertion_layer:
        item = int(item)
        if item == 0:
            add_cfg_on_logits = True
        elif item < 0:
            abs_cfg_insertion_layers.append(leng + item)
        elif item != 1:
            raise ValueError(f"cfg_insertion_layer={item} is not supported")

    for block_chunk in infinity.block_chunks:
        for module in block_chunk.module:
            attn = getattr(module, "sa", None) or getattr(module, "attn")
            attn.kv_caching(True)

    ret = []
    idx_Bl_list = []
    idx_Bld_list = []
    summed_codes = 0
    cur_L = 0
    last_stage = cond_cache[0]["sos"]
    num_stages_minus_1 = len(scale_schedule) - 1

    try:
        for si, pn in enumerate(scale_schedule):
            cur_L += int(np.array(pn).prod())
            cfg = float(cfg_list[si])
            cond = cond_cache[si]
            bs = int(cond["bs"])
            if last_stage.shape[0] != bs:
                last_stage = last_stage[:B].repeat(bs // B, 1, 1)

            attn_fn = None
            if infinity.use_flex_attn:
                attn_fn = infinity.attn_fn_compile_dict.get(tuple(scale_schedule[: si + 1]), None)

            layer_idx = 0
            for block_idx, block_chunk in enumerate(infinity.block_chunks):
                if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule)
                if not infinity.add_lvl_embeding_only_first_block:
                    last_stage = infinity.add_lvl_embeding(last_stage, si, scale_schedule)

                for module in block_chunk.module:
                    last_stage = module(
                        x=last_stage,
                        cond_BD=cond["cond_BD_or_gss"],
                        ca_kv=cond["ca_kv"],
                        attn_bias_or_two_vector=None,
                        attn_fn=attn_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=infinity.rope2d_freqs_grid,
                        scale_ind=si,
                    )
                    if cfg != 1.0 and layer_idx in abs_cfg_insertion_layers:
                        last_stage = cfg * last_stage[:B] + (1.0 - cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

            if cfg != 1.0 and add_cfg_on_logits:
                logits_BlV = infinity.get_logits(last_stage, cond["cond_BD"]).mul(1 / float(tau_list[si]))
                logits_BlV = cfg * logits_BlV[:B] + (1.0 - cfg) * logits_BlV[B:]
            else:
                logits_BlV = infinity.get_logits(last_stage[:B], cond["cond_BD"][:B]).mul(1 / float(tau_list[si]))

            if infinity.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_bits = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_bits,
                    rng=rng,
                    top_k=top_k or infinity.top_k,
                    top_p=top_p or infinity.top_p,
                    num_samples=1,
                )[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k or infinity.top_k,
                    top_p=top_p or infinity.top_p,
                    num_samples=1,
                )[:, :, 0]

            if vae_type != 0:
                assert pn[0] == 1
                idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
                if infinity.apply_spatial_patchify:
                    idx_Bld = idx_Bld.permute(0, 3, 1, 2)
                    idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
                    idx_Bld = idx_Bld.permute(0, 2, 3, 1)
                idx_Bld = idx_Bld.unsqueeze(1)
                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(
                        codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up
                    )
                    last_stage = F.interpolate(
                        summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up
                    )
                    last_stage = last_stage.squeeze(-3)
                    if infinity.apply_spatial_patchify:
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)
                else:
                    summed_codes += codes
            else:
                h_BChw = infinity.quant_only_used_in_inference[0].embedding(idx_Bl).float()
                h_BChw = h_BChw.transpose_(1, 2).reshape(
                    B, infinity.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2]
                )
                ret.append(h_BChw)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    _, last_stage = infinity.quant_only_used_in_inference[0].one_step_fuse(
                        si, num_stages_minus_1 + 1, None, h_BChw, scale_schedule
                    )

            if si != num_stages_minus_1:
                last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
                last_stage = last_stage.repeat(cond_cache[si + 1]["bs"] // B, 1, 1)
    finally:
        for block_chunk in infinity.block_chunks:
            for module in block_chunk.module:
                attn = getattr(module, "sa", None) or getattr(module, "attn")
                attn.kv_caching(False)

    if vae_type != 0:
        img = vae.decode(summed_codes.squeeze(-3))
    else:
        img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)
    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    return ret, idx_Bl_list, img


def main() -> None:
    parser = argparse.ArgumentParser(description="Mid-scale IDResampler inference")
    add_common_arguments(parser)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--face_image", type=str, required=True)
    parser.add_argument("--subject_token", type=str, default="sks")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs/id_resampler_midscale")
    parser.add_argument("--out_prefix", type=str, default="")
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--resampler_ckpt", type=str, default="")
    parser.add_argument("--resampler_ckpt_glob", type=str, default="",
                        help="若提供，會在同一個 Python process 內依序跑所有 matching ckpt")
    parser.add_argument("--resampler_n_tokens", type=int, default=1)
    parser.add_argument("--resampler_n_id_ctx", type=int, default=4)
    parser.add_argument("--resampler_n_layers", type=int, default=2)
    parser.add_argument("--resampler_n_heads", type=int, default=8)
    parser.add_argument("--resampler_use_prompt_ctx", type=int, default=1, choices=[0, 1])
    parser.add_argument("--resampler_anchor_word", type=str, default="person")
    parser.add_argument("--resampler_delta_max_norm", type=float, default=-1.0)
    parser.add_argument("--resampler_out_norm_match", type=str, default="none",
                        choices=["none", "anchor", "base"])
    parser.add_argument("--resampler_residual_base", type=str, default="anchor",
                        choices=["anchor", "orig"])
    parser.add_argument("--resampler_match_orig_norm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--id_scale_start", type=int, default=4)
    parser.add_argument("--id_scale_end", type=int, default=6)
    parser.add_argument("--inject_alpha", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)

    args = parser.parse_args()
    args.cfg = list(map(float, str(args.cfg).split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    if not os.path.exists(args.face_image):
        raise FileNotFoundError(args.face_image)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Mid-scale IDResampler inference")
    print("=" * 80)
    print(f"prompt        : {args.prompt}")
    print(f"face_image    : {args.face_image}")
    print(f"id scales     : {args.id_scale_start}..{args.id_scale_end} (0-based)")
    print(f"non-id scales : remove '{args.subject_token}' condition token")
    print(f"ckpt          : {args.resampler_ckpt or '(uninitialized)'}")
    print(f"ckpt glob     : {args.resampler_ckpt_glob or '(none)'}")
    print("=" * 80 + "\n")

    print("[Init] Loading models...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    print("[Init] Done.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    if args.id_scale_start < 0 or args.id_scale_end < args.id_scale_start:
        raise ValueError("invalid id scale range")
    if args.id_scale_end >= len(scale_schedule):
        raise ValueError(f"id_scale_end={args.id_scale_end} exceeds total scales {len(scale_schedule)}")

    client = AdaFaceClient(url=args.adaface_url)
    client.health()
    e_A = client.embed_files([args.face_image])[0]
    e_A_t = torch.from_numpy(e_A.astype(np.float32))

    sks_idx = find_focus_token_indices(
        text_tokenizer, args.prompt, [args.subject_token], verbose=False
    )
    if not sks_idx:
        raise ValueError(f"subject_token '{args.subject_token}' not found in prompt")
    print(f"[Tokens] subject positions: {sks_idx}")

    text_features, mask = _t5_forward(text_tokenizer, text_encoder, args.prompt, device)
    orig_sks = text_features[0, sks_idx, :].detach().clone()
    no_sks_cond = _remove_indices_condition(text_features, mask, sks_idx)

    cfg_list = args.cfg if isinstance(args.cfg, list) else [float(args.cfg)] * len(scale_schedule)
    if len(cfg_list) < len(scale_schedule):
        cfg_list = cfg_list + [cfg_list[-1]] * (len(scale_schedule) - len(cfg_list))
    tau_list = [float(args.tau)] * len(scale_schedule)

    ckpts = []
    if args.resampler_ckpt_glob.strip():
        ckpts = sorted(glob.glob(args.resampler_ckpt_glob.strip()))
        if not ckpts:
            print(f"[Ckpt] no matches for glob: {args.resampler_ckpt_glob}; fallback to --resampler_ckpt")
    if not ckpts:
        ckpts = [args.resampler_ckpt.strip()]
    print(f"[Ckpt] will run {len(ckpts)} checkpoint(s)")

    root_out_dir = args.out_dir
    multi_ckpt_mode = bool(args.resampler_ckpt_glob.strip()) or len(ckpts) > 1
    for ckpt_i, ckpt in enumerate(ckpts):
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0] if ckpt else "uninitialized"
        if ckpt and not os.path.exists(ckpt):
            print(f"[Skip] ckpt not found: {ckpt}")
            continue

        args.resampler_ckpt = ckpt
        this_out_dir = os.path.join(root_out_dir, ckpt_name) if multi_ckpt_mode else root_out_dir
        os.makedirs(this_out_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"[Ckpt {ckpt_i + 1}/{len(ckpts)}] {ckpt or '(uninitialized)'}")
        print(f"[Ckpt] output dir: {this_out_dir}")
        print("=" * 80)

        resampler = build_resampler(args, text_tokenizer, text_encoder, device)
        orig_sks_emb = extract_orig_sks_from_text_features(
            text_features, sks_idx, resampler.n_tokens
        ).to(device)
        base_emb_arg = orig_sks_emb if resampler.residual_base == "orig" else None

        with torch.no_grad():
            r_out = resampler(
                id_feat=e_A_t.to(device).unsqueeze(0),
                prompt_ctx=text_features if resampler.use_prompt_ctx else None,
                prompt_mask=mask.bool() if resampler.use_prompt_ctx else None,
                base_emb=base_emb_arg,
            )
        replacement = r_out[0]
        _print_diagnostics(
            orig_sks=orig_sks,
            anchor=resampler.anchor.detach().to(text_features.device),
            replacement_vec=replacement,
            sks_idx=sks_idx,
            label="resampler",
            base_emb=orig_sks_emb,
        )

        id_tf = text_features.clone()
        alpha = float(args.inject_alpha)
        if replacement.shape[0] == 1:
            for idx in sks_idx:
                src = replacement[0]
                if bool(args.resampler_match_orig_norm):
                    eps = 1e-8
                    orig_norm = text_features[0, idx, :].norm(p=2).clamp_min(eps)
                    src = src / src.norm(p=2).clamp_min(eps) * orig_norm
                id_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
        elif replacement.shape[0] == len(sks_idx):
            for k, idx in enumerate(sks_idx):
                src = replacement[k]
                if bool(args.resampler_match_orig_norm):
                    eps = 1e-8
                    orig_norm = text_features[0, idx, :].norm(p=2).clamp_min(eps)
                    src = src / src.norm(p=2).clamp_min(eps) * orig_norm
                id_tf[0, idx, :] = (1.0 - alpha) * text_features[0, idx, :] + alpha * src
        else:
            raise ValueError(
                f"replacement n_tokens={replacement.shape[0]} must be 1 or len(sks_idx)={len(sks_idx)}"
            )

        id_cond = _compact_from_features(id_tf, mask)
        cond_by_scale = {
            si: (id_cond if args.id_scale_start <= si <= args.id_scale_end else no_sks_cond)
            for si in range(len(scale_schedule))
        }

        base_name = args.out_prefix.strip() or ckpt_name
        for i in range(int(args.n_samples)):
            seed = int(args.seed) + i
            out_path = os.path.join(this_out_dir, f"{base_name}__midscale__seed{seed}.jpg")
            t0 = time.time()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                _, _, img_list = autoregressive_infer_midscale_conditions(
                    infinity=infinity,
                    vae=vae,
                    scale_schedule=scale_schedule,
                    cond_by_scale=cond_by_scale,
                    B=1,
                    g_seed=seed,
                    cfg_list=cfg_list,
                    tau_list=tau_list,
                    cfg_insertion_layer=[int(args.cfg_insertion_layer)],
                    vae_type=int(args.vae_type),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    sampling_per_bits=int(args.sampling_per_bits),
                )
            img = img_list[0].cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, img)
            print(f"  [{i + 1}/{args.n_samples}] seed={seed} saved {out_path} ({time.time() - t0:.1f}s)")

        del resampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Done]")


if __name__ == "__main__":
    main()
