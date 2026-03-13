"""
Extract IQR-Filtered Cross-Attention Maps (Block 33 only)

Reads prompts from a JSONL file, generates images, and outputs ONLY the
IQR-filtered (non-outlier) averaged cross-attention map for each word × scale.

Output structure per prompt:
    prompt_XXX/
        generated.jpg
        prompt_info.json
        words/
            <word>/
                attention/
                    scale_00.npy          # raw (H, W) float32
                    scale_01.npy
                    ...
                overlay/                  # optional visualisation
                    scale_00.jpg
                    ...
                metadata.json

Usage:
    python -m attention_map.run \
        --input  exp_prompts/prompts.jsonl \
        --output outputs/filtered_attention
"""

import sys
import os
import re
import json
import time
import argparse
import traceback

# ---------------------------------------------------------------------------
# Path setup — works whether this file lives at <project>/attention_map/run.py
# or is invoked via  python -m attention_map.run  from the project root.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tools.run_infinity import (
    load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from attention_map.extractor import CrossAttentionExtractor


# ============================================================================
# Filesystem helpers
# ============================================================================

_SANITIZE_MAP = str.maketrans({
    '.': '_DOT_',
    '/': '_SLASH_',
    '\\': '_BSLASH_',
    ':': '_COLON_',
    '"': '_DQUOTE_',
    "'": '_SQUOTE_',
    ' ': '_',
})


def sanitize_dirname(name: str) -> str:
    """Convert a word/token into a safe directory name."""
    safe = name.translate(_SANITIZE_MAP)
    # Strip leading/trailing underscores and collapse repeats
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe or '_EMPTY_'


# ============================================================================
# IQR filtering (Block 33 logic)
# ============================================================================

def iqr_filtered_mean(
    attn_stack: torch.Tensor,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute IQR-filtered (non-outlier) mean of attention maps.

    Args:
        attn_stack: (num_blocks, H, W)

    Returns:
        filtered_attn: (H, W) numpy float32
        num_outliers: number of blocks removed
        num_used: number of blocks kept
    """
    num_blocks = attn_stack.shape[0]
    attn_mean = attn_stack.mean(dim=0)  # (H, W)

    # MSE per block
    mse = torch.sum((attn_stack - attn_mean.unsqueeze(0)) ** 2, dim=[1, 2])

    # IQR outlier detection
    q1 = torch.quantile(mse, 0.25)
    q3 = torch.quantile(mse, 0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    keep = mse <= threshold
    num_outliers = int((~keep).sum().item())
    num_used = int(keep.sum().item())

    if num_used > 0:
        filtered = attn_stack[keep].mean(dim=0).cpu().numpy()
    else:
        # Fallback: all flagged → use plain mean
        filtered = attn_mean.cpu().numpy()
        num_used = num_blocks

    return filtered, num_outliers, num_used


# ============================================================================
# Visualisation helper
# ============================================================================

def overlay_attention(
    attn_map: np.ndarray,
    background: np.ndarray,
    alpha: float = 0.4,
    target_size: int = 1024,
) -> np.ndarray:
    """Create a JET-coloured attention overlay on the background image."""
    a_min, a_max = attn_map.min(), attn_map.max()
    if a_max > a_min:
        norm = (attn_map - a_min) / (a_max - a_min)
    else:
        norm = attn_map

    attn_resized = cv2.resize(norm, (target_size, target_size),
                              interpolation=cv2.INTER_NEAREST)
    bg_resized = cv2.resize(background, (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    return cv2.addWeighted(bg_resized, 1 - alpha, heatmap, alpha, 0)


# ============================================================================
# Prompt word grouping (copied from separated_batch_extract_attention.py)
# ============================================================================

def group_prompt_words(
    prompt: str,
    token_ids: List[int],
    tokenizer,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """
    Group T5 sub-word token indices by the original words in the prompt.

    Returns:
        word_groups:   {"word": [token_idx, ...], ...}
        id_to_text:    {"token_idx": "decoded_text", ...}
    """
    id_to_text: Dict[str, str] = {}
    for i in range(len(token_ids)):
        id_to_text[str(i)] = tokenizer.decode([token_ids[i]])

    raw_words = re.findall(r'\w+|[^\w\s]', prompt)

    word_groups: Dict[str, List[int]] = {}
    tidx = 0
    occurrence: Dict[str, int] = {}

    for word in raw_words:
        wl = word.lower()
        matched: List[int] = []
        reconstructed = ""

        while tidx < len(token_ids) and len(reconstructed.replace('▁', '').replace(' ', '')) < len(word):
            txt = id_to_text.get(str(tidx), "")
            if txt.strip() in ('<pad>', '</s>', '<unk>', ''):
                tidx += 1
                continue
            matched.append(tidx)
            reconstructed += txt.replace('▁', '').replace(' ', '')
            tidx += 1
            if reconstructed.lower() == wl:
                break

        if not matched:
            continue

        base = word
        if base in occurrence:
            key = f"{base}_{occurrence[base]}"
            occurrence[base] += 1
        else:
            if base == '"':
                key = f'{base}_1'
                occurrence[base] = 2
            else:
                key = base
                occurrence[base] = 1

        word_groups[key] = matched

    return word_groups, id_to_text


# ============================================================================
# CLI arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract IQR-filtered cross-attention maps (Block 33)")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSONL file with prompts")
    p.add_argument("--output", type=str, default="outputs/filtered_attention",
                   help="Output directory")
    p.add_argument("--blocks", nargs="+", type=int, default=list(range(32)),
                   help="Block indices to extract (default: 0-31)")
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Overlay transparency")
    p.add_argument("--save_overlay", action="store_true", default=True,
                   help="Save JPG overlays (default: True)")
    p.add_argument("--no_overlay", action="store_true",
                   help="Skip saving JPG overlays")
    # Model paths
    p.add_argument("--pn", type=str, default="1M")
    p.add_argument("--model_path", type=str,
                   default="/home/avlab/Infinity/weights/infinity_2b_reg.pth")
    p.add_argument("--vae_path", type=str,
                   default="/media/avlab/ee303_4T/SceneTxtVAR/weights/infinity_vae_d32_reg.pth")
    p.add_argument("--text_encoder_path", type=str,
                   default="/media/avlab/ee303_4T/SceneTxtVAR/weights/"
                           "models--google--flan-t5-xl/snapshots/"
                           "7d6315df2c2fb742f0f5b556879d730926ca9001")
    return p.parse_args()


# ============================================================================
# Model setup
# ============================================================================

def setup_model(args):
    print("⏳ Loading models...")
    model_args = argparse.Namespace(
        pn=args.pn,
        model_path=args.model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=args.vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type="infinity_2b",
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=args.text_encoder_path,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir="/dev/shm",
        checkpoint_type="torch",
        bf16=1,
        enable_model_cache=True,
    )
    text_tokenizer, text_encoder = load_tokenizer(t5_path=model_args.text_encoder_ckpt)
    vae = load_visual_tokenizer(model_args)
    infinity = load_transformer(vae, model_args)
    print("✅ Models loaded")
    return infinity, vae, text_tokenizer, text_encoder, model_args


# ============================================================================
# Process one prompt
# ============================================================================

def process_one_prompt(
    prompt_data: Dict,
    prompt_idx: int,
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    args,
    model_args,
    output_dir: Path,
):
    prompt = prompt_data.get("prompt", "")
    seed = prompt_data.get("seed", prompt_idx * 1000)
    cfg = prompt_data.get("cfg", args.cfg)
    tau = prompt_data.get("tau", args.tau)
    h_div_w = prompt_data.get("h_div_w", 1.0)

    # Tokenize
    token_ids = text_tokenizer([prompt], return_tensors="pt")["input_ids"][0].tolist()
    word_groups, id_to_text = group_prompt_words(prompt, token_ids, text_tokenizer)

    print(f"\n{'='*60}")
    print(f"Prompt {prompt_idx}: {prompt[:80]}...")
    print(f"  Words: {len(word_groups)}, Tokens: {len(token_ids)}, Seed: {seed}")
    print(f"{'='*60}")

    # Scale schedule
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    num_scales = len(scale_schedule)

    # Extractor (all blocks needed for IQR)
    extractor = CrossAttentionExtractor(
        model=infinity,
        block_indices=args.blocks,
        batch_idx=1,
    )
    extractor.register_patches()
    extractor.hook_vae_decoder(vae, scale_schedule, infinity_model=infinity)

    try:
        # Generate image
        print("🎨 Generating image...")
        with torch.inference_mode():
            generated_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder, prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=cfg, tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[0],
                vae_type=32, sampling_per_bits=1,
                enable_positive_prompt=0,
            )
        print("✅ Image generated")

        # Decode intermediate images
        extractor.decode_intermediate_images()
        extractor.get_summary()

        # Output dirs
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        image_np = generated_image.cpu().numpy()
        cv2.imwrite(str(prompt_dir / "generated.jpg"), image_np)

        save_overlay = args.save_overlay and not args.no_overlay

        # ---------------------------------------------------------------
        # For each word: collect all blocks → IQR filter → save
        # ---------------------------------------------------------------
        for word, token_indices in word_groups.items():
            token_texts = [id_to_text.get(str(i), "").strip() for i in token_indices]
            print(f"\n  📝 Word '{word}' (tokens: {token_texts})")

            word_dir = prompt_dir / "words" / sanitize_dirname(word)
            attn_dir = word_dir / "attention"
            attn_dir.mkdir(parents=True, exist_ok=True)
            if save_overlay:
                overlay_dir = word_dir / "overlay"
                overlay_dir.mkdir(parents=True, exist_ok=True)

            word_meta: Dict[str, object] = {
                "word": word,
                "token_indices": token_indices,
                "token_texts": token_texts,
                "scales": [],
            }

            for scale_idx in range(num_scales):
                _, h, w = scale_schedule[scale_idx]

                # Collect attention from every block for this scale
                block_attns: List[torch.Tensor] = []
                for bidx in args.blocks:
                    raw = extractor.extract_word_attention(
                        block_idx=bidx,
                        scale_idx=scale_idx,
                        token_indices=token_indices,
                        spatial_size=(h, w),
                    )
                    if raw is not None:
                        block_attns.append(torch.from_numpy(raw))

                if len(block_attns) < 3:
                    print(f"    Scale {scale_idx} ({h}×{w}): "
                          f"only {len(block_attns)} blocks, skipping IQR")
                    if block_attns:
                        filtered = torch.stack(block_attns).mean(dim=0).numpy()
                        n_out, n_used = 0, len(block_attns)
                    else:
                        continue
                else:
                    stack = torch.stack(block_attns)  # (num_blocks, H, W)
                    filtered, n_out, n_used = iqr_filtered_mean(stack)

                # Save raw numpy
                npy_path = attn_dir / f"scale_{scale_idx:02d}.npy"
                np.save(str(npy_path), filtered.astype(np.float32))

                # Save overlay
                if save_overlay:
                    bg = (extractor.intermediate_images[scale_idx]
                          if scale_idx < len(extractor.intermediate_images)
                          else image_np)
                    vis = overlay_attention(filtered, bg, alpha=args.alpha)
                    cv2.imwrite(str(overlay_dir / f"scale_{scale_idx:02d}.jpg"), vis)

                word_meta["scales"].append({
                    "scale_idx": scale_idx,
                    "spatial_size": [h, w],
                    "num_outliers_removed": n_out,
                    "num_blocks_used": n_used,
                })
                print(f"    Scale {scale_idx:2d} ({h:3d}×{w:3d}): "
                      f"removed {n_out}/{len(block_attns)} outliers → "
                      f"used {n_used} blocks")

            # ---------------------------------------------------------------
            # attn_mean: average of scale_02 ~ scale_12 (resized to max res)
            # ---------------------------------------------------------------
            MEAN_START, MEAN_END = 2, min(num_scales - 1, 12)
            # Determine target resolution from the largest scale in the range
            _, target_h, target_w = scale_schedule[MEAN_END]

            maps_to_avg: List[np.ndarray] = []
            for si in range(MEAN_START, MEAN_END + 1):
                npy_file = attn_dir / f"scale_{si:02d}.npy"
                if npy_file.exists():
                    m = np.load(str(npy_file))
                    # Resize to target resolution using bilinear interpolation
                    if m.shape != (target_h, target_w):
                        m = cv2.resize(m, (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR)
                    maps_to_avg.append(m)

            if maps_to_avg:
                attn_mean = np.mean(maps_to_avg, axis=0).astype(np.float32)
                np.save(str(attn_dir / "attn_mean.npy"), attn_mean)

                if save_overlay:
                    bg = (extractor.intermediate_images[-1]
                          if extractor.intermediate_images else image_np)
                    vis = overlay_attention(attn_mean, bg, alpha=args.alpha)
                    cv2.imwrite(str(overlay_dir / "attn_mean.jpg"), vis)

                word_meta["attn_mean"] = {
                    "scales_used": list(range(MEAN_START, MEAN_END + 1)),
                    "target_spatial_size": [target_h, target_w],
                    "num_scales_averaged": len(maps_to_avg),
                }
                print(f"    📊 attn_mean: avg of scale_{MEAN_START:02d}~"
                      f"scale_{MEAN_END:02d} → ({target_h}×{target_w})")
            else:
                print(f"    ⚠️  attn_mean: no scale data in range "
                      f"{MEAN_START}–{MEAN_END}")

            # Save per-word metadata
            with open(word_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(word_meta, f, indent=2, ensure_ascii=False)

        # Save prompt-level info
        prompt_info = {
            "prompt": prompt,
            "seed": seed,
            "cfg": cfg,
            "tau": tau,
            "h_div_w": h_div_w,
            "num_words": len(word_groups),
            "num_scales": num_scales,
            "num_blocks_source": len(args.blocks),
            "word_groups": {w: idxs for w, idxs in word_groups.items()},
            "scale_schedule": [[s, h, w] for s, h, w in scale_schedule],
        }
        with open(prompt_dir / "prompt_info.json", "w", encoding="utf-8") as f:
            json.dump(prompt_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Prompt {prompt_idx} done → {prompt_dir}")

    finally:
        extractor.restore_vae_decoder(vae)
        extractor.remove_patches()
        extractor.clear_maps()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Input:  {input_path}")
    print(f"📂 Output: {output_dir}")
    print(f"🎯 Blocks: {args.blocks[0]}–{args.blocks[-1]} ({len(args.blocks)} blocks)")
    print(f"📊 Only IQR-filtered (block 33) attention will be saved\n")

    infinity, vae, text_tokenizer, text_encoder, model_args = setup_model(args)

    # Read JSONL
    prompts: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"📝 Loaded {len(prompts)} prompts\n")

    for idx, prompt_data in enumerate(prompts):
        # Skip already processed
        if (output_dir / f"prompt_{idx:03d}").exists():
            print(f"➡️  Prompt {idx} already exists, skipping")
            continue
        try:
            process_one_prompt(
                prompt_data, idx,
                infinity, vae, text_tokenizer, text_encoder,
                args, model_args, output_dir,
            )
        except Exception as e:
            print(f"\n❌ Error on prompt {idx}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"✅ All done! Results → {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
