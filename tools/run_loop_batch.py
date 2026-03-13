import os
import sys
# Add the parent directory to Python path to find the infinity module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add tools/ directory so we can import from run_loop
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import json
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

from infinity.models.infinityLoopFloat import Infinity
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

# ── reuse helpers from run_loop ──────────────────────────────────────────────
from run_loop import (
    encode_prompt,
    gen_one_img,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    add_common_arguments,
)


# =====================================================================
#  Helper functions
# =====================================================================

def build_all_schedules(args):
    """Return a list of (schedule, description) tuples to iterate over.

    Sources (checked in order, first non-empty wins):
      1. --schedule_file : path to a JSON file
      2. --schedules     : inline JSON string

    JSON structure (both sources):
    [
      {
        "name": "optional human-readable label",
        "rules": [{"scale":4,"rollback_to":2,"times":3}, ...]
      },
      ...
    ]

    A shorthand is also accepted – a plain list-of-rules is treated as
    a single experiment:
      [{"scale":4,"rollback_to":2,"times":3}]
    """
    raw = None

    # 1. file source
    if args.schedule_file:
        with open(args.schedule_file, 'r') as f:
            raw = json.load(f)
        print(f'[Schedule] Loaded from file: {args.schedule_file}')

    # 2. inline string
    if raw is None and args.schedules:
        raw = json.loads(args.schedules)
        print(f'[Schedule] Loaded from --schedules argument')

    if raw is None:
        return []

    # Normalise ──────────────────────────────────────────────────────────────
    experiments = []

    # Case A: list of experiment objects  [{"name":..., "rules":[...]}, ...]
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict) and 'rules' in raw[0]:
        for i, exp in enumerate(raw):
            rules = exp['rules']
            name = exp.get('name', None)
            if not name:
                name = '_'.join(f's{r["scale"]}rb{r["rollback_to"]}x{r["times"]}' for r in rules)
            experiments.append((rules, name))

    # Case B: list of lists  [[rule, ...], [rule, ...], ...]
    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        for i, rules in enumerate(raw):
            name = '_'.join(f's{r["scale"]}rb{r["rollback_to"]}x{r["times"]}' for r in rules)
            experiments.append((rules, name))

    # Case C: single flat list of rules  [{"scale":...}, ...]
    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        rules = raw
        name = '_'.join(f's{r["scale"]}rb{r["rollback_to"]}x{r["times"]}' for r in rules)
        experiments.append((rules, name))

    else:
        raise ValueError(f'Unrecognised schedule format: {type(raw)}')

    return experiments


def load_prompts_from_jsonl(jsonl_path):
    """Load prompts from a JSONL file. Each line: {"prompt": "..."}.

    Returns a list of dicts, each with at least a 'prompt' key.
    Extra fields in the JSONL are preserved.
    """
    prompts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'[Warning] Skipping line {line_no} in {jsonl_path}: {e}')
                continue
            if 'prompt' not in entry:
                print(f'[Warning] Skipping line {line_no}: no "prompt" key')
                continue
            prompts.append(entry)
    return prompts


def extract_render_texts(prompt):
    """Extract quoted text from a prompt for text-rendering evaluation.

    Looks for patterns like:
      reads "OPEN"
      says "HELLO WORLD"
      text "FOO"
      "BAR" (standalone quotes)

    Returns a list of extracted strings.
    """
    # Match text inside double quotes (escaped or regular)
    texts = re.findall(r'"([^"]+)"', prompt)
    # Also try escaped quotes from JSON
    texts += re.findall(r'\\"([^\\"]+)\\"', prompt)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def make_prompt_dir_name(prompt_idx, prompt_text):
    """Create a filesystem-safe directory name from a prompt.

    Format: prompt_{idx:03d}_{short_label}
    The short_label is derived from the first quoted text (render target)
    or from the first few words of the prompt.
    """
    # Try to extract the first quoted text as a label
    render_texts = extract_render_texts(prompt_text)
    if render_texts:
        label = render_texts[0]
    else:
        # Use first few words
        label = ' '.join(prompt_text.split()[:4])

    # Make filesystem-safe: keep alphanumeric, spaces→underscores
    label = re.sub(r'[^\w\s-]', '', label)
    label = re.sub(r'\s+', '_', label.strip())
    label = label[:40]  # truncate

    return f"prompt_{prompt_idx:03d}_{label}"


def save_prompt_info(prompt_dir, prompt_text, prompt_idx, extra_fields=None):
    """Save prompt metadata as JSON for later text-rendering evaluation.

    Creates prompt_info.json with:
      - prompt: full prompt text
      - prompt_index: 1-based index
      - render_texts: list of texts expected to appear in the image
      - extra fields from the JSONL entry
    """
    render_texts = extract_render_texts(prompt_text)
    info = {
        "prompt_index": prompt_idx,
        "prompt": prompt_text,
        "render_texts": render_texts,
    }
    if extra_fields:
        for k, v in extra_fields.items():
            if k != 'prompt':
                info[k] = v

    info_path = os.path.join(prompt_dir, "prompt_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info_path


def generate_and_save(
    infinity, vae, text_tokenizer, text_encoder,
    prompt, seed, args, scale_schedule,
    rollback_schedule, rollback_merge_mode,
    save_path,
):
    """Generate one image and save it. Returns elapsed time."""
    t0 = time.time()
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=args.cfg, tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                rollback_schedule=rollback_schedule,
                rollback_merge_mode=rollback_merge_mode,
            )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_np = generated_image.cpu().numpy()
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img_np)
    elapsed = time.time() - t0

    del generated_image, img_np
    torch.cuda.empty_cache()
    return elapsed


# =====================================================================
#  Single-prompt mode (original behavior)
# =====================================================================

def run_single_prompt(args, infinity, vae, text_tokenizer, text_encoder,
                      scale_schedule, rollback_merge_mode, experiments, save_dir):
    """Original single-prompt batch: base + each schedule → one image each."""
    total_runs = (0 if args.skip_base else 1) + len(experiments)
    print(f'\n[Plan] {total_runs} image(s) to generate '
          f'(base={"skip" if args.skip_base else "yes"}, '
          f'rollback experiments={len(experiments)})')
    for i, (rules, name) in enumerate(experiments):
        print(f'  [{i+1}] {name}')
        for r in rules:
            print(f'       scale {r["scale"]} → rollback to {r["rollback_to"]}, '
                  f'{r["times"]} time(s)')

    run_idx = 0

    # Base image
    if not args.skip_base:
        run_idx += 1
        print(f'\n{"="*80}')
        print(f'[{run_idx}/{total_runs}] Generating BASE image (no rollback)')
        print('='*80)
        save_path = os.path.join(save_dir, "output_base.jpg")
        elapsed = generate_and_save(
            infinity, vae, text_tokenizer, text_encoder,
            args.prompt, args.seed, args, scale_schedule,
            None, rollback_merge_mode, save_path,
        )
        print(f'[✓] Base image saved to {save_path}  ({elapsed:.1f}s)')

    # Rollback experiments
    for exp_i, (rules, name) in enumerate(experiments):
        run_idx += 1
        print(f'\n{"="*80}')
        print(f'[{run_idx}/{total_runs}] Experiment: {name}')
        for r in rules:
            print(f'    scale {r["scale"]} → rollback to {r["rollback_to"]}, '
                  f'{r["times"]} time(s)')
        print('='*80)

        save_path = os.path.join(save_dir, f"output_rollback_{name}.jpg")
        elapsed = generate_and_save(
            infinity, vae, text_tokenizer, text_encoder,
            args.prompt, args.seed, args, scale_schedule,
            rules, rollback_merge_mode, save_path,
        )
        print(f'[✓] Saved: {save_path}  ({elapsed:.1f}s)')

    return run_idx


# =====================================================================
#  Multi-prompt mode
# =====================================================================

def run_multi_prompt(args, infinity, vae, text_tokenizer, text_encoder,
                     scale_schedule, rollback_merge_mode, experiments, save_dir):
    """Multi-prompt batch: for each prompt → for each schedule → N images.

    Output structure:
        save_dir/
        ├── prompt_001_OPEN/
        │   ├── prompt_info.json
        │   ├── base/
        │   │   ├── seed_1.jpg
        │   │   ├── seed_2.jpg
        │   │   └── ...
        │   ├── s4rb2x3/
        │   │   ├── seed_1.jpg
        │   │   └── ...
        │   └── ...
        ├── prompt_002_LATTE/
        │   └── ...
        └── experiment_summary.json
    """
    prompts = load_prompts_from_jsonl(args.prompt_file)
    num_images = args.num_images
    base_seed = args.seed
    seeds = [base_seed + i for i in range(num_images)]

    num_settings = (0 if args.skip_base else 1) + len(experiments)
    total_images = len(prompts) * num_settings * num_images
    print(f'\n[Multi-Prompt Plan]')
    print(f'  Prompts:              {len(prompts)}')
    print(f'  Settings (base+exp):  {num_settings}')
    print(f'  Images per setting:   {num_images}')
    print(f'  Seeds:                {seeds}')
    print(f'  Total images:         {total_images}')
    print()

    global_img_idx = 0
    t_total_start = time.time()

    for p_idx, prompt_entry in enumerate(prompts, 1):
        prompt_text = prompt_entry['prompt']
        prompt_dir_name = make_prompt_dir_name(p_idx, prompt_text)
        prompt_dir = os.path.join(save_dir, prompt_dir_name)
        os.makedirs(prompt_dir, exist_ok=True)

        # Save prompt metadata
        info_path = save_prompt_info(prompt_dir, prompt_text, p_idx, prompt_entry)

        print(f'\n{"#"*80}')
        print(f'  PROMPT [{p_idx}/{len(prompts)}]: {prompt_text[:80]}...'
              if len(prompt_text) > 80 else
              f'  PROMPT [{p_idx}/{len(prompts)}]: {prompt_text}')
        print(f'  Dir: {prompt_dir_name}')
        render_texts = extract_render_texts(prompt_text)
        if render_texts:
            print(f'  Render texts: {render_texts}')
        print(f'{"#"*80}')

        # ── Base images ──────────────────────────────────────────────────
        if not args.skip_base:
            base_dir = os.path.join(prompt_dir, "base")
            os.makedirs(base_dir, exist_ok=True)
            for s_i, seed in enumerate(seeds):
                global_img_idx += 1
                save_path = os.path.join(base_dir, f"seed_{seed}.jpg")
                print(f'  [{global_img_idx}/{total_images}] base / seed={seed}', end=' ')
                elapsed = generate_and_save(
                    infinity, vae, text_tokenizer, text_encoder,
                    prompt_text, seed, args, scale_schedule,
                    None, rollback_merge_mode, save_path,
                )
                print(f'→ {elapsed:.1f}s')

        # ── Rollback experiments ─────────────────────────────────────────
        for exp_i, (rules, name) in enumerate(experiments):
            exp_dir = os.path.join(prompt_dir, name)
            os.makedirs(exp_dir, exist_ok=True)
            for s_i, seed in enumerate(seeds):
                global_img_idx += 1
                save_path = os.path.join(exp_dir, f"seed_{seed}.jpg")
                print(f'  [{global_img_idx}/{total_images}] {name} / seed={seed}', end=' ')
                elapsed = generate_and_save(
                    infinity, vae, text_tokenizer, text_encoder,
                    prompt_text, seed, args, scale_schedule,
                    rules, rollback_merge_mode, save_path,
                )
                print(f'→ {elapsed:.1f}s')

    # ── Save experiment summary ──────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    summary = {
        "total_prompts": len(prompts),
        "total_settings": num_settings,
        "images_per_setting": num_images,
        "total_images": global_img_idx,
        "seeds": seeds,
        "merge_mode": rollback_merge_mode,
        "skip_base": args.skip_base,
        "schedule_file": args.schedule_file,
        "prompt_file": args.prompt_file,
        "elapsed_seconds": round(total_elapsed, 1),
        "experiments": [
            {"name": name, "rules": rules}
            for rules, name in experiments
        ],
    }
    summary_path = os.path.join(save_dir, "experiment_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[✓] Experiment summary saved to {summary_path}')

    return global_img_idx


# =====================================================================
#  Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch Loop-Rollback: load model ONCE, then sweep over '
                    'multiple rollback schedules and/or multiple prompts.')
    add_common_arguments(parser)

    # ── prompt source (mutually exclusive in practice) ───────────────────
    parser.add_argument('--prompt', type=str, default='a dog',
                        help='Single prompt for single-prompt mode.')
    parser.add_argument('--prompt_file', type=str, default='',
                        help='Path to a JSONL file with multiple prompts. '
                             'Each line: {"prompt": "..."}. '
                             'Enables multi-prompt mode.')

    # ── output ───────────────────────────────────────────────────────────
    parser.add_argument('--save_file', type=str, default='./outputs')

    # ── schedule source ──────────────────────────────────────────────────
    parser.add_argument('--schedule_file', type=str, default='',
                        help='Path to a JSON file containing a list of '
                             'rollback schedule experiments.')
    parser.add_argument('--schedules', type=str, default='',
                        help='Inline JSON string with a list of schedule '
                             'experiments (same format as the file).')

    # ── batch options ────────────────────────────────────────────────────
    parser.add_argument('--skip_base', action='store_true',
                        help='Skip generating the base (no-rollback) image.')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images per prompt×setting in multi-prompt '
                             'mode. Each uses seed, seed+1, ..., seed+N-1. '
                             '(default: 5)')
    args = parser.parse_args()

    # ── parse cfg ────────────────────────────────────────────────────────
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # =====================================================================
    #  LOAD MODELS ONCE
    # =====================================================================
    print('\n' + '='*80)
    print('LOADING MODELS (once for all experiments)')
    print('='*80)
    t_load_start = time.time()

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    rollback_merge_mode = args.rollback_merge_mode if args.rollback_merge_mode >= 0 else None

    print(f'[✓] All models loaded in {time.time() - t_load_start:.1f}s')
    print(f'    Scale schedule ({len(scale_schedule)} scales): {scale_schedule}')
    print(f'    Merge mode: {rollback_merge_mode}')

    # ── prepare output dir ───────────────────────────────────────────────
    save_dir = args.save_file
    if save_dir.startswith('/') and not save_dir.startswith(os.path.expanduser('~')):
        save_dir = '.' + save_dir
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ── collect experiments ──────────────────────────────────────────────
    experiments = build_all_schedules(args)
    for i, (rules, name) in enumerate(experiments):
        print(f'  Schedule [{i+1}] {name}')
        for r in rules:
            print(f'       scale {r["scale"]} → rollback to {r["rollback_to"]}, '
                  f'{r["times"]} time(s)')

    # =====================================================================
    #  Dispatch: multi-prompt or single-prompt
    # =====================================================================
    if args.prompt_file:
        print(f'\n[Mode] Multi-prompt (from {args.prompt_file})')
        total_images = run_multi_prompt(
            args, infinity, vae, text_tokenizer, text_encoder,
            scale_schedule, rollback_merge_mode, experiments, save_dir,
        )
    else:
        print(f'\n[Mode] Single-prompt: "{args.prompt}"')
        total_images = run_single_prompt(
            args, infinity, vae, text_tokenizer, text_encoder,
            scale_schedule, rollback_merge_mode, experiments, save_dir,
        )

    # =====================================================================
    #  DONE
    # =====================================================================
    print(f'\n{"="*80}')
    print('BATCH LOOP ROLLBACK COMPLETE')
    print('='*80)
    print(f'Total images: {total_images}')
    print(f'Results saved to: {save_dir}/')
    print('='*80 + '\n')
