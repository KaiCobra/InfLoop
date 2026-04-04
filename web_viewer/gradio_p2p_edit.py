"""
P2P-Edit Gradio Demo — 模型常駐記憶體，互動式圖像編輯

啟動方式：
    cd /home/avlab/Documents/InfLoop
    python web_viewer/gradio_p2p_edit.py

模型只在啟動時載入一次，之後每次互動都直接推論。
"""

import os
import sys
import time
import datetime

# 將專案根目錄加入 Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image
from torch.cuda.amp import autocast

torch._dynamo.config.cache_size_limit = 64

# ── 從 run_p2p_edit.py 匯入核心元件 ──
from tools.run_p2p_edit import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
    encode_image_to_raw_features,
    encode_image_to_scale_tokens,
    find_focus_token_indices,
    parse_focus_words_arg,
    merge_focus_terms,
    derive_focus_terms_from_prompt_diff,
    collect_attention_text_masks,
    combine_and_store_masks,
    build_cumulative_replacement_prob_masks,
    CrossAttentionExtractor,
)
from infinity.utils.bitwise_token_storage import BitwiseTokenStorage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# ============================================================
# 固定模型參數（與 infer_p2p_edit.sh 一致）
# ============================================================
MODEL_CONFIG = dict(
    pn='1M',
    model_type='infinity_2b',
    model_path='weights/infinity_2b_reg.pth',
    vae_type=32,
    vae_path='weights/infinity_vae_d32reg.pth',
    cfg=4.0,
    tau=0.5,
    text_encoder_ckpt='weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001',
    text_channels=2048,
    use_scale_schedule_embedding=0,
    use_bit_label=1,
    add_lvl_embeding_only_first_block=1,
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    apply_spatial_patchify=0,
    checkpoint_type='torch',
    cfg_insertion_layer=0,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    bf16=1,
)

# ============================================================
# 全域模型物件（啟動時載入一次）
# ============================================================
text_tokenizer = None
text_encoder = None
vae = None
infinity = None


def load_models():
    """載入所有模型到 GPU，只呼叫一次。"""
    global text_tokenizer, text_encoder, vae, infinity

    print("=" * 60)
    print("[Gradio] 載入模型中 ...")
    print("=" * 60)

    # 建立 args-like 物件供 load 函式使用
    class Args:
        pass
    args = Args()
    for k, v in MODEL_CONFIG.items():
        setattr(args, k, v)
    args.cache_dir = '/dev/shm'
    args.enable_model_cache = 0
    args.use_flex_attn = 0
    args.h_div_w_template = 1.0

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    print("=" * 60)
    print("[Gradio] 模型載入完成！")
    print("=" * 60)


# ============================================================
# 推論主函式
# ============================================================

def run_p2p_edit(
    source_image_pil,
    source_prompt,
    target_prompt,
    source_focus_words,
    target_focus_words,
    image_injection_scales,
    attn_threshold_percentile,
    use_cumulative_prob_mask,
    h_div_w_template,
    seed,
):
    """執行一次 P2P-Edit 推論，回傳 (source_img, target_img, status)。"""
    if source_image_pil is None:
        return None, None, "請上傳 source image。"

    t_start = time.time()

    # ── 參數準備 ──
    cfg = MODEL_CONFIG['cfg']
    tau = MODEL_CONFIG['tau']
    pn = MODEL_CONFIG['pn']
    seed = int(seed)
    image_injection_scales = int(image_injection_scales)
    num_full_replace_scales = image_injection_scales  # 保持一致
    phase17_fallback_replace_scales = 4
    attn_block_start = 2
    attn_block_end = -1

    # Scale schedule
    h_div_w = float(h_div_w_template)
    scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    total_scales = len(scale_schedule)

    # Attention block 範圍
    depth = len(infinity.unregistered_blocks)
    abs_start = (depth // 2) if attn_block_start < 0 else min(attn_block_start, depth - 1)
    abs_end = (depth - 1) if attn_block_end < 0 else min(attn_block_end, depth - 1)
    attn_block_indices = list(range(abs_start, abs_end + 1))

    # Focus words
    source_focus_words_list = parse_focus_words_arg(source_focus_words)
    target_focus_words_list = parse_focus_words_arg(target_focus_words)

    # Auto diff
    auto_src, auto_tgt = derive_focus_terms_from_prompt_diff(source_prompt, target_prompt)
    source_focus_words_list = merge_focus_terms(source_focus_words_list, auto_src)
    target_focus_words_list = merge_focus_terms(target_focus_words_list, auto_tgt)

    source_focus_token_indices = find_focus_token_indices(
        text_tokenizer, source_prompt, source_focus_words_list, verbose=True,
    ) if source_focus_words_list else []
    target_focus_token_indices = find_focus_token_indices(
        text_tokenizer, target_prompt, target_focus_words_list, verbose=True,
    ) if target_focus_words_list else []

    device_cuda = torch.device('cuda')

    # ── Phase 0：編碼 Source Image ──
    source_pil = source_image_pil.convert('RGB')
    image_raw_features = encode_image_to_raw_features(
        vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
        device=device_cuda, apply_spatial_patchify=False,
    )
    image_scale_tokens = encode_image_to_scale_tokens(
        vae=vae, pil_img=source_pil, scale_schedule=scale_schedule,
        device=device_cuda, apply_spatial_patchify=False,
    )

    inject_schedule = [
        0.0 if si < image_injection_scales else 1.0
        for si in range(total_scales)
    ]

    # ── Phase 1：Source 生成 + Attention 擷取 ──
    p2p_token_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')

    source_extractor = CrossAttentionExtractor(
        model=infinity, block_indices=attn_block_indices,
        batch_idx=0, aggregate_method="mean",
    )
    if source_focus_token_indices:
        source_extractor.register_patches()

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            source_img_tensor = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                source_prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=cfg, tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                vae_type=MODEL_CONFIG['vae_type'],
                sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=True,
                p2p_attn_full_replace_scales=0,
                inject_image_features=image_raw_features,
                inject_schedule=inject_schedule,
            )

    if source_focus_token_indices:
        source_extractor.remove_patches()

    # ── Phase 1.5：Source Focus Mask ──
    source_text_masks = {}
    source_low_attn_masks = {}
    if source_focus_token_indices and len(source_extractor.attention_maps) > 0:
        source_text_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=attn_threshold_percentile,
            label="source", low_attn=False,
        )
        source_low_attn_masks = collect_attention_text_masks(
            extractor=source_extractor,
            focus_token_indices=source_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=attn_threshold_percentile,
            label="source_preserve", low_attn=True,
        )

    # ── Phase 1.6：Preserve Storage ──
    phase17_storage = None
    if source_low_attn_masks and image_scale_tokens:
        phase17_storage = BitwiseTokenStorage(num_scales=total_scales, device='cpu')
        for si, low_mask in source_low_attn_masks.items():
            if si not in image_scale_tokens:
                continue
            phase17_storage.tokens[si] = image_scale_tokens[si].clone()
            mask_tensor = torch.tensor(
                low_mask, dtype=torch.bool
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            phase17_storage.masks[si] = mask_tensor.cpu()

    # ── Phase 1.7：Target 生成 → Target Focus Mask ──
    target_text_masks = {}
    if target_focus_token_indices:
        target_extractor = CrossAttentionExtractor(
            model=infinity, block_indices=attn_block_indices,
            batch_idx=0, aggregate_method="mean",
        )
        target_extractor.register_patches()

        _phase17_storage = phase17_storage
        _phase17_use_mask = (phase17_storage is not None)
        _phase17_full_replace = num_full_replace_scales
        if _phase17_storage is None and phase17_fallback_replace_scales > 0 and p2p_token_storage.tokens:
            _phase17_storage = p2p_token_storage
            _phase17_use_mask = False
            _phase17_full_replace = phase17_fallback_replace_scales

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                _phase17_img = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder,
                    target_prompt,
                    g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                    cfg_list=cfg, tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                    vae_type=MODEL_CONFIG['vae_type'],
                    sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                    enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                    p2p_token_storage=_phase17_storage,
                    p2p_token_replace_prob=0.0,
                    p2p_use_mask=_phase17_use_mask,
                    p2p_save_tokens=False,
                    p2p_attn_full_replace_scales=_phase17_full_replace,
                    inject_image_features=None,
                    inject_schedule=None,
                )
        del _phase17_img

        target_extractor.remove_patches()
        target_text_masks = collect_attention_text_masks(
            extractor=target_extractor,
            focus_token_indices=target_focus_token_indices,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
            attn_block_indices=attn_block_indices,
            threshold_percentile=attn_threshold_percentile,
            label="target",
        )
        torch.cuda.empty_cache()

    # ── Phase 1.9：合併 Mask + 覆寫 storage ──
    if source_text_masks or target_text_masks:
        combine_and_store_masks(
            source_text_masks=source_text_masks,
            target_text_masks=target_text_masks,
            scale_schedule=scale_schedule,
            p2p_token_storage=p2p_token_storage,
            num_full_replace_scales=num_full_replace_scales,
        )

    if use_cumulative_prob_mask and p2p_token_storage.masks:
        p2p_token_storage.masks = build_cumulative_replacement_prob_masks(
            masks=p2p_token_storage.masks,
            scale_schedule=scale_schedule,
            num_full_replace_scales=num_full_replace_scales,
        )

    if image_scale_tokens:
        for si_tok, tok in image_scale_tokens.items():
            p2p_token_storage.tokens[si_tok] = tok

    del source_img_tensor
    torch.cuda.empty_cache()

    # ── Phase 2：Target 生成 ──
    has_mask = len(p2p_token_storage.masks) > 0

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            target_img_tensor = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                target_prompt,
                g_seed=seed, gt_leak=0, gt_ls_Bl=None,
                cfg_list=cfg, tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[MODEL_CONFIG['cfg_insertion_layer']],
                vae_type=MODEL_CONFIG['vae_type'],
                sampling_per_bits=MODEL_CONFIG['sampling_per_bits'],
                enable_positive_prompt=MODEL_CONFIG['enable_positive_prompt'],
                p2p_token_storage=p2p_token_storage,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=has_mask,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=num_full_replace_scales,
                inject_image_features=None,
                inject_schedule=None,
            )

    # ── 輸出圖片轉換 ──
    target_np = target_img_tensor.cpu().numpy()
    if target_np.dtype != np.uint8:
        target_np = np.clip(target_np, 0, 255).astype(np.uint8)
    # OpenCV BGR → RGB for Gradio
    target_rgb = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)

    # 同時存檔
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "outputs", f"gradio_p2p_edit_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    source_pil.save(os.path.join(save_dir, "source_input.jpg"))
    cv2.imwrite(os.path.join(save_dir, "target.jpg"), target_np)

    del target_img_tensor, target_np, p2p_token_storage
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    status = (
        f"完成！耗時 {elapsed:.1f}s\n"
        f"Source focus: {source_focus_words_list}\n"
        f"Target focus: {target_focus_words_list}\n"
        f"Scale schedule: {total_scales} scales, h/w={h_div_w}\n"
        f"結果已存於: {save_dir}/"
    )
    return source_pil, Image.fromarray(target_rgb), status


# ============================================================
# Gradio UI
# ============================================================

def build_ui():
    with gr.Blocks(title="P2P-Edit Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# P2P-Edit 互動式圖像編輯")
        gr.Markdown("上傳 source image，設定 prompt 與參數，即可產生編輯結果。模型已常駐記憶體。")

        with gr.Row():
            # ── 左欄：輸入 ──
            with gr.Column(scale=1):
                source_image = gr.Image(
                    label="Source Image（拖曳上傳）",
                    type="pil",
                    height=350,
                )
                source_prompt = gr.Textbox(
                    label="Source Prompt",
                    value="A oil paint of Girl with a Pearl Earring.",
                )
                target_prompt = gr.Textbox(
                    label="Target Prompt",
                    value="A oil paint of Green Frog with a Pearl Earring.",
                )
                source_focus_words = gr.Textbox(
                    label="Source Focus Words",
                    value="Girl",
                    placeholder="以逗號分隔多個 phrase",
                )
                target_focus_words = gr.Textbox(
                    label="Target Focus Words",
                    value="Green Frog",
                    placeholder="以逗號分隔多個 phrase",
                )
                seed = gr.Number(label="Seed", value=1, precision=0)

            # ── 右欄：參數 + 輸出 ──
            with gr.Column(scale=1):
                image_injection_scales = gr.Slider(
                    minimum=0, maximum=6, step=1, value=2,
                    label="Image Injection Scales（前 N 個 scale 注入 source image）",
                )
                attn_threshold_percentile = gr.Slider(
                    minimum=0, maximum=100, step=1, value=20,
                    label="Attention Threshold Percentile（愈低 → focus 區域愈大）",
                )
                use_cumulative_prob_mask = gr.Checkbox(
                    label="使用跨尺度累積機率遮罩 (Cumulative Prob Mask)",
                    value=False,
                )
                h_div_w_template = gr.Radio(
                    choices=[1.0, 1.25, 1.333, 1.5, 1.75, 2.0, 2.5, 3.0],
                    value=1.0,
                    label="H / W 比例（圖片長寬比）",
                )

                run_btn = gr.Button("開始編輯", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            source_output = gr.Image(label="Source 重建", height=400)
            target_output = gr.Image(label="Target 編輯結果", height=400)

        status_box = gr.Textbox(label="執行狀態", lines=4, interactive=False)

        run_btn.click(
            fn=run_p2p_edit,
            inputs=[
                source_image,
                source_prompt,
                target_prompt,
                source_focus_words,
                target_focus_words,
                image_injection_scales,
                attn_threshold_percentile,
                use_cumulative_prob_mask,
                h_div_w_template,
                seed,
            ],
            outputs=[source_output, target_output, status_box],
        )

    return demo


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    load_models()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
