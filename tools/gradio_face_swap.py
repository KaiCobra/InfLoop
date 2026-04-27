#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gradio_face_swap.py — Face-Swap pipeline 的 Gradio Demo Page

UI：
  • Prompt (text)
  • Source face image (single)
  • λ₁ / λ₂ sliders（phase 2 線性混合：new_e_I = λ₁ * e_I + λ₂ * proj(e_A)）
  • Subject word（預設 "boy"）
  • Seed
  • Run 按鈕

Backend：模型只在啟動時載入一次；每次 Run 共用單一 lock 序列化 GPU 推論。

啟動：
  bash scripts/run_gradio_face_swap.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from typing import List, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.run_p2p_edit import (  # noqa: E402
    add_common_arguments,
    encode_prompt,
    find_focus_token_indices,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from tools.run_pie_edit_faceSwap import gen_one_img_kv, run_one_case_faceSwap  # noqa: E402
from tools.face_swap_utils import (  # noqa: E402
    AdaFaceClient,
    encode_prompt_with_face_op,
)
from tools.optimize_face_token import (  # noqa: E402
    build_bsc,
    optimize_v_A,
    save_v_A_cache,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # noqa: E402


DEFAULT_PROMPT = "a boy turned his head to his left over the shuolder and tilted up"


# ============================================================
# App: 模型載入一次，共享在所有 request 之間
# ============================================================

class FaceSwapApp:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()

        print("[Init] Loading models once...")
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args)
        self.infinity = load_transformer(self.vae, args)
        print("[Init] Model load complete.")

        scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
        self.scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        self.total_scales = len(self.scale_schedule)

        depth = len(self.infinity.unregistered_blocks)
        attn_block_start = (depth // 2) if args.attn_block_start < 0 else min(args.attn_block_start, depth - 1)
        attn_block_end = (depth - 1) if args.attn_block_end < 0 else min(args.attn_block_end, depth - 1)
        self.attn_block_indices = list(range(attn_block_start, attn_block_end + 1))
        self.device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.client = AdaFaceClient(url=args.adaface_url)
        try:
            h = self.client.health()
            print(f"[AdaFace] server OK: {h}")
        except Exception as exc:
            print(f"[AdaFace] ⚠ cannot reach server at {args.adaface_url}: {exc}")
            print("           （demo 仍會啟動，但 Run 時會失敗）")

        # B 圖快取：(prompt, seed) -> path on disk
        self._b_cache: dict = {}
        self._b_cache_lock = threading.Lock()

        # BitwiseSelfCorrection（textual inversion 用）
        self.bsc = build_bsc(self.vae, args)

        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(args.identity_cache_dir, exist_ok=True)

    # ────────────────────────────────────────
    # B 圖：依 (prompt, seed) 快取
    # ────────────────────────────────────────
    def _generate_or_get_B(self, prompt_t: str, seed: int) -> str:
        key = (prompt_t, int(seed))
        with self._b_cache_lock:
            cached = self._b_cache.get(key)
            if cached and os.path.exists(cached) and os.path.getsize(cached) > 0:
                return cached

        # 不在 cache 內 → 生成
        os.makedirs(self.args.work_dir, exist_ok=True)
        b_path = os.path.join(
            self.args.work_dir,
            f"B_{abs(hash((prompt_t, int(seed)))) & 0xffffffff:08x}.jpg",
        )
        kv_raw = encode_prompt(self.text_tokenizer, self.text_encoder, prompt_t)
        with torch.no_grad():
            img = gen_one_img_kv(
                self.infinity, self.vae,
                text_cond_tuple=kv_raw,
                g_seed=int(seed),
                gt_leak=0, gt_ls_Bl=None,
                cfg_list=self.args.cfg, tau_list=self.args.tau,
                scale_schedule=self.scale_schedule,
                cfg_insertion_layer=[self.args.cfg_insertion_layer],
                vae_type=self.args.vae_type,
                sampling_per_bits=self.args.sampling_per_bits,
                p2p_token_storage=None,
                p2p_token_replace_prob=0.0,
                p2p_use_mask=False,
                p2p_save_tokens=False,
                p2p_attn_full_replace_scales=0,
                inject_image_features=None,
                inject_schedule=None,
            )
        img_np = img.cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        cv2.imwrite(b_path, img_np)
        with self._b_cache_lock:
            self._b_cache[key] = b_path
        return b_path

    # ────────────────────────────────────────
    # 一次 face-swap inference
    # ────────────────────────────────────────
    def run(
        self,
        prompt_t: str,
        source_image: Optional[Image.Image],
        lam1: float,
        lam2: float,
        subject_word: str,
        seed: int,
        phase2_mode: str,
        identity_name: str,
    ):
        prompt_t = (prompt_t or "").strip()
        if not prompt_t:
            raise gr.Error("Prompt 不能為空")
        subject_word = (subject_word or "").strip()
        if not subject_word:
            raise gr.Error("subject_word 不能為空（例如 'boy'）")
        identity_name = (identity_name or "").strip()

        # learned 模式不需要 source image（直接讀 cache）；linear 模式需要
        if phase2_mode == "linear" and source_image is None:
            raise gr.Error("Linear 模式需要上傳 source face image（用來算 e_A）")
        if phase2_mode == "learned" and not identity_name:
            raise gr.Error("Learned 模式需要 identity name 來定位 weights/identities/<name>/v_A.pt")

        with self.lock:
            t0 = time.time()
            session_id = uuid.uuid4().hex[:10]
            session_dir = os.path.join(self.args.work_dir, f"session_{session_id}")
            os.makedirs(session_dir, exist_ok=True)

            # 1. 存上傳的 source face（若有）
            src_path = None
            if source_image is not None:
                src_path = os.path.join(session_dir, "src.png")
                source_image.convert("RGB").save(src_path)

            # 2. B 圖（依 prompt + seed cache）
            print(f"[run] prompt='{prompt_t}' seed={seed} mode={phase2_mode} id={identity_name}")
            b_path = self._generate_or_get_B(prompt_t, int(seed))

            # 3. e_B 永遠要算（phase 1.7 要做 subtract）
            e_b = self.client.embed_files([b_path])[0]
            e_b_norm = float(np.linalg.norm(e_b))
            e_b_t = torch.from_numpy(e_b.astype(np.float32))

            # 4. subject token indices
            subject_token_indices = find_focus_token_indices(
                self.text_tokenizer, prompt_t, [subject_word]
            )
            if not subject_token_indices:
                raise gr.Error(
                    f"subject_word='{subject_word}' 在 prompt 中找不到對應 token"
                )

            # 5. 三組 kv：phase 1 / 1.7 不變；phase 2 依 mode 切
            kv_phase1 = encode_prompt_with_face_op(
                self.text_tokenizer, self.text_encoder,
                prompt=prompt_t,
                face_emb_512=None, op_mode=None,
                subject_token_indices=subject_token_indices,
                verbose=bool(self.args.debug_face_op),
            )
            kv_phase17 = encode_prompt_with_face_op(
                self.text_tokenizer, self.text_encoder,
                prompt=prompt_t,
                face_emb_512=e_b_t, op_mode="subtract",
                subject_token_indices=subject_token_indices,
                verbose=bool(self.args.debug_face_op),
            )

            extra_info = {}
            if phase2_mode == "learned":
                v_A_path = os.path.join(self.args.identity_cache_dir, identity_name, "v_A.pt")
                if not os.path.exists(v_A_path):
                    raise gr.Error(
                        f"找不到 v_A.pt：{v_A_path}。請先按下 Train v_A 訓練，或改回 Linear 模式。"
                    )
                blob = torch.load(v_A_path, map_location="cpu", weights_only=False)
                v_A_tensor = blob["v_A"].float()
                kv_phase2 = encode_prompt_with_face_op(
                    self.text_tokenizer, self.text_encoder,
                    prompt=prompt_t,
                    op_mode="learned",
                    subject_token_indices=subject_token_indices,
                    learned_v_A=v_A_tensor,
                    verbose=bool(self.args.debug_face_op),
                )
                extra_info.update({
                    "phase2_mode": "learned",
                    "v_A_path": v_A_path,
                    "v_A_subject_word": blob.get("subject_word"),
                    "v_A_prompt_t": blob.get("prompt_t"),
                })
            else:  # linear
                e_a = self.client.embed_files([src_path])[0]
                e_a_norm = float(np.linalg.norm(e_a))
                cos_ab = float(np.dot(e_a, e_b))
                e_a_t = torch.from_numpy(e_a.astype(np.float32))
                kv_phase2 = encode_prompt_with_face_op(
                    self.text_tokenizer, self.text_encoder,
                    prompt=prompt_t,
                    face_emb_512=e_a_t, op_mode="linear",
                    subject_token_indices=subject_token_indices,
                    lam1=float(lam1),
                    lam2=float(lam2),
                    verbose=bool(self.args.debug_face_op),
                )
                extra_info.update({
                    "phase2_mode": "linear",
                    "lam1": float(lam1),
                    "lam2": float(lam2),
                    "e_A_norm": round(e_a_norm, 6),
                    "cos(e_A,e_B)": round(cos_ab, 6),
                })

            # 6. 把 seed 寫進 args 副本，避免污染原物件
            args_for_run = argparse.Namespace(**vars(self.args))
            args_for_run.seed = int(seed)

            # 7. 跑 face-swap pipeline
            run_one_case_faceSwap(
                infinity=self.infinity,
                vae=self.vae,
                text_tokenizer=self.text_tokenizer,
                text_encoder=self.text_encoder,
                source_image_path=b_path,
                prompt_text=prompt_t,
                subject_word=subject_word,
                kv_phase1=kv_phase1,
                kv_phase17=kv_phase17,
                kv_phase2=kv_phase2,
                save_dir=session_dir,
                args=args_for_run,
                scale_schedule=self.scale_schedule,
                attn_block_indices=self.attn_block_indices,
                total_scales=self.total_scales,
                device_cuda=self.device_cuda,
            )
            target_path = os.path.join(session_dir, "target.jpg")
            elapsed = time.time() - t0

            info = {
                "session": session_id,
                "session_dir": session_dir,
                "prompt": prompt_t,
                "subject_word": subject_word,
                "subject_token_indices": subject_token_indices,
                "seed": int(seed),
                "B_path": b_path,
                "target_path": target_path,
                "e_B_norm": round(e_b_norm, 6),
                "elapsed_sec": round(elapsed, 2),
                **extra_info,
            }

            b_pil = Image.open(b_path).convert("RGB")
            tgt_pil = Image.open(target_path).convert("RGB") if os.path.exists(target_path) else None
            return b_pil, tgt_pil, info

    # ────────────────────────────────────────
    # Textual Inversion: 訓練 v_A
    # ────────────────────────────────────────
    def train_v_A(
        self,
        source_image: Optional[Image.Image],
        prompt_t: str,
        subject_word: str,
        identity_name: str,
        steps: int,
        lr: float,
        l2_reg: float,
    ):
        if source_image is None:
            raise gr.Error("請先上傳 source face image")
        prompt_t = (prompt_t or "").strip()
        subject_word = (subject_word or "").strip()
        identity_name = (identity_name or "").strip()
        if not prompt_t or not subject_word or not identity_name:
            raise gr.Error("prompt / subject_word / identity_name 都必填")

        with self.lock:
            session_id = uuid.uuid4().hex[:10]
            stage_dir = os.path.join(self.args.work_dir, f"train_{identity_name}_{session_id}")
            os.makedirs(stage_dir, exist_ok=True)
            # 把上傳的圖存進 stage_dir，當成 identity_dir 給 optimize_v_A 讀
            src_path = os.path.join(stage_dir, "src.png")
            source_image.convert("RGB").save(src_path)

            # 用一個 namespace 包 args（optimize_v_A 用 args.steps/lr/l2_reg/log_every/seed/apply_spatial_patchify）
            args_ti = argparse.Namespace(**vars(self.args))
            args_ti.steps = int(steps)
            args_ti.lr = float(lr)
            args_ti.l2_reg = float(l2_reg)
            args_ti.log_every = max(1, int(steps) // 10)

            t0 = time.time()
            result = optimize_v_A(
                infinity=self.infinity,
                vae=self.vae,
                text_tokenizer=self.text_tokenizer,
                text_encoder=self.text_encoder,
                bsc=self.bsc,
                identity_dir=stage_dir,
                prompt_t=prompt_t,
                subject_word=subject_word,
                scale_schedule=self.scale_schedule,
                args=args_ti,
                device=self.device_cuda,
            )
            if result is None:
                raise gr.Error("Textual Inversion 失敗（看 console log 了解原因）")

            pt_path, meta_path = save_v_A_cache(
                self.args.identity_cache_dir, identity_name, result
            )
            elapsed = time.time() - t0
            print(f"[train_v_A] saved {pt_path} ({elapsed:.1f}s)")

            return {
                "identity": identity_name,
                "v_A_path": pt_path,
                "meta_path": meta_path,
                "init_loss": result["init_loss"],
                "final_loss": result["final_loss"],
                "iters": result["iters"],
                "lr": result["lr"],
                "l2_reg": result["l2_reg"],
                "elapsed_sec": round(elapsed, 2),
                "n_images": result["n_images"],
            }


# ============================================================
# UI
# ============================================================

def build_ui(app: FaceSwapApp) -> gr.Blocks:
    with gr.Blocks(title="Face-Swap (P2P-Edit) Demo") as demo:
        gr.Markdown(
            "## Face-Swap (P2P-Edit) — Gradio Demo\n"
            "**Phase 2 兩種模式**：\n"
            "- **linear**：`new_e_I = λ₁·e_I + λ₂·proj(e_A)`（AdaFace embedding，可能與 Infinity 空間不對齊）\n"
            "- **learned**：直接用 Textual Inversion 訓練好的 `v_A`（已對齊 Infinity 空間）— 先按 Train v_A"
        )
        with gr.Row():
            # ── 左欄：輸入 ──
            with gr.Column(scale=1):
                prompt_box = gr.Textbox(
                    label="Prompt T_t",
                    value=app.args.default_prompt,
                    lines=2,
                )
                source_img = gr.Image(
                    label="Source face image",
                    type="pil",
                    image_mode="RGB",
                    height=320,
                )
                subject_box = gr.Textbox(
                    label="Subject word（要操作的 token，例如 'boy'）",
                    value=app.args.default_subject,
                )
                identity_name_box = gr.Textbox(
                    label="Identity name（learned 模式必填；對應 weights/identities/<name>/）",
                    value="",
                    placeholder="例如 smith",
                )
                phase2_mode_radio = gr.Radio(
                    choices=["linear", "learned"],
                    value="linear",
                    label="Phase 2 mode",
                )
                with gr.Group():
                    gr.Markdown("**Linear-mode 參數（AdaFace 路徑）**")
                    lam1_slider = gr.Slider(
                        label="λ₁ (e_I weight)", minimum=0.0, maximum=2.0,
                        value=0.0, step=0.05,
                    )
                    lam2_slider = gr.Slider(
                        label="λ₂ (proj(e_A) weight)", minimum=0.0, maximum=2.0,
                        value=1.0, step=0.05,
                    )
                seed_box = gr.Number(
                    label="Seed", value=int(app.args.seed), precision=0,
                )
                run_btn = gr.Button("Run face swap", variant="primary")

            # ── 右欄：輸出 ──
            with gr.Column(scale=1):
                b_out = gr.Image(label="Base image B（從 prompt 生成）", type="pil", height=320)
                tgt_out = gr.Image(label="Face-swap result（target.jpg）", type="pil", height=320)
                info_out = gr.JSON(label="Run info")

        # ── Textual Inversion 區塊 ──
        with gr.Accordion("Train v_A (Textual Inversion)", open=False):
            gr.Markdown(
                "對左邊上傳的 source face image 跑 Textual Inversion，\n"
                "把 prompt 中 subject token 在 Infinity 自身 embedding 空間裡優化成代表這張臉的 `v_A`，\n"
                "存到 `weights/identities/<identity_name>/v_A.pt`。訓練完將 Phase 2 mode 切成 learned 重跑即可。"
            )
            with gr.Row():
                ti_steps = gr.Slider(
                    label="steps", minimum=20, maximum=500, value=150, step=10,
                )
                ti_lr = gr.Slider(
                    label="learning rate", minimum=1e-4, maximum=5e-3, value=1e-3, step=1e-4,
                )
                ti_l2 = gr.Slider(
                    label="L2 reg ||v_A−v_init||²", minimum=0.0, maximum=1e-2, value=1e-4, step=1e-4,
                )
            train_btn = gr.Button("Train v_A on this face", variant="secondary")
            train_info_out = gr.JSON(label="Training result")

        # ── 事件繫結 ──
        run_btn.click(
            fn=app.run,
            inputs=[
                prompt_box, source_img,
                lam1_slider, lam2_slider,
                subject_box, seed_box,
                phase2_mode_radio, identity_name_box,
            ],
            outputs=[b_out, tgt_out, info_out],
        )
        train_btn.click(
            fn=app.train_v_A,
            inputs=[
                source_img, prompt_box, subject_box, identity_name_box,
                ti_steps, ti_lr, ti_l2,
            ],
            outputs=[train_info_out],
        )

    return demo


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Face-Swap Gradio Demo")
    add_common_arguments(parser)

    # Server / app 設定
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", type=int, default=0, choices=[0, 1])
    parser.add_argument("--work_dir", type=str, default="./outputs/gradio_face_swap")
    parser.add_argument("--adaface_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--identity_cache_dir", type=str, default="./weights/identities",
                        help="存放 v_A.pt 的目錄；Train 按鈕寫入這裡，learned 模式從這裡讀")

    # Face-swap 預設值
    parser.add_argument("--default_prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--default_subject", type=str, default="boy")
    parser.add_argument("--debug_face_op", type=int, default=0, choices=[0, 1])

    # P2P-Edit 設定（與 batch_run_pie_edit_faceSwap 對齊）
    parser.add_argument("--num_full_replace_scales", type=int, default=2)
    parser.add_argument("--attn_threshold_percentile", type=float, default=80.0)
    parser.add_argument("--attn_block_start", type=int, default=2)
    parser.add_argument("--attn_block_end", type=int, default=-1)
    parser.add_argument("--attn_batch_idx", type=int, default=0)
    parser.add_argument("--p2p_token_replace_prob", type=float, default=0.0)
    parser.add_argument("--save_attn_vis", type=int, default=0, choices=[0, 1])
    parser.add_argument("--image_injection_scales", type=int, default=2)
    parser.add_argument("--inject_weights", type=str, default="")
    parser.add_argument("--use_normalized_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_last_scale_mask", type=int, default=0, choices=[0, 1])
    parser.add_argument("--last_scale_majority_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_method", type=int, default=1, choices=list(range(1, 15)))
    parser.add_argument("--absolute_high", type=float, default=0.7)
    parser.add_argument("--absolute_low", type=float, default=0.3)
    parser.add_argument("--debug_mode", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    print("\n" + "=" * 80)
    print("Face-Swap Gradio Demo")
    print("=" * 80)
    print(f"work_dir       : {args.work_dir}")
    print(f"adaface_url    : {args.adaface_url}")
    print(f"default_prompt : {args.default_prompt}")
    print(f"server         : {args.host}:{args.port}  share={bool(args.share)}")
    print("=" * 80 + "\n")

    app = FaceSwapApp(args)
    demo = build_ui(app)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
