"""
Cross-Attention Extractor for Infinity Model (Simplified)

Extracts cross-attention maps from all blocks using monkey patching.
Only cross-attention is supported. Self-attention is removed for simplicity.

This is a simplified version of tools/attention_extractor.py,
designed to be self-contained and portable across Infinity projects.
"""

import os
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class CrossAttentionExtractor:
    """
    Extract cross-attention maps from all blocks of the Infinity model.

    Uses monkey patching to intercept attention computations during inference.
    Stores raw attention weights: {block_idx: [attn_scale_0, attn_scale_1, ...]}
    Each tensor has shape [1, num_heads, query_len, key_len].

    Usage:
        extractor = CrossAttentionExtractor(model=infinity, block_indices=list(range(32)))
        extractor.register_patches()
        # ... run inference ...
        extractor.remove_patches()
        # access extractor.attention_maps
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: List[int],
        batch_idx: int = 1,
        aggregate_method: str = "mean",
    ):
        """
        Args:
            model: Infinity transformer model
            block_indices: Block indices to hook (e.g. list(range(32)))
            batch_idx: Which batch to extract (0=uncond, 1=cond for CFG)
            aggregate_method: How to aggregate heads ("mean", "max")
        """
        self.model = model
        self.block_indices = block_indices
        self.batch_idx = batch_idx
        self.aggregate_method = aggregate_method

        self.attention_maps: Dict[int, List[torch.Tensor]] = {}
        self.original_forwards: Dict[str, object] = {}

        # VAE intermediate decoding
        self.intermediate_images: List[np.ndarray] = []
        self.vae_hooked = False
        self.vae = None
        self.infinity_model = None
        self.scale_schedule: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Monkey-patching
    # ------------------------------------------------------------------

    def _create_patched_forward(self, original_forward, block_idx: int):
        """Patched CrossAttention.forward that captures attention weights."""
        extractor = self  # closure reference

        def patched_forward(q, ca_kv):
            kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
            N = kv_compact.shape[0]
            B, L, C = q.shape
            module = original_forward.__self__
            batch_idx = getattr(module, "_extractor_batch_idx", 0)

            try:
                # Q projection
                if isinstance(module.mat_q, nn.Parameter):
                    q_proj = module.mat_q.expand(B, -1, -1)
                else:
                    q_proj = module.mat_q(q)

                # K, V projections
                kv = F.linear(
                    kv_compact,
                    weight=module.mat_kv.weight,
                    bias=torch.cat((module.zero_k_bias, module.v_bias)),
                )
                kv = kv.view(N, 2, module.num_heads, module.head_dim)
                k, v = kv[:, 0], kv[:, 1]

                # Reshape Q → (B, H, L, c)
                q_proj = q_proj.view(B, L, module.num_heads, module.head_dim).transpose(1, 2)

                # Select K for the corresponding prompt segment
                if len(cu_seqlens_k) > batch_idx + 1:
                    k_start = cu_seqlens_k[batch_idx].item()
                    k_end = cu_seqlens_k[batch_idx + 1].item()
                else:
                    k_start = 0
                    k_end = N if len(cu_seqlens_k) <= 1 else cu_seqlens_k[1].item()

                k_selected = k[k_start:k_end]  # (k_len, H, c)

                # Attention scores: (H, L, c) @ (H, c, k_len) → (H, L, k_len)
                q_selected = q_proj[batch_idx]  # (H, L, c)
                k_selected_t = k_selected.permute(1, 2, 0)  # (H, c, k_len)
                attn_scores = torch.bmm(q_selected, k_selected_t) * module.scale
                attn_weights = F.softmax(attn_scores, dim=-1)  # (H, L, k_len)

                # Store with batch dim: (1, H, L, k_len)
                attn_weights = attn_weights.unsqueeze(0)
                if block_idx not in extractor.attention_maps:
                    extractor.attention_maps[block_idx] = []
                extractor.attention_maps[block_idx].append(attn_weights.detach().cpu())

            except Exception as e:
                print(f"Warning: Failed to capture cross-attention in block {block_idx}: {e}")

            return original_forward(q, ca_kv)

        return patched_forward

    def register_patches(self):
        """Patch CrossAttention modules in the specified blocks."""
        print(f"\n🔧 Patching CROSS attention in blocks: "
              f"{self.block_indices[0]}–{self.block_indices[-1]} "
              f"({len(self.block_indices)} blocks)")
        print(f"   batch_idx={self.batch_idx} (0=uncond, 1=cond)")

        patched = 0
        for name, module in self.model.named_modules():
            if ".ca" not in name or "CrossAttention" not in type(module).__name__:
                continue
            if "blocks." not in name and "block_chunks." not in name:
                continue

            try:
                parts = name.split(".")
                if "block_chunks" in name:
                    chunk_idx = int(parts[1])
                    local_idx = int(parts[3])
                    block_num = chunk_idx * 4 + local_idx
                else:
                    block_num = int(parts[1])

                if block_num not in self.block_indices:
                    continue

                self.original_forwards[name] = module.forward
                module._extractor_batch_idx = self.batch_idx
                module.forward = self._create_patched_forward(module.forward, block_num)
                patched += 1
            except (ValueError, IndexError):
                continue

        if patched == 0:
            print("⚠️  Warning: No CrossAttention modules were patched!")
        else:
            print(f"✓ Patched {patched} CrossAttention modules")

    def remove_patches(self):
        """Restore original forward methods."""
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards = {}
        self.remove_vae_hook()
        print("✓ All patches removed")

    # ------------------------------------------------------------------
    # VAE intermediate image decoding
    # ------------------------------------------------------------------

    def hook_vae_decoder(self, vae, scale_schedule, infinity_model=None):
        """Register VAE for post-generation intermediate decoding."""
        if self.vae_hooked:
            return
        self.vae = vae
        self.infinity_model = infinity_model
        self.scale_schedule = scale_schedule
        self.intermediate_images = []
        self.vae_hooked = True
        print(f"✓ VAE hook registered ({len(scale_schedule)} scales)")

    def remove_vae_hook(self):
        self.vae_hooked = False

    def restore_vae_decoder(self, vae):
        self.vae_hooked = False

    def decode_intermediate_images(self):
        """Decode all intermediate summed_codes after generation."""
        if not self.vae_hooked or self.infinity_model is None:
            print("Warning: VAE hook not ready")
            return
        if not hasattr(self.infinity_model, "summed_codes_list_for_vis"):
            print("Warning: model has no summed_codes_list_for_vis")
            return

        codes_list = self.infinity_model.summed_codes_list_for_vis
        if not codes_list:
            print("Warning: summed_codes_list_for_vis is empty")
            return

        print(f"🎨 Decoding {len(codes_list)} intermediate scales...")
        self.intermediate_images = []
        with torch.no_grad():
            for summed_codes in codes_list:
                result = self.vae.decode(summed_codes.squeeze(-3))
                img = result[0].permute(1, 2, 0).cpu().float().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.intermediate_images.append(img_bgr)
        print(f"✓ Decoded {len(self.intermediate_images)} intermediate images")

    # ------------------------------------------------------------------
    # Attention aggregation helpers
    # ------------------------------------------------------------------

    def aggregate_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Aggregate across attention heads.
        Input : (1, H, L, k_len)
        Output: (1, L, k_len)
        """
        if self.aggregate_method == "max":
            return attn.max(dim=1)[0]
        return attn.mean(dim=1)  # default: mean

    def extract_word_attention(
        self,
        block_idx: int,
        scale_idx: int,
        token_indices: List[int],
        spatial_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract the merged attention map for a set of token indices
        at a given block and scale.

        Returns:
            2-D numpy array of shape (H, W), or None on failure.
        """
        if block_idx not in self.attention_maps:
            return None
        attn_list = self.attention_maps[block_idx]
        if scale_idx >= len(attn_list):
            return None

        attn_tensor = attn_list[scale_idx]  # (1, H_heads, L, k_len)
        attn_agg = self.aggregate_heads(attn_tensor)  # (1, L, k_len)

        H, W = spatial_size
        target_len = H * W
        merged = None
        count = 0

        for tidx in token_indices:
            if tidx >= attn_agg.shape[2]:
                continue
            token_attn = attn_agg[0, :, tidx].cpu().numpy()  # (L,)
            if token_attn.shape[0] < target_len:
                continue
            attn_map = token_attn[-target_len:].reshape(H, W)
            if merged is None:
                merged = attn_map
                count = 1
            else:
                merged = merged + attn_map
                count += 1

        if merged is None:
            return None
        return merged / count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_maps(self):
        self.attention_maps = {}

    def get_summary(self):
        print("\n" + "=" * 50)
        print(f"Cross-Attention Maps Summary")
        print("=" * 50)
        for bidx in sorted(self.attention_maps.keys()):
            steps = self.attention_maps[bidx]
            if steps:
                print(f"  Block {bidx:2d}: {len(steps):2d} scales, shape {tuple(steps[0].shape)}")
        print("=" * 50)
