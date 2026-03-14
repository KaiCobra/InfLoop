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
from typing import Dict, List, Optional, Tuple, Set


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
        capture_attention: bool = True,
        replacement_maps: Optional[Dict[int, Dict[int, Dict[int, torch.Tensor]]]] = None,
        replace_scales: Optional[List[int]] = None,
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
        self.capture_attention = capture_attention

        self.attention_maps: Dict[int, List[torch.Tensor]] = {}
        self.original_forwards: Dict[str, object] = {}
        self.call_counters: Dict[int, int] = {}
        self.replacement_maps: Dict[int, Dict[int, Dict[int, torch.Tensor]]] = replacement_maps or {}
        self.replace_scales: Set[int] = set(replace_scales or [])

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
            module = original_forward.__self__
            batch_idx = getattr(module, "_extractor_batch_idx", 0)
            scale_idx = extractor.call_counters.get(block_idx, 0)
            replace_for_scale = extractor.replacement_maps.get(block_idx, {}).get(scale_idx, {})
            should_replace = bool(replace_for_scale) and (
                not extractor.replace_scales or scale_idx in extractor.replace_scales
            )

            try:
                kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
                N = kv_compact.shape[0]
                B, L, C = q.shape

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

                def _span_for_batch(cur_batch_idx: int) -> Tuple[int, int]:
                    if len(cu_seqlens_k) > cur_batch_idx + 1:
                        return cu_seqlens_k[cur_batch_idx].item(), cu_seqlens_k[cur_batch_idx + 1].item()
                    return 0, N if len(cu_seqlens_k) <= 1 else cu_seqlens_k[1].item()

                def _compute_attn(cur_batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                    k_start, k_end = _span_for_batch(cur_batch_idx)
                    k_selected = k[k_start:k_end]  # (k_len, H, c)
                    v_selected = v[k_start:k_end]  # (k_len, H, c)
                    q_selected = q_proj[cur_batch_idx]  # (H, L, c)
                    attn_scores = torch.bmm(
                        q_selected.float(),
                        k_selected.permute(1, 2, 0).float(),
                    ) * module.scale
                    attn_weights = F.softmax(attn_scores, dim=-1).to(dtype=q_selected.dtype)
                    return attn_weights, v_selected

                def _apply_replacement(attn_weights: torch.Tensor) -> torch.Tensor:
                    if not replace_for_scale:
                        return attn_weights
                    replaced = attn_weights.clone()
                    for tgt_token_idx, src_map in replace_for_scale.items():
                        if tgt_token_idx >= replaced.shape[-1]:
                            continue
                        src_map_t = src_map.to(device=replaced.device, dtype=replaced.dtype)
                        if src_map_t.dim() == 1:
                            src_map_t = src_map_t.unsqueeze(0)
                        if src_map_t.shape[0] == 1 and replaced.shape[0] > 1:
                            src_map_t = src_map_t.expand(replaced.shape[0], -1)
                        if src_map_t.shape[0] != replaced.shape[0]:
                            continue
                        if src_map_t.shape[-1] != replaced.shape[1]:
                            src_map_t = F.interpolate(
                                src_map_t.unsqueeze(0),
                                size=replaced.shape[1],
                                mode='linear',
                                align_corners=False,
                            ).squeeze(0)
                        replaced[:, :, tgt_token_idx] = src_map_t
                    norm = replaced.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    return replaced / norm

                if not should_replace:
                    if extractor.capture_attention:
                        attn_weights, _ = _compute_attn(batch_idx)
                        if block_idx not in extractor.attention_maps:
                            extractor.attention_maps[block_idx] = []
                        extractor.attention_maps[block_idx].append(
                            attn_weights.unsqueeze(0).detach().cpu()
                        )
                    extractor.call_counters[block_idx] = scale_idx + 1
                    return original_forward(q, ca_kv)

                outputs = []
                for cur_batch_idx in range(B):
                    attn_weights, v_selected = _compute_attn(cur_batch_idx)
                    if cur_batch_idx == batch_idx:
                        attn_weights = _apply_replacement(attn_weights)
                        if extractor.capture_attention:
                            if block_idx not in extractor.attention_maps:
                                extractor.attention_maps[block_idx] = []
                            extractor.attention_maps[block_idx].append(
                                attn_weights.unsqueeze(0).detach().cpu()
                            )

                    out = torch.bmm(
                        attn_weights.to(dtype=v_selected.dtype),
                        v_selected.permute(1, 0, 2),
                    )
                    out = out.transpose(0, 1).reshape(L, C)
                    outputs.append(out)

                oup = torch.stack(outputs, dim=0)
                extractor.call_counters[block_idx] = scale_idx + 1
                return module.proj_drop(module.proj(oup))

            except Exception as e:
                print(f"Warning: Failed to capture cross-attention in block {block_idx}: {e}")
                extractor.call_counters[block_idx] = scale_idx + 1

            return original_forward(q, ca_kv)

        return patched_forward

    def register_patches(self):
        """Patch CrossAttention modules in the specified blocks."""
        print(f"\n🔧 Patching CROSS attention in blocks: "
              f"{self.block_indices[0]}–{self.block_indices[-1]} "
              f"({len(self.block_indices)} blocks)")
        print(f"   batch_idx={self.batch_idx} (0=uncond, 1=cond)")
        self.call_counters = {}

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
            token_attn = attn_agg[0, :, tidx].float().cpu().numpy()  # (L,)
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

    def set_replacement_maps(
        self,
        replacement_maps: Optional[Dict[int, Dict[int, Dict[int, torch.Tensor]]]],
        replace_scales: Optional[List[int]] = None,
    ):
        self.replacement_maps = replacement_maps or {}
        self.replace_scales = set(replace_scales or [])

    def clear_replacement_maps(self):
        self.replacement_maps = {}
        self.replace_scales = set()

    def extract_single_token_attention(
        self,
        block_idx: int,
        scale_idx: int,
        token_index: int,
        spatial_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract attention map for a single text token at a given block and scale.

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

        if token_index >= attn_agg.shape[2]:
            return None
        token_attn = attn_agg[0, :, token_index].float().cpu().numpy()  # (L,)
        if token_attn.shape[0] < target_len:
            return None
        return token_attn[-target_len:].reshape(H, W)

    def get_summary(self):
        print("\n" + "=" * 50)
        print(f"Cross-Attention Maps Summary")
        print("=" * 50)
        for bidx in sorted(self.attention_maps.keys()):
            steps = self.attention_maps[bidx]
            if steps:
                print(f"  Block {bidx:2d}: {len(steps):2d} scales, shape {tuple(steps[0].shape)}")
        print("=" * 50)


class AttentionCacheInjector:
    """
    Prompt-to-Prompt style attention cache injector.

    During target generation, injects cached source cross-attention maps
    to transfer the spatial layout from source to target.

    For aligned tokens (same word or swapped words):
        Use source attention map (preserves spatial layout)
    For new tokens (only in target prompt):
        Keep target's own attention (generates freely)

    This is implemented as a separate patching layer that wraps existing
    CrossAttention forwards. It can coexist with CrossAttentionExtractor
    patches (register extractor first, then injector).
    """

    def __init__(
        self,
        model: nn.Module,
        source_attention_maps: Dict[int, List[torch.Tensor]],
        block_indices: List[int],
        token_alignment: Dict[int, int],
        max_scale: int = -1,
        batch_idx: int = 0,
    ):
        """
        Args:
            model: Infinity transformer model
            source_attention_maps: Cached source attention maps
                {block_idx: [attn_scale_0, attn_scale_1, ...]}
                Each tensor shape: [1, H_heads, L_visual, text_len]
            block_indices: Block indices to patch (should match source extractor's)
            token_alignment: target_token_idx -> source_token_idx mapping
                Tokens not in this dict are "new" and keep target attention.
            max_scale: Maximum scale index to inject (-1 = all scales)
            batch_idx: Which batch to inject (0=conditioned usually)
        """
        self.model = model
        self.source_attention_maps = source_attention_maps
        self.block_indices = block_indices
        self.token_alignment = token_alignment
        self.max_scale = max_scale
        self.batch_idx = batch_idx

        self.original_forwards: Dict[str, object] = {}
        self._scale_counters: Dict[int, int] = {}

    def get_source_attn(self, block_idx: int, scale_idx: int) -> Optional[torch.Tensor]:
        """Get cached source attention for a specific block and scale."""
        if block_idx not in self.source_attention_maps:
            return None
        attn_list = self.source_attention_maps[block_idx]
        if scale_idx >= len(attn_list):
            return None
        return attn_list[scale_idx]  # [1, H, L, k_len]

    def _create_injection_forward(self, original_forward, block_idx: int):
        """Create a patched forward that injects source attention."""
        injector = self

        def patched_forward(q, ca_kv):
            scale_idx = injector._scale_counters.get(block_idx, 0)
            injector._scale_counters[block_idx] = scale_idx + 1

            # Check if we should inject at this scale
            if injector.max_scale >= 0 and scale_idx > injector.max_scale:
                return original_forward(q, ca_kv)

            cached_attn = injector.get_source_attn(block_idx, scale_idx)
            if cached_attn is None:
                return original_forward(q, ca_kv)

            if not injector.token_alignment:
                return original_forward(q, ca_kv)

            # Get normal output from the wrapped forward (may include extractor)
            normal_output = original_forward(q, ca_kv)

            # Now recompute with injection for the target batch
            kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
            N = kv_compact.shape[0]
            B, L, C = q.shape
            target_batch = injector.batch_idx

            try:
                # Access the underlying CrossAttention module
                # Walk up the closure chain to find the module
                fwd = original_forward
                module = None
                while hasattr(fwd, '__self__'):
                    module = fwd.__self__
                    break
                if module is None:
                    # Try to find module from closure
                    if hasattr(original_forward, '__closure__') and original_forward.__closure__:
                        for cell in original_forward.__closure__:
                            try:
                                val = cell.cell_contents
                                if hasattr(val, '__self__') and hasattr(val.__self__, 'mat_q'):
                                    module = val.__self__
                                    break
                            except (ValueError, AttributeError):
                                continue
                if module is None or not hasattr(module, 'mat_q'):
                    return normal_output

                # Q projection
                if isinstance(module.mat_q, nn.Parameter):
                    q_proj = module.mat_q.expand(B, -1, -1)
                else:
                    q_proj = module.mat_q(q)

                # KV projection
                kv = F.linear(
                    kv_compact,
                    weight=module.mat_kv.weight,
                    bias=torch.cat((module.zero_k_bias, module.v_bias)),
                )
                kv = kv.view(N, 2, module.num_heads, module.head_dim)
                k, v = kv[:, 0], kv[:, 1]

                # Get K/V span for target batch
                if len(cu_seqlens_k) > target_batch + 1:
                    k_start = cu_seqlens_k[target_batch].item()
                    k_end = cu_seqlens_k[target_batch + 1].item()
                else:
                    k_start = 0
                    k_end = N if len(cu_seqlens_k) <= 1 else cu_seqlens_k[1].item()

                k_selected = k[k_start:k_end]  # (k_len, H, c)
                v_selected = v[k_start:k_end]  # (k_len, H, c)

                # Q for target batch: (B, L, H, c) -> select batch -> (H, L, c)
                q_view = q_proj.view(B, L, module.num_heads, module.head_dim)
                q_b = q_view[target_batch].transpose(0, 1)  # (H, L, c)

                # Compute target attention
                k_sel_t = k_selected.permute(1, 2, 0)  # (H, c, k_len)
                target_scores = torch.bmm(q_b.float(), k_sel_t.float()) * module.scale
                target_attn = F.softmax(target_scores, dim=-1)  # (H, L, k_len)

                # Source attention
                src_attn = cached_attn[0].to(target_attn.device).float()  # (H, L', src_k_len)

                # Inject: replace text token columns for aligned tokens
                injected_attn = target_attn.clone()
                if src_attn.shape[1] == target_attn.shape[1]:  # same visual length
                    for tgt_j, src_j in injector.token_alignment.items():
                        if tgt_j < target_attn.shape[2] and src_j < src_attn.shape[2]:
                            injected_attn[:, :, tgt_j] = src_attn[:, :, src_j]

                    # Re-normalize so attention sums to 1
                    attn_sum = injected_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    injected_attn = injected_attn / attn_sum

                # Compute output: (H, L, k_len) @ (H, k_len, c) -> (H, L, c)
                v_sel_perm = v_selected.permute(1, 0, 2).float()  # (H, k_len, c)
                injected_out = torch.bmm(injected_attn, v_sel_perm)  # (H, L, c)
                # (H, L, c) -> (L, H, c) -> (L, C) -> (1, L, C)
                injected_out = injected_out.transpose(0, 1).reshape(L, -1).unsqueeze(0)

                # Apply proj and dropout
                injected_out = module.proj_drop(
                    module.proj(injected_out.to(module.proj.weight.dtype))
                )

                # Replace the target batch output
                normal_output[target_batch] = injected_out[0]

            except Exception as e:
                print(f"[AttentionCacheInjector] Warning: injection failed at "
                      f"block {block_idx} scale {scale_idx}: {e}")

            return normal_output

        return patched_forward

    def register_patches(self):
        """Patch CrossAttention modules for attention injection."""
        self._scale_counters = {bidx: 0 for bidx in self.block_indices}

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
                module.forward = self._create_injection_forward(module.forward, block_num)
                patched += 1
            except (ValueError, IndexError):
                continue

        aligned_count = len(self.token_alignment)
        print(f"[AttentionCacheInjector] Patched {patched} CrossAttention modules "
              f"(max_scale={self.max_scale}, alignment={aligned_count} token pairs)")

    def remove_patches(self):
        """Restore original forward methods (to whatever was before injection)."""
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards = {}
        self._scale_counters = {}
        print("[AttentionCacheInjector] All injection patches removed")

    def reset_scale_counters(self):
        """Reset scale counters for a new generation pass."""
        self._scale_counters = {bidx: 0 for bidx in self.block_indices}
