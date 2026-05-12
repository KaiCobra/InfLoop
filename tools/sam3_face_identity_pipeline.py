#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 face identity conditioning pipeline for Infinity.

This is model plumbing only. It intentionally avoids dataloaders, losses, and
training loops so the dataset work can be added later without entangling the
core architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from tools.sam3_face_features import SAM3PyramidFeatureExtractor


def _load_face_identity_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "infinity" / "models" / "face_identity_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "_infloop_face_identity_adapter",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load face identity adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_face_adapter = _load_face_identity_adapter_module()
SAM3FacePyramidAdapter = _face_adapter.SAM3FacePyramidAdapter
attach_visual_identity_adapter = _face_adapter.attach_visual_identity_adapter
set_visual_id_tokens = _face_adapter.set_visual_id_tokens


@dataclass(frozen=True)
class SAM3FaceIdentityConfig:
    sam3_path: str = "weights/sam3_hf"
    sam_dim: int = 1024
    adapter_dim: int = 1024
    model_dim: int = 2048
    num_regions: int = 10
    num_id_tokens: int = 12
    adapter_layers: int = 2
    adapter_heads: int = 8
    branch_heads: int = 16
    target_scales: Sequence[int] = tuple(range(4, 13))
    block_indices: Sequence[int] = tuple(range(16, 32))
    sam_dtype: torch.dtype = torch.bfloat16
    freeze_sam3: bool = True


class SAM3FaceIdentityPipeline(nn.Module):
    """End-to-end model-side pipeline for SAM3 identity conditioning.

    Input:
        image tensor, shaped ``(B, 3, H, W)``.

    Output:
        A dict mapping Infinity scale index to visual identity tokens,
        each shaped ``(B, num_id_tokens, model_dim)``.

    The returned tokens can be injected by calling ``install_on_infinity`` once,
    then ``condition_infinity`` before each generation/forward pass.
    """

    def __init__(
        self,
        config: SAM3FaceIdentityConfig = SAM3FaceIdentityConfig(),
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sam3 = SAM3PyramidFeatureExtractor(
            model_path=config.sam3_path,
            device=device,
            dtype=config.sam_dtype,
            freeze=config.freeze_sam3,
        )
        self.adapter = SAM3FacePyramidAdapter(
            sam_dim=config.sam_dim,
            model_dim=config.model_dim,
            adapter_dim=config.adapter_dim,
            num_regions=config.num_regions,
            num_id_tokens=config.num_id_tokens,
            num_layers=config.adapter_layers,
            num_heads=config.adapter_heads,
            target_scales=config.target_scales,
        )
        if device is not None:
            self.adapter.to(device)

    def forward(
        self,
        image: torch.Tensor,
        face_masks: Optional[torch.Tensor] = None,
        target_scales: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        pyramid = self.sam3(image)
        adapter_device = next(self.adapter.parameters()).device
        pyramid = [feat.to(adapter_device) for feat in pyramid]
        if face_masks is not None:
            face_masks = face_masks.to(adapter_device)
        return self.adapter(
            pyramid,
            face_masks=face_masks,
            target_scales=target_scales,
        )

    def install_on_infinity(self, infinity_model: nn.Module) -> nn.Module:
        """Attach zero-gated visual cross-attention branches to Infinity."""
        return attach_visual_identity_adapter(
            infinity_model,
            block_indices=self.config.block_indices,
            scale_start=min(self.config.target_scales),
            scale_end=max(self.config.target_scales),
            model_dim=self.config.model_dim,
            num_heads=self.config.branch_heads,
        )

    @torch.no_grad()
    def condition_infinity(
        self,
        infinity_model: nn.Module,
        image: torch.Tensor,
        face_masks: Optional[torch.Tensor] = None,
        target_scales: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Encode a source face and set tokens on an already-attached Infinity."""
        tokens_by_scale = self.forward(
            image=image,
            face_masks=face_masks,
            target_scales=target_scales,
        )
        set_visual_id_tokens(infinity_model, tokens_by_scale)
        return tokens_by_scale


def build_sam3_face_identity_pipeline(
    infinity_model: Optional[nn.Module] = None,
    *,
    sam3_path: str = "weights/sam3_hf",
    model_dim: int = 2048,
    target_scales: Sequence[int] = tuple(range(4, 13)),
    block_indices: Sequence[int] = tuple(range(16, 32)),
    device: Optional[torch.device | str] = None,
    **kwargs,
) -> SAM3FaceIdentityPipeline:
    """Build the SAM3 -> adapter -> Infinity hook pipeline.

    If ``infinity_model`` is passed, the visual identity branches are attached
    immediately. Otherwise the caller can later call ``pipeline.install_on_infinity``.
    """
    config = SAM3FaceIdentityConfig(
        sam3_path=sam3_path,
        model_dim=model_dim,
        target_scales=tuple(int(x) for x in target_scales),
        block_indices=tuple(int(x) for x in block_indices),
        **kwargs,
    )
    pipeline = SAM3FaceIdentityPipeline(config=config, device=device)
    if infinity_model is not None:
        pipeline.install_on_infinity(infinity_model)
    return pipeline
