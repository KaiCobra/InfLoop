#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 image feature extraction utilities.

This module intentionally contains no dataloader or training logic.  It wraps a
local SAM3 Hugging Face checkpoint and returns a feature pyramid suitable for
the face identity adapter.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple_hw(size) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, Sequence) and len(size) >= 2:
        return (int(size[-2]), int(size[-1]))
    return None


class SAM3PyramidFeatureExtractor(nn.Module):
    """Frozen SAM3 feature extractor returning List[(B, D, h_i, w_i)].

    The local checkpoint is expected to be a Hugging Face directory, e.g.
    ``weights/sam3_hf``.  The wrapper is deliberately defensive because SAM3
    output field names can differ across Transformers versions.
    """

    def __init__(
        self,
        model_path: str = "weights/sam3_hf",
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        trust_remote_code: bool = True,
        output_sizes: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "SAM3PyramidFeatureExtractor requires transformers with SAM3 support"
            ) from exc

        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        if freeze:
            self.model.requires_grad_(False)
        self.output_sizes = list(output_sizes) if output_sizes is not None else None

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Extract SAM3 feature pyramid from an image tensor.

        Args:
            image: Tensor in ``(B, 3, H, W)``.  Values may be either [0, 1] or
                already normalized; the HF processor handles PIL/np best, but
                this wrapper keeps tensor input as the canonical project API.
        """
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError(f"image must be (B, 3, H, W), got {tuple(image.shape)}")

        image = image.to(self.device)
        outputs = self._forward_model(image)
        pyramid = self._extract_pyramid(outputs)
        if not pyramid:
            raise RuntimeError(
                "Could not find SAM3 feature maps in model outputs. "
                "Inspect the output object and update _extract_pyramid()."
            )
        pyramid = [feat.float() for feat in pyramid]
        if self.output_sizes is not None:
            pyramid = [
                F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
                for feat, size in zip(pyramid, self.output_sizes)
            ]
        return pyramid

    def _forward_model(self, image: torch.Tensor):
        # Prefer direct pixel_values path; fall back to processor if needed.
        try:
            return self.model(pixel_values=image, output_hidden_states=True, return_dict=True)
        except TypeError:
            pass

        # Processor fallback converts tensors to CPU numpy arrays. This path is
        # slower, but keeps the wrapper usable across SAM3 HF variants.
        imgs = image.detach().float().cpu().clamp(0, 1)
        imgs = [img.permute(1, 2, 0).numpy() for img in imgs]
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def _extract_pyramid(self, outputs) -> List[torch.Tensor]:
        candidates = []
        for name in (
            "feature_maps",
            "features",
            "image_features",
            "vision_features",
            "backbone_feature_maps",
            "fpn_features",
        ):
            if hasattr(outputs, name):
                val = getattr(outputs, name)
                if val is not None:
                    candidates.append(val)
        if isinstance(outputs, dict):
            for name in (
                "feature_maps",
                "features",
                "image_features",
                "vision_features",
                "backbone_feature_maps",
                "fpn_features",
            ):
                if outputs.get(name) is not None:
                    candidates.append(outputs[name])

        for val in candidates:
            maps = self._coerce_feature_list(val)
            if maps:
                return maps

        hidden = getattr(outputs, "vision_hidden_states", None)
        if hidden is None and hasattr(outputs, "hidden_states"):
            hidden = getattr(outputs, "hidden_states")
        maps = self._coerce_feature_list(hidden)
        return maps

    def _coerce_feature_list(self, value) -> List[torch.Tensor]:
        if value is None:
            return []
        if torch.is_tensor(value):
            return self._tensor_to_maps(value)
        if isinstance(value, (list, tuple)):
            out: List[torch.Tensor] = []
            for item in value:
                if torch.is_tensor(item):
                    out.extend(self._tensor_to_maps(item))
            # Keep only spatial maps and deduplicate by shape.
            spatial = [x for x in out if x.dim() == 4]
            seen = set()
            uniq = []
            for feat in spatial:
                key = tuple(feat.shape)
                if key not in seen:
                    seen.add(key)
                    uniq.append(feat)
            return uniq
        return []

    def _tensor_to_maps(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        if tensor.dim() == 4:
            # Accept either BCHW or BHWC.
            if tensor.shape[1] in (256, 512, 768, 1024, 2048):
                return [tensor]
            if tensor.shape[-1] in (256, 512, 768, 1024, 2048):
                return [tensor.permute(0, 3, 1, 2).contiguous()]
        if tensor.dim() == 3:
            # Convert BLC to BCHW if L is square.
            B, L, D = tensor.shape
            side = int(L ** 0.5)
            if side * side == L:
                return [tensor.transpose(1, 2).reshape(B, D, side, side).contiguous()]
        return []
