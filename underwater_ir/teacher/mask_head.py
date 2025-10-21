from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskHeadOutput:
    scores: torch.Tensor  # (B, K, H, W)
    probs: torch.Tensor  # (B, K, H, W)
    masks: torch.Tensor  # (B, K, H, W)
    intensity: torch.Tensor  # (B, K, H, W)
    avg_intensity: torch.Tensor  # (B, K)
    global_prob: torch.Tensor  # (B, K)
    confidence: torch.Tensor  # (B, K)
    raw_probs: torch.Tensor  # (B, K, H, W)
    used_crf: bool


class MaskHead(nn.Module):
    """Generate degradation-aware masks and statistics from dense vision-language features."""

    def __init__(
        self,
        temperature: float = 10.0,
        alpha: float = 5.0,
        default_threshold: float = 0.5,
        confidence_delta: float = 0.1,
        morphology_kernel: int = 3,
        use_crf: bool = False,
        crf_iterations: int = 5,
        crf_spatial_kernel: int = 5,
        crf_spatial_sigma: float = 3.0,
        crf_edge_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.default_threshold = default_threshold
        self.confidence_delta = confidence_delta
        self.morphology_kernel = morphology_kernel
        self.use_crf = use_crf
        if use_crf:
            from .differentiable_crf import DifferentiableGuidedCRF

            self.crf = DifferentiableGuidedCRF(
                num_iterations=crf_iterations,
                spatial_kernel=crf_spatial_kernel,
                spatial_sigma=crf_spatial_sigma,
                edge_weight=crf_edge_weight,
            )
        else:
            self.register_module("crf", None)

    @staticmethod
    def _prepare_image_tokens(tokens: torch.Tensor, grid_size: Optional[Tuple[int, int]]) -> Tuple[torch.Tensor, Tuple[int, int]]:

        if tokens.dim() == 4:  # (B, C, H, W)
            b, c, h, w = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, HW, C)
            return tokens, (h, w)

        if tokens.dim() == 3 and tokens.shape[1] > tokens.shape[2]:  # (B, N, C)
            if grid_size is None:
                length = tokens.shape[1]
                side = int(length**0.5)
                grid_size = (side, side)
            return tokens, grid_size

        if tokens.dim() == 3:  # (N, B, C)
            tokens = tokens.transpose(0, 1)
            if grid_size is None:
                length = tokens.shape[1]
                side = int(length**0.5)
                grid_size = (side, side)
            return tokens, grid_size

        raise ValueError("Unsupported token shape for MaskHead input.")

    @staticmethod
    def _morphological_close(mask: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return mask
        b, k, h, w = mask.shape
        flat = mask.view(b * k, 1, h, w)
        padding = kernel // 2
        dilated = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=padding)
        inverted = 1 - dilated
        eroded = 1 - F.max_pool2d(inverted, kernel_size=kernel, stride=1, padding=padding)
        return eroded.view(b, k, h, w)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        thresholds: Optional[Iterable[float]] = None,
        temperature: Optional[float] = None,
        alpha: Optional[float] = None,
        confidence_delta: Optional[float] = None,
        image: Optional[torch.Tensor] = None,
    ) -> MaskHeadOutput:
        tokens, grid = self._prepare_image_tokens(image_tokens, grid_size)
        h, w = grid
        b, n, c = tokens.shape
        if n != h * w:
            raise ValueError(f"Token count {n} does not match grid size {h}x{w}.")

        tokens = F.normalize(tokens, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        k = text_embeddings.shape[0]

        sim = torch.einsum("bnc,kc->bkn", tokens, text_embeddings)
        tau = temperature or self.temperature
        scores = sim.view(b, k, h, w)
        probs = F.softmax(sim * tau, dim=1).view(b, k, h, w)
        raw_probs = probs.clone()

        alpha_val = alpha or self.alpha
        intensity = torch.sigmoid(scores * alpha_val)
        avg_intensity = intensity.view(b, k, -1).mean(dim=-1)

        thresholds_tensor = torch.tensor(
            list(thresholds) if thresholds is not None else [self.default_threshold] * k,
            dtype=scores.dtype,
            device=scores.device,
        ).view(1, k, 1, 1)

        delta = confidence_delta or self.confidence_delta
        upper = thresholds_tensor + delta
        initial_conf = ((probs >= thresholds_tensor) & (probs <= upper)).float()
        confidence_raw = initial_conf.view(b, k, -1).mean(dim=-1)

        used_crf = self.use_crf and self.crf is not None
        if used_crf:
            if image is None:
                raise ValueError("MaskHead with CRF enabled requires the guidance image.")
            confidence_map = confidence_raw if confidence_raw is not None else None
            probs = self.crf(probs, image, confidence=confidence_map)

        masks = (probs >= thresholds_tensor).float()
        if self.morphology_kernel > 1:
            masks = self._morphological_close(masks, self.morphology_kernel)

        global_prob = probs.view(b, k, -1).mean(dim=-1)

        conf = ((probs >= thresholds_tensor) & (probs <= upper)).float()
        confidence = conf.view(b, k, -1).mean(dim=-1)

        return MaskHeadOutput(
            scores=scores,
            probs=probs,
            masks=masks,
            intensity=intensity,
            avg_intensity=avg_intensity,
            global_prob=global_prob,
            confidence=confidence,
             raw_probs=raw_probs,
             used_crf=used_crf,
        )
