from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(ks: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(ks, dtype=dtype, device=device) - ks // 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel.view(1, 1, ks, ks)


class DifferentiableGuidedCRF(nn.Module):
    """Lightweight differentiable guided CRF using mean-field iterations with edge-aware smoothing."""

    def __init__(
        self,
        num_iterations: int = 5,
        spatial_kernel: int = 5,
        spatial_sigma: float = 3.0,
        edge_weight: float = 5.0,
    ) -> None:
        super().__init__()
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd.")
        self.num_iterations = num_iterations
        self.spatial_kernel = spatial_kernel
        self.spatial_sigma = spatial_sigma
        self.edge_weight = edge_weight
        self.register_buffer("kernel", torch.empty(1))
        self.pad = spatial_kernel // 2

    def _ensure_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.kernel.numel() != self.spatial_kernel * self.spatial_kernel or self.kernel.device != device:
            kernel = _gaussian_kernel(self.spatial_kernel, self.spatial_sigma, device, dtype)
            self.kernel = kernel
        return self.kernel

    def forward(self, probs: torch.Tensor, image: torch.Tensor, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            probs: (B, K, H, W) softmax probabilities.
            image: (B, C, H, W) guidance image in [0,1].
            confidence: optional (B, K) weighting per class.
        Returns:
            (B, K, H, W) refined probabilities.
        """
        if image.shape[-2:] != probs.shape[-2:]:
            image = F.interpolate(image, size=probs.shape[-2:], mode="bilinear", align_corners=False)

        kernel = self._ensure_kernel(probs.device, probs.dtype)
        num_classes = probs.shape[1]
        depthwise_kernel = kernel.expand(num_classes, 1, -1, -1)

        unary = -torch.log(probs.clamp(min=1e-6))
        q = probs.clone()

        gray = image.mean(dim=1, keepdim=True)
        grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        edge_mask = torch.exp(-self.edge_weight * grad_mag)
        edge_mask = edge_mask.expand(-1, num_classes, -1, -1)

        if confidence is not None:
            conf_map = confidence.unsqueeze(-1).unsqueeze(-1)
        else:
            conf_map = 1.0

        for _ in range(self.num_iterations):
            smoothed = F.conv2d(q, depthwise_kernel, padding=self.pad, groups=num_classes)
            pairwise = edge_mask * (smoothed - q)
            message = conf_map * pairwise
            q = F.softmax(-(unary + message), dim=1)
        return q
