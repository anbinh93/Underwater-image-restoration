from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mask_head import MaskHeadOutput


@dataclass
class DegradationCode:
    code: torch.Tensor  # (B, 2K)
    logits: torch.Tensor  # (B, K)
    avg_intensity: torch.Tensor  # (B, K)
    global_prob: torch.Tensor  # (B, K)
    confidence: torch.Tensor  # (B, K)
    confidence_scale: torch.Tensor  # (B,)
    broadcast: Optional[torch.Tensor] = None  # (B, 2K, H, W) when requested


class DegradationCoder(nn.Module):
    """Assemble global degradation descriptors from VLM mask statistics."""

    def __init__(self, eps: float = 1e-5, broadcast: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.broadcast_enabled = broadcast

    def forward(
        self,
        mask_output: MaskHeadOutput,
        spatial_shape: Optional[Tuple[int, int]] = None,
        enable_broadcast: Optional[bool] = None,
    ) -> DegradationCode:
        logits = torch.logit(
            mask_output.global_prob.clamp(self.eps, 1.0 - self.eps)
        )
        code = torch.cat([logits, mask_output.avg_intensity], dim=-1)
        confidence_scale = 1.0 - mask_output.confidence.mean(dim=-1)

        broadcast_tensor = None
        should_broadcast = (
            self.broadcast_enabled if enable_broadcast is None else enable_broadcast
        )
        if should_broadcast and spatial_shape is not None:
            h, w = spatial_shape
            broadcast_tensor = code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        return DegradationCode(
            code=code,
            logits=logits,
            avg_intensity=mask_output.avg_intensity,
            global_prob=mask_output.global_prob,
            confidence=mask_output.confidence,
            confidence_scale=confidence_scale,
            broadcast=broadcast_tensor,
        )
