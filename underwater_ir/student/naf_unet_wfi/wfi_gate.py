from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .utils import broadcast_condition


class WFIGate(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_masks: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, channels)
        in_channels = channels * 2 + (num_masks if num_masks > 0 else 0) + channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hf: torch.Tensor,
        lf: torch.Tensor,
        cond: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_map = broadcast_condition(self.cond_proj(cond), hf.shape[-2:])
        features = [hf, lf, cond_map]
        if masks is not None:
            if masks.shape[2:] != hf.shape[2:]:
                masks = torch.nn.functional.interpolate(masks, size=hf.shape[-2:], mode="bilinear", align_corners=False)
            features.append(masks)
        x = torch.cat(features, dim=1)
        alpha = self.net(x)
        out = alpha * hf + (1 - alpha) * lf
        return out, alpha
