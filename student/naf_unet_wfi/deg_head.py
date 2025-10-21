from __future__ import annotations

import torch
import torch.nn as nn


class DegradationHead(nn.Module):
    """Predicts degradation distribution p_d^{Stu} from intermediate features."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        logits = self.net(pooled).flatten(1)
        return logits
