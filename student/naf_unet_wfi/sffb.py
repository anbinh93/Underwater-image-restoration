from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .dwt_fft import fft2, ifft2
from .utils import AdaLayerNorm2d, DepthwiseSeparableConv, LayerNorm2d


class SpatialFrequencyFusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int = 0,
        kernel_sizes: Sequence[int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.norm = AdaLayerNorm2d(channels, cond_dim) if cond_dim > 0 else LayerNorm2d(channels)
        self.spatial_convs = nn.ModuleList(
            [
                DepthwiseSeparableConv(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.merge = nn.Conv2d(len(kernel_sizes) * channels, channels, kernel_size=1)
        self.amp_mod = nn.Conv2d(channels, channels, kernel_size=1)
        self.phase_mod = nn.Conv2d(channels, channels, kernel_size=1)
        self.reproj = nn.Conv2d(channels, channels, kernel_size=1)

    def _apply_norm(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.norm, AdaLayerNorm2d):
            if cond is None:
                raise ValueError("Condition vector required for AdaLayerNorm2d.")
            return self.norm(x, cond)
        return self.norm(x)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self._apply_norm(x, cond)
        spatial = torch.cat([conv(x_norm) for conv in self.spatial_convs], dim=1)
        spatial = self.merge(spatial)

        amp, phase = fft2(spatial)
        amp_mod = self.amp_mod(amp)
        phase_mod = self.phase_mod(phase)
        freq = ifft2(amp_mod, phase_mod)

        fused = self.reproj(spatial + freq)
        return fused
