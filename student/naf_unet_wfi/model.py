from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .cfc import CrossFrequencyConditioner
from .deg_head import DegradationHead
from .dwt_fft import dwt2, idwt2
from .sffb import SpatialFrequencyFusionBlock
from .utils import LayerNorm2d
from .wfi_gate import WFIGate
from .wtb import WideTransformerBlock


class WFIBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, num_masks: int, num_heads: int = 4) -> None:
        super().__init__()
        self.hf_block = WideTransformerBlock(channels * 3, cond_dim, num_heads=max(1, num_heads * 3))
        self.lf_block = SpatialFrequencyFusionBlock(channels, cond_dim)
        self.cfc = CrossFrequencyConditioner(channels, num_heads)
        self.gate = WFIGate(channels, cond_dim, num_masks)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh = dwt2(x)
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.hf_block(hf, cond)
        lh, hl, hh = torch.chunk(hf, 3, dim=1)

        lf = self.lf_block(ll, cond)
        hf_sum = lh + hl + hh
        t_out, f_out = self.cfc(hf_sum, lf)
        merged, alpha = self.gate(t_out, f_out, cond, masks)
        out = idwt2(merged, lh, hl, hh)
        return out, alpha


class WFIStage(nn.Module):
    def __init__(self, channels: int, cond_dim: int, num_masks: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [WFIBlock(channels, cond_dim, num_masks) for _ in range(num_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        alphas: List[torch.Tensor] = []
        for block in self.blocks:
            x, alpha = block(x, cond, masks)
            alphas.append(alpha)
        return x, alphas


class NAFNetWFIGate(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 3,
        blocks_per_level: int = 2,
        cond_dim: int = 16,
        num_masks: int = 0,
        num_degradation_types: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.num_masks = num_masks

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.final_norm = LayerNorm2d(base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        channels = [base_channels * (2 ** i) for i in range(num_levels)]

        self.encoder_stages = nn.ModuleList(
            [
                WFIStage(channels[i], cond_dim, num_masks, blocks_per_level)
                for i in range(num_levels)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
                for i in range(num_levels - 1)
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2)
                for i in reversed(range(num_levels - 1))
            ]
        )
        # Fusion convs for concatenated skip connections (UNet-style)
        self.skip_fusions = nn.ModuleList(
            [
                nn.Conv2d(channels[i] * 2, channels[i], kernel_size=1)
                for i in reversed(range(num_levels - 1))
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [
                WFIStage(channels[i], cond_dim, num_masks, blocks_per_level)
                for i in reversed(range(num_levels - 1))
            ]
        )
        self.degradation_head = (
            DegradationHead(channels[-1], num_degradation_types)
            if num_degradation_types is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        if cond is None:
            raise ValueError("Condition vector z_d must be provided.")

        residual = x
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        alpha_maps: List[torch.Tensor] = []

        for idx, stage in enumerate(self.encoder_stages):
            x, alphas = stage(x, cond, masks)
            alpha_maps.extend(alphas)
            if idx < len(self.encoder_stages) - 1:
                skips.append(x)
                x = self.downsamples[idx](x)

        deg_logits = self.degradation_head(x) if self.degradation_head is not None else None

        for up, fusion, stage, skip in zip(self.upsamples, self.skip_fusions, self.decoder_stages, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # UNet-style concatenation instead of addition
            x = torch.cat([x, skip], dim=1)
            x = fusion(x)
            x, alphas = stage(x, cond, masks)
            alpha_maps.extend(alphas)

        x = self.final_norm(x)
        out = self.head(x)
        return out + residual, alpha_maps, deg_logits
