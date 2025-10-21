from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class AdaLayerNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(num_features)
        self.linear = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.linear(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        normalized = self.norm(x)
        return normalized * (1 + scale) + shift


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def broadcast_condition(cond: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
    return cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_shape[0], spatial_shape[1])


def stack_masks(masks: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [m.float() for m in masks]
    return torch.cat(tensors, dim=1) if tensors else None
