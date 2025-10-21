from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class CrossFrequencyConditioner(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        head_dim = channels // num_heads
        if head_dim * num_heads != channels:
            raise ValueError("channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.vt_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.vf_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, hf: torch.Tensor, lf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = hf.shape
        q = self.q_conv(hf).view(b, self.num_heads, self.head_dim, h * w)
        k = self.k_conv(lf).view(b, self.num_heads, self.head_dim, h * w)
        vt = self.vt_conv(hf).view(b, self.num_heads, self.head_dim, h * w)
        vf = self.vf_conv(lf).view(b, self.num_heads, self.head_dim, h * w)

        attn = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        t_out = torch.matmul(attn, vt.transpose(-2, -1)).transpose(-2, -1)
        f_out = torch.matmul(attn, vf.transpose(-2, -1)).transpose(-2, -1)

        t_out = t_out.reshape(b, c, h, w)
        f_out = f_out.reshape(b, c, h, w)
        return t_out, f_out
