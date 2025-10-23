from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AdaLayerNorm2d, LayerNorm2d


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        weights = self.net(pooled).view(b, c, 1, 1)
        return x * weights


class WideTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int = 0,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dw_kernel_sizes: tuple = (3, 5, 7),
        attn_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        head_dim = channels // num_heads
        if head_dim * num_heads != channels:
            raise ValueError("channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_chunk_size = attn_chunk_size

        self.norm1 = AdaLayerNorm2d(channels, cond_dim) if cond_dim > 0 else LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        
        # Multi-scale depth-wise convolution branches (WFI2-net spec)
        self.dw_branches = nn.ModuleList([
            nn.Conv2d(
                channels * 3,
                channels * 3,
                kernel_size=k,
                padding=k // 2,
                groups=channels * 3,
            )
            for k in dw_kernel_sizes
        ])
        self.dw_merge = nn.Conv2d(len(dw_kernel_sizes) * channels * 3, channels * 3, kernel_size=1)
        
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_attn = ChannelAttention(channels)

        hidden_dim = int(channels * mlp_ratio)
        self.norm2 = AdaLayerNorm2d(channels, cond_dim) if cond_dim > 0 else LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )

    def _apply_norm(self, norm: nn.Module, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(norm, AdaLayerNorm2d):
            if cond is None:
                raise ValueError("Condition vector required for AdaLayerNorm2d.")
            return norm(x, cond)
        return norm(x)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape

        residual = x
        x_norm = self._apply_norm(self.norm1, x, cond)

        # Multi-scale depth-wise processing
        qkv_base = self.qkv(x_norm)
        multi_scale = torch.cat([branch(qkv_base) for branch in self.dw_branches], dim=1)
        qkv = self.dw_merge(multi_scale)
        
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # Memory-efficient attention: compute in chunks to avoid OOM
        # For 256x256 input -> 32x32 feature -> 1024 tokens
        # chunk_size=256: max memory = batch * heads * 256 * 1024 * 4 bytes
        attn_out_list = []
        
        for i in range(0, h * w, self.attn_chunk_size):
            end_i = min(i + self.attn_chunk_size, h * w)
            q_chunk = q[:, :, :, i:end_i]  # [b, heads, head_dim, chunk]
            
            # Compute attention for this chunk
            attn_chunk = torch.matmul(q_chunk.transpose(-2, -1), k) / math.sqrt(self.head_dim)  # [b, heads, chunk, h*w]
            attn_chunk = torch.softmax(attn_chunk, dim=-1)
            attn_out_chunk = torch.matmul(attn_chunk, v.transpose(-2, -1)).transpose(-2, -1)  # [b, heads, head_dim, chunk]
            attn_out_list.append(attn_out_chunk)
        
        attn_out = torch.cat(attn_out_list, dim=-1)  # [b, heads, head_dim, h*w]
        attn_out = attn_out.reshape(b, c, h, w)
        attn_out = self.proj(attn_out)

        channel_out = self.channel_attn(x_norm)
        x = residual + attn_out + channel_out

        residual = x
        x_norm = self._apply_norm(self.norm2, x, cond)
        x = self.ffn(x_norm)
        return residual + x
