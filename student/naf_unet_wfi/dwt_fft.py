from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F


SQRT2_INV = 2 ** -0.5


@lru_cache(maxsize=8)
def _haar_filter(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    low = torch.tensor([SQRT2_INV, SQRT2_INV], dtype=dtype, device=device)
    high = torch.tensor([SQRT2_INV, -SQRT2_INV], dtype=dtype, device=device)
    ll = torch.einsum("i,j->ij", low, low)
    lh = torch.einsum("i,j->ij", low, high)
    hl = torch.einsum("i,j->ij", high, low)
    hh = torch.einsum("i,j->ij", high, high)
    return ll, lh, hl, hh


def _prepare_kernels(channels: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ll, lh, hl, hh = _haar_filter(device, dtype)
    kernels = []
    for kernel in (ll, lh, hl, hh):
        weight = kernel.view(1, 1, 2, 2).repeat(channels, 1, 1, 1)
        kernels.append(weight)
    return tuple(kernels)


def dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError("Input to dwt2 must be 4D (B, C, H, W).")
    b, c, h, w = x.shape
    kernels = _prepare_kernels(c, x.device, x.dtype)
    outputs = [F.conv2d(x, weight, stride=2, groups=c) for weight in kernels]
    return tuple(outputs)


def idwt2(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    if ll.dim() != 4:
        raise ValueError("Subbands must be 4D tensors.")
    b, c, h, w = ll.shape
    device, dtype = ll.device, ll.dtype
    kernels = _prepare_kernels(c, device, dtype)
    subbands = (ll, lh, hl, hh)
    upsampled = [
        F.conv_transpose2d(
            subbands[idx],
            kernels[idx],
            stride=2,
            groups=c,
        )
        for idx in range(4)
    ]
    return sum(upsampled)


def fft2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.fft2(x, norm="ortho")
    amplitude = spectrum.abs()
    phase = torch.atan2(spectrum.imag, spectrum.real)
    return amplitude, phase


def ifft2(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    spectrum = torch.complex(real, imag)
    return torch.fft.ifft2(spectrum, norm="ortho").real
