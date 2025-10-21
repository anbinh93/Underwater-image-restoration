from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..student.naf_unet_wfi.dwt_fft import dwt2, idwt2


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_tensor(image: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = torch.from_numpy(image.astype(np.float32) / 255.0).to(device=device, dtype=dtype)
    return img.permute(2, 0, 1).contiguous()


def histogram_equalization(image: torch.Tensor) -> torch.Tensor:
    np_img = _to_numpy(image)
    out_channels = []
    for ch in cv2.split(np_img):
        hist, bins = np.histogram(ch.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
        equalized = np.interp(ch.flatten(), bins[:-1], cdf).reshape(ch.shape)
        out_channels.append(equalized.astype(np.uint8))
    merged = cv2.merge(out_channels)
    return _to_tensor(merged, image.device, image.dtype)


def clahe_equalization(image: torch.Tensor, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> torch.Tensor:
    np_img = _to_numpy(image)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _to_tensor(rgb, image.device, image.dtype)


def wavelet_denoise(image: torch.Tensor, threshold_scale: float = 1.0) -> torch.Tensor:
    x = image.unsqueeze(0)
    ll, lh, hl, hh = dwt2(x)
    detail_coeffs = [lh, hl, hh]

    denoised_coeffs = []
    for coeff in detail_coeffs:
        sigma = torch.median(coeff.abs()) / 0.6745
        var = torch.mean(coeff**2)
        threshold = threshold_scale * (sigma**2 / (var.sqrt() + 1e-6))
        denoised = torch.sign(coeff) * torch.relu(coeff.abs() - threshold)
        denoised_coeffs.append(denoised)
    output = idwt2(ll, *denoised_coeffs)
    return output.squeeze(0)


def _box_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    size = 2 * radius + 1
    kernel = torch.ones(1, 1, size, size, device=device, dtype=dtype)
    return kernel / (size * size)


def guided_filter(input_img: torch.Tensor, guidance: torch.Tensor, radius: int = 4, eps: float = 1e-3) -> torch.Tensor:
    if input_img.dim() == 3:
        input_img = input_img.unsqueeze(0)
    if guidance.dim() == 3:
        guidance = guidance.unsqueeze(0)
    kernel = _box_kernel(radius, input_img.device, input_img.dtype)
    mean_I = F.conv2d(guidance, kernel, padding=radius)
    mean_P = F.conv2d(input_img, kernel, padding=radius)
    mean_IP = F.conv2d(guidance * input_img, kernel, padding=radius)
    cov_IP = mean_IP - mean_I * mean_P
    mean_II = F.conv2d(guidance * guidance, kernel, padding=radius)
    var_I = mean_II - mean_I**2

    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a = F.conv2d(a, kernel, padding=radius)
    mean_b = F.conv2d(b, kernel, padding=radius)
    q = mean_a * guidance + mean_b
    return q.squeeze(0)


def gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def multi_scale_retinex(image: torch.Tensor, sigmas: Sequence[float] = (15, 80, 250)) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    x = image.clamp_min(1e-3)
    log_x = torch.log(x)
    result = torch.zeros_like(x)
    for sigma in sigmas:
        size = int(2 * math.ceil(3 * sigma) + 1)
        kernel = gaussian_kernel(size, sigma, x.device, x.dtype).repeat(x.shape[1], 1, 1, 1)
        blurred = F.conv2d(x, kernel, padding=size // 2, groups=x.shape[1])
        result += log_x - torch.log(blurred + 1e-6)
    retinex = result / len(sigmas)
    retinex = torch.exp(retinex) - 1.0
    retinex = torch.clamp(retinex, 0.0, 1.0)
    return retinex.squeeze(0)


def gamma_correction(image: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = max(gamma, 1e-3)
    return image.clamp(0, 1) ** gamma


def lf_hf_mix(low_freq: torch.Tensor, high_freq: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp(alpha * high_freq + (1 - alpha) * low_freq, 0.0, 1.0)
