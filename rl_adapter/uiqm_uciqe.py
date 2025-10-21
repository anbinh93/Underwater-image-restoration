from __future__ import annotations

import cv2
import numpy as np
import torch


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def compute_uicm(image: np.ndarray) -> float:
    r, g, b = [image[:, :, i].astype(np.float32) for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b

    mu_rg, sigma_rg = rg.mean(), rg.std()
    mu_yb, sigma_yb = yb.mean(), yb.std()

    return (-0.0268 * mu_rg + 0.1586 * sigma_rg) - (0.0504 * mu_yb - 0.164 * sigma_yb)


def compute_uism(image: np.ndarray) -> float:
    sobel_params = dict(dx=1, dy=0, ksize=3)
    total = 0.0
    for i in range(3):
        channel = image[:, :, i].astype(np.float32)
        gx = cv2.Sobel(channel, cv2.CV_32F, **sobel_params)
        gy = cv2.Sobel(channel, cv2.CV_32F, dx=0, dy=1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        total += np.mean(grad)
    return total / 3.0


def compute_uiconm(image: np.ndarray) -> float:
    l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    max_l = l_channel.max()
    min_l = l_channel.min()
    return (max_l - min_l) / (max_l + min_l + 1e-6)


def compute_uiqm(image: torch.Tensor) -> float:
    np_img = _image_to_numpy(image)
    coeffs = (0.0282, 0.2953, 3.5753)
    uicm = compute_uicm(np_img)
    uism = compute_uism(np_img)
    uiconm = compute_uiconm(np_img)
    return coeffs[0] * uicm + coeffs[1] * uism + coeffs[2] * uiconm


def compute_uciqe(image: torch.Tensor) -> float:
    np_img = _image_to_numpy(image)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    chroma = np.sqrt(a**2 + b**2)
    chroma_std = chroma.std()
    saturation = (chroma / (l + 1e-6)).mean()
    contrast = l.std()

    coeffs = (0.4680, 0.2745, 0.2576)
    return coeffs[0] * chroma_std + coeffs[1] * contrast + coeffs[2] * saturation
