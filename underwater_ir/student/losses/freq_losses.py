from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..naf_unet_wfi.dwt_fft import dwt2, fft2


def _masked_l1(diff: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return diff.abs().mean()
    weight = mask.float()
    while weight.dim() < diff.dim():
        weight = weight.unsqueeze(1)
    norm = weight.sum().clamp_min(1e-6)
    return (diff.abs() * weight).sum() / norm


class FrequencyLoss(nn.Module):
    def __init__(self, lambda_hf: float = 0.5, lambda_lf: float = 0.5) -> None:
        super().__init__()
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        hf_mask: Optional[torch.Tensor] = None,
        lf_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_ll, pred_lh, pred_hl, pred_hh = dwt2(pred)
        tgt_ll, tgt_lh, tgt_hl, tgt_hh = dwt2(target)

        hf_loss = (
            _masked_l1(pred_lh - tgt_lh, hf_mask)
            + _masked_l1(pred_hl - tgt_hl, hf_mask)
            + _masked_l1(pred_hh - tgt_hh, hf_mask)
        )

        amp_pred, _ = fft2(pred_ll)
        amp_tgt, _ = fft2(tgt_ll)
        lf_loss = _masked_l1(amp_pred - amp_tgt, lf_mask)

        return self.lambda_hf * hf_loss, self.lambda_lf * lf_loss


class RegionLoss(nn.Module):
    def __init__(self, weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        masks: Optional[Iterable[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if masks is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        total = 0.0
        weight_sum = 0.0
        for idx, mask in enumerate(masks):
            w = self.weights[idx] if self.weights is not None else 1.0
            total = total + w * _masked_l1(pred - target, mask)
            weight_sum += w
        return total / max(weight_sum, 1e-6)


class TotalVariationLoss(nn.Module):
    def __init__(self, weight: float = 1e-4) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return self.weight * (tv_h + tv_w)
