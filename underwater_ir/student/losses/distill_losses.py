from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        confidence_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-6))
        loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        loss = loss.sum(dim=-1)
        if confidence_scale is not None:
            loss = loss * confidence_scale
        return loss.mean() if self.reduction == "mean" else loss.sum() / student_logits.shape[0]


class FeatureAlignmentLoss(nn.Module):
    def __init__(self, weight: float = 0.05) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = student_feats - teacher_feats
        if mask is not None:
            while mask.dim() < diff.dim():
                mask = mask.unsqueeze(1)
            diff = diff * mask
            norm = mask.sum().clamp_min(1e-6)
            value = diff.abs().sum() / norm
        else:
            value = diff.abs().mean()
        return self.weight * value


class ContrastiveInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, weight: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, positive_index: torch.Tensor) -> torch.Tensor:
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.t() / self.temperature
        loss = F.cross_entropy(logits, positive_index)
        return self.weight * loss
