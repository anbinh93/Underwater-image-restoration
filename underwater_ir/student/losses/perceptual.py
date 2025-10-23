from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

try:
    from torchvision.models import vgg19
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for PerceptualLoss.") from exc


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers: Sequence[str]) -> None:
        super().__init__()
        self.layers = layers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.features = vgg19(pretrained=True).features
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.ReLU):
                self.features[idx] = nn.ReLU(inplace=False)
        for param in self.features.parameters():
            param.requires_grad = False

        mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 20,
            "relu4_1": 23,
            "relu4_2": 25,
            "relu4_3": 27,
            "relu4_4": 32,
            "relu5_1": 35,
        }
        self.layer_indices = [mapping[name] for name in layers]
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        # Debug: print shapes if mismatch occurs
        try:
            x = (x - self.mean) / self.std
        except RuntimeError as e:
            print(f"❌ Shape mismatch in VGGFeatureExtractor:")
            print(f"   Input x shape: {x.shape}, device: {x.device}")
            print(f"   self.mean shape: {self.mean.shape}, device: {self.mean.device}")
            print(f"   self.std shape: {self.std.shape}, device: {self.std.device}")
            raise e
        
        feats = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    def __init__(self, layers: Sequence[str] = ("relu2_2", "relu3_4"), weight: float = 0.05) -> None:
        super().__init__()
        self.weight = weight
        self.extractor = VGGFeatureExtractor(layers)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self.extractor(pred)
        target_feats = self.extractor(target)
        loss = pred.new_tensor(0.0)
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + (pf - tf).abs().mean()
        return self.weight * loss
