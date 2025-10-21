from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


EncoderOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]


def _maybe_freeze(module: Optional[nn.Module], freeze: bool) -> None:
    if module is None or not freeze:
        return
    for param in module.parameters():
        param.requires_grad = False


class EncoderAdapter(nn.Module):
    """Wraps arbitrary image/text encoders and projects them into a common embedding space."""

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        target_dim: Optional[int] = 512,
        freeze_backbones: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.target_dim = target_dim
        self.normalize = normalize

        self.image_projection: Optional[nn.Linear] = None
        self.text_projection: Optional[nn.Linear] = None

        _maybe_freeze(self.image_encoder, freeze_backbones)
        _maybe_freeze(self.text_encoder, freeze_backbones)

    def _ensure_projection(self, attr: str, in_dim: int, device: torch.device) -> Optional[nn.Linear]:
        projection = getattr(self, attr)
        if self.target_dim is None or in_dim == self.target_dim:
            return None
        if projection is None:
            projection = nn.Linear(in_dim, self.target_dim, bias=False)
            nn.init.xavier_normal_(projection.weight)
            setattr(self, attr, projection)
        return projection.to(device)

    @staticmethod
    def _split_encoder_output(output: EncoderOutput) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(output, dict):
            if "pooled" in output:
                pooled = output["pooled"]
            elif "image_features" in output:
                pooled = output["image_features"]
            else:
                pooled = next((v for v in output.values() if isinstance(v, torch.Tensor)), None)
                if pooled is None:
                    raise ValueError("Encoder output dictionary does not contain tensors.")
            tokens = output.get("tokens") or output.get("patch_tokens") or output.get("dense")
            return pooled, tokens
        if isinstance(output, (tuple, list)):
            pooled = output[0]
            tokens = output[1] if len(output) > 1 else None
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[-1]
            return pooled, tokens
        return output, None

    def forward_image(
        self,
        images: torch.Tensor,
        return_tokens: bool = True,
        **encoder_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        raw_output = self.image_encoder(images, **encoder_kwargs)
        pooled, tokens = self._split_encoder_output(raw_output)
        pooled = pooled.float()

        proj = self._ensure_projection("image_projection", pooled.shape[-1], pooled.device)
        if proj is not None:
            pooled = proj(pooled)

        if self.normalize:
            pooled = F.normalize(pooled, dim=-1)

        result: Dict[str, torch.Tensor] = {"pooled": pooled}

        if return_tokens and tokens is not None:
            tokens = tokens.float()
            if tokens.dim() == 3 and tokens.shape[0] == pooled.shape[0]:
                flat_tokens = tokens
            elif tokens.dim() == 3 and tokens.shape[1] == pooled.shape[0]:
                flat_tokens = tokens.transpose(0, 1)
            else:
                flat_tokens = tokens

            proj_tokens = self._ensure_projection(
                "image_projection", flat_tokens.shape[-1], flat_tokens.device
            )
            if proj_tokens is not None:
                flat_tokens = proj_tokens(flat_tokens)

            if self.normalize:
                flat_tokens = F.normalize(flat_tokens, dim=-1)

            result["tokens"] = flat_tokens
        return result

    def forward_text(
        self,
        tokenized_prompts: torch.Tensor,
        **encoder_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        raw_output = self.text_encoder(tokenized_prompts, **encoder_kwargs)
        pooled, tokens = self._split_encoder_output(raw_output)
        pooled = pooled.float()

        proj = self._ensure_projection("text_projection", pooled.shape[-1], pooled.device)
        if proj is not None:
            pooled = proj(pooled)

        if self.normalize:
            pooled = F.normalize(pooled, dim=-1)

        result: Dict[str, torch.Tensor] = {"pooled": pooled}

        if tokens is not None:
            tokens = tokens.float()
            proj_tokens = self._ensure_projection(
                "text_projection", tokens.shape[-1], tokens.device
            )
            if proj_tokens is not None:
                tokens = proj_tokens(tokens)
            if self.normalize:
                tokens = F.normalize(tokens, dim=-1)
            result["tokens"] = tokens
        return result

    def extra_repr(self) -> str:
        return f"target_dim={self.target_dim}, normalize={self.normalize}"
