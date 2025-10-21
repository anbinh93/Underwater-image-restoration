#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rl_adapter.q_learning import EnhancementAction, QLearningAgent
from rl_adapter.uiep_cwgr_ops import (
    clahe_equalization,
    gamma_correction,
    guided_filter,
    lf_hf_mix,
    multi_scale_retinex,
    wavelet_denoise,
)
from rl_adapter.uiqm_uciqe import compute_uciqe, compute_uiqm
from student.naf_unet_wfi import NAFNetWFIGate


@dataclass
class EnhancementKnobs:
    gamma: float = 1.0
    clahe_clip: float = 2.0
    msr_mix: float = 0.5
    lf_hf_alpha: float = 0.5

    def apply_delta(self, action: EnhancementAction) -> None:
        self.gamma = float(np.clip(self.gamma + action.gamma_delta, 0.5, 2.5))
        self.clahe_clip = float(np.clip(self.clahe_clip + action.clahe_delta, 1.0, 8.0))
        self.msr_mix = float(np.clip(self.msr_mix + action.msr_mix_delta, 0.0, 1.0))
        self.lf_hf_alpha = float(np.clip(self.lf_hf_alpha + action.lf_hf_alpha_delta, 0.0, 1.0))


def pil_to_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def compute_sharpness(image: torch.Tensor) -> float:
    gray = image.mean(dim=1, keepdim=True)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    sobel_x = F.conv2d(gray, sobel_kernel_x, padding=1)
    sobel_y = F.conv2d(gray, sobel_kernel_y, padding=1)
    grad = torch.sqrt(sobel_x**2 + sobel_y**2)
    return float(grad.mean().item())


def build_state(image: torch.Tensor, probs: torch.Tensor) -> List[float]:
    uiqm = compute_uiqm(image.squeeze(0))
    uciqe = compute_uciqe(image.squeeze(0))
    hist_mean = float(image.mean().item())
    hist_std = float(image.std().item())
    sharpness = compute_sharpness(image)
    return [uiqm, uciqe, hist_mean, hist_std, sharpness] + probs.tolist()


def apply_pipeline(image: torch.Tensor, knobs: EnhancementKnobs) -> torch.Tensor:
    img = gamma_correction(image.squeeze(0), knobs.gamma).unsqueeze(0)
    clahe = clahe_equalization(img.squeeze(0), clip_limit=knobs.clahe_clip).unsqueeze(0)
    denoised = wavelet_denoise(clahe.squeeze(0)).unsqueeze(0)
    guided = guided_filter(denoised.squeeze(0), clahe.squeeze(0)).unsqueeze(0)
    retinex = multi_scale_retinex(guided.squeeze(0)).unsqueeze(0)
    mixed = lf_hf_mix(guided, retinex, knobs.msr_mix)
    blended = lf_hf_mix(mixed, retinex, knobs.lf_hf_alpha)
    return blended


def adapt_image(
    model: NAFNetWFIGate,
    image: torch.Tensor,
    z_d: torch.Tensor,
    masks: torch.Tensor,
    steps: int = 5,
    epsilon: float = 0.1,
) -> Tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        restored, _, deg_logits = model(image, z_d, masks=masks)
    probs = F.softmax(deg_logits, dim=-1).squeeze(0)

    action_space = [
        EnhancementAction(gamma_delta=-0.1, clahe_delta=0.0, msr_mix_delta=0.05, lf_hf_alpha_delta=0.05),
        EnhancementAction(gamma_delta=0.1, clahe_delta=0.0, msr_mix_delta=-0.05, lf_hf_alpha_delta=-0.05),
        EnhancementAction(gamma_delta=0.0, clahe_delta=0.5, msr_mix_delta=0.0, lf_hf_alpha_delta=0.0),
        EnhancementAction(gamma_delta=0.0, clahe_delta=-0.5, msr_mix_delta=0.0, lf_hf_alpha_delta=0.0),
        EnhancementAction(gamma_delta=0.0, clahe_delta=0.0, msr_mix_delta=0.1, lf_hf_alpha_delta=-0.1),
    ]

    agent = QLearningAgent(action_space, epsilon=epsilon)
    knobs = EnhancementKnobs()

    current_image = restored.clone()
    best_image = current_image.clone()
    best_score = -float("inf")

    state = build_state(current_image, probs)
    current_uiqm = state[0]
    current_uciqe = state[1]

    for _ in range(steps):
        action_idx = agent.select_action(state)
        knobs.apply_delta(action_space[action_idx])
        candidate = apply_pipeline(restored, knobs)
        next_state = build_state(candidate, probs)
        next_uiqm, next_uciqe = next_state[0], next_state[1]

        reward = 0.6 * (next_uiqm - current_uiqm) + 0.4 * (next_uciqe - current_uciqe)
        agent.update(state, action_idx, reward, next_state)

        if reward >= 0:
            current_image = candidate
            current_uiqm, current_uciqe = next_uiqm, next_uciqe
            state = next_state
            score = current_uiqm + current_uciqe
            if score > best_score:
                best_score = score
                best_image = current_image.clone()

    return best_image, current_uiqm, current_uciqe


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with RL-based enhancement adapter.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained student model weights.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--pseudo", type=str, required=True, help="Pseudo-label file (.pt) for the image.")
    parser.add_argument("--output", type=str, required=True, help="Output image path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    pseudo = torch.load(args.pseudo)
    z_d = pseudo["z_d"].unsqueeze(0).to(device)
    masks = pseudo["masks"].unsqueeze(0).to(device)
    num_masks = masks.shape[1]
    cond_dim = z_d.shape[-1]
    num_degradations = pseudo["global_prob"].numel()

    model = NAFNetWFIGate(
        cond_dim=cond_dim,
        num_masks=num_masks,
        num_degradation_types=num_degradations,
    ).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    image = pil_to_tensor(Path(args.image), device)
    enhanced, uiqm, uciqe = adapt_image(model, image, z_d, masks, steps=args.steps)

    output_image = tensor_to_image(enhanced)
    cv2.imwrite(args.output, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Enhanced image saved to {args.output}. UIQM={uiqm:.4f}, UCIQE={uciqe:.4f}")


if __name__ == "__main__":
    main()
