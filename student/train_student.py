#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "universal-image-restoration"))

import options as option  # noqa: E402
from data import create_dataset  # noqa: E402

from student.naf_unet_wfi import NAFNetWFIGate  # noqa: E402
from student.losses.freq_losses import FrequencyLoss, RegionLoss, TotalVariationLoss  # noqa: E402
from student.losses.distill_losses import ContrastiveInfoNCELoss, FeatureAlignmentLoss, KLDivergenceLoss  # noqa: E402
from student.losses.perceptual import PerceptualLoss  # noqa: E402


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + epsilon**2).mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = torch.nn.functional.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = torch.nn.functional.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = torch.nn.functional.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x**2
    sigma_y = torch.nn.functional.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y**2
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-6)
    return (1 - ssim_map).mean()


def load_pseudo_label(root: Path, image_path: str) -> Dict[str, torch.Tensor]:
    rel = Path(image_path).with_suffix(".pt")
    candidate = root / rel
    if not candidate.exists():
        raise FileNotFoundError(f"Pseudo-label file not found for {image_path}")
    data = torch.load(candidate)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAFNet-WFI student model with DACLiP pseudo-labels.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for datasets.")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset key (train/val).")
    parser.add_argument("--pseudo-root", type=str, required=True, help="Directory containing pseudo-label .pt files.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-config", type=str, default=None, help="Optional JSON overriding loss weights.")
    parser.add_argument("--save-path", type=str, default="student_model.pt")
    parser.add_argument("--num-degradations", type=int, default=None, help="Number of degradation categories for student head.")
    args = parser.parse_args()

    pseudo_root = Path(args.pseudo_root)
    opt = option.parse(args.config, is_train=True)
    opt = option.dict_to_nonedict(opt)
    dataset_opt = opt["datasets"][args.dataset]
    dataset = create_dataset(dataset_opt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pseudo_example_path = next(pseudo_root.rglob("*.pt"))
    pseudo_sample = torch.load(pseudo_example_path)
    cond_dim = pseudo_sample["z_d"].numel()
    num_masks = pseudo_sample["masks"].shape[0]
    num_degradation_types = args.num_degradations or (pseudo_sample["global_prob"].numel())

    model = NAFNetWFIGate(
        in_channels=dataset_opt.get("in_channels", 3),
        out_channels=dataset_opt.get("out_channels", 3),
        base_channels=64,
        num_levels=3,
        blocks_per_level=2,
        cond_dim=cond_dim,
        num_masks=num_masks,
        num_degradation_types=num_degradation_types,
    ).to(args.device)

    l1_weight = 1.0
    ssim_weight = 0.2
    perc_weight = 0.05
    hf_weight = 0.5
    lf_weight = 0.5
    region_weight = 0.3
    tv_weight = 1e-4
    kd_weight = 0.1
    feat_weight = 0.05
    ctr_weight = 0.05
    id_weight = 0.0

    if args.loss_config:
        with open(args.loss_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        l1_weight = cfg.get("rec", l1_weight)
        ssim_weight = cfg.get("ssim", ssim_weight)
        perc_weight = cfg.get("perc", perc_weight)
        hf_weight = cfg.get("hf", hf_weight)
        lf_weight = cfg.get("lf", lf_weight)
        region_weight = cfg.get("region", region_weight)
        tv_weight = cfg.get("tv", tv_weight)
        kd_weight = cfg.get("kd", kd_weight)
        feat_weight = cfg.get("feat", feat_weight)
        ctr_weight = cfg.get("ctr", ctr_weight)
        id_weight = cfg.get("id", id_weight)

    frequency_loss = FrequencyLoss(hf_weight, lf_weight)
    region_loss = RegionLoss()
    tv_loss = TotalVariationLoss(tv_weight)
    perceptual_loss = PerceptualLoss(weight=perc_weight)
    kd_loss = KLDivergenceLoss()
    feat_align_loss = FeatureAlignmentLoss(weight=feat_weight)
    contrastive_loss = ContrastiveInfoNCELoss(weight=ctr_weight)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader:
            lq = batch["LQ"].to(args.device)
            gt = batch.get("GT")
            if gt is not None:
                gt = gt.to(args.device)

            pseudo = [load_pseudo_label(pseudo_root, path) for path in batch["LQ_path"]]
            masks = torch.stack([item["masks"] for item in pseudo], dim=0).to(args.device)
            z_d = torch.stack([item["z_d"] for item in pseudo], dim=0).to(args.device)
            teacher_prob = torch.stack([item["global_prob"] for item in pseudo], dim=0).to(args.device)
            confidence_scale = torch.stack([item["confidence_scale"] for item in pseudo], dim=0).to(args.device)

            output, alpha_maps, student_logits = model(lq, z_d, masks=masks)

            total_loss = torch.tensor(0.0, device=args.device)

            if gt is not None:
                total_loss = total_loss + l1_weight * charbonnier_loss(output, gt)
                total_loss = total_loss + ssim_weight * ssim_loss(output, gt)
                total_loss = total_loss + perceptual_loss(output, gt)

                hf_mask = masks[:, :1, :, :]
                lf_mask = masks[:, -1:, :, :]
                hf_term, lf_term = frequency_loss(output, gt, hf_mask=hf_mask, lf_mask=lf_mask)
                total_loss = total_loss + hf_term + lf_term

                region_term = region_loss(output, gt, masks=masks)
                total_loss = total_loss + region_weight * region_term

                total_loss = total_loss + tv_loss(output)

            distill = kd_loss(student_logits, teacher_prob, confidence_scale=confidence_scale)
            total_loss = total_loss + kd_weight * distill

            if alpha_maps:
                student_feat = alpha_maps[-1]
                teacher_feat = masks.mean(dim=1, keepdim=True)
                total_loss = total_loss + feat_align_loss(student_feat, teacher_feat)

            if "text_embeddings" in pseudo[0]:
                image_embed = output.mean(dim=(2, 3))
                text_embed = pseudo[0]["text_embeddings"].to(args.device)
                positive_index = torch.arange(image_embed.shape[0], device=args.device)
                total_loss = total_loss + contrastive_loss(image_embed, text_embed, positive_index)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        torch.save(model.state_dict(), args.save_path)
        print(f"Epoch {epoch + 1}/{args.epochs} completed. Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    main()
