#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from underwater_ir.data import (
    create_paired_eval_loader,
    create_paired_train_loader,
    create_unpaired_eval_loader,
)
from underwater_ir.rl.uiqm_uciqe import compute_uciqe, compute_uiqm
from .naf_unet_wfi import NAFNetWFIGate
from .losses.distill_losses import ContrastiveInfoNCELoss, FeatureAlignmentLoss, KLDivergenceLoss
from .losses.freq_losses import FrequencyLoss, RegionLoss, TotalVariationLoss
from .losses.perceptual import PerceptualLoss


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + epsilon**2).mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x**2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-6)
    return (1 - ssim_map).mean()


def ssim_value(x: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - ssim_loss(x, y).item()


def psnr_value(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def load_pseudo_label(root: Path, rel_path: str) -> Dict[str, torch.Tensor]:
    """Load pseudo-label from .pt or .npy format (V2 compatibility)"""
    base_path = root / Path(rel_path)
    
    # Try .pt first (original format)
    pt_path = base_path.with_suffix(".pt")
    if pt_path.exists():
        data = torch.load(pt_path)
        return data
    
    # Try .npy format (V2 export format)
    # V2 exports: {name}_features.npy, {name}_masks.npy, {name}_probs.npy
    stem = base_path.stem
    parent = base_path.parent
    
    features_path = parent / f"{stem}_features.npy"
    masks_path = parent / f"{stem}_masks.npy"
    probs_path = parent / f"{stem}_probs.npy"
    
    if features_path.exists() and masks_path.exists() and probs_path.exists():
        # Load numpy arrays and convert to tensors
        features = torch.from_numpy(np.load(features_path))
        masks = torch.from_numpy(np.load(masks_path))
        probs = torch.from_numpy(np.load(probs_path))
        
        # Reconstruct pseudo-label dict in expected format
        # Assuming features is the z_d code
        data = {
            "z_d": features,
            "masks": masks,
            "global_prob": probs,
            "confidence": torch.ones_like(probs) * 0.8,  # Default confidence
            "confidence_scale": torch.tensor(0.2),  # Default scale
        }
        return data
    
    # If neither format found, raise error
    raise FileNotFoundError(
        f"Pseudo-label not found for {rel_path}. "
        f"Tried: {pt_path}, {features_path}, {masks_path}, {probs_path}"
    )


@dataclass
class EvalEntry:
    name: str
    loader: torch.utils.data.DataLoader
    pseudo_root: Path


def build_reference_eval_entries(root: Path, pseudo_root: Path, batch_size: int, num_workers: int) -> List[EvalEntry]:
    entries: List[EvalEntry] = []
    root = Path(root).expanduser()
    pseudo_root = Path(pseudo_root).expanduser()
    if not root.exists():
        return entries
    root = root.resolve()
    if pseudo_root.exists():
        pseudo_root = pseudo_root.resolve()
    root = root.resolve()
    if pseudo_root.exists():
        pseudo_root = pseudo_root.resolve()
    for subdir in sorted([d for d in root.iterdir() if d.is_dir()]):
        input_dir = subdir / "input"
        target_dir = subdir / "target"
        if not input_dir.exists() or not target_dir.exists():
            continue
        loader = create_paired_eval_loader(subdir, batch_size=batch_size, num_workers=num_workers)
        entries.append(EvalEntry(subdir.name, loader, pseudo_root / subdir.name))
    return entries


def build_nonref_eval_entries(root: Path, pseudo_root: Path, batch_size: int, num_workers: int) -> List[EvalEntry]:
    entries: List[EvalEntry] = []
    root = Path(root).expanduser()
    pseudo_root = Path(pseudo_root).expanduser()
    if not root.exists():
        return entries
    for subdir in sorted([d for d in root.iterdir() if d.is_dir()]):
        data_root = subdir / "input" if (subdir / "input").exists() else subdir
        loader = create_unpaired_eval_loader(data_root, batch_size=batch_size, num_workers=num_workers)
        entries.append(EvalEntry(subdir.name, loader, pseudo_root / subdir.name))
    return entries


def evaluate_reference(
    model: NAFNetWFIGate,
    entries: Iterable[EvalEntry],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for entry in entries:
            if not entry.pseudo_root.exists():
                print(f"[eval][ref] Skipping {entry.name}: pseudo labels not found at {entry.pseudo_root}")
                continue
            psnr_sum = 0.0
            ssim_sum = 0.0
            count = 0
            for batch in entry.loader:
                lq = batch["LQ"].to(device)
                gt = batch["GT"].to(device)
                rel_paths = batch["rel_path"]
                pseudo = [load_pseudo_label(entry.pseudo_root, rel) for rel in rel_paths]
                masks = torch.stack([item["masks"] for item in pseudo], dim=0).to(device)
                z_d = torch.stack([item["z_d"] for item in pseudo], dim=0).to(device)
                output, _, _ = model(lq, z_d, masks=masks)
                for pred_img, gt_img in zip(output, gt):
                    psnr_sum += psnr_value(pred_img.clamp(0, 1), gt_img.clamp(0, 1))
                    ssim_sum += ssim_value(pred_img.clamp(0, 1).unsqueeze(0), gt_img.clamp(0, 1).unsqueeze(0))
                    count += 1
            if count > 0:
                results[entry.name] = {"psnr": psnr_sum / count, "ssim": ssim_sum / count}
    return results


def evaluate_non_reference(
    model: NAFNetWFIGate,
    entries: Iterable[EvalEntry],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for entry in entries:
            if not entry.pseudo_root.exists():
                print(f"[eval][non-ref] Skipping {entry.name}: pseudo labels not found at {entry.pseudo_root}")
                continue
            uiqm_sum = 0.0
            uciqe_sum = 0.0
            count = 0
            for batch in entry.loader:
                lq = batch["LQ"].to(device)
                rel_paths = batch["rel_path"]
                pseudo = [load_pseudo_label(entry.pseudo_root, rel) for rel in rel_paths]
                masks = torch.stack([item["masks"] for item in pseudo], dim=0).to(device)
                z_d = torch.stack([item["z_d"] for item in pseudo], dim=0).to(device)
                output, _, _ = model(lq, z_d, masks=masks)
                for pred_img in output:
                    pred_img = pred_img.clamp(0, 1).cpu()
                    uiqm_sum += compute_uiqm(pred_img)
                    uciqe_sum += compute_uciqe(pred_img)
                    count += 1
            if count > 0:
                results[entry.name] = {"uiqm": uiqm_sum / count, "uciqe": uciqe_sum / count}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAFNet-WFI student model with DACLiP pseudo-labels.")
    parser.add_argument("--train-root", type=str, default="Dataset/train", help="Path to paired training dataset root.")
    parser.add_argument("--val-ref-root", type=str, default="Dataset/testset(ref)", help="Path to reference benchmark datasets.")
    parser.add_argument("--val-nonref-root", type=str, default="Dataset/testset(non-ref)", help="Path to non-reference benchmark datasets.")
    parser.add_argument(
        "--pseudo-root",
        type=str,
        required=True,
        help="Base directory containing pseudo-labels (expects subfolders 'train', 'testset_ref', 'testset_nonref' unless overrides are provided).",
    )
    parser.add_argument("--pseudo-train-root", type=str, default=None, help="Optional override for training pseudo labels.")
    parser.add_argument("--pseudo-val-ref-root", type=str, default=None, help="Optional override for reference benchmark pseudo labels.")
    parser.add_argument("--pseudo-val-nonref-root", type=str, default=None, help="Optional override for non-ref benchmark pseudo labels.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-config", type=str, default=None, help="Optional JSON overriding loss weights.")
    parser.add_argument("--save-path", type=str, default="student_model.pt")
    parser.add_argument("--num-degradations", type=int, default=None, help="Number of degradation categories for student head.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    pseudo_root = Path(args.pseudo_root).expanduser().resolve()
    pseudo_train_root = Path(args.pseudo_train_root).expanduser().resolve() if args.pseudo_train_root else pseudo_root / "train"
    pseudo_val_ref_root = Path(args.pseudo_val_ref_root).expanduser().resolve() if args.pseudo_val_ref_root else pseudo_root / "testset_ref"
    pseudo_val_nonref_root = Path(args.pseudo_val_nonref_root).expanduser().resolve() if args.pseudo_val_nonref_root else pseudo_root / "testset_nonref"

    if not pseudo_train_root.exists():
        raise FileNotFoundError(f"Training pseudo-label directory not found: {pseudo_train_root}")
    
    # Check for .pt files first, then .npy files (V2 format)
    pseudo_example_path = next(pseudo_train_root.rglob("*.pt"), None)
    if pseudo_example_path is None:
        # Try V2 format: look for *_features.npy files
        pseudo_example_path = next(pseudo_train_root.rglob("*_features.npy"), None)
        if pseudo_example_path is None:
            raise RuntimeError(
                f"No pseudo-label files found in {pseudo_train_root}. "
                f"Expected either *.pt or *_features.npy files."
            )
        
        # Load V2 format
        stem = pseudo_example_path.stem.replace("_features", "")
        parent = pseudo_example_path.parent
        
        masks_path = parent / f"{stem}_masks.npy"
        probs_path = parent / f"{stem}_probs.npy"
        
        if not masks_path.exists() or not probs_path.exists():
            raise RuntimeError(
                f"Incomplete V2 pseudo-label set for {stem}. "
                f"Need: *_features.npy, *_masks.npy, *_probs.npy"
            )
        
        features = torch.from_numpy(np.load(pseudo_example_path))
        masks = torch.from_numpy(np.load(masks_path))
        probs = torch.from_numpy(np.load(probs_path))
        
        pseudo_sample = {
            "z_d": features,
            "masks": masks,
            "global_prob": probs,
        }
    else:
        # Load original .pt format
        pseudo_sample = torch.load(pseudo_example_path)
    cond_dim = pseudo_sample["z_d"].numel()
    num_masks = pseudo_sample["masks"].shape[0]
    num_degradation_types = args.num_degradations or pseudo_sample["global_prob"].numel()

    train_loader = create_paired_train_loader(args.train_root, batch_size=args.batch_size, num_workers=args.num_workers)
    ref_entries = build_reference_eval_entries(Path(args.val_ref_root), pseudo_val_ref_root, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    nonref_entries = build_nonref_eval_entries(Path(args.val_nonref_root), pseudo_val_nonref_root, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    model = NAFNetWFIGate(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_levels=3,
        blocks_per_level=2,
        cond_dim=cond_dim,
        num_masks=num_masks,
        num_degradation_types=num_degradation_types,
    ).to(device)

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
        running_loss = 0.0
        for batch in train_loader:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)
            rel_paths = batch["rel_path"]

            pseudo = [load_pseudo_label(pseudo_train_root, rel) for rel in rel_paths]
            masks = torch.stack([item["masks"] for item in pseudo], dim=0).to(device)
            z_d = torch.stack([item["z_d"] for item in pseudo], dim=0).to(device)
            teacher_prob = torch.stack([item["global_prob"] for item in pseudo], dim=0).to(device)
            confidence_scale = torch.stack(
                [item.get("confidence_scale", torch.ones_like(item["global_prob"])) for item in pseudo], dim=0
            ).to(device)

            output, alpha_maps, student_logits = model(lq, z_d, masks=masks)

            total_loss = lq.new_tensor(0.0)
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

            if all("text_embeddings" in item for item in pseudo):
                image_embed = output.mean(dim=(2, 3))
                text_embed = torch.stack([item["text_embeddings"] for item in pseudo], dim=0).to(device)
                positive_index = torch.arange(image_embed.shape[0], device=device)
                total_loss = total_loss + contrastive_loss(image_embed, text_embed, positive_index)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += total_loss.item()

        torch.save(model.state_dict(), args.save_path)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {avg_loss:.4f}")

        if ref_entries:
            ref_metrics = evaluate_reference(model, ref_entries, device)
            for name, metrics in ref_metrics.items():
                print(f"[Ref] {name}: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
        else:
            print("No reference evaluation datasets found.")

        if nonref_entries:
            nonref_metrics = evaluate_non_reference(model, nonref_entries, device)
            for name, metrics in nonref_metrics.items():
                print(f"[Non-Ref] {name}: UIQM={metrics['uiqm']:.3f}, UCIQE={metrics['uciqe']:.3f}")
        else:
            print("No non-reference evaluation datasets found.")


if __name__ == "__main__":
    main()
