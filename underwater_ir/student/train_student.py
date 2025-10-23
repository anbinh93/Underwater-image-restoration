#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes"""
    return dist.get_world_size() if dist.is_initialized() else 1


def ensure_device_consistency(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Ensure all tensors in dict are on the same device"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + epsilon**2).mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x**2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    # Clamp variance to avoid negative values due to numerical errors
    sigma_x = sigma_x.clamp(min=0)
    sigma_y = sigma_y.clamp(min=0)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)  # Increased epsilon for safety
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
        # Load numpy arrays and convert to tensors with consistent dtype (float32)
        features = torch.from_numpy(np.load(features_path)).float()
        masks = torch.from_numpy(np.load(masks_path)).float()
        probs = torch.from_numpy(np.load(probs_path)).float()
        
        # Reconstruct pseudo-label dict in expected format
        # Assuming features is the z_d code
        data = {
            "z_d": features,
            "masks": masks,
            "global_prob": probs,
            "confidence": torch.ones_like(probs) * 0.8,  # Default confidence
            "confidence_scale": torch.tensor(0.2, dtype=probs.dtype),  # Same dtype as probs
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


def build_reference_eval_entries(root: Path, pseudo_root: Path, batch_size: int, num_workers: int, img_size: int = 256) -> List[EvalEntry]:
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
        loader = create_paired_eval_loader(subdir, batch_size=batch_size, num_workers=num_workers, img_size=img_size)
        entries.append(EvalEntry(subdir.name, loader, pseudo_root / subdir.name))
    return entries


def build_nonref_eval_entries(root: Path, pseudo_root: Path, batch_size: int, num_workers: int, img_size: int = 256) -> List[EvalEntry]:
    entries: List[EvalEntry] = []
    root = Path(root).expanduser()
    pseudo_root = Path(pseudo_root).expanduser()
    if not root.exists():
        return entries
    for subdir in sorted([d for d in root.iterdir() if d.is_dir()]):
        data_root = subdir / "input" if (subdir / "input").exists() else subdir
        loader = create_unpaired_eval_loader(data_root, batch_size=batch_size, num_workers=num_workers, img_size=img_size)
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
                # Ensure device consistency
                pseudo = [ensure_device_consistency(p, device) for p in pseudo]
                masks = torch.stack([item["masks"] for item in pseudo], dim=0)
                z_d = torch.stack([item["z_d"] for item in pseudo], dim=0)
                output, _, _ = model(lq, z_d, masks=masks)
                for pred_img, gt_img in zip(output, gt):
                    pred_clamped = pred_img.clamp(0, 1)
                    gt_clamped = gt_img.clamp(0, 1)
                    
                    # Debug: Check if output is identical to GT
                    if count == 0:  # First sample only
                        mse = torch.mean((pred_clamped - gt_clamped) ** 2).item()
                        if mse < 1e-6:
                            print(f"[eval][ref] WARNING: {entry.name} - output≈GT (MSE={mse:.2e})")
                            print(f"  pred range: [{pred_clamped.min():.4f}, {pred_clamped.max():.4f}]")
                            print(f"  gt range: [{gt_clamped.min():.4f}, {gt_clamped.max():.4f}]")
                    
                    psnr_sum += psnr_value(pred_clamped, gt_clamped)
                    ssim_sum += ssim_value(pred_clamped.unsqueeze(0), gt_clamped.unsqueeze(0))
                    count += 1
            if count > 0:
                avg_psnr = psnr_sum / count
                avg_ssim = ssim_sum / count
                results[entry.name] = {"psnr": avg_psnr, "ssim": avg_ssim}
                
                # Warn if metrics are suspicious
                if avg_ssim > 0.999:
                    print(f"[eval][ref] ⚠️  {entry.name}: SSIM={avg_ssim:.4f} (suspiciously high!)")
                    print(f"  This may indicate model is not learning or output≈input")
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
                # Ensure device consistency
                pseudo = [ensure_device_consistency(p, device) for p in pseudo]
                masks = torch.stack([item["masks"] for item in pseudo], dim=0)
                z_d = torch.stack([item["z_d"] for item in pseudo], dim=0)
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
    parser.add_argument("--img-size", type=int, default=256, help="Input image size (all images will be resized to this)")
    parser.add_argument("--attn-chunk-size", type=int, default=256, help="Chunk size for memory-efficient attention (lower=less memory)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss-config", type=str, default=None, help="Optional JSON overriding loss weights.")
    parser.add_argument("--save-path", type=str, default="student_model.pt")
    parser.add_argument("--num-degradations", type=int, default=None, help="Number of degradation categories for student head.")
    
    # DDP arguments
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training (auto-set by torch.distributed.launch)")
    parser.add_argument("--world-size", type=int, default=1, help="Number of processes for distributed training")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup DDP if enabled
    if args.ddp:
        # Get rank from environment (set by torchrun or torch.distributed.launch)
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        setup_ddp(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")
        args.local_rank = local_rank
        args.world_size = world_size
    else:
        device = torch.device(args.device)
        local_rank = 0
        world_size = 1

    # Only print from main process
    def print_main(*msg):
        if is_main_process():
            print(*msg)

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
        
        # Load V2 format with consistent dtype (float32)
        stem = pseudo_example_path.stem.replace("_features", "")
        parent = pseudo_example_path.parent
        
        masks_path = parent / f"{stem}_masks.npy"
        probs_path = parent / f"{stem}_probs.npy"
        
        if not masks_path.exists() or not probs_path.exists():
            raise RuntimeError(
                f"Incomplete V2 pseudo-label set for {stem}. "
                f"Need: *_features.npy, *_masks.npy, *_probs.npy"
            )
        
        features = torch.from_numpy(np.load(pseudo_example_path)).float()
        masks = torch.from_numpy(np.load(masks_path)).float()
        probs = torch.from_numpy(np.load(probs_path)).float()
        
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

    # Create DistributedSampler for DDP
    train_sampler = None
    if args.ddp:
        from torch.utils.data import DataLoader
        from underwater_ir.data.datasets import PairedImageDataset
        from torchvision import transforms as T
        
        train_root = Path(args.train_root)
        transform = T.Compose([
            T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        train_dataset = PairedImageDataset(train_root / "input", train_root / "target", transform=transform)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = create_paired_train_loader(args.train_root, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)
    
    # Only main process evaluates
    if is_main_process():
        ref_entries = build_reference_eval_entries(Path(args.val_ref_root), pseudo_val_ref_root, batch_size=args.eval_batch_size, num_workers=args.num_workers, img_size=args.img_size)
        nonref_entries = build_nonref_eval_entries(Path(args.val_nonref_root), pseudo_val_nonref_root, batch_size=args.eval_batch_size, num_workers=args.num_workers, img_size=args.img_size)
    else:
        ref_entries = []
        nonref_entries = []

    model = NAFNetWFIGate(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_levels=3,
        blocks_per_level=2,
        cond_dim=cond_dim,
        num_masks=num_masks,
        num_degradation_types=num_degradation_types,
        attn_chunk_size=args.attn_chunk_size,
    ).to(device)
    
    # Initialize weights properly to avoid NaN
    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    print_main(f"✅ Model weights initialized with Kaiming normal")
    
    # Check for NaN in initial weights
    nan_params_init = [name for name, param in model.named_parameters() if torch.isnan(param).any()]
    if nan_params_init:
        raise RuntimeError(f"NaN detected in initial model parameters: {nan_params_init}")
    
    print_main(f"✅ Model initialized with attn_chunk_size={args.attn_chunk_size} (lower=less memory)")
    
    # Wrap model with DDP if enabled
    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print_main(f"✅ DDP enabled: training on {world_size} GPUs")
    
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

    frequency_loss = FrequencyLoss(hf_weight, lf_weight).to(device)
    region_loss = RegionLoss().to(device)
    tv_loss = TotalVariationLoss(tv_weight).to(device)
    perceptual_loss = PerceptualLoss(weight=perc_weight).to(device)
    kd_loss = KLDivergenceLoss().to(device)
    feat_align_loss = FeatureAlignmentLoss(weight=feat_weight).to(device)
    contrastive_loss = ContrastiveInfoNCELoss(weight=ctr_weight).to(device)

    print_main(f"✅ All loss modules moved to device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        epoch_metrics = {'loss': 0.0, 'l1': 0.0, 'ssim': 0.0, 'perc': 0.0, 'freq': 0.0}
        num_batches = 0
        
        # Progress bar (only on main process)
        if is_main_process():
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', 
                       ncols=120, dynamic_ncols=True)
        else:
            pbar = train_loader
        
        for batch in pbar:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)
            rel_paths = batch["rel_path"]

            pseudo = [load_pseudo_label(pseudo_train_root, rel) for rel in rel_paths]
            # Ensure all pseudo-label tensors are on correct device
            pseudo = [ensure_device_consistency(p, device) for p in pseudo]
            
            masks = torch.stack([item["masks"] for item in pseudo], dim=0)
            z_d = torch.stack([item["z_d"] for item in pseudo], dim=0)
            teacher_prob = torch.stack([item["global_prob"] for item in pseudo], dim=0)
            confidence_scale = torch.stack(
                [item.get("confidence_scale", torch.ones_like(item["global_prob"])) for item in pseudo], dim=0
            )

            # CRITICAL: Normalize masks to prevent overflow
            masks = masks / (masks.max() + 1e-8)
            masks = torch.clamp(masks, 0, 1)

            output, alpha_maps, student_logits = model(lq, z_d, masks=masks)

            # Quick NaN check
            if torch.isnan(output).any():
                if is_main_process():
                    print(f"\n[Rank {get_rank()}] ⚠️  NaN detected in output! Skipping batch.")
                continue

            total_loss = lq.new_tensor(0.0)
            
            # Compute losses
            l1_loss_val = charbonnier_loss(output, gt)
            ssim_loss_val = ssim_loss(output, gt)
            perc_loss_val = perceptual_loss(output, gt)
            
            total_loss = total_loss + l1_weight * l1_loss_val
            total_loss = total_loss + ssim_weight * ssim_loss_val
            total_loss = total_loss + perc_loss_val

            # Resize masks to match output/gt dimensions if needed
            if masks.shape[-2:] != output.shape[-2:]:
                masks_resized = F.interpolate(masks, size=output.shape[-2:], mode='nearest')
            else:
                masks_resized = masks
            
            hf_mask = masks_resized[:, :1, :, :]
            lf_mask = masks_resized[:, -1:, :, :]
            hf_term, lf_term = frequency_loss(output, gt, hf_mask=hf_mask, lf_mask=lf_mask)
            freq_loss_val = hf_term + lf_term
            total_loss = total_loss + freq_loss_val

            # Split masks along channel dimension for RegionLoss
            mask_list = [masks_resized[:, i:i+1, :, :] for i in range(masks_resized.shape[1])]
            region_term = region_loss(output, gt, masks=mask_list)
            total_loss = total_loss + region_weight * region_term

            tv_loss_val = tv_loss(output)
            total_loss = total_loss + tv_loss_val

            distill = kd_loss(student_logits, teacher_prob, confidence_scale=confidence_scale)
            total_loss = total_loss + kd_weight * distill

            if alpha_maps:
                student_feat = alpha_maps[-1]
                teacher_feat = masks_resized.mean(dim=1, keepdim=True)
                
                # Resize student_feat to match teacher_feat if dimensions differ
                if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                    student_feat = F.interpolate(
                        student_feat, size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                total_loss = total_loss + feat_align_loss(student_feat, teacher_feat)

            if all("text_embeddings" in item for item in pseudo):
                image_embed = output.mean(dim=(2, 3))
                text_embed = torch.stack([item["text_embeddings"] for item in pseudo], dim=0).to(device)
                positive_index = torch.arange(image_embed.shape[0], device=device)
                total_loss = total_loss + contrastive_loss(image_embed, text_embed, positive_index)

            # Check for NaN/Inf in total loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if is_main_process():
                    print(f"[Rank {get_rank()}] ⚠️  NaN/Inf in loss! Skipping batch.")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping with NaN check
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if is_main_process():
                    print(f"[Rank {get_rank()}] ⚠️  Invalid gradient! Skipping.")
                optimizer.zero_grad()
                continue
            
            optimizer.step()

            # Track metrics
            running_loss += total_loss.item()
            epoch_metrics['loss'] += total_loss.item()
            epoch_metrics['l1'] += l1_loss_val.item()
            epoch_metrics['ssim'] += ssim_loss_val.item()
            epoch_metrics['perc'] += perc_loss_val.item()
            epoch_metrics['freq'] += freq_loss_val.item()
            num_batches += 1
            
            # Update progress bar (main process only)
            if is_main_process():
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'l1': f'{l1_loss_val.item():.3f}',
                    'ssim': f'{ssim_loss_val.item():.3f}'
                })

        # Synchronize loss across all processes for DDP
        if args.ddp:
            avg_loss_tensor = torch.tensor(running_loss / max(num_batches, 1), device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        else:
            avg_loss = running_loss / max(num_batches, 1)
        
        # Calculate epoch average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        # Only main process saves and evaluates
        if is_main_process():
            # Save model (unwrap DDP if needed)
            model_to_save = model.module if args.ddp else model
            torch.save(model_to_save.state_dict(), args.save_path)
            
            # Print comprehensive epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{args.epochs} Summary")
            print(f"{'='*80}")
            print(f"  Total Loss:      {avg_loss:.4f}")
            print(f"  L1 Loss:         {epoch_metrics['l1']:.4f}")
            print(f"  SSIM Loss:       {epoch_metrics['ssim']:.4f}")
            print(f"  Perceptual Loss: {epoch_metrics['perc']:.4f}")
            print(f"  Frequency Loss:  {epoch_metrics['freq']:.4f}")
            print(f"  Batches:         {num_batches}")
            
            # Evaluation
            if ref_entries:
                print(f"\n{'-'*80}")
                print("Reference Evaluation:")
                print(f"{'-'*80}")
                ref_metrics = evaluate_reference(model, ref_entries, device)
                for name, metrics in ref_metrics.items():
                    print(f"  {name:25s} | PSNR: {metrics['psnr']:6.2f} dB | SSIM: {metrics['ssim']:.4f}")

            if nonref_entries:
                print(f"\n{'-'*80}")
                print("Non-Reference Evaluation:")
                print(f"{'-'*80}")
                nonref_metrics = evaluate_non_reference(model, nonref_entries, device)
                for name, metrics in nonref_metrics.items():
                    print(f"  {name:25s} | UIQM: {metrics['uiqm']:6.3f} | UCIQE: {metrics['uciqe']:6.3f}")
            
            print(f"{'='*80}\n")
        
        # Synchronize all processes after evaluation
        if args.ddp:
            dist.barrier()
    
    # Cleanup DDP
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
