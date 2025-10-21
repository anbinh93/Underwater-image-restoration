#!/usr/bin/env python3
"""
Export Pseudo-labels V2 - Using Unified CLIP Loader
Supports both HuggingFace CLIP and custom DACLiP models via config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, InterpolationMode

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from underwater_ir.model_loader import load_clip_model
from underwater_ir.data.datasets import (
    PairedImageDataset,
    UnpairedImageDataset,
    create_dataloader as create_simple_dataloader,
)

# Import teacher modules
try:
    from .deg_coder import DegradationCoder
    from .mask_head import MaskHead
except ImportError:
    from deg_coder import DegradationCoder
    from mask_head import MaskHead


def build_clip_transform(resolution: int = 224) -> Compose:
    """Build standard CLIP preprocessing transform"""
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_prompts(prompt_file: Path) -> List[str]:
    """Load degradation prompts from JSON file"""
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        prompts = list(data.values())
    elif isinstance(data, list):
        prompts = list(data)
    else:
        raise ValueError("Prompt file must contain a JSON list or dictionary of prompts.")
    
    return prompts


def preprocess_for_hf_clip(images: torch.Tensor, processor) -> torch.Tensor:
    """
    Preprocess images for HuggingFace CLIP
    
    Args:
        images: Tensor of shape (B, C, H, W) in [0, 1]
        processor: CLIPProcessor from transformers
        
    Returns:
        Preprocessed tensor ready for CLIP
    """
    batch_images = []
    for img in images:
        # Convert to PIL
        img = img.clamp(0, 1)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        batch_images.append(pil_img)
    
    # Use processor
    inputs = processor(images=batch_images, return_tensors="pt")
    return inputs["pixel_values"]


def extract_features_hf_clip(model, processor, tokenizer, images: torch.Tensor, 
                             texts: List[str], device: torch.device):
    """
    Extract features using HuggingFace CLIP model
    
    Returns:
        image_features, text_features, logits_per_image
    """
    # Preprocess images
    pixel_values = preprocess_for_hf_clip(images, processor).to(device)
    
    # Tokenize texts
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    
    # Get features
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        logits_per_image = outputs.logits_per_image
    
    return image_features, text_features, logits_per_image


def compute_degradation_masks(
    logits: torch.Tensor,
    temperature: float = 10.0,
    threshold: float = 0.5,
    use_crf: bool = False,
    images: Optional[torch.Tensor] = None,
    crf_params: Optional[dict] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute degradation masks from CLIP logits
    
    Args:
        logits: Logits per image (B, N_prompts)
        temperature: Softmax temperature
        threshold: Binary threshold for masks
        use_crf: Whether to apply CRF refinement
        images: Original images for CRF (B, C, H, W)
        crf_params: CRF parameters
        
    Returns:
        masks: Binary masks (B, N_prompts, H, W)
        probs: Probability scores (B, N_prompts)
    """
    # Compute probabilities
    probs = torch.softmax(logits / temperature, dim=-1)
    
    # Simple thresholding for masks
    masks = (probs > threshold).float()
    
    # Expand to spatial dimensions (placeholder - actual implementation may vary)
    B, N = masks.shape
    H, W = 224, 224  # Standard CLIP size
    
    if images is not None:
        H, W = images.shape[2], images.shape[3]
    
    # Expand masks to spatial dimensions
    spatial_masks = masks.unsqueeze(-1).unsqueeze(-1).expand(B, N, H, W)
    
    # TODO: Implement CRF refinement if use_crf is True
    if use_crf and crf_params is not None:
        print("⚠️  CRF refinement not yet implemented in V2")
    
    return spatial_masks, probs


def export_pseudolabels(
    model,
    processor,
    tokenizer,
    dataloader,
    prompts: List[str],
    output_dir: Path,
    device: torch.device,
    model_type: str = "hf_clip",
    temperature: float = 10.0,
    threshold: float = 0.5,
    use_crf: bool = False,
    crf_params: Optional[dict] = None
):
    """
    Export pseudo-labels using CLIP model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    print(f"Processing {len(dataloader)} batches...")
    
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            images = batch.get("lq", batch.get("input"))
            gt_images = batch.get("gt", batch.get("target"))
            names = batch.get("name", [f"image_{i}" for i in range(len(images))])
        else:
            images = batch[0]
            gt_images = batch[1] if len(batch) > 1 else None
            names = [f"image_{batch_idx}_{i}" for i in range(len(images))]
        
        images = images.to(device)
        
        # Extract features based on model type
        if model_type in ["hf_clip", "openai_clip"]:
            image_features, text_features, logits = extract_features_hf_clip(
                model, processor, tokenizer, images, prompts, device
            )
        else:
            # TODO: Implement for custom DACLiP if needed
            raise NotImplementedError(f"Model type {model_type} not yet supported")
        
        # Compute masks
        masks, probs = compute_degradation_masks(
            logits,
            temperature=temperature,
            threshold=threshold,
            use_crf=use_crf,
            images=images,
            crf_params=crf_params
        )
        
        # Save outputs
        for i, name in enumerate(names):
            if isinstance(name, (list, tuple)):
                name = name[0]
            
            # Remove extension if present
            name = Path(name).stem
            
            # Save masks
            mask_np = masks[i].cpu().numpy()  # (N_prompts, H, W)
            np.save(output_dir / f"{name}_masks.npy", mask_np)
            
            # Save probabilities
            prob_np = probs[i].cpu().numpy()  # (N_prompts,)
            np.save(output_dir / f"{name}_probs.npy", prob_np)
            
            # Save image features (z_d - degradation encoding)
            feat_np = image_features[i].cpu().numpy()
            np.save(output_dir / f"{name}_features.npy", feat_np)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    print(f"✅ Pseudo-labels saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export pseudo-labels using unified CLIP loader"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Path to input (degraded) image directory"
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=None,
        help="Path to target (clean) image directory for paired data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for pseudo-labels"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(ROOT / "prompts/degradation_prompts.json"),
        help="Path to degradation prompts JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Softmax temperature"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for masks"
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Enable CRF refinement (placeholder)"
    )
    parser.add_argument(
        "--crf-iters",
        type=int,
        default=5,
        help="CRF iterations"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Pseudo-label Export V2 with Unified CLIP Loader")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Input: {args.input_root}")
    print(f"Output: {args.output}")
    print("")
    
    # Load CLIP model
    print("[1/4] Loading CLIP model...")
    model, processor, tokenizer = load_clip_model(config_path=args.config)
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Load prompts
    print("[2/4] Loading prompts...")
    prompts = load_prompts(Path(args.prompts))
    print(f"  Loaded {len(prompts)} prompts")
    
    # Create dataset and dataloader
    print("[3/4] Creating dataloader...")
    input_root = Path(args.input_root)
    target_root = Path(args.target_root) if args.target_root else None
    
    if target_root and target_root.exists():
        dataset = PairedImageDataset(input_root, target_root)
        print(f"  Using paired dataset")
    else:
        dataset = UnpairedImageDataset(input_root)
        print(f"  Using unpaired dataset")
    
    dataloader = create_simple_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"  Dataset size: {len(dataset)}")
    
    # Export pseudo-labels
    print("[4/4] Exporting pseudo-labels...")
    output_dir = Path(args.output)
    
    crf_params = {
        "iters": args.crf_iters,
    } if args.use_crf else None
    
    # Determine model type from config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_type = config.get('clip', {}).get('model_type', 'openai_clip')
    
    export_pseudolabels(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        dataloader=dataloader,
        prompts=prompts,
        output_dir=output_dir,
        device=device,
        model_type=model_type,
        temperature=args.temperature,
        threshold=args.threshold,
        use_crf=args.use_crf,
        crf_params=crf_params
    )
    
    print("")
    print("=" * 80)
    print("✅ Pseudo-label export completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
