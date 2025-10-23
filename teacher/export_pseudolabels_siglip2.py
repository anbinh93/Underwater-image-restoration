#!/usr/bin/env python3
"""
Export pseudo-labels using SigLIP v2 (google/siglip2-large-patch16-512)
This replaces the DACLiP teacher model with SigLIP v2 for better feature extraction.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# Degradation prompts for underwater images
DEGRADATION_PROMPTS = [
    "clear water with good visibility",
    "slight blue-green color cast", 
    "moderate color distortion underwater",
    "heavy color cast and low contrast",
    "very murky water with particles",
    "backscatter and suspended particles",
    "low light conditions underwater",
    "artificial lighting with color shift",
    "extreme turbidity and poor visibility",
    "mixed degradations with noise"
]


class SigLIP2PseudoLabelExtractor:
    """Extract pseudo-labels using SigLIP v2"""
    
    def __init__(self, model_name="google/siglip2-large-patch16-512", device="cuda"):
        print(f"Loading SigLIP v2 model: {model_name}")
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        
        # Prepare text embeddings for degradation types
        print("Encoding degradation prompts...")
        self.text_features = self._encode_text_prompts(DEGRADATION_PROMPTS)
        print(f"✅ Text features shape: {self.text_features.shape}")
        
    def _encode_text_prompts(self, prompts):
        """Encode text prompts to get text embeddings"""
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            text_embeds = self.model.get_text_features(**inputs)
            # Normalize
            text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds
    
    def extract_image_features(self, image_path):
        """
        Extract features from a single image
        
        Returns:
            features: Image features (1D vector)
            masks: Degradation masks (C, H, W) - one per degradation type
            probs: Degradation probabilities (C,)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get image features
            outputs = self.model.get_image_features(**inputs)
            image_features = F.normalize(outputs, dim=-1)  # (1, D)
            
            # Compute similarity with text prompts
            similarity = (image_features @ self.text_features.T)  # (1, C)
            
            # Convert to probabilities using softmax
            probs = F.softmax(similarity * 10.0, dim=-1)  # Temperature scaling
            
            # Create spatial masks based on probabilities
            # SigLIP v2 outputs 512x512 patches for large model
            # We'll create degradation-aware spatial masks
            masks = self._create_spatial_masks(image, probs, size=(224, 224))
        
        return (
            image_features.squeeze(0).cpu().numpy(),  # (D,)
            masks.cpu().numpy(),  # (C, H, W)
            probs.squeeze(0).cpu().numpy()  # (C,)
        )
    
    def _create_spatial_masks(self, image, probs, size=(224, 224)):
        """
        Create spatial masks for each degradation type.
        
        For now, we create uniform masks weighted by probabilities,
        but you can extend this to use attention maps or patch features.
        """
        C = probs.shape[1]  # Number of degradation types
        H, W = size
        
        # Create masks: each channel weighted by its probability
        masks = probs.view(C, 1, 1).expand(C, H, W)  # (C, H, W)
        
        # Optional: Add spatial variation based on image statistics
        # This helps the student learn spatial degradation patterns
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        img_resized = F.interpolate(img_tensor, size=size, mode='bilinear', align_corners=False)
        
        # Use image intensity as spatial prior
        intensity = img_resized.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        
        # Modulate masks with intensity (darker areas = more degradation)
        spatial_weight = 1.0 - intensity.squeeze(0)  # (1, H, W)
        masks = masks * (0.5 + 0.5 * spatial_weight)  # Soft modulation
        
        # Ensure masks are properly normalized per channel
        for c in range(C):
            mask_c = masks[c]
            if mask_c.max() > 0:
                masks[c] = mask_c / (mask_c.sum() + 1e-8) * (H * W)
        
        return masks


def export_pseudo_labels(
    data_root: str,
    output_root: str,
    model_name: str = "google/siglip2-large-patch16-512",
    device: str = "cuda",
    batch_size: int = 1,  # Process one at a time for now
):
    """
    Export pseudo-labels for all images in data_root
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SIGLIP V2 PSEUDO-LABEL EXPORT")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Model: {model_name}")
    print()
    
    # Initialize extractor
    extractor = SigLIP2PseudoLabelExtractor(model_name=model_name, device=device)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_root.rglob(f"*{ext}"))
        image_files.extend(data_root.rglob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    print(f"Found {len(image_files)} images")
    print()
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc="Exporting pseudo-labels"):
        try:
            # Extract features and masks
            features, masks, probs = extractor.extract_image_features(img_path)
            
            # Validate that masks are NOT all zeros
            if masks.max() == 0:
                print(f"⚠️  Warning: {img_path.name} produced zero masks!")
                error_count += 1
                continue
            
            # Save as .npy files
            stem = img_path.stem
            np.save(output_root / f"{stem}_features.npy", features)
            np.save(output_root / f"{stem}_masks.npy", masks)
            np.save(output_root / f"{stem}_probs.npy", probs)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            error_count += 1
            continue
    
    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully exported: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Output directory: {output_root}")
    print()
    
    # Validate a few samples
    print("Validating samples...")
    sample_masks = list(output_root.glob("*_masks.npy"))[:3]
    for mask_path in sample_masks:
        masks = np.load(mask_path)
        print(f"  {mask_path.name}")
        print(f"    Shape: {masks.shape}")
        print(f"    Min/Max/Mean: {masks.min():.4f} / {masks.max():.4f} / {masks.mean():.4f}")
        if masks.max() == 0:
            print(f"    ❌ WARNING: Still zero masks!")
        else:
            print(f"    ✅ OK")


def main():
    parser = argparse.ArgumentParser(description="Export pseudo-labels using SigLIP v2")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing images"
    )
    parser.add_argument(
        "--output-root", 
        type=str,
        required=True,
        help="Output directory for pseudo-labels"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/siglip2-large-patch16-512",
        help="SigLIP v2 model name from HuggingFace"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (keep at 1 for stability)"
    )
    
    args = parser.parse_args()
    
    export_pseudo_labels(
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
