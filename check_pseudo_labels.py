#!/usr/bin/env python3
"""
Check pseudo-label files for common issues.
Run this to diagnose why masks are all zeros.
"""

import sys
from pathlib import Path
import numpy as np
import torch

def check_pseudo_labels(pseudo_root: str, num_samples: int = 10):
    """Check pseudo-label files for issues"""
    pseudo_path = Path(pseudo_root)
    
    if not pseudo_path.exists():
        print(f"❌ Pseudo-label directory not found: {pseudo_path}")
        return False
    
    print(f"🔍 Checking pseudo-labels in: {pseudo_path}")
    print()
    
    # Find sample files
    pt_files = list(pseudo_path.rglob("*.pt"))[:num_samples]
    npy_features = list(pseudo_path.rglob("*_features.npy"))[:num_samples]
    npy_masks = list(pseudo_path.rglob("*_masks.npy"))[:num_samples]
    
    if not pt_files and not npy_features:
        print(f"❌ No pseudo-label files found!")
        print(f"   Looked for: *.pt or *_features.npy")
        return False
    
    print(f"✅ Found {len(pt_files)} .pt files, {len(npy_features)} .npy files")
    print()
    
    has_issues = False
    
    # Check .pt format
    if pt_files:
        print(f"📋 Checking .pt format:")
        for i, pt_file in enumerate(pt_files[:3]):
            try:
                data = torch.load(pt_file)
                print(f"\n  File {i+1}: {pt_file.name}")
                print(f"    Keys: {list(data.keys())}")
                
                if "masks" in data:
                    masks = data["masks"]
                    print(f"    masks shape: {masks.shape}")
                    print(f"    masks dtype: {masks.dtype}")
                    print(f"    masks min/max/mean: {masks.min():.4f} / {masks.max():.4f} / {masks.mean():.4f}")
                    print(f"    masks non-zero: {(masks != 0).sum()} / {masks.numel()}")
                    
                    if masks.max() == 0:
                        print(f"    ❌ WARNING: All masks are ZERO!")
                        has_issues = True
                    elif masks.abs().max() < 1e-6:
                        print(f"    ⚠️  WARNING: Masks are nearly zero!")
                        has_issues = True
                    else:
                        print(f"    ✅ Masks look OK")
                
                if "z_d" in data:
                    z_d = data["z_d"]
                    print(f"    z_d shape: {z_d.shape}")
                    print(f"    z_d min/max/mean: {z_d.min():.4f} / {z_d.max():.4f} / {z_d.mean():.4f}")
                    
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                has_issues = True
    
    # Check .npy format  
    if npy_masks:
        print(f"\n📋 Checking .npy format:")
        for i, mask_file in enumerate(npy_masks[:3]):
            try:
                masks = np.load(mask_file)
                print(f"\n  File {i+1}: {mask_file.name}")
                print(f"    masks shape: {masks.shape}")
                print(f"    masks dtype: {masks.dtype}")
                print(f"    masks min/max/mean: {masks.min():.4f} / {masks.max():.4f} / {masks.mean():.4f}")
                print(f"    masks non-zero: {(masks != 0).sum()} / {masks.size}")
                
                if masks.max() == 0:
                    print(f"    ❌ WARNING: All masks are ZERO!")
                    has_issues = True
                elif np.abs(masks).max() < 1e-6:
                    print(f"    ⚠️  WARNING: Masks are nearly zero!")
                    has_issues = True
                else:
                    print(f"    ✅ Masks look OK")
                    
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
                has_issues = True
    
    print()
    print("=" * 70)
    
    if has_issues:
        print("❌ ISSUES FOUND in pseudo-labels!")
        print()
        print("Common causes:")
        print("  1. Pseudo-labels were exported incorrectly")
        print("  2. Export script had bugs")
        print("  3. Source teacher model outputs were zero")
        print()
        print("Solution:")
        print("  Re-export pseudo-labels with correct teacher model")
        print("  Check teacher model is loaded and working properly")
        return False
    else:
        print("✅ Pseudo-labels look good!")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-root", type=str, default="pseudo_labels_v2/train",
                       help="Path to pseudo-label directory")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of files to check")
    args = parser.parse_args()
    
    success = check_pseudo_labels(args.pseudo_root, args.num_samples)
    sys.exit(0 if success else 1)
