#!/usr/bin/env python3
"""
Quick validation script for pseudo-label export.
Use this to check if teacher model is producing valid outputs before exporting.
"""

import sys
import torch
from pathlib import Path

def validate_teacher_export(teacher_checkpoint: str, sample_image_path: str):
    """
    Validate that teacher model produces non-zero masks.
    """
    print("=" * 70)
    print("PSEUDO-LABEL EXPORT VALIDATION")
    print("=" * 70)
    print()
    
    # Check checkpoint exists
    ckpt_path = Path(teacher_checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Teacher checkpoint not found: {ckpt_path}")
        return False
    print(f"✅ Teacher checkpoint found: {ckpt_path}")
    
    # Check sample image
    img_path = Path(sample_image_path)
    if not img_path.exists():
        print(f"❌ Sample image not found: {img_path}")
        return False
    print(f"✅ Sample image found: {img_path}")
    print()
    
    # Try to load and run teacher model
    try:
        print("Loading teacher model...")
        # TODO: Add your teacher model loading code here
        # Example:
        # from teacher.deg_coder import TeacherModel
        # teacher = TeacherModel.load_from_checkpoint(teacher_checkpoint)
        # teacher.eval()
        
        print("⚠️  TODO: Implement teacher model loading in this script")
        print()
        print("You need to:")
        print("  1. Import your teacher model class")
        print("  2. Load checkpoint")
        print("  3. Run inference on sample image")
        print("  4. Check that masks are NOT all zeros")
        print()
        print("Example code:")
        print("""
    from PIL import Image
    import torchvision.transforms as T
    from your_teacher_module import TeacherModel
    
    teacher = TeacherModel.load_from_checkpoint(teacher_checkpoint)
    teacher.eval()
    teacher = teacher.cuda()
    
    img = Image.open(sample_image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = teacher(img_tensor)
        masks = output['masks']  # or however your model outputs masks
        
    print(f"Masks shape: {masks.shape}")
    print(f"Masks min/max/mean: {masks.min():.4f} / {masks.max():.4f} / {masks.mean():.4f}")
    print(f"Masks non-zero: {(masks != 0).sum()} / {masks.numel()}")
    
    if masks.max() == 0:
        print("❌ PROBLEM: Masks are all zeros!")
        print("Check:")
        print("  - Teacher model loaded correctly?")
        print("  - Model in eval mode?")
        print("  - Input image normalized correctly?")
        return False
    else:
        print("✅ Masks look good!")
        return True
        """)
        
        return None  # Cannot validate without implementation
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_pseudo_labels(pseudo_root: str):
    """Check if existing pseudo-labels have the zero-mask problem"""
    print()
    print("=" * 70)
    print("CHECKING EXISTING PSEUDO-LABELS")
    print("=" * 70)
    print()
    
    root = Path(pseudo_root)
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        return False
    
    # Find some samples
    pt_files = list(root.rglob("*.pt"))[:3]
    npy_files = list(root.rglob("*_masks.npy"))[:3]
    
    if not pt_files and not npy_files:
        print(f"❌ No pseudo-label files found")
        return False
    
    has_zeros = False
    
    # Check .pt files
    for pt_file in pt_files:
        data = torch.load(pt_file)
        if "masks" in data:
            masks = data["masks"]
            if masks.max() == 0:
                print(f"❌ {pt_file.name}: masks are ALL ZEROS")
                has_zeros = True
            else:
                print(f"✅ {pt_file.name}: masks OK (max={masks.max():.4f})")
    
    # Check .npy files
    if npy_files:
        import numpy as np
        for npy_file in npy_files:
            masks = np.load(npy_file)
            if masks.max() == 0:
                print(f"❌ {npy_file.name}: masks are ALL ZEROS")
                has_zeros = True
            else:
                print(f"✅ {npy_file.name}: masks OK (max={masks.max():.4f})")
    
    print()
    if has_zeros:
        print("=" * 70)
        print("❌ FOUND ZERO-MASK PROBLEM!")
        print("=" * 70)
        print()
        print("Solution: RE-EXPORT pseudo-labels with working teacher model")
        print()
        print("Steps:")
        print("  1. Verify teacher model checkpoint is valid")
        print("  2. Test teacher model on a few images manually")
        print("  3. Check export script for bugs")
        print("  4. Re-run export with verbose logging")
        print("  5. Validate exported files with check_pseudo_labels.py")
        return False
    else:
        print("✅ Pseudo-labels look OK!")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, 
                       help="Path to teacher model checkpoint")
    parser.add_argument("--sample-image", type=str,
                       help="Path to a sample image for testing")
    parser.add_argument("--pseudo-root", type=str, default="pseudo_labels_v2/train",
                       help="Path to existing pseudo-labels to check")
    args = parser.parse_args()
    
    if args.teacher_ckpt and args.sample_image:
        result = validate_teacher_export(args.teacher_ckpt, args.sample_image)
        if result is False:
            sys.exit(1)
    
    if args.pseudo_root:
        result = check_existing_pseudo_labels(args.pseudo_root)
        if not result:
            sys.exit(1)
    
    print()
    print("For now, use this to check existing pseudo-labels:")
    print(f"  python3 {sys.argv[0]} --pseudo-root pseudo_labels_v2/train")
