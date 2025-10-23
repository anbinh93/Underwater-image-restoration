#!/usr/bin/env python3
"""
Quick diagnostic to verify all loss modules are properly moved to device.
Run this after git pull to ensure the fix is applied.
"""

import sys
import re
from pathlib import Path

def check_loss_device_fix():
    """Check if loss modules have .to(device) calls."""
    train_file = Path("underwater_ir/student/train_student.py")
    
    if not train_file.exists():
        print(f"❌ File not found: {train_file}")
        print("   Make sure you're in the repository root directory")
        return False
    
    content = train_file.read_text()
    
    # Check for loss module definitions with .to(device)
    loss_modules = [
        "frequency_loss",
        "region_loss", 
        "tv_loss",
        "perceptual_loss",
        "kd_loss",
        "feat_align_loss",
        "contrastive_loss"
    ]
    
    print("🔍 Checking loss module device placement...")
    print()
    
    all_good = True
    for loss_name in loss_modules:
        # Look for pattern: loss_name = SomeLoss(...).to(device)
        pattern = rf"{loss_name}\s*=\s*\w+\([^)]*\)\.to\(device\)"
        if re.search(pattern, content):
            print(f"   ✅ {loss_name}: properly moved to device")
        else:
            print(f"   ❌ {loss_name}: NOT moved to device (will cause DDP errors)")
            all_good = False
    
    print()
    
    # Check for the confirmation message
    if "All loss modules moved to device" in content:
        print("✅ Confirmation message present")
    else:
        print("⚠️  Confirmation message not found (minor issue)")
    
    print()
    
    if all_good:
        print("=" * 60)
        print("✅ SUCCESS: All loss modules are properly configured!")
        print("=" * 60)
        print()
        print("You can now run training with:")
        print("  bash train_ddp.sh 8 2 20 128 128")
        return True
    else:
        print("=" * 60)
        print("❌ ISSUE DETECTED: Some loss modules not moved to device")
        print("=" * 60)
        print()
        print("This will cause: 'Expected all tensors to be on the same device'")
        print()
        print("Solution:")
        print("  1. Run: git pull origin main")
        print("  2. Run this script again to verify")
        return False

if __name__ == "__main__":
    success = check_loss_device_fix()
    sys.exit(0 if success else 1)
