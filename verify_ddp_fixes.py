#!/usr/bin/env python3
"""
Quick verification script to check if all DDP fixes are applied.
Run this on the server before training.
"""

import sys
from pathlib import Path

def check_file_contains(filepath: Path, search_strings: list[str], description: str) -> bool:
    """Check if file contains all search strings"""
    if not filepath.exists():
        print(f"❌ {description}: File not found - {filepath}")
        return False
    
    content = filepath.read_text()
    all_found = True
    
    for search_str in search_strings:
        if search_str in content:
            print(f"   ✅ Found: {search_str[:60]}...")
        else:
            print(f"   ❌ Missing: {search_str[:60]}...")
            all_found = False
    
    return all_found

def main():
    print("=" * 70)
    print("DDP Training - Fix Verification")
    print("=" * 70)
    print()
    
    base_path = Path("underwater_ir/student")
    all_ok = True
    
    # Check 1: Loss modules moved to device
    print("1️⃣  Checking: Loss modules moved to device...")
    train_file = base_path / "train_student.py"
    if check_file_contains(
        train_file,
        [
            "perceptual_loss = PerceptualLoss(weight=perc_weight).to(device)",
            "frequency_loss = FrequencyLoss(hf_weight, lf_weight).to(device)",
            "All loss modules moved to device",
        ],
        "Loss device placement"
    ):
        print("   ✅ PASS: Loss modules properly moved to device\n")
    else:
        print("   ❌ FAIL: Loss modules not moved to device\n")
        all_ok = False
    
    # Check 2: Masks resized to match output
    print("2️⃣  Checking: Masks resized to match output dimensions...")
    if check_file_contains(
        train_file,
        [
            "if masks.shape[-2:] != output.shape[-2:]:",
            "masks_resized = F.interpolate(masks, size=output.shape[-2:], mode='nearest')",
        ],
        "Mask resizing"
    ):
        print("   ✅ PASS: Masks resized correctly\n")
    else:
        print("   ❌ FAIL: Mask resizing not implemented\n")
        all_ok = False
    
    # Check 3: Frequency loss with wavelet resizing
    print("3️⃣  Checking: FrequencyLoss handles wavelet dimensions...")
    freq_file = base_path / "losses/freq_losses.py"
    if check_file_contains(
        freq_file,
        [
            "if hf_mask is not None and hf_mask.shape[-2:] != pred_lh.shape[-2:]:",
            "hf_mask = torch.nn.functional.interpolate",
        ],
        "Wavelet mask resizing"
    ):
        print("   ✅ PASS: FrequencyLoss handles wavelet dimensions\n")
    else:
        print("   ❌ FAIL: Wavelet dimension handling missing\n")
        all_ok = False
    
    # Check 4: Region loss with channel splitting
    print("4️⃣  Checking: RegionLoss gets channel-split masks...")
    if check_file_contains(
        train_file,
        [
            "mask_list = [masks_resized[:, i:i+1, :, :] for i in range(masks_resized.shape[1])]",
            "region_term = region_loss(output, gt, masks=mask_list)",
        ],
        "Region loss mask splitting"
    ):
        print("   ✅ PASS: Masks split by channel for RegionLoss\n")
    else:
        print("   ❌ FAIL: Mask channel splitting not implemented\n")
        all_ok = False
    
    # Check 5: Feature alignment with resizing
    print("5️⃣  Checking: Feature alignment handles dimension mismatch...")
    if check_file_contains(
        train_file,
        [
            "teacher_feat = masks_resized.mean(dim=1, keepdim=True)",
            "if student_feat.shape[-2:] != teacher_feat.shape[-2:]:",
            "student_feat = F.interpolate",
        ],
        "Feature alignment resizing"
    ):
        print("   ✅ PASS: Feature alignment handles dimension mismatch\n")
    else:
        print("   ❌ FAIL: Feature alignment resizing missing\n")
        all_ok = False
    
    # Final summary
    print("=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready to train!")
        print("=" * 70)
        print()
        print("Start training with:")
        print("  bash train_ddp.sh 8 2 20 128 128")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Need to git pull!")
        print("=" * 70)
        print()
        print("Run these commands:")
        print("  git fetch origin")
        print("  git reset --hard origin/main")
        print("  python3 verify_ddp_fixes.py")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
