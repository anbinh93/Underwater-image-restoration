#!/usr/bin/env python3
"""Verify all DDP and memory fixes are properly applied"""

import sys
from pathlib import Path

def check_file_contains(filepath: Path, patterns: list[str], name: str) -> bool:
    """Check if file contains all required patterns"""
    if not filepath.exists():
        print(f"❌ {name}: File not found: {filepath}")
        return False
    
    content = filepath.read_text()
    missing = []
    for pattern in patterns:
        if pattern not in content:
            missing.append(pattern)
    
    if missing:
        print(f"❌ {name}: Missing patterns:")
        for p in missing:
            print(f"     - {p}")
        return False
    
    print(f"✅ {name}: All checks passed")
    return True

def main():
    base = Path(__file__).parent
    all_ok = True
    
    print("Verifying DDP + Memory fixes...")
    print("=" * 60)
    
    # Check 1: WTB chunked attention
    all_ok &= check_file_contains(
        base / "underwater_ir/student/naf_unet_wfi/wtb.py",
        [
            "attn_chunk_size: int = 256",
            "self.attn_chunk_size = attn_chunk_size",
            "for i in range(0, h * w, self.attn_chunk_size)",
        ],
        "WTB chunked attention"
    )
    
    # Check 2: Model propagates attn_chunk_size
    all_ok &= check_file_contains(
        base / "underwater_ir/student/naf_unet_wfi/model.py",
        [
            "attn_chunk_size: int = 256",
            "attn_chunk_size=attn_chunk_size",
        ],
        "Model attn_chunk_size"
    )
    
    # Check 3: Train script DDP support
    all_ok &= check_file_contains(
        base / "underwater_ir/student/train_student.py",
        [
            "import torch.distributed as dist",
            "from torch.nn.parallel import DistributedDataParallel as DDP",
            "def setup_ddp(",
            "ensure_device_consistency",
            "--attn-chunk-size",
        ],
        "Training script DDP"
    )
    
    # Check 4: Device consistency helper
    all_ok &= check_file_contains(
        base / "underwater_ir/student/train_student.py",
        [
            "def ensure_device_consistency",
            "ensure_device_consistency(p, device)",
        ],
        "Device consistency"
    )
    
    # Check 5: Datasets sampler support
    all_ok &= check_file_contains(
        base / "underwater_ir/data/datasets.py",
        [
            "sampler = None",
            "sampler=sampler",
        ],
        "Dataset sampler"
    )
    
    # Check 6: DDP wrapper script
    all_ok &= check_file_contains(
        base / "train_ddp.sh",
        [
            "torchrun",
            "--attn-chunk-size",
            "--ddp",
        ],
        "DDP wrapper script"
    )
    
    print("=" * 60)
    if all_ok:
        print("✅ All fixes verified! Ready for DDP training.")
        print("")
        print("Start training with:")
        print("  bash train_ddp.sh 4 2 20 128 128")
        return 0
    else:
        print("❌ Some fixes are missing. Run: git pull origin main")
        return 1

if __name__ == "__main__":
    sys.exit(main())
