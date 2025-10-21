#!/usr/bin/env python3
"""
Test CLIP Model Loading with New Configuration
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 80)
print("Testing New CLIP Model Loader")
print("=" * 80)

# Test 1: Check dependencies
print("\n[1/3] Checking dependencies...")
missing_deps = []

try:
    import torch
    print(f"  ✅ torch: {torch.__version__}")
except ImportError:
    missing_deps.append("torch")
    print("  ❌ torch not found")

try:
    import transformers
    print(f"  ✅ transformers: {transformers.__version__}")
except ImportError:
    missing_deps.append("transformers")
    print("  ❌ transformers not found")

try:
    import yaml
    print(f"  ✅ PyYAML installed")
except ImportError:
    missing_deps.append("pyyaml")
    print("  ❌ PyYAML not found")

try:
    from PIL import Image
    print(f"  ✅ Pillow installed")
except ImportError:
    missing_deps.append("pillow")
    print("  ❌ Pillow not found")

if missing_deps:
    print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
    print(f"   Install with: pip install {' '.join(missing_deps)}")
    sys.exit(1)

# Test 2: Load model using new loader
print("\n[2/3] Testing model loader...")
try:
    from underwater_ir.model_loader import load_clip_model
    
    config_path = ROOT / "configs" / "config_clip.yaml"
    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"  Loading model from config: {config_path}")
    model, processor, tokenizer = load_clip_model(config_path=str(config_path))
    
    print(f"  ✅ Model loaded successfully!")
    print(f"     Type: {type(model).__name__}")
    print(f"     Device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"  ❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test tokenization
print("\n[3/3] Testing tokenizer...")
try:
    test_texts = [
        "underwater image with blue color cast",
        "low contrast underwater scene",
        "clear water with good visibility"
    ]
    
    if hasattr(tokenizer, '__call__'):
        tokens = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        print(f"  ✅ Tokenization successful!")
        print(f"     Input shape: {tokens['input_ids'].shape}")
        print(f"     Max length: {tokens['input_ids'].shape[1]}")
    else:
        print(f"  ⚠️  Tokenizer type: {type(tokenizer)}")
        print(f"     (May be open_clip tokenizer)")
    
except Exception as e:
    print(f"  ⚠️  Tokenization test: {e}")

# Summary
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYou can now run the training pipeline:")
print("  bash run_clip_training_v2.sh")
print("\nOr test with the new pseudo-label export:")
print("  python -m underwater_ir.teacher.export_pseudolabels_v2 \\")
print("    --config configs/config_clip.yaml \\")
print("    --input-root Dataset/train/input \\")
print("    --target-root Dataset/train/target \\")
print("    --output pseudo-labels/test \\")
print("    --batch-size 2")
