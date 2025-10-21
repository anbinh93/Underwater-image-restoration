#!/usr/bin/env python3
"""
Quick test to verify CLIP model can be loaded from Hugging Face
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy" / "third_party" / "universal-image-restoration"))

print("=" * 80)
print("Testing CLIP Model Loading from Hugging Face")
print("=" * 80)

# Test import
print("\n[1/3] Importing open_clip...")
try:
    import open_clip
    print(f"✅ open_clip imported from: {open_clip.__file__}")
except Exception as e:
    print(f"❌ Failed to import open_clip: {e}")
    sys.exit(1)

# Test list available models
print("\n[2/3] Listing available models...")
try:
    available = open_clip.list_pretrained()
    print(f"✅ Found {len(available)} available model configurations")
    
    # Check for HF models
    hf_models = [m for m in available if 'hf-hub' in str(m)]
    if hf_models:
        print(f"   HF Hub models available: {len(hf_models)}")
        for model in hf_models[:3]:
            print(f"     - {model}")
    else:
        print("   ⚠️ No HF hub models listed (but they should still work)")
except Exception as e:
    print(f"⚠️ Could not list models: {e}")

# Test loading OpenAI's official CLIP model
print("\n[3/3] Testing model load: ViT-B-32 with openai checkpoint")
try:
    import torch
    
    model_name = "ViT-B-32"
    checkpoint = "openai"
    print(f"   Loading {model_name} with {checkpoint} checkpoint...")
    
    with torch.no_grad():
        model, preprocess = open_clip.create_model_from_pretrained(
            model_name, pretrained=checkpoint
        )
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Has visual encoder: {hasattr(model, 'visual')}")
    print(f"   Has text encoder: {hasattr(model, 'encode_text')}")
    
    if hasattr(model, 'visual') and hasattr(model.visual, 'grid_size'):
        print(f"   Visual grid size: {model.visual.grid_size}")
    
    # Test tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)
    test_text = ["a photo of underwater scene", "clear water"]
    tokens = tokenizer(test_text)
    print(f"✅ Tokenizer works! Shape: {tokens.shape}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print(f"\nThe model '{model_name}' with '{checkpoint}' checkpoint loaded successfully!")
print("This is OpenAI's official CLIP ViT-B/32 model.")
print("\nYou can now run: bash run_clip_training.sh")
