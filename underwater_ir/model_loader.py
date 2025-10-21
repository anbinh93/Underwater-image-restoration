#!/usr/bin/env python3
"""
CLIP Model Loader
Provides a unified interface to load CLIP models from different sources
- HuggingFace Transformers (recommended)
- Custom OpenCLIP (legacy DACLiP)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml

import torch
import torch.nn as nn


class CLIPModelLoader:
    """
    Unified CLIP model loader supporting multiple backends
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        """
        Initialize CLIP model loader
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration (alternative to config_path)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        self.clip_config = self.config.get('clip', {})
        self.model_type = self.clip_config.get('model_type', 'openai_clip')
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Determine the device to use"""
        device_config = self.clip_config.get('device', 'auto')
        
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_config)
    
    def load_model(self) -> Tuple[nn.Module, object, object]:
        """
        Load CLIP model based on configuration
        
        Returns:
            Tuple of (model, processor/transform, tokenizer)
        """
        if self.model_type == 'openai_clip':
            return self._load_hf_clip()
        elif self.model_type == 'daclip_custom':
            return self._load_daclip()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _load_hf_clip(self) -> Tuple[nn.Module, object, object]:
        """
        Load CLIP model from HuggingFace Transformers
        
        Returns:
            Tuple of (model, processor, tokenizer)
        """
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers library not found. Install it with: pip install transformers"
            ) from e
        
        hf_config = self.clip_config.get('hf_model', {})
        model_name = hf_config.get('model_name', 'openai/clip-vit-base-patch32')
        cache_dir = hf_config.get('cache_dir', './cache/clip_models')
        use_fp16 = self.clip_config.get('use_fp16', True)
        
        print(f"Loading CLIP model from HuggingFace: {model_name}")
        print(f"Device: {self.device}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load processor and tokenizer
        processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Load model
        model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Convert to fp16 if requested
        if use_fp16 and self.device.type == 'cuda':
            model = model.half()
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ CLIP model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {self.device}")
        print(f"   FP16: {use_fp16 and self.device.type == 'cuda'}")
        
        return model, processor, tokenizer
    
    def _load_daclip(self) -> Tuple[nn.Module, object, object]:
        """
        Load custom DACLiP model (legacy)
        
        Returns:
            Tuple of (model, transform, tokenizer)
        """
        daclip_config = self.clip_config.get('daclip_model', {})
        checkpoint_path = daclip_config.get('checkpoint_path', 'pretrained/daclip_ViT-B-32.pt')
        legacy_path = daclip_config.get('legacy_path', 'legacy/third_party/universal-image-restoration')
        
        # Add legacy path to sys.path
        root = Path(__file__).resolve().parent.parent
        legacy_full_path = root / legacy_path
        
        if not legacy_full_path.exists():
            raise FileNotFoundError(
                f"Legacy DACLiP path not found: {legacy_full_path}\n"
                "Please ensure the legacy code is available or use openai_clip model_type"
            )
        
        sys.path.insert(0, str(legacy_full_path))
        
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip dependencies missing (ftfy, sentencepiece). "
                "Install them with: pip install ftfy sentencepiece"
            ) from e
        
        print(f"Loading DACLiP model: {checkpoint_path}")
        
        # Check if checkpoint exists
        checkpoint_full_path = root / checkpoint_path
        if not checkpoint_full_path.exists():
            raise FileNotFoundError(
                f"DACLiP checkpoint not found: {checkpoint_full_path}\n"
                "Please download the checkpoint as described in the README"
            )
        
        # Load model
        model_name = daclip_config.get('model_name', 'daclip_ViT-B-32')
        model, preprocess = open_clip.create_model_from_pretrained(
            model_name,
            pretrained=str(checkpoint_full_path)
        )
        
        # Get tokenizer
        tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ DACLiP model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {self.device}")
        
        return model, preprocess, tokenizer


def load_clip_model(config_path: Optional[str] = None, 
                    config_dict: Optional[dict] = None) -> Tuple[nn.Module, object, object]:
    """
    Convenience function to load CLIP model
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Dictionary with configuration
        
    Returns:
        Tuple of (model, processor/transform, tokenizer)
        
    Example:
        >>> model, processor, tokenizer = load_clip_model(config_path='configs/config_clip.yaml')
    """
    loader = CLIPModelLoader(config_path=config_path, config_dict=config_dict)
    return loader.load_model()


if __name__ == "__main__":
    """Test the loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CLIP model loader')
    parser.add_argument('--config', type=str, default='configs/config_clip.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing CLIP Model Loader")
    print("=" * 80)
    
    try:
        model, processor, tokenizer = load_clip_model(config_path=args.config)
        print("\n✅ Model loaded successfully!")
        
        # Test tokenizer
        test_texts = ["a photo of underwater scene", "clear water"]
        if hasattr(tokenizer, '__call__'):
            tokens = tokenizer(test_texts)
            print(f"\n✅ Tokenizer test passed! Token shape: {tokens.shape if hasattr(tokens, 'shape') else 'variable'}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
