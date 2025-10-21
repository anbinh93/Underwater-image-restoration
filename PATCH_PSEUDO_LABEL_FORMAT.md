# Patch: Pseudo-Label Format Compatibility (V1 .pt ↔ V2 .npy)

## Problem
The training script `train_student.py` expected pseudo-labels in `.pt` format (PyTorch saved dicts), but the server's export script generated `.npy` files (NumPy arrays) in V2 format:
- `{filename}_features.npy` - Degradation code (z_d)
- `{filename}_masks.npy` - Spatial attention masks
- `{filename}_probs.npy` - Global degradation probabilities

This caused training to fail with: `RuntimeError: No pseudo-label files found`

## Solution
Modified `underwater_ir/student/train_student.py` to support both formats:

### 1. Updated `load_pseudo_label()` function (lines 55-95)
**Before:**
```python
def load_pseudo_label(root: Path, rel_path: str) -> Dict[str, torch.Tensor]:
    candidate = (root / Path(rel_path)).with_suffix(".pt")
    if not candidate.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {candidate}")
    data = torch.load(candidate)
    return data
```

**After:**
```python
def load_pseudo_label(root: Path, rel_path: str) -> Dict[str, torch.Tensor]:
    """Load pseudo-label from .pt or .npy format (V2 compatibility)"""
    base_path = root / Path(rel_path)
    
    # Try .pt first (original format)
    pt_path = base_path.with_suffix(".pt")
    if pt_path.exists():
        data = torch.load(pt_path)
        return data
    
    # Try .npy format (V2 export format)
    stem = base_path.stem
    parent = base_path.parent
    
    features_path = parent / f"{stem}_features.npy"
    masks_path = parent / f"{stem}_masks.npy"
    probs_path = parent / f"{stem}_probs.npy"
    
    if features_path.exists() and masks_path.exists() and probs_path.exists():
        # Load numpy arrays and convert to tensors
        features = torch.from_numpy(np.load(features_path))
        masks = torch.from_numpy(np.load(masks_path))
        probs = torch.from_numpy(np.load(probs_path))
        
        # Reconstruct pseudo-label dict in expected format
        data = {
            "z_d": features,
            "masks": masks,
            "global_prob": probs,
            "confidence": torch.ones_like(probs) * 0.8,  # Default confidence
            "confidence_scale": torch.tensor(0.2),  # Default scale
        }
        return data
    
    # If neither format found, raise error
    raise FileNotFoundError(
        f"Pseudo-label not found for {rel_path}. "
        f"Tried: {pt_path}, {features_path}, {masks_path}, {probs_path}"
    )
```

### 2. Updated startup validation (lines 241-278)
**Before:**
```python
pseudo_example_path = next(pseudo_train_root.rglob("*.pt"), None)
if pseudo_example_path is None:
    raise RuntimeError(f"No pseudo-label files found in {pseudo_train_root}")
pseudo_sample = torch.load(pseudo_example_path)
```

**After:**
```python
# Check for .pt files first, then .npy files (V2 format)
pseudo_example_path = next(pseudo_train_root.rglob("*.pt"), None)
if pseudo_example_path is None:
    # Try V2 format: look for *_features.npy files
    pseudo_example_path = next(pseudo_train_root.rglob("*_features.npy"), None)
    if pseudo_example_path is None:
        raise RuntimeError(
            f"No pseudo-label files found in {pseudo_train_root}. "
            f"Expected either *.pt or *_features.npy files."
        )
    
    # Load V2 format
    stem = pseudo_example_path.stem.replace("_features", "")
    parent = pseudo_example_path.parent
    
    masks_path = parent / f"{stem}_masks.npy"
    probs_path = parent / f"{stem}_probs.npy"
    
    if not masks_path.exists() or not probs_path.exists():
        raise RuntimeError(
            f"Incomplete V2 pseudo-label set for {stem}. "
            f"Need: *_features.npy, *_masks.npy, *_probs.npy"
        )
    
    features = torch.from_numpy(np.load(pseudo_example_path))
    masks = torch.from_numpy(np.load(masks_path))
    probs = torch.from_numpy(np.load(probs_path))
    
    pseudo_sample = {
        "z_d": features,
        "masks": masks,
        "global_prob": probs,
    }
else:
    # Load original .pt format
    pseudo_sample = torch.load(pseudo_example_path)
```

### 3. Added NumPy import (line 11)
```python
import numpy as np
```

## Key Features
- **Backward compatible**: Still supports original `.pt` format
- **Forward compatible**: Handles new `.npy` V2 format from server
- **Graceful fallback**: Tries `.pt` first, then `.npy`
- **Clear error messages**: Tells user exactly which files were tried and what's missing
- **Format detection**: Automatically detects format based on file existence

## Testing
To test on server, run:
```bash
cd /home/ec2-user/SageMaker/daclip-uir
bash run_clip_training.sh
```

Expected behavior:
1. Stage 1 export creates `*_features.npy`, `*_masks.npy`, `*_probs.npy` files
2. Stage 2 training successfully loads these files and starts training
3. No more "No pseudo-label files found" error

## V2 Format Specification
Each sample consists of 3 NumPy files:
- **{filename}_features.npy**: Shape `[D]` - Degradation code from CLIP encoder
- **{filename}_masks.npy**: Shape `[M, H, W]` - M spatial attention masks
- **{filename}_probs.npy**: Shape `[N]` - N-class degradation probabilities

Where:
- `D` = degradation code dimension (e.g., 512 for CLIP)
- `M` = number of mask layers (e.g., 4)
- `N` = number of degradation categories (e.g., 11)
- `H, W` = mask spatial dimensions (e.g., 64x64)

## Migration Path
For users with old `.pt` files:
- No action needed - code still supports `.pt` format
- Can continue using existing pseudo-labels

For users generating new pseudo-labels:
- Use latest export script from server
- Generates `.npy` files automatically
- Training script now handles both formats seamlessly
