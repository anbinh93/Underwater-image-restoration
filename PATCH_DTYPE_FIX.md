# Patch: Fixed dtype Mismatch (float16 → float32)

## Problem
Training crashed with a dtype mismatch error:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
```

This occurred in the `AdaLayerNorm2d` module when trying to apply the linear transformation to the conditioning vector (`z_d`).

## Root Cause
When pseudo-labels are exported to `.npy` files, NumPy preserves the original dtype (likely float16 from the CLIP model for memory efficiency). When these `.npy` files are loaded with `torch.from_numpy()`, the tensors retain the float16 dtype.

However, the student model (NAFNet-WFIGate) is initialized in float32 by default. This creates a dtype mismatch:
- **Conditioning tensors** (`z_d`, `masks`, `probs`): float16 (from .npy files)
- **Model weights** (Linear layers in AdaLayerNorm2d): float32

## Solution
Modified the `load_pseudo_label()` function to explicitly convert all tensors to float32 after loading from NumPy.

### Changes Made

#### Updated `underwater_ir/student/train_student.py`

**1. In `load_pseudo_label()` function (lines 75-80):**
```python
# Before:
features = torch.from_numpy(np.load(features_path))
masks = torch.from_numpy(np.load(masks_path))
probs = torch.from_numpy(np.load(probs_path))

# After:
features = torch.from_numpy(np.load(features_path)).float()
masks = torch.from_numpy(np.load(masks_path)).float()
probs = torch.from_numpy(np.load(probs_path)).float()
```

**2. In initialization code (lines 267-269):**
```python
# Before:
features = torch.from_numpy(np.load(pseudo_example_path))
masks = torch.from_numpy(np.load(masks_path))
probs = torch.from_numpy(np.load(probs_path))

# After:
features = torch.from_numpy(np.load(pseudo_example_path)).float()
masks = torch.from_numpy(np.load(masks_path)).float()
probs = torch.from_numpy(np.load(probs_path)).float()
```

#### Updated `run_clip_training.sh`

**1. Post-export verification (lines 82-109):**
- Now checks for both `.pt` and `.npy` formats
- Provides informative messages about which format was found
- Shows sample files for both formats

**2. Pre-training verification (lines 163-180):**
- Updated to accept both `.pt` and `.npy` formats
- Clear messages about which format is being used
- Removed redundant `.pt`-only checks

## Technical Details

### dtype Conversion Chain
1. **Export**: CLIP model saves embeddings as float16 → `.npy` files
2. **Load**: `np.load()` reads float16 arrays
3. **Convert**: `torch.from_numpy()` creates float16 tensors
4. **Cast**: `.float()` converts to float32 tensors ← **This is the fix**
5. **Transfer**: `.to(device)` moves to GPU (preserving float32)

### Why float32?
- **Model default**: PyTorch models default to float32
- **Numerical stability**: float32 provides better precision for gradients
- **Compatibility**: All loss functions and optimizers expect float32
- **Memory trade-off**: Slightly more memory, but ensures training stability

### Alternative Solutions Considered
1. ❌ **Convert model to float16**: Would require mixed precision training setup
2. ❌ **Save .npy as float32**: Would require changing export script (breaks server)
3. ✅ **Cast on load**: Simple, backward compatible, clear intent

## Verification

The fix ensures:
- ✅ All pseudo-label tensors are float32 before training
- ✅ Model weights remain float32 (default)
- ✅ No dtype mismatch in matrix multiplication
- ✅ Compatible with both `.pt` (already float32) and `.npy` (needs cast) formats

## Testing

### On Server:
```bash
cd /home/ec2-user/SageMaker/daclip-uir
git pull  # Get the updated files
bash run_clip_training.sh
```

Expected behavior:
1. ✅ Export creates `.npy` files (float16 internally)
2. ✅ Training loads `.npy` files and casts to float32
3. ✅ No dtype mismatch errors
4. ✅ Training proceeds normally

### Verification Commands:
```python
# Check dtype of loaded pseudo-labels
import numpy as np
import torch

# Load .npy file
features = np.load("pseudo-labels/daclip/train/image_features.npy")
print(f"NumPy dtype: {features.dtype}")  # Expected: float16 or float32

# Convert to tensor
tensor = torch.from_numpy(features)
print(f"Tensor dtype (before): {tensor.dtype}")  # Expected: torch.float16 or torch.float32

# Apply fix
tensor = torch.from_numpy(features).float()
print(f"Tensor dtype (after): {tensor.dtype}")  # Expected: torch.float32 ✅
```

## Performance Impact
- **Memory**: ~2x increase for pseudo-labels in memory (float16 → float32)
- **Training speed**: Negligible (tensors are small compared to images)
- **Accuracy**: Improved numerical stability

## Backward Compatibility
- ✅ `.pt` files (already float32): No change
- ✅ `.npy` files (float16 or float32): Explicit cast to float32
- ✅ Mixed formats in same directory: Works correctly

## Related Files Modified
1. `underwater_ir/student/train_student.py` - Added `.float()` casts
2. `run_clip_training.sh` - Updated validation to support both formats
