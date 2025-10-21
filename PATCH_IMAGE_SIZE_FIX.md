# Patch: Fixed Image Size Mismatch in DataLoader

## Problem
Training crashed with a DataLoader batching error:
```
RuntimeError: stack expects each tensor to be equal size, but got [3, 256, 256] at entry 0 and [3, 670, 1080] at entry 1
```

This occurred because images in the dataset have different resolutions, and the DataLoader cannot batch tensors of different sizes.

## Root Cause
The `default_transform()` function in `underwater_ir/data/datasets.py` only converted images to tensors without resizing them:
```python
def default_transform() -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose([T.ToTensor()])
```

## Solution
Modified the data loading pipeline to resize all images to a consistent size before batching.

### Changes Made

#### 1. Updated `underwater_ir/data/datasets.py`

**Added torch import:**
```python
import torch
```

**Updated `default_transform()` function:**
```python
def default_transform(img_size: int = 256) -> Callable[[Image.Image], torch.Tensor]:
    """Default transform: resize to square and convert to tensor"""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
```

**Updated loader creation functions to support `img_size` parameter:**
- `create_paired_train_loader(..., img_size: int = 256)`
- `create_paired_eval_loader(..., img_size: int = 256)`
- `create_unpaired_eval_loader(..., img_size: int = 256)`

All three functions now:
1. Accept an `img_size` parameter (default: 256)
2. Create a transform with `default_transform(img_size=img_size)`
3. Pass the transform to the dataset constructor

#### 2. Updated `underwater_ir/student/train_student.py`

**Added command-line argument:**
```python
parser.add_argument("--img-size", type=int, default=256, 
                   help="Input image size (all images will be resized to this)")
```

**Updated helper functions:**
- `build_reference_eval_entries(..., img_size: int = 256)`
- `build_nonref_eval_entries(..., img_size: int = 256)`

**Updated loader creation calls:**
```python
train_loader = create_paired_train_loader(
    args.train_root, 
    batch_size=args.batch_size, 
    num_workers=args.num_workers, 
    img_size=args.img_size
)

ref_entries = build_reference_eval_entries(
    Path(args.val_ref_root), 
    pseudo_val_ref_root, 
    batch_size=args.eval_batch_size, 
    num_workers=args.num_workers, 
    img_size=args.img_size
)

nonref_entries = build_nonref_eval_entries(
    Path(args.val_nonref_root), 
    pseudo_val_nonref_root, 
    batch_size=args.eval_batch_size, 
    num_workers=args.num_workers, 
    img_size=args.img_size
)
```

## Usage

### Default (256x256):
```bash
python -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --batch-size 4
```

### Custom image size (512x512):
```bash
python -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --batch-size 4 \
  --img-size 512
```

### Via run_clip_training.sh:
The shell script automatically uses default 256x256. To change it, modify the script:
```bash
python -m underwater_ir.student.train_student \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --num-workers "${WORKERS}" \
  --img-size 512 \  # Add this line
  --save-path "${SAVE_PATH}"
```

## Technical Details

### Resize Strategy
- **Mode**: Bilinear interpolation (smooth resizing)
- **Shape**: Square images (img_size × img_size)
- **Aspect ratio**: Not preserved (images are stretched/squeezed to square)

### Memory Considerations
| Image Size | Memory/Image (FP32) | Batch Size 4 | Batch Size 8 |
|------------|---------------------|--------------|--------------|
| 256×256    | ~0.75 MB           | ~3 MB        | ~6 MB        |
| 512×512    | ~3 MB              | ~12 MB       | ~24 MB       |
| 1024×1024  | ~12 MB             | ~48 MB       | ~96 MB       |

### Performance Impact
- **Smaller sizes (128-256)**: Faster training, lower memory, may lose detail
- **Medium sizes (256-512)**: Balanced performance and quality
- **Larger sizes (512-1024)**: Better quality, slower training, higher memory

## Recommendation
- **For experimentation**: Use 256×256 (default)
- **For production**: Use 512×512 or higher if GPU memory allows
- **For low-memory systems**: Use 128×128 or 192×192

## Testing
Run the training pipeline on the server:
```bash
cd /home/ec2-user/SageMaker/daclip-uir
bash run_clip_training.sh
```

The training should now successfully batch images without size mismatch errors! 🎉
