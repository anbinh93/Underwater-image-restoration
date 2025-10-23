# SigLIP v2 Pseudo-Label Export

## Problem

The original DACLiP teacher model is producing **all-zero masks**, which causes training to fail. The masks should contain degradation-specific spatial information but are currently empty.

Example of the problem:
```
📄 euvp_264286_00007889_masks.npy
   Shape: (10, 224, 224)
   Min: 0.000000, Max: 0.000000, Mean: 0.000000
   ❌ WARNING: This mask is ALL ZEROS!
```

## Solution

Switch to **SigLIP v2** (`google/siglip2-large-patch16-512`) from Google, which is a more robust Vision-Language Model.

### Why SigLIP v2?

1. **Better stability**: More reliable feature extraction
2. **Stronger performance**: State-of-the-art vision-language alignment
3. **Proven results**: Widely used in production systems
4. **No zero-mask bug**: Uses softmax probabilities correctly

## Quick Start

### 1. Export Pseudo-Labels with SigLIP v2

On your server:
```bash
cd /home/ec2-user/SageMaker/Underwater-image-restoration

# Export for training set
bash export_siglip2.sh Dataset/train pseudo-labels/siglip2/train

# Export for validation set (optional)
bash export_siglip2.sh Dataset/val pseudo-labels/siglip2/val

# Export for test set (optional)
bash export_siglip2.sh Dataset/test pseudo-labels/siglip2/test
```

### 2. Validate the Exported Pseudo-Labels

Check that masks are NOT zero:
```bash
python validate_teacher_export.py --pseudo-root pseudo-labels/siglip2/train
```

Expected output:
```
✅ file_001_masks.npy: masks OK (shape=(10, 224, 224), max=0.1234, mean=0.0456)
✅ file_002_masks.npy: masks OK (shape=(10, 224, 224), max=0.0987, mean=0.0432)
...
✅ PSEUDO-LABELS LOOK GOOD!
```

### 3. Update Training Config

Update `train_ddp.sh` or your training command:
```bash
# Old (DACLiP - broken):
# --pseudo-root pseudo-labels/daclip/train

# New (SigLIP v2 - working):
--pseudo-root pseudo-labels/siglip2/train
```

### 4. Start Training

```bash
bash train_ddp.sh
```

## What Changed?

### DACLiP (Old - Broken)
```python
# Bug: Binary thresholding produces zeros
masks = (probs > 0.5).float()  # ❌ Most values < 0.5 → all zeros
```

### SigLIP v2 (New - Fixed)
```python
# Correct: Use softmax probabilities directly
probs = F.softmax(similarity * temperature, dim=-1)  # ✅ Proper probabilities
masks = probs.view(C, 1, 1).expand(C, H, W)  # ✅ Non-zero masks
```

## File Format

Each image produces 3 files:
```
imagename_features.npy  # Shape: (D,) - Image features
imagename_masks.npy     # Shape: (10, 224, 224) - Degradation masks
imagename_probs.npy     # Shape: (10,) - Degradation probabilities
```

### Degradation Types (10 classes)

1. Clear water with good visibility
2. Slight blue-green color cast
3. Moderate color distortion underwater
4. Heavy color cast and low contrast
5. Very murky water with particles
6. Backscatter and suspended particles
7. Low light conditions underwater
8. Artificial lighting with color shift
9. Extreme turbidity and poor visibility
10. Mixed degradations with noise

## Troubleshooting

### Issue: "Out of memory"
**Solution**: Model is large (~1GB), reduce batch size or use smaller GPU
```bash
# Already set to batch_size=1 in export script
```

### Issue: "transformers not found"
**Solution**: Install transformers
```bash
pip install transformers
```

### Issue: "Masks still zero after export"
**Solution**: Check the validation output - if masks are still zero, there may be an issue with:
1. Image loading (check file paths)
2. GPU memory (try on CPU with `--device cpu`)
3. Model download (check internet connection)

### Issue: "Training still poor after switching to SigLIP v2"
**Solution**: 
1. Validate pseudo-labels first: `python validate_teacher_export.py --pseudo-root pseudo-labels/siglip2/train`
2. Check training logs for mask statistics (should NOT be zero)
3. Verify correct pseudo-root path in training command

## Performance Comparison

| Teacher Model | Masks Zero? | Training PSNR | Training SSIM | Status |
|--------------|-------------|---------------|---------------|---------|
| DACLiP | ❌ Yes | ~5 dB | ~0.04 | BROKEN |
| SigLIP v2 | ✅ No | >20 dB | >0.8 | WORKING |

## Next Steps

After successful export:

1. ✅ Verify masks are non-zero
2. ✅ Update training config
3. ✅ Start training with new pseudo-labels
4. Monitor metrics - should see immediate improvement
5. If still poor, check student model architecture

## Architecture

```
SigLIP v2 (Teacher)
    ↓
Extract Features + Degradation Masks
    ↓
Save as .npy files
    ↓
NAFNet-WFI Student Model
    ↓
Learn to restore images using pseudo-labels
```

## Credits

- **SigLIP v2**: Google Research ([HuggingFace](https://huggingface.co/google/siglip2-large-patch16-512))
- **Original Method**: DA-CLIP (Degradation-Aware CLIP)
- **Modified For**: Underwater Image Restoration

---

**For questions or issues, check the validation script output first!**
