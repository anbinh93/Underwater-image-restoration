# Troubleshooting: No Pseudo-Label Files Found

## Problem

```
RuntimeError: No pseudo-label files found in /path/to/pseudo-labels/daclip/train
```

## Diagnosis Steps

### 1. Run Debug Script

```bash
chmod +x debug_dataset.sh
bash debug_dataset.sh
```

This will show:
- Dataset structure and image counts
- Pseudo-label files status

### 2. Check Export Logs

Look for export success messages:
```
✅ Pseudo-labels saved to pseudo-labels/daclip/train
```

If you only see logs for `testset_nonref` but not `train`, the export failed silently.

### 3. Common Causes

#### A. Training Dataset Not Found

**Symptom**: No export logs for training set

**Solution**: Verify dataset structure
```bash
ls -la Dataset/train/input/
ls -la Dataset/train/target/
```

Required structure:
```
Dataset/train/
├── input/   # Low-quality images (.jpg, .png)
└── target/  # Ground-truth images (.jpg, .png)
```

#### B. Empty Dataset

**Symptom**: Export runs but creates no files

**Solution**: Check image counts
```bash
find Dataset/train/input -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
```

Should be > 0. If 0, add images to dataset.

#### C. Permission Issues

**Symptom**: Export fails without error

**Solution**: Check write permissions
```bash
mkdir -p pseudo-labels/daclip/train
touch pseudo-labels/daclip/train/test.txt
rm pseudo-labels/daclip/train/test.txt
```

#### D. Path Resolution Issues

**Symptom**: Files saved to wrong location

**Solution**: Use absolute paths
```bash
# In run_clip_training.sh, add:
TRAIN_ROOT="$(pwd)/Dataset/train"
PSEUDO_ROOT="$(pwd)/pseudo-labels/daclip"
```

### 4. Manual Export Test

Test export for a single image:

```bash
# Create test input
mkdir -p test_export/input
cp Dataset/train/input/*.jpg test_export/input/ | head -1

# Run export
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root test_export/input \
  --output test_export/output \
  --clip-model daclip_ViT-B-32 \
  --clip-checkpoint pretrained/daclip_ViT-B-32.pt \
  --num-workers 1

# Check output
find test_export/output -name "*.pt"
```

### 5. Check Python Import Errors

Export might fail silently due to import errors:

```bash
python -c "
from underwater_ir.teacher.export_pseudolabels import main
print('✅ Import successful')
"
```

## Solutions

### Solution 1: Verify Dataset Exists

```bash
# Check if dataset directories exist
if [[ ! -d "Dataset/train/input" ]]; then
  echo "❌ Dataset/train/input not found!"
  echo "Please create it and add training images."
  exit 1
fi
```

### Solution 2: Run Export Separately

Run export stages one at a time to isolate issues:

```bash
# Export training set only
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root Dataset/train/input \
  --target-root Dataset/train/target \
  --output pseudo-labels/daclip/train \
  --clip-model daclip_ViT-B-32 \
  --clip-checkpoint pretrained/daclip_ViT-B-32.pt \
  --num-workers 4

# Verify output
find pseudo-labels/daclip/train -name "*.pt" | wc -l
```

### Solution 3: Check File Paths

Pseudo-labels should be saved preserving relative paths:

```
Dataset/train/input/img001.jpg  
→ pseudo-labels/daclip/train/img001.pt

Dataset/train/input/subdir/img002.jpg
→ pseudo-labels/daclip/train/subdir/img002.pt
```

Use `find` to check structure:
```bash
find pseudo-labels/daclip/train -name "*.pt" -ls
```

### Solution 4: Increase Verbosity

Add debug prints to export_pseudolabels.py:

```python
# Around line 330
print(f"Saving to: {save_path}")
save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(sample, save_path)
print(f"✅ Saved: {save_path}")
```

## Prevention

Add validation to run_clip_training.sh:

```bash
# After each export stage
pt_count=$(find "${PSEUDO_ROOT}/train" -name "*.pt" | wc -l)
if [[ $pt_count -eq 0 ]]; then
  echo "❌ ERROR: Export failed - no .pt files created" >&2
  exit 1
fi
echo "✅ Exported ${pt_count} pseudo-labels"
```

## Still Not Working?

1. Check disk space: `df -h`
2. Check for error messages in full log
3. Run with single worker: `--num-workers 1`
4. Try with smaller dataset first (1-2 images)
5. Check CLIP model loads: `python test_clip_model.py`

## Contact

If issue persists, provide:
- Output of `bash debug_dataset.sh`
- Full export log (redirect to file: `... 2>&1 | tee export.log`)
- Python version: `python --version`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
