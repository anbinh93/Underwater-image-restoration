# Quick Fix Guide: NaN in Training# 🚨 HƯỚNG DẪN FIX NaN VÀ CÁC LỖI DDP



## Problem Summary## Tình trạng hiện tại

- ✅ Đã fix 5 lỗi dimension mismatch

After switching to SigLIP v2, masks are no longer zero ✅, but now getting **NaN in model parameters**:- ✅ Đã thêm NaN detection

- ✅ Đã fix SSIM loss numerical stability

```- ⚠️ Server cần pull code mới nhất

masks: min=0.5798, max=1.2519, mean=1.0000  ← Problem: max > 1.0!

NaN found in model parameters: ['module.stem.weight', 'module.final_norm.bias', ...]---

```

## Chạy trên Server (AWS EC2)

## Root Cause

### Bước 1: Stop toàn bộ training đang chạy

The SigLIP v2 export script was normalizing masks incorrectly:```bash

```pythonpkill -9 -f train_student

# OLD (BUGGY):pkill -9 -f torchrun

masks[c] = mask_c / (mask_c.sum() + 1e-8) * (H * W)  # Multiplies by 50176!nvidia-smi  # Xác nhận GPU đã free

# Result: masks can be >> 1.0```

```

### Bước 2: Update code từ GitHub

When masks > 1.0, they multiply with features causing overflow → NaN.```bash

cd /home/ec2-user/SageMaker/Underwater-image-restoration

## Solution Applied

# Xem code hiện tại đang ở commit nào

**1. Fixed export script** to use min-max normalization:git log --oneline -3

```python

# NEW (CORRECT):# Pull code mới nhất (có NaN fixes)

masks[c] = (mask_c - mask_c.min()) / (mask_c.max() - mask_c.min() + 1e-8)git fetch origin

masks = torch.clamp(masks, 0, 1)  # Ensure [0, 1]git reset --hard origin/main

```

# Verify đã pull thành công

**2. Added safety in training** (fallback normalization):git log --oneline -3

```python# Phải thấy commits: 30f7ac3, d67eabe, 395dd06

masks = masks / (masks.max() + 1e-8)```

masks = torch.clamp(masks, 0, 1)

```### Bước 3: Verify tất cả fixes đã được áp dụng

```bash

**3. Added gradient protection**:python3 verify_ddp_fixes.py

```python```

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if torch.isnan(grad_norm):**Expected output:**

    skip_batch()  # Don't update with bad gradients```

```✅ ALL CHECKS PASSED - Ready to train!

```

## Action Required

Nếu FAIL → chạy:

### On Server (AWS EC2)```bash

bash URGENT_FIX_SERVER.sh

```bash```

# 1. Pull latest fixes

cd /home/ec2-user/SageMaker/Underwater-image-restoration### Bước 4: Clear cache và khởi động training

git pull```bash

# Clear Python cache

# 2. RE-EXPORT pseudo-labels with fixed scriptfind . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# This is CRITICAL - old pseudo-labels still have masks > 1.0find . -name "*.pyc" -delete

rm -rf pseudo-labels/siglip2/train  # Remove old broken pseudo-labels

bash export_siglip2.sh Dataset/train pseudo-labels/siglip2/train# Start training với NaN detection

bash train_ddp.sh 8 2 20 128 128

# 3. Validate new pseudo-labels```

python validate_teacher_export.py --pseudo-root pseudo-labels/siglip2/train

---

# Expected output:

#   ✅ masks OK (shape=(10, 224, 224), max=1.0000, mean=0.456)## Các Fix đã áp dụng

#   ✅ All sample masks are valid!

### Fix 1-5: Dimension Mismatch (đã fix trước đó)

# 4. Resume training with new pseudo-labels- ✅ Loss modules → device

bash train_ddp.sh- ✅ Masks resizing 

```- ✅ Wavelet dimensions

- ✅ Region loss iteration

## Validation Checklist- ✅ Feature alignment



Before starting training, verify:### Fix 6: NaN Detection (MỚI)

**File:** `train_student.py`

- [ ] **Git pull successful** - latest code downloaded

- [ ] **Old pseudo-labels deleted** - no leftover masks > 1.0Kiểm tra NaN tại 3 điểm:

- [ ] **Export completed** - no errors during SigLIP v2 export1. **Input check:** LQ, GT, pseudo-labels

- [ ] **Masks validated** - all in [0, 1] range2. **Output check:** Model output

- [ ] **Training starts** - no immediate NaN3. **Loss check:** Total loss before backward



## Expected Behavior After FixNếu phát hiện NaN → skip batch và in warning.



### During Export:### Fix 7: SSIM Numerical Stability (MỚI)

```**File:** `train_student.py`, function `ssim_loss()`

✅ Successfully exported: 1000

✅ All sample masks are valid!```python

   file_001_masks.npy: Range: [0.000000, 1.000000], Mean: 0.456789 ✅ OK# Clamp variance to avoid negative values

   file_002_masks.npy: Range: [0.000000, 1.000000], Mean: 0.512345 ✅ OKsigma_x = sigma_x.clamp(min=0)

```sigma_y = sigma_y.clamp(min=0)



### During Training:# Increased epsilon

```ssim_map = numerator / (denominator + 1e-8)  # Was 1e-6

[DEBUG] Input statistics (after normalization):```

  masks: min=0.0000, max=1.0000, mean=0.4567  ← Should be ≤ 1.0!

  ---

Epoch 1: Loss=2.3456, PSNR=18.5, SSIM=0.65  ← Should improve!

Epoch 2: Loss=1.8934, PSNR=21.2, SSIM=0.74## Monitoring Training

...

```### Khi training chạy, check:



### What to Watch:1. **Không có WARNING về NaN:**

```

✅ **Good signs:**# Nếu thấy messages này → có vấn đề

- Masks max = 1.0 (not 1.25+)[Rank X] WARNING: NaN detected in input LQ!

- No NaN warnings[Rank X] WARNING: NaN in model output!

- Loss decreasing[Rank X] WARNING: NaN/Inf in total_loss!

- PSNR > 15 dB after epoch 1```



❌ **Bad signs (stop and debug):**2. **Loss giảm bình thường:**

- Masks max > 1.0 → Re-export needed```

- NaN in parameters → Check model architectureEpoch 1/20 - Train loss: 0.XXXX  # NOT nan

- PSNR < 10 dB → Something still wrong[Ref] UIEB: PSNR=XX.XX dB        # NOT nan

```

## Troubleshooting

3. **GPU memory stable:**

### Issue: "Masks still > 1.0 after export"```bash

**Solution:** Make sure you pulled latest code and deleted old pseudo-labelswatch -n 1 nvidia-smi

```bash# All 8 GPUs should show similar memory usage

git log --oneline -1  # Should show "fix: Normalize masks to prevent NaN"```

rm -rf pseudo-labels/siglip2/train

bash export_siglip2.sh Dataset/train pseudo-labels/siglip2/train---

```

## Troubleshooting

### Issue: "NaN still appears in training"

**Solution:** Check which layer is producing NaN### Vấn đề: Vẫn thấy dimension mismatch errors

```bash**Nguyên nhân:** Server chưa pull code mới  

# Look for this in logs:**Giải pháp:**

# "NaN found in model parameters: ['module.XXX.weight']"```bash

# Then check that specific layer's implementationgit fetch origin

```git log HEAD..origin/main --oneline  # Xem có commits mới không

git reset --hard origin/main

### Issue: "Export is too slow"python3 verify_ddp_fixes.py

**Solution:** SigLIP v2 is large (~1GB), but should process ~10 images/sec on V100```

- If much slower: Check GPU utilization

- If OOM: Reduce batch size (already at 1)### Vấn đề: Training vẫn bị NaN

**Check:**

## Technical Details```bash

# 1. Xem có batch nào bị skip không

### Why Min-Max Normalization?grep "WARNING: NaN" <log_file>



**Sum-based normalization** (old):# 2. Kiểm tra pseudo-labels

```pythonpython3 -c "

mask = mask / mask.sum() * (H * W)import torch

# Example: sum=10000, H*W=50176import numpy as np

# → mask *= 5.0176  ← Can exceed 1.0!from pathlib import Path

```

pseudo_root = Path('pseudo_labels_v2/train')

**Min-max normalization** (new):sample = next(pseudo_root.rglob('*_features.npy'))

```pythonfeatures = np.load(sample)

mask = (mask - mask.min()) / (mask.max() - mask.min())print(f'Has NaN in pseudo-labels: {np.isnan(features).any()}')

# Always maps to [0, 1] rangeprint(f'Has Inf in pseudo-labels: {np.isinf(features).any()}')

# Preserves relative differencesprint(f'Min value: {features.min()}')

```print(f'Max value: {features.max()}')

"

### Why Does NaN Happen?```



1. Masks > 1.0 multiply with features**Nếu pseudo-labels có NaN/Inf:**

2. Features become very large (overflow)→ Cần re-export pseudo-labels

3. Gradient computation hits float32 limits

4. Backprop produces NaN/Inf### Vấn đề: OOM (Out of Memory)

5. Parameters updated with NaN → model broken```bash

# Giảm batch size

### Gradient Clippingbash train_ddp.sh 8 1 20 128 128



```python# Hoặc giảm image size

grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)bash train_ddp.sh 8 2 20 96 128

``````



This prevents gradients from becoming too large, but doesn't fix root cause (masks > 1.0). **Re-export is still required.**---



## Summary## Expected Training Output



| Step | Status | Action |```

|------|--------|--------|✅ Model initialized with attn_chunk_size=128 (lower=less memory)

| 1. Pull latest code | ⏳ TODO | `git pull` |✅ DDP enabled: training on 8 GPUs

| 2. Delete old pseudo-labels | ⏳ TODO | `rm -rf pseudo-labels/siglip2/train` |✅ All loss modules moved to device: cuda:0

| 3. Re-export with fixed script | ⏳ TODO | `bash export_siglip2.sh ...` |Epoch 1/20 - Train loss: 0.1234

| 4. Validate masks [0,1] | ⏳ TODO | `python validate_teacher_export.py ...` |[Ref] UIEB: PSNR=22.45 dB, SSIM=0.8234

| 5. Start training | ⏳ TODO | `bash train_ddp.sh` |[Non-Ref] UCCS: UIQM=3.145, UCIQE=0.534

| 6. Monitor for NaN | ⏳ TODO | Check logs for "NaN" warnings |Epoch 2/20 - Train loss: 0.1123

...

---```



**Bottom Line:** Must re-export pseudo-labels with fixed script. Old pseudo-labels have masks > 1.0 which cause NaN.**Metrics should NOT be NaN!**


---

## Quick Commands Reference

```bash
# Full workflow
cd /home/ec2-user/SageMaker/Underwater-image-restoration
git reset --hard origin/main
python3 verify_ddp_fixes.py
bash train_ddp.sh 8 2 20 128 128

# Monitor
watch -n 1 nvidia-smi
tail -f <training_log>

# Stop training
pkill -9 -f train_student

# Check running processes
ps aux | grep train_student
```

---

## Contacts

Nếu vẫn gặp vấn đề:
1. Chụp screenshot error message
2. Copy output của `python3 verify_ddp_fixes.py`
3. Copy output của `git log --oneline -5`

Good luck! 🚀
