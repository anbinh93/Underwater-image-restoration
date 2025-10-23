# 🚨 HƯỚNG DẪN FIX NaN VÀ CÁC LỖI DDP

## Tình trạng hiện tại
- ✅ Đã fix 5 lỗi dimension mismatch
- ✅ Đã thêm NaN detection
- ✅ Đã fix SSIM loss numerical stability
- ⚠️ Server cần pull code mới nhất

---

## Chạy trên Server (AWS EC2)

### Bước 1: Stop toàn bộ training đang chạy
```bash
pkill -9 -f train_student
pkill -9 -f torchrun
nvidia-smi  # Xác nhận GPU đã free
```

### Bước 2: Update code từ GitHub
```bash
cd /home/ec2-user/SageMaker/Underwater-image-restoration

# Xem code hiện tại đang ở commit nào
git log --oneline -3

# Pull code mới nhất (có NaN fixes)
git fetch origin
git reset --hard origin/main

# Verify đã pull thành công
git log --oneline -3
# Phải thấy commits: 30f7ac3, d67eabe, 395dd06
```

### Bước 3: Verify tất cả fixes đã được áp dụng
```bash
python3 verify_ddp_fixes.py
```

**Expected output:**
```
✅ ALL CHECKS PASSED - Ready to train!
```

Nếu FAIL → chạy:
```bash
bash URGENT_FIX_SERVER.sh
```

### Bước 4: Clear cache và khởi động training
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Start training với NaN detection
bash train_ddp.sh 8 2 20 128 128
```

---

## Các Fix đã áp dụng

### Fix 1-5: Dimension Mismatch (đã fix trước đó)
- ✅ Loss modules → device
- ✅ Masks resizing 
- ✅ Wavelet dimensions
- ✅ Region loss iteration
- ✅ Feature alignment

### Fix 6: NaN Detection (MỚI)
**File:** `train_student.py`

Kiểm tra NaN tại 3 điểm:
1. **Input check:** LQ, GT, pseudo-labels
2. **Output check:** Model output
3. **Loss check:** Total loss before backward

Nếu phát hiện NaN → skip batch và in warning.

### Fix 7: SSIM Numerical Stability (MỚI)
**File:** `train_student.py`, function `ssim_loss()`

```python
# Clamp variance to avoid negative values
sigma_x = sigma_x.clamp(min=0)
sigma_y = sigma_y.clamp(min=0)

# Increased epsilon
ssim_map = numerator / (denominator + 1e-8)  # Was 1e-6
```

---

## Monitoring Training

### Khi training chạy, check:

1. **Không có WARNING về NaN:**
```
# Nếu thấy messages này → có vấn đề
[Rank X] WARNING: NaN detected in input LQ!
[Rank X] WARNING: NaN in model output!
[Rank X] WARNING: NaN/Inf in total_loss!
```

2. **Loss giảm bình thường:**
```
Epoch 1/20 - Train loss: 0.XXXX  # NOT nan
[Ref] UIEB: PSNR=XX.XX dB        # NOT nan
```

3. **GPU memory stable:**
```bash
watch -n 1 nvidia-smi
# All 8 GPUs should show similar memory usage
```

---

## Troubleshooting

### Vấn đề: Vẫn thấy dimension mismatch errors
**Nguyên nhân:** Server chưa pull code mới  
**Giải pháp:**
```bash
git fetch origin
git log HEAD..origin/main --oneline  # Xem có commits mới không
git reset --hard origin/main
python3 verify_ddp_fixes.py
```

### Vấn đề: Training vẫn bị NaN
**Check:**
```bash
# 1. Xem có batch nào bị skip không
grep "WARNING: NaN" <log_file>

# 2. Kiểm tra pseudo-labels
python3 -c "
import torch
import numpy as np
from pathlib import Path

pseudo_root = Path('pseudo_labels_v2/train')
sample = next(pseudo_root.rglob('*_features.npy'))
features = np.load(sample)
print(f'Has NaN in pseudo-labels: {np.isnan(features).any()}')
print(f'Has Inf in pseudo-labels: {np.isinf(features).any()}')
print(f'Min value: {features.min()}')
print(f'Max value: {features.max()}')
"
```

**Nếu pseudo-labels có NaN/Inf:**
→ Cần re-export pseudo-labels

### Vấn đề: OOM (Out of Memory)
```bash
# Giảm batch size
bash train_ddp.sh 8 1 20 128 128

# Hoặc giảm image size
bash train_ddp.sh 8 2 20 96 128
```

---

## Expected Training Output

```
✅ Model initialized with attn_chunk_size=128 (lower=less memory)
✅ DDP enabled: training on 8 GPUs
✅ All loss modules moved to device: cuda:0
Epoch 1/20 - Train loss: 0.1234
[Ref] UIEB: PSNR=22.45 dB, SSIM=0.8234
[Non-Ref] UCCS: UIQM=3.145, UCIQE=0.534
Epoch 2/20 - Train loss: 0.1123
...
```

**Metrics should NOT be NaN!**

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
