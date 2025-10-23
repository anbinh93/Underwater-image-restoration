# ⚠️ URGENT: OOM Fix - GPU hết memory khi train

## Tình trạng hiện tại
- **Lỗi**: `torch.OutOfMemoryError: Tried to allocate 24.00 GiB`
- **GPU**: 39.49GB total, nhưng cần allocate 24GB cho một attention matrix
- **Nguyên nhân**: Attention matrix quá lớn với image size lớn

## 🔥 Giải pháp khẩn cấp (Chọn 1 trong 3)

### Giải pháp 1: Giảm image size (Nhanh nhất - 30 giây)
```bash
# Trên server, chạy ngay:
cd /home/ec2-user/SageMaker/Underwater-image-restoration

# Train với image size nhỏ hơn
bash train_ddp.sh 4 2 20 96 64
#                  │ │ │  │  └─ attn_chunk=64 (memory-efficient)
#                  │ │ │  └──── img_size=96 (thay vì 128)
#                  │ │ └─────── epochs=20
#                  │ └───────── batch=2/GPU
#                  └─────────── 4 GPUs
```

### Giải pháp 2: Git pull + retry (Khuyến nghị - 2 phút)
```bash
cd /home/ec2-user/SageMaker/Underwater-image-restoration
git pull origin main

# Verify chunked attention được update
grep -n "attn_chunk_size" underwater_ir/student/naf_unet_wfi/wtb.py
# Expected: Phải thấy self.attn_chunk_size trong code

# Retry với config an toàn
bash train_ddp.sh 4 2 20 128 128
```

### Giải pháp 3: Giảm batch size + chunk size (An toàn nhất)
```bash
# Ultra conservative settings cho GPU 40GB
bash train_ddp.sh 4 1 20 128 64
#                  │ │      │  └─ chunk=64 (4x memory efficient)
#                  │ │      └──── img=128
#                  │ └─────────── batch=1/GPU (4 total)
#                  └───────────── 4 GPUs
```

## 📊 Memory Breakdown

### Với IMG_SIZE=128, BATCH=2
```
Input: [2, 3, 128, 128]
After stem: [2, 64, 128, 128]
After downsample 1: [2, 128, 64, 64]   ← 4096 tokens
After downsample 2: [2, 256, 32, 32]   ← 1024 tokens

Attention matrix cho 1024 tokens:
- Size: [batch, heads, tokens, tokens] = [2, 12, 1024, 1024]
- Memory: 2 × 12 × 1024² × 4 bytes = 100MB ✅

Nhưng có nhiều WTB blocks → tổng memory cao!
```

### Attention Chunk Size Impact
| Chunk Size | Max Attention Memory | Speed | Recommendation |
|------------|---------------------|-------|----------------|
| 1024 (full)| 100MB per block     | Fast  | ❌ OOM with big images |
| 256        | 25MB per block      | 90%   | ✅ Default |
| 128        | 12.5MB per block    | 80%   | ✅ Safe for 16GB GPU |
| 64         | 6.25MB per block    | 70%   | ✅ Ultra safe |

## 🎯 Recommended Settings by GPU

### GPU 40GB (A100) - Bạn đang dùng
```bash
# Option A: Max performance
bash train_ddp.sh 4 4 20 128 256

# Option B: Balanced
bash train_ddp.sh 4 2 20 160 128

# Option C: Safe (current issue)
bash train_ddp.sh 4 2 20 128 128
```

### GPU 24GB (RTX 3090/4090)
```bash
bash train_ddp.sh 4 2 20 128 128
```

### GPU 16GB (V100)
```bash
bash train_ddp.sh 4 2 20 96 64
```

## 🔍 Debug: Kiểm tra server có code mới không

```bash
# Check wtb.py có chunked attention chưa
cd /home/ec2-user/SageMaker/Underwater-image-restoration
head -50 underwater_ir/student/naf_unet_wfi/wtb.py | grep -A 5 "def __init__"

# Expected output phải có:
# def __init__(self, ..., attn_chunk_size: int = 256):
#     self.attn_chunk_size = attn_chunk_size

# Check forward có dùng chunk không
grep -A 20 "def forward" underwater_ir/student/naf_unet_wfi/wtb.py | grep chunk

# Expected: phải thấy "for i in range(0, h * w, self.attn_chunk_size)"
```

## ⚡ Immediate Action

**Chạy ngay lệnh này để test:**
```bash
cd /home/ec2-user/SageMaker/Underwater-image-restoration

# Kill training đang chạy (nếu có)
pkill -f "torchrun"

# Git pull để lấy code mới
git stash  # Save local changes if any
git pull origin main

# Verify update
python3 -c "
from underwater_ir.student.naf_unet_wfi.wtb import WideTransformerBlock
import inspect
sig = inspect.signature(WideTransformerBlock.__init__)
print('✅ Updated!' if 'attn_chunk_size' in sig.parameters else '❌ Old version!')
"

# Start training với safe config
bash train_ddp.sh 4 2 20 128 128
```

## 📈 Monitor Memory Usage

### Terminal 1: Training
```bash
bash train_ddp.sh 4 2 20 128 128
```

### Terminal 2: GPU Monitor
```bash
watch -n 1 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

### Terminal 3: Process Memory
```bash
watch -n 2 "ps aux | grep python | grep -v grep | awk '{print \$2, \$4, \$11}' | head -8"
```

## 🚨 Nếu vẫn OOM sau khi git pull

### Reduce everything:
```bash
bash train_ddp.sh 4 1 20 96 64
```

### Hoặc train single GPU trước:
```bash
python -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --epochs 20 \
  --batch-size 4 \
  --img-size 128 \
  --attn-chunk-size 128 \
  --save-path experiments/student_single_gpu.pt
```

## 💡 Why Still OOM?

1. **Server chưa có code mới**: Git pull để update
2. **Python cache**: `rm -rf __pycache__` và `rm -rf underwater_ir/**/__pycache__`
3. **Multiple forward passes**: Gradient accumulation không được implement
4. **Memory fragmentation**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 🔧 Advanced: Set environment variables

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1  # For debugging only, slower

bash train_ddp.sh 4 2 20 128 128
```

## ✅ Success Criteria

Training should show:
```
✅ Model initialized with attn_chunk_size=128 (lower=less memory)
✅ DDP enabled: training on 4 GPUs
Epoch 1/20 - Train loss: 0.xxxx
```

GPU memory should be:
```
| GPU | Memory-Usage |
|  0  | 15GB/40GB    |  ← Should be < 30GB
|  1  | 15GB/40GB    |
|  2  | 15GB/40GB    |
|  3  | 15GB/40GB    |
```
