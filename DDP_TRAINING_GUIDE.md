# Distributed Data Parallel (DDP) Training Guide

## Giới thiệu
Script training đã được nâng cấp để hỗ trợ Distributed Data Parallel (DDP) training trên nhiều GPU. DDP giúp tăng tốc training bằng cách phân phối dữ liệu và gradient trên nhiều GPU.

## Yêu cầu
- PyTorch >= 1.10 (có `torchrun`)
- NCCL backend cho CUDA
- Nhiều GPU CUDA-compatible (test với 4 GPU)
- Pseudo-labels đã được export sẵn

## Quick Start - Train trên 4 GPU

### Cách 1: Sử dụng script wrapper (Khuyến nghị)
```bash
# Train với cấu hình mặc định (4 GPU, batch=4/GPU, 20 epochs)
bash train_ddp.sh

# Tùy chỉnh số GPU và batch size
bash train_ddp.sh 4 8        # 4 GPU, 8 batch/GPU = 32 total batch
bash train_ddp.sh 2 16       # 2 GPU, 16 batch/GPU = 32 total batch

# Tùy chỉnh đầy đủ
bash train_ddp.sh 4 4 20 256  # 4 GPU, 4 batch/GPU, 20 epochs, 256x256 images
```

### Cách 2: Sử dụng torchrun trực tiếp
```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --val-ref-root "Dataset/testset(ref)" \
  --val-nonref-root "Dataset/testset(non-ref)" \
  --pseudo-root pseudo-labels/daclip \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 4 \
  --img-size 256 \
  --save-path experiments/daclip_student_ddp.pt \
  --ddp
```

### Cách 3: Sử dụng python -m torch.distributed.launch (PyTorch < 1.10)
```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=12355 \
  -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --epochs 20 \
  --batch-size 4 \
  --ddp
```

## Cấu hình DDP

### Arguments mới
- `--ddp`: Enable DDP training (bắt buộc cho multi-GPU)
- `--local-rank`: Rank của process (tự động set bởi torchrun)
- `--world-size`: Tổng số GPU (tự động set bởi torchrun)

### Environment Variables (tự động set)
- `MASTER_ADDR`: localhost (mặc định)
- `MASTER_PORT`: 12355 (mặc định)
- `LOCAL_RANK`: GPU rank (0, 1, 2, 3)
- `WORLD_SIZE`: Số GPU (4)
- `RANK`: Global rank (same as LOCAL_RANK cho single node)

## Batch Size và Memory

### Effective Batch Size
```
Total Batch Size = num_gpus × batch_size_per_gpu
```

### Ví dụ cấu hình
| GPUs | Batch/GPU | Total Batch | Memory/GPU (256px) | Memory/GPU (512px) |
|------|-----------|-------------|--------------------|--------------------|
| 4    | 4         | 16          | ~6 GB              | ~12 GB             |
| 4    | 8         | 32          | ~12 GB             | ~24 GB             |
| 4    | 16        | 64          | ~24 GB             | ~48 GB             |
| 2    | 16        | 32          | ~24 GB             | ~48 GB             |

### Khuyến nghị
- **GPU 24GB (RTX 3090/4090, A10)**: batch=8, img_size=256 hoặc batch=4, img_size=512
- **GPU 16GB (V100, RTX 4000)**: batch=4, img_size=256
- **GPU 40GB+ (A100)**: batch=16, img_size=512

## Tính năng DDP

### ✅ Tự động
- **Data partitioning**: Mỗi GPU train trên phần khác nhau của dataset
- **Gradient synchronization**: Gradient được average qua tất cả GPU
- **Model replication**: Model được copy sang tất cả GPU
- **Sampler shuffle**: Mỗi epoch shuffle khác nhau trên mỗi GPU
- **Loss averaging**: Training loss được average qua tất cả GPU

### ✅ Optimization
- **Only main process (rank 0)**:
  - Prints training logs
  - Evaluates validation metrics
  - Saves model checkpoints
- **All processes**:
  - Train forward/backward pass
  - Compute gradients
  - Synchronize at barriers

### ✅ Fault tolerance
- Graceful shutdown với `cleanup_ddp()`
- Barrier synchronization sau mỗi epoch
- Safe model saving (unwrap DDP wrapper)

## Performance

### Speedup với DDP
Với dataset 5602 images, batch_size=4/GPU:

| GPUs | Images/GPU | Steps/Epoch | Time/Epoch* | Speedup |
|------|------------|-------------|-------------|---------|
| 1    | 5602       | 1400        | ~20 min     | 1.0x    |
| 2    | 2801       | 700         | ~11 min     | 1.8x    |
| 4    | 1400       | 350         | ~6 min      | 3.3x    |

*Estimate, actual time depends on GPU model

### Efficiency
- **Linear speedup** không đạt được do communication overhead
- **Expected efficiency**: 80-90% với 4 GPU
- **Bottlenecks**: 
  - Gradient synchronization (NCCL)
  - Pseudo-label loading (I/O)
  - Image augmentation (CPU)

### Tips tối ưu
1. **Tăng num_workers**: 4-8 workers/GPU để giảm I/O bottleneck
2. **Pin memory**: Đã enable mặc định
3. **Batch size**: Tăng batch/GPU để tăng GPU utilization
4. **Async loading**: DataLoader tự động prefetch

## Monitoring

### GPU Usage
```bash
# Terminal 1: Run training
bash train_ddp.sh

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi
```

### Expected output
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     12345      C   python                          6000MiB |
|    1   N/A  N/A     12346      C   python                          6000MiB |
|    2   N/A  N/A     12347      C   python                          6000MiB |
|    3   N/A  N/A     12348      C   python                          6000MiB |
+-----------------------------------------------------------------------------+
```

### Logs
Chỉ rank 0 print logs:
```
✅ DDP enabled: training on 4 GPUs
Epoch 1/20 - Train loss: 0.1234
[Ref] UIEB: PSNR=23.45 dB, SSIM=0.8765
[Non-Ref] UIEB-unpaired: UIQM=0.456, UCIQE=0.789
```

## Troubleshooting

### Error: "NCCL error"
**Giải pháp**:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Nếu không có InfiniBand
```

### Error: "Address already in use"
**Nguyên nhân**: Port 12355 đang được sử dụng  
**Giải pháp**:
```bash
export MASTER_PORT=12356  # Đổi port
```

### Error: "CUDA out of memory"
**Giải pháp**:
- Giảm `--batch-size`
- Giảm `--img-size`
- Giảm số GPU (nhưng tăng batch/GPU)

### Slow training
**Kiểm tra**:
- `num_workers` đủ cao (4-8/GPU)
- Pseudo-labels trên SSD, không phải HDD
- Network bandwidth (NFS/shared storage có thể chậm)

### Hang at initialization
**Nguyên nhân**: DDP không thể communicate  
**Giải pháp**:
```bash
# Check network
ping localhost

# Check NCCL
python -c "import torch; print(torch.cuda.nccl.version())"

# Use gloo backend (slower but more stable)
# Modify train_student.py: dist.init_process_group("gloo", ...)
```

## So sánh Single-GPU vs Multi-GPU

### Single GPU
```bash
python -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --epochs 20 \
  --batch-size 16 \
  --device cuda
```

### Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=4 \
  -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --pseudo-root pseudo-labels/daclip \
  --epochs 20 \
  --batch-size 4 \
  --ddp
```

**Lưu ý**: Total batch size = 16 (same) nhưng DDP nhanh hơn ~3.3x

## Advanced: Multi-Node Training

Nếu có nhiều máy (nodes) với multiple GPU:

### Node 0 (master)
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=12355 \
  -m underwater_ir.student.train_student \
  --ddp \
  [other args...]
```

### Node 1 (worker)
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr=192.168.1.100 \
  --master_port=12355 \
  -m underwater_ir.student.train_student \
  --ddp \
  [other args...]
```

## Files Modified

1. **`underwater_ir/student/train_student.py`**
   - Added DDP imports and setup functions
   - Added `--ddp`, `--local-rank`, `--world-size` arguments
   - Created DistributedSampler for training data
   - Wrapped model with DDP
   - Added rank-aware logging and saving
   - Added barrier synchronization

2. **`underwater_ir/data/datasets.py`**
   - Added `sampler` parameter to dataloaders
   - Updated `create_paired_train_loader` signature

3. **`train_ddp.sh`** (new)
   - Wrapper script for easy DDP training
   - Validates environment and data
   - Launches torchrun with correct arguments

## Best Practices

1. ✅ **Always use `--ddp` flag** khi train multi-GPU
2. ✅ **Use torchrun** thay vì python -m torch.distributed.launch
3. ✅ **Batch size = memory_per_gpu / 4** as starting point
4. ✅ **Monitor all GPUs** để đảm bảo balanced load
5. ✅ **Save checkpoints regularly** (đã auto save mỗi epoch)
6. ✅ **Test single-GPU first** trước khi scale lên multi-GPU

## References
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
