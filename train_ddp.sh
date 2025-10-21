#!/usr/bin/env bash
set -euo pipefail

# Distributed training wrapper script for 4 GPUs
# Usage: bash train_ddp.sh [num_gpus] [batch_size_per_gpu]

NUM_GPUS="${1:-4}"
BATCH_SIZE="${2:-4}"
EPOCHS="${3:-20}"
IMG_SIZE="${4:-256}"

TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/daclip"
SAVE_PATH="experiments/daclip_student_ddp.pt"
WORKERS=4

echo "================================================================================"
echo "🚀 Starting Distributed Training with DDP"
echo "================================================================================"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Total batch size: $((NUM_GPUS * BATCH_SIZE))"
echo "  Epochs: ${EPOCHS}"
echo "  Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "  Workers per GPU: ${WORKERS}"
echo "================================================================================"
echo ""

# Verify CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available: {torch.cuda.device_count()} GPUs')"

# Verify pseudo-labels exist
if [[ ! -d "${PSEUDO_ROOT}/train" ]]; then
  echo "❌ ERROR: Pseudo-label directory not found: ${PSEUDO_ROOT}/train" >&2
  echo "   Please run export stage first (bash run_clip_training.sh)" >&2
  exit 1
fi

pt_count=$(find "${PSEUDO_ROOT}/train" -name "*.pt" -type f 2>/dev/null | wc -l)
npy_count=$(find "${PSEUDO_ROOT}/train" -name "*_features.npy" -type f 2>/dev/null | wc -l)
total_count=$((pt_count + npy_count))

if [[ $total_count -eq 0 ]]; then
  echo "❌ ERROR: No pseudo-label files found in ${PSEUDO_ROOT}/train" >&2
  exit 1
fi

if [[ $pt_count -gt 0 ]]; then
  echo "✅ Found ${pt_count} pseudo-label files (.pt format)"
fi
if [[ $npy_count -gt 0 ]]; then
  echo "✅ Found ${npy_count} pseudo-label files (.npy format)"
fi
echo ""

# Create output directory
mkdir -p experiments

# Launch distributed training using torchrun (PyTorch >= 1.10)
echo "Starting training with torchrun..."
echo ""

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  -m underwater_ir.student.train_student \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${WORKERS}" \
  --img-size "${IMG_SIZE}" \
  --save-path "${SAVE_PATH}" \
  --ddp

echo ""
echo "================================================================================"
echo "✅ Training completed!"
echo "   Model saved to: ${SAVE_PATH}"
echo "================================================================================"
