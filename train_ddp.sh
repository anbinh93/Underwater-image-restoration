#!/usr/bin/env bash
set -euo pipefail

# Distributed training wrapper script for multi-GPU training
# Usage: bash train_ddp.sh [num_gpus] [batch_size_per_gpu] [epochs] [img_size] [attn_chunk_size]

NUM_GPUS="${1:-8}"
BATCH_SIZE="${2:-2}"
EPOCHS="${3:-200}"
IMG_SIZE="${4:-128}"
ATTN_CHUNK="${5:-128}"

TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset_ref"
VAL_NONREF_ROOT="Dataset/testset_nonref"
PSEUDO_ROOT="pseudo-labels/siglip2"
SAVE_PATH="experiments/student_siglip2_ddp.pt"
WORKERS=4

echo "================================================================================"
echo "🚀 DDP Training - Underwater Image Restoration"
echo "================================================================================"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch/GPU: ${BATCH_SIZE} → Total: $((NUM_GPUS * BATCH_SIZE))"
echo "  Epochs: ${EPOCHS}"
echo "  Image size: ${IMG_SIZE}×${IMG_SIZE}"
echo "  Attention chunk: ${ATTN_CHUNK}"
echo "  Workers/GPU: ${WORKERS}"
echo "  Pseudo-labels: ${PSEUDO_ROOT}"
echo "================================================================================"
echo ""

# Verify CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA: {torch.cuda.device_count()} GPUs')"

# Verify pseudo-labels exist (with better error message)
if [[ ! -d "${PSEUDO_ROOT}/train" ]]; then
  echo "❌ ERROR: Pseudo-labels not found at: ${PSEUDO_ROOT}/train" >&2
  echo "" >&2
  echo "   Please export pseudo-labels first:" >&2
  echo "   bash export_all_siglip2.sh Dataset ${PSEUDO_ROOT}" >&2
  echo "" >&2
  exit 1
fi

# Count pseudo-label files
npy_count=$(find "${PSEUDO_ROOT}/train" -name "*_masks.npy" -type f 2>/dev/null | wc -l)

if [[ $npy_count -eq 0 ]]; then
  echo "❌ ERROR: No mask files found in ${PSEUDO_ROOT}/train" >&2
  echo "   Expected: *_masks.npy, *_features.npy, *_probs.npy" >&2
  exit 1
fi

echo "✅ Found ${npy_count} training samples"
echo ""

# Create experiment directory
mkdir -p experiments

echo "Starting training..."
echo ""

# Launch distributed training
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
  --attn-chunk-size "${ATTN_CHUNK}" \
  --save-path "${SAVE_PATH}" \
  --ddp

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "================================================================================"
  echo "✅ Training Completed Successfully!"
  echo "================================================================================"
  echo "  Model: ${SAVE_PATH}"
  echo "  Epochs: ${EPOCHS}"
  echo "  GPUs: ${NUM_GPUS}"
  echo ""
else
  echo "================================================================================"
  echo "❌ Training Failed (exit code: ${EXIT_CODE})"
  echo "================================================================================"
  echo "  Check logs above for errors"
  echo ""
  exit ${EXIT_CODE}
fi
echo "================================================================================"
