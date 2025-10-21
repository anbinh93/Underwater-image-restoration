#!/usr/bin/env bash
set -euo pipefail

# Train the underwater student using OpenAI CLIP ViT-B/32
# 1) Export pseudo-labels with OpenAI CLIP checkpoint.
# 2) Train the student against Dataset/train with benchmark evaluation.

# NOTE: Model "openai/clip-vit-base-patch32" from HF doesn't exist in open_clip format.
# The correct way to use OpenAI's ViT-B/32 CLIP with open_clip is:
#   Model: "ViT-B-32" with Checkpoint: "openai"
# This will auto-download from OpenAI's official release.
CLIP_MODEL="ViT-B-32"
CLIP_CKPT="openai"
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/openai_clip"
SAVE_PATH="experiments/openai_clip_student.pt"
EPOCHS=20
BATCH=4
WORKERS=4

# Ensure we're in the project root
cd "$(dirname "$0")"

# Stage 1: export pseudo labels (train split here; add val splits if needed)
mkdir -p "${PSEUDO_ROOT}/train"
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root "${TRAIN_ROOT}/input" \
  --target-root "${TRAIN_ROOT}/target" \
  --output "${PSEUDO_ROOT}/train" \
  --clip-model "${CLIP_MODEL}" \
  --clip-checkpoint "${CLIP_CKPT}" \
  --use-crf \
  --num-workers "${WORKERS}"

# Stage 1b: export pseudo labels for reference benchmarks
for subset_dir in "${VAL_REF_ROOT}"/*; do
  [ -d "${subset_dir}" ] || continue
  subset="$(basename "${subset_dir}")"
  mkdir -p "${PSEUDO_ROOT}/testset_ref/${subset}"
  python -m underwater_ir.teacher.export_pseudolabels \
    --input-root "${subset_dir}/input" \
    --target-root "${subset_dir}/target" \
    --output "${PSEUDO_ROOT}/testset_ref/${subset}" \
    --clip-model "${CLIP_MODEL}" \
    --clip-checkpoint "${CLIP_CKPT}" \
    --use-crf \
    --num-workers "${WORKERS}"
done

# Stage 1c: export pseudo labels for non-reference benchmarks
for subset_dir in "${VAL_NONREF_ROOT}"/*; do
  [ -d "${subset_dir}" ] || continue
  subset="$(basename "${subset_dir}")"
  input_dir="${subset_dir}/input"
  if [[ ! -d "${input_dir}" ]]; then
    input_dir="${subset_dir}"
  fi
  mkdir -p "${PSEUDO_ROOT}/testset_nonref/${subset}"
  python -m underwater_ir.teacher.export_pseudolabels \
    --input-root "${input_dir}" \
    --output "${PSEUDO_ROOT}/testset_nonref/${subset}" \
    --clip-model "${CLIP_MODEL}" \
    --clip-checkpoint "${CLIP_CKPT}" \
    --use-crf \
    --num-workers "${WORKERS}"
done

# Stage 2: student training + evaluation
python -m underwater_ir.student.train_student \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --num-workers "${WORKERS}" \
  --save-path "${SAVE_PATH}"
