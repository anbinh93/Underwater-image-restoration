#!/usr/bin/env bash
set -euo pipefail

# Train the underwater student using CLIP encoder prompts exported from a Hugging Face checkpoint.
# 1) Export pseudo-labels with the CLIP HF weight.
# 2) Train the student against Dataset/train with benchmark evaluation.

HF_MODEL="hf-hub:openai/clip-vit-base-patch32"
CLIP_CKPT="hf-hub:openai/clip-vit-base-patch32"
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/clip_hf"
SAVE_PATH="experiments/clip_hf_student.pt"
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
  --clip-model "${HF_MODEL}" \
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
    --clip-model "${HF_MODEL}" \
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
    --clip-model "${HF_MODEL}" \
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
