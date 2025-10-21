#!/usr/bin/env bash
set -euo pipefail

# Train the underwater student using CLIP encoder prompts exported from a Hugging Face checkpoint.
# 1) Export pseudo-labels with the CLIP HF weight.
# 2) Train the student against Dataset/train with benchmark evaluation.

HF_MODEL="hf-hub:openai/clip-vit-base-patch32"
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/clip_hf"
SAVE_PATH="experiments/clip_hf_student.pt"
EPOCHS=20
BATCH=4
WORKERS=4

# Stage 1: export pseudo labels (train split here; add val splits if needed)
mkdir -p "${PSEUDO_ROOT}/train"
python underwater_ir/teacher/export_pseudolabels.py \
  --input-root "${TRAIN_ROOT}/input" \
  --target-root "${TRAIN_ROOT}/target" \
  --output "${PSEUDO_ROOT}/train" \
  --clip-model "${HF_MODEL}" \
  --use-crf \
  --num-workers "${WORKERS}"

# Stage 2: student training + evaluation
python underwater_ir/student/train_student.py \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --num-workers "${WORKERS}" \
  --save-path "${SAVE_PATH}"
