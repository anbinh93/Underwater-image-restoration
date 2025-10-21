#!/usr/bin/env bash
set -euo pipefail

# Train the underwater student using DACLiP ViT-B/32 (per README/app.py).
# 1) Export pseudo-labels with DACLiP checkpoints.
# 2) Train the student with evaluation on reference/non-reference benchmarks.

CLIP_MODEL="daclip_ViT-B-32"
CLIP_CKPT="pretrained/daclip_ViT-B-32.pt"
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/daclip"
SAVE_PATH="experiments/daclip_student.pt"
EPOCHS=20
BATCH=4
WORKERS=4

if [[ ! -f "${CLIP_CKPT}" ]]; then
  echo "Expected DACLiP checkpoint at ${CLIP_CKPT}. Please download per README." >&2
  exit 1
fi

python - <<'PY'
try:
    import ftfy, sentencepiece  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing dependencies (ftfy, sentencepiece). Install them before running.") from exc
PY

mkdir -p "${PSEUDO_ROOT}/train"
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root "${TRAIN_ROOT}/input" \
  --target-root "${TRAIN_ROOT}/target" \
  --output "${PSEUDO_ROOT}/train" \
  --clip-model "${CLIP_MODEL}" \
  --clip-checkpoint "${CLIP_CKPT}" \
  --use-crf \
  --num-workers "${WORKERS}"

for subset_dir in "${VAL_REF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
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

for subset_dir in "${VAL_NONREF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
  subset="$(basename "${subset_dir}")"
  input_dir="${subset_dir}/input"
  [[ -d "${input_dir}" ]] || input_dir="${subset_dir}"
  mkdir -p "${PSEUDO_ROOT}/testset_nonref/${subset}"
  python -m underwater_ir.teacher.export_pseudolabels \
    --input-root "${input_dir}" \
    --output "${PSEUDO_ROOT}/testset_nonref/${subset}" \
    --clip-model "${CLIP_MODEL}" \
    --clip-checkpoint "${CLIP_CKPT}" \
    --use-crf \
    --num-workers "${WORKERS}"
done

python -m underwater_ir.student.train_student \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --num-workers "${WORKERS}" \
  --save-path "${SAVE_PATH}"
