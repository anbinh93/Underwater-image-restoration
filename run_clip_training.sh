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

echo ""
echo "================================================================================"
echo "[Stage 1a] Exporting pseudo-labels for TRAINING set..."
echo "================================================================================"
echo "  Input: ${TRAIN_ROOT}/input"
echo "  Target: ${TRAIN_ROOT}/target"
echo "  Output: ${PSEUDO_ROOT}/train"
echo "  CLIP Model: ${CLIP_MODEL}"
echo "  Checkpoint: ${CLIP_CKPT}"
echo ""

# Verify input directories exist
if [[ ! -d "${TRAIN_ROOT}/input" ]]; then
  echo "❌ ERROR: Training input directory not found: ${TRAIN_ROOT}/input" >&2
  exit 1
fi

if [[ ! -d "${TRAIN_ROOT}/target" ]]; then
  echo "❌ ERROR: Training target directory not found: ${TRAIN_ROOT}/target" >&2
  exit 1
fi

echo "✅ Training directories verified"
input_count=$(find "${TRAIN_ROOT}/input" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
target_count=$(find "${TRAIN_ROOT}/target" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "   Input images: ${input_count}"
echo "   Target images: ${target_count}"
echo ""

mkdir -p "${PSEUDO_ROOT}/train"
echo "Running export_pseudolabels..."
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root "${TRAIN_ROOT}/input" \
  --target-root "${TRAIN_ROOT}/target" \
  --output "${PSEUDO_ROOT}/train" \
  --clip-model "${CLIP_MODEL}" \
  --clip-checkpoint "${CLIP_CKPT}" \
  --num-workers "${WORKERS}"

export_status=$?
if [[ $export_status -ne 0 ]]; then
  echo "❌ ERROR: Pseudo-label export failed with exit code ${export_status}" >&2
  exit 1
fi

echo ""
echo "Verifying exported pseudo-labels..."
if [[ ! -d "${PSEUDO_ROOT}/train" ]]; then
  echo "❌ ERROR: Output directory was not created: ${PSEUDO_ROOT}/train" >&2
  exit 1
fi

pt_files=$(find "${PSEUDO_ROOT}/train" -name "*.pt" -type f)
pt_count=$(echo "$pt_files" | grep -c "\.pt$" || echo 0)

if [[ $pt_count -eq 0 ]]; then
  echo "❌ ERROR: No .pt files found in ${PSEUDO_ROOT}/train" >&2
  echo "   Export may have failed silently. Check logs above." >&2
  exit 1
fi

echo "✅ Export successful!"
echo "   Output directory: ${PSEUDO_ROOT}/train"
echo "   Pseudo-label files: ${pt_count}"
echo "   Sample files:"
echo "$pt_files" | head -3
echo ""

echo "================================================================================"
echo "[Stage 1b] Exporting pseudo-labels for REFERENCE validation sets..."
echo "================================================================================"
for subset_dir in "${VAL_REF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
  subset="$(basename "${subset_dir}")"
  echo "  Processing: ${subset}"
  mkdir -p "${PSEUDO_ROOT}/testset_ref/${subset}"
  python -m underwater_ir.teacher.export_pseudolabels \
    --input-root "${subset_dir}/input" \
    --target-root "${subset_dir}/target" \
    --output "${PSEUDO_ROOT}/testset_ref/${subset}" \
    --clip-model "${CLIP_MODEL}" \
    --clip-checkpoint "${CLIP_CKPT}" \
    --num-workers "${WORKERS}"
echo ""

echo "================================================================================"
echo "[Stage 1c] Exporting pseudo-labels for NON-REFERENCE validation sets..."
echo "================================================================================"
for subset_dir in "${VAL_NONREF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
  subset="$(basename "${subset_dir}")"
  input_dir="${subset_dir}/input"
  [[ -d "${input_dir}" ]] || input_dir="${subset_dir}"
  echo "  Processing: ${subset}"
  mkdir -p "${PSEUDO_ROOT}/testset_nonref/${subset}"
  python -m underwater_ir.teacher.export_pseudolabels \
    --input-root "${input_dir}" \
    --output "${PSEUDO_ROOT}/testset_nonref/${subset}" \
    --clip-model "${CLIP_MODEL}" \
    --clip-checkpoint "${CLIP_CKPT}" \
    --num-workers "${WORKERS}"
done
echo ""

echo "================================================================================  "
echo "[Stage 2] Training student model..."
echo "  Pseudo-labels root: ${PSEUDO_ROOT}"
echo "  Model save path: ${SAVE_PATH}"
echo "================================================================================"

# Verify pseudo-labels exist before training
if [[ ! -d "${PSEUDO_ROOT}/train" ]]; then
  echo "❌ ERROR: Training pseudo-label directory not found: ${PSEUDO_ROOT}/train" >&2
  exit 1
fi

pt_count=$(find "${PSEUDO_ROOT}/train" -name "*.pt" -type f | wc -l)
if [[ $pt_count -eq 0 ]]; then
  echo "❌ ERROR: No .pt files found in ${PSEUDO_ROOT}/train" >&2
  echo "   Please check the export stage for errors." >&2
  exit 1
fi

echo "✅ Found ${pt_count} pseudo-label files in training set"
echo ""

python -m underwater_ir.student.train_student \
  --train-root "${TRAIN_ROOT}" \
  --val-ref-root "${VAL_REF_ROOT}" \
  --val-nonref-root "${VAL_NONREF_ROOT}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --num-workers "${WORKERS}" \
  --save-path "${SAVE_PATH}"
