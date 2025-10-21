#!/usr/bin/env bash
set -euo pipefail

# Train the underwater student using DACLiP or standard CLIP
# This script uses the new unified CLIP model loader with config support

# Configuration
CONFIG_FILE="configs/config_clip.yaml"
CLIP_MODEL="openai_clip"  # Options: openai_clip, daclip_custom
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
PSEUDO_ROOT="pseudo-labels/daclip"
SAVE_PATH="experiments/daclip_student.pt"
EPOCHS=20
BATCH=4
WORKERS=4

echo "================================================================================"
echo "DACLiP-UIR Training Pipeline with Unified CLIP Loader"
echo "================================================================================"
echo "Configuration file: ${CONFIG_FILE}"
echo "CLIP model type: ${CLIP_MODEL}"
echo ""

# Check if config exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "❌ Configuration file not found: ${CONFIG_FILE}" >&2
  echo "   Please ensure the config file exists." >&2
  exit 1
fi

# Check Python dependencies
echo "[1/5] Checking dependencies..."
python3 - <<'PY'
import sys

missing = []
try:
    import torch
except ImportError:
    missing.append("torch")

try:
    import yaml
except ImportError:
    missing.append("pyyaml")

try:
    import transformers
except ImportError:
    missing.append("transformers")

try:
    from PIL import Image
except ImportError:
    missing.append("pillow")

if missing:
    print(f"❌ Missing dependencies: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("✅ All basic dependencies are installed")
PY

if [[ $? -ne 0 ]]; then
  exit 1
fi

# Test CLIP model loading
echo ""
echo "[2/5] Testing CLIP model loading..."
python3 -m underwater_ir.model_loader --config "${CONFIG_FILE}"

if [[ $? -ne 0 ]]; then
  echo "❌ CLIP model loading failed. Please check your configuration." >&2
  exit 1
fi

# Check dataset directories
echo ""
echo "[3/5] Checking dataset directories..."
if [[ ! -d "${TRAIN_ROOT}/input" ]]; then
  echo "❌ Training input directory not found: ${TRAIN_ROOT}/input" >&2
  exit 1
fi

if [[ ! -d "${TRAIN_ROOT}/target" ]]; then
  echo "❌ Training target directory not found: ${TRAIN_ROOT}/target" >&2
  exit 1
fi

echo "✅ Dataset directories found"

# Export pseudo-labels for training set
echo ""
echo "[4/5] Exporting pseudo-labels..."
mkdir -p "${PSEUDO_ROOT}/train"

if [[ -f "underwater_ir/teacher/export_pseudolabels_v2.py" ]]; then
  # Use new version with unified loader
  echo "Using new export_pseudolabels_v2.py with unified CLIP loader..."
  python3 -m underwater_ir.teacher.export_pseudolabels_v2 \
    --config "${CONFIG_FILE}" \
    --input-root "${TRAIN_ROOT}/input" \
    --target-root "${TRAIN_ROOT}/target" \
    --output "${PSEUDO_ROOT}/train" \
    --num-workers "${WORKERS}"
else
  # Fallback to original version
  echo "⚠️  Using original export_pseudolabels.py"
  echo "    Consider updating to export_pseudolabels_v2.py for better compatibility"
  
  # Check if using DACLiP custom model
  if [[ "${CLIP_MODEL}" == "daclip_custom" ]]; then
    CLIP_CHECKPOINT="pretrained/daclip_ViT-B-32.pt"
    if [[ ! -f "${CLIP_CHECKPOINT}" ]]; then
      echo "❌ DACLiP checkpoint not found: ${CLIP_CHECKPOINT}" >&2
      echo "   Please download it as described in README.md" >&2
      exit 1
    fi
    
    python3 -m underwater_ir.teacher.export_pseudolabels \
      --input-root "${TRAIN_ROOT}/input" \
      --target-root "${TRAIN_ROOT}/target" \
      --output "${PSEUDO_ROOT}/train" \
      --clip-model "daclip_ViT-B-32" \
      --clip-checkpoint "${CLIP_CHECKPOINT}" \
      --use-crf \
      --num-workers "${WORKERS}"
  else
    echo "❌ Cannot use openai_clip with original export_pseudolabels.py" >&2
    echo "   Please create export_pseudolabels_v2.py or switch to daclip_custom model" >&2
    exit 1
  fi
fi

# Export pseudo-labels for validation (reference)
echo ""
echo "Exporting pseudo-labels for validation sets (reference)..."
for subset_dir in "${VAL_REF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
  subset="$(basename "${subset_dir}")"
  echo "  Processing: ${subset}"
  mkdir -p "${PSEUDO_ROOT}/testset_ref/${subset}"
  
  if [[ -f "underwater_ir/teacher/export_pseudolabels_v2.py" ]]; then
  python3 -m underwater_ir.teacher.export_pseudolabels_v2 \
    --config "${CONFIG_FILE}" \
    --input-root "${subset_dir}/input" \
    --target-root "${subset_dir}/target" \
    --output "${PSEUDO_ROOT}/testset_ref/${subset}" \
    --num-workers "${WORKERS}"
  else
    if [[ "${CLIP_MODEL}" == "daclip_custom" ]]; then
      python3 -m underwater_ir.teacher.export_pseudolabels \
        --input-root "${subset_dir}/input" \
        --target-root "${subset_dir}/target" \
        --output "${PSEUDO_ROOT}/testset_ref/${subset}" \
        --clip-model "daclip_ViT-B-32" \
        --clip-checkpoint "${CLIP_CHECKPOINT}" \
        --use-crf \
        --num-workers "${WORKERS}"
    fi
  fi
done

# Export pseudo-labels for validation (non-reference)
echo ""
echo "Exporting pseudo-labels for validation sets (non-reference)..."
for subset_dir in "${VAL_NONREF_ROOT}"/*; do
  [[ -d "${subset_dir}" ]] || continue
  subset="$(basename "${subset_dir}")"
  echo "  Processing: ${subset}"
  input_dir="${subset_dir}/input"
  [[ -d "${input_dir}" ]] || input_dir="${subset_dir}"
  mkdir -p "${PSEUDO_ROOT}/testset_nonref/${subset}"
  
  if [[ -f "underwater_ir/teacher/export_pseudolabels_v2.py" ]]; then
  python3 -m underwater_ir.teacher.export_pseudolabels_v2 \
    --config "${CONFIG_FILE}" \
    --input-root "${input_dir}" \
    --output "${PSEUDO_ROOT}/testset_nonref/${subset}" \
    --num-workers "${WORKERS}"
  else
    if [[ "${CLIP_MODEL}" == "daclip_custom" ]]; then
      python3 -m underwater_ir.teacher.export_pseudolabels \
        --input-root "${input_dir}" \
        --output "${PSEUDO_ROOT}/testset_nonref/${subset}" \
        --clip-model "daclip_ViT-B-32" \
        --clip-checkpoint "${CLIP_CHECKPOINT}" \
        --use-crf \
        --num-workers "${WORKERS}"
    fi
  fi
done

# Train student model
echo ""
echo "[5/5] Training student model..."
if [[ -f "underwater_ir/student/train_student_v2.py" ]]; then
  python3 -m underwater_ir.student.train_student_v2 \
    --config "${CONFIG_FILE}" \
    --train-root "${TRAIN_ROOT}" \
    --val-ref-root "${VAL_REF_ROOT}" \
    --val-nonref-root "${VAL_NONREF_ROOT}" \
    --pseudo-root "${PSEUDO_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --num-workers "${WORKERS}" \
    --save-path "${SAVE_PATH}"
else
  echo "⚠️  Using original train_student.py"
  python3 -m underwater_ir.student.train_student \
    --train-root "${TRAIN_ROOT}" \
    --val-ref-root "${VAL_REF_ROOT}" \
    --val-nonref-root "${VAL_NONREF_ROOT}" \
    --pseudo-root "${PSEUDO_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --num-workers "${WORKERS}" \
    --save-path "${SAVE_PATH}"
fi

echo ""
echo "================================================================================"
echo "✅ Training pipeline completed!"
echo "================================================================================"
echo "Model saved to: ${SAVE_PATH}"
echo "Pseudo-labels saved to: ${PSEUDO_ROOT}"
