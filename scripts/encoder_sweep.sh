#!/usr/bin/env bash
# Compare different VLM encoders (CLIP2, BERT, SigLIP2, etc.) using the DACLiP teacher + student pipeline.
# Usage:
#   scripts/encoder_sweep.sh \
#       --config legacy/third_party/universal-image-restoration/config/daclip-sde/options/train.yml \
#       --dataset train \
#       --pseudo-root pseudo-labels \
#       --output-root experiments/encoder_sweep \
#       --epochs 10
#
# Edit the ENCODER_MATRIX below to add/remove encoders or provide adapter-specific overrides.
set -euo pipefail

CONFIG=""
DATASET="train"
TEST_DATASET=""
PSEUDO_ROOT="pseudo-labels"
OUTPUT_ROOT="experiments/encoder_sweep"
TRAIN_ROOT="Dataset/train"
VAL_REF_ROOT="Dataset/testset(ref)"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
EPOCHS=10
BATCH_SIZE=""
NUM_WORKERS=""
EXTRA_EXPORT_ARGS=()
EXTRA_TRAIN_ARGS=()
SKIP_EXPORT=0
SKIP_TRAIN=0

declare -a SELECTED_ENCODERS=()

usage() {
  cat <<'EOF'
Run teacher pseudo-label export + student training for multiple VLM encoders.

Options:
  --config <path>        YAML options file (default: required)
  --dataset <name>       Dataset key for training (default: train)
  --test-dataset <name>  Optional dataset key dedicated to pseudo-label export (default: dataset)
  --pseudo-root <dir>    Directory to store pseudo-labels (default: pseudo-labels)
  --output-root <dir>    Directory to store trained checkpoints (default: experiments/encoder_sweep)
  --train-root <path>    Paired training dataset root (default: Dataset/train)
  --val-ref-root <path>  Reference benchmark root (default: Dataset/testset(ref))
  --val-nonref-root <path> Non-reference benchmark root (default: Dataset/testset(non-ref))
  --epochs <int>         Training epochs for student (default: 10)
  --batch-size <int>     Override batch size passed to student/train_student.py
  --num-workers <int>    Override dataloader workers for pseudo export and student training
  --encoder <name>       Limit sweep to a specific encoder entry (can be provided multiple times)
  --skip-export          Do not re-run teacher export (expects existing pseudo-labels)
  --skip-train           Do not run student training (e.g., export only)
  --extra-export <args>  Additional args forwarded to underwater_ir/teacher/export_pseudolabels.py (repeatable)
  --extra-train <args>   Additional args forwarded to underwater_ir/student/train_student.py (repeatable)
  -h, --help             Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --test-dataset) TEST_DATASET="$2"; shift 2;;
    --pseudo-root) PSEUDO_ROOT="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --train-root) TRAIN_ROOT="$2"; shift 2;;
    --val-ref-root) VAL_REF_ROOT="$2"; shift 2;;
    --val-nonref-root) VAL_NONREF_ROOT="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --encoder) SELECTED_ENCODERS+=("$2"); shift 2;;
    --skip-export) SKIP_EXPORT=1; shift;;
    --skip-train) SKIP_TRAIN=1; shift;;
    --extra-export) EXTRA_EXPORT_ARGS+=("$2"); shift 2;;
    --extra-train) EXTRA_TRAIN_ARGS+=("$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Error: --config is required." >&2
  exit 1
fi

TEST_DATASET="${TEST_DATASET:-$DATASET}"

# Encoder entries: name|clip_model|tag|export_args|train_args
# - name: human-readable label (used in logging)
# - clip_model: identifier passed to --clip-model (for pure OpenCLIP stacks)
# - tag: folder name for pseudo labels + checkpoint
# - export_args: extra CLI args appended when exporting
# - train_args: extra CLI args appended when training
ENCODER_MATRIX=(
  "CLIP2|clip_ViT-L-14|clip2||"
  "DACLiP|daclip_ViT-B-32|daclip||"
  "SigLIP2|siglip_So400m_patch16-384|siglip2||"
  # To evaluate a custom adapter (e.g., CLIP image + BERT text), add an entry like:
  # \"BERT-Adapter|daclip_ViT-B-32|bert_adapter|--adapter-config configs/encoders/bert_adapter.json|\"
)

should_run_encoder() {
  local name="$1"
  if [[ ${#SELECTED_ENCODERS[@]} -eq 0 ]]; then
    return 0
  fi
  for sel in "${SELECTED_ENCODERS[@]}"; do
    if [[ "${sel}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

python_exec() {
  local cmd=("$@")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

mkdir -p "${PSEUDO_ROOT}" "${OUTPUT_ROOT}"

for entry in "${ENCODER_MATRIX[@]}"; do
  IFS="|" read -r name clip_model tag export_overrides train_overrides <<< "${entry}"
  if ! should_run_encoder "${name}"; then
    continue
  fi

  pseudo_dir="${PSEUDO_ROOT}/${tag}"
  model_path="${OUTPUT_ROOT}/${tag}.pt"

  echo "=== Encoder: ${name} (${clip_model}) => tag=${tag} ==="

  if [[ ${SKIP_EXPORT} -eq 0 ]]; then
    mkdir -p "${pseudo_dir}/train"
    export_args=(
      python underwater_ir/teacher/export_pseudolabels.py
      --config "${CONFIG}"
      --dataset "${TEST_DATASET}"
      --output "${pseudo_dir}/train"
      --clip-model "${clip_model}"
    )
    if [[ -n "${NUM_WORKERS}" ]]; then
      export_args+=(--num-workers "${NUM_WORKERS}")
    fi
    if [[ -n "${export_overrides}" ]]; then
      read -r -a tmp <<< "${export_overrides}"
      export_args+=("${tmp[@]}")
    fi
    export_args+=("${EXTRA_EXPORT_ARGS[@]}")
    python_exec "${export_args[@]}"
  else
    echo "[skip-export] Using existing pseudo labels in ${pseudo_dir}"
  fi

  if [[ ${SKIP_TRAIN} -eq 0 ]]; then
    train_args=(
      python underwater_ir/student/train_student.py
      --train-root "${TRAIN_ROOT}"
      --val-ref-root "${VAL_REF_ROOT}"
      --val-nonref-root "${VAL_NONREF_ROOT}"
      --pseudo-root "${pseudo_dir}"
      --epochs "${EPOCHS}"
      --save-path "${model_path}"
    )
    if [[ -n "${BATCH_SIZE}" ]]; then
      train_args+=(--batch-size "${BATCH_SIZE}")
    fi
    if [[ -n "${NUM_WORKERS}" ]]; then
      train_args+=(--num-workers "${NUM_WORKERS}")
    fi
    if [[ -n "${train_overrides}" ]]; then
      read -r -a tmp <<< "${train_overrides}"
      train_args+=("${tmp[@]}")
    fi
    train_args+=("${EXTRA_TRAIN_ARGS[@]}")
    python_exec "${train_args[@]}"
  else
    echo "[skip-train] Not running student training for ${name}."
  fi

  echo "=== Completed encoder: ${name} ==="
done
