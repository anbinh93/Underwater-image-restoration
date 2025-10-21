#!/usr/bin/env bash
# Debug script to check dataset and pseudo-label structure

set -euo pipefail

echo "================================================================================"
echo "Dataset and Pseudo-Label Structure Debug"
echo "================================================================================"

# Check training dataset
echo ""
echo "[1] Training Dataset:"
echo "--------------------------------------------------------------------------------"
TRAIN_ROOT="Dataset/train"
if [[ -d "${TRAIN_ROOT}" ]]; then
    echo "✅ ${TRAIN_ROOT} exists"
    
    if [[ -d "${TRAIN_ROOT}/input" ]]; then
        input_count=$(find "${TRAIN_ROOT}/input" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
        echo "  Input images: ${input_count}"
        if [[ $input_count -gt 0 ]]; then
            echo "  Sample files:"
            find "${TRAIN_ROOT}/input" -type f \( -name "*.jpg" -o -name "*.png" \) | head -3
        fi
    else
        echo "❌ ${TRAIN_ROOT}/input does not exist"
    fi
    
    if [[ -d "${TRAIN_ROOT}/target" ]]; then
        target_count=$(find "${TRAIN_ROOT}/target" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
        echo "  Target images: ${target_count}"
    else
        echo "❌ ${TRAIN_ROOT}/target does not exist"
    fi
else
    echo "❌ ${TRAIN_ROOT} does not exist"
fi

# Check validation datasets
echo ""
echo "[2] Reference Validation Datasets:"
echo "--------------------------------------------------------------------------------"
VAL_REF_ROOT="Dataset/testset(ref)"
if [[ -d "${VAL_REF_ROOT}" ]]; then
    echo "✅ ${VAL_REF_ROOT} exists"
    for subset_dir in "${VAL_REF_ROOT}"/*; do
        if [[ -d "${subset_dir}" ]]; then
            subset="$(basename "${subset_dir}")"
            input_count=0
            target_count=0
            if [[ -d "${subset_dir}/input" ]]; then
                input_count=$(find "${subset_dir}/input" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
            fi
            if [[ -d "${subset_dir}/target" ]]; then
                target_count=$(find "${subset_dir}/target" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
            fi
            echo "  ${subset}: input=${input_count}, target=${target_count}"
        fi
    done
else
    echo "⚠️  ${VAL_REF_ROOT} does not exist"
fi

echo ""
echo "[3] Non-Reference Validation Datasets:"
echo "--------------------------------------------------------------------------------"
VAL_NONREF_ROOT="Dataset/testset(non-ref)"
if [[ -d "${VAL_NONREF_ROOT}" ]]; then
    echo "✅ ${VAL_NONREF_ROOT} exists"
    for subset_dir in "${VAL_NONREF_ROOT}"/*; do
        if [[ -d "${subset_dir}" ]]; then
            subset="$(basename "${subset_dir}")"
            input_dir="${subset_dir}/input"
            [[ -d "${input_dir}" ]] || input_dir="${subset_dir}"
            input_count=$(find "${input_dir}" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
            echo "  ${subset}: input=${input_count}"
        fi
    done
else
    echo "⚠️  ${VAL_NONREF_ROOT} does not exist"
fi

# Check pseudo-labels
echo ""
echo "[4] Pseudo-Labels:"
echo "--------------------------------------------------------------------------------"
PSEUDO_ROOT="pseudo-labels/daclip"
if [[ -d "${PSEUDO_ROOT}" ]]; then
    echo "✅ ${PSEUDO_ROOT} exists"
    
    # Training
    if [[ -d "${PSEUDO_ROOT}/train" ]]; then
        pt_count=$(find "${PSEUDO_ROOT}/train" -name "*.pt" -type f | wc -l)
        echo "  train: ${pt_count} .pt files"
        if [[ $pt_count -gt 0 ]]; then
            echo "    Sample files:"
            find "${PSEUDO_ROOT}/train" -name "*.pt" -type f | head -3
        fi
    else
        echo "  ❌ train directory missing"
    fi
    
    # Reference validation
    if [[ -d "${PSEUDO_ROOT}/testset_ref" ]]; then
        for subset_dir in "${PSEUDO_ROOT}/testset_ref"/*; do
            if [[ -d "${subset_dir}" ]]; then
                subset="$(basename "${subset_dir}")"
                pt_count=$(find "${subset_dir}" -name "*.pt" -type f | wc -l)
                echo "  testset_ref/${subset}: ${pt_count} .pt files"
            fi
        done
    else
        echo "  ⚠️  testset_ref directory missing"
    fi
    
    # Non-reference validation
    if [[ -d "${PSEUDO_ROOT}/testset_nonref" ]]; then
        for subset_dir in "${PSEUDO_ROOT}/testset_nonref"/*; do
            if [[ -d "${subset_dir}" ]]; then
                subset="$(basename "${subset_dir}")"
                pt_count=$(find "${subset_dir}" -name "*.pt" -type f | wc -l)
                echo "  testset_nonref/${subset}: ${pt_count} .pt files"
            fi
        done
    else
        echo "  ⚠️  testset_nonref directory missing"
    fi
else
    echo "❌ ${PSEUDO_ROOT} does not exist"
fi

echo ""
echo "================================================================================"
echo "Debug Complete"
echo "================================================================================"
