#!/usr/bin/env bash
# Validate dataset structure and script configuration

set -euo pipefail

echo "================================================================================"
echo "Validating Dataset Structure and Script Configuration"
echo "================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

errors=0
warnings=0

# Function to check directory
check_dir() {
    local dir="$1"
    local description="$2"
    if [[ -d "$dir" ]]; then
        echo -e "${GREEN}✓${NC} Found: $description ($dir)"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $description ($dir)"
        ((errors++))
        return 1
    fi
}

# Function to check file
check_file() {
    local file="$1"
    local description="$2"
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} Found: $description ($file)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Optional: $description ($file)"
        ((warnings++))
        return 1
    fi
}

echo ""
echo "Checking Python modules..."
echo "--------------------------------------------------------------------------------"

# Check Python modules
python test_imports.py
import_result=$?

if [[ $import_result -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} All Python imports successful"
else
    echo -e "${RED}✗${NC} Python import errors detected"
    ((errors++))
fi

echo ""
echo "Checking dataset directories..."
echo "--------------------------------------------------------------------------------"

# Check training data
check_dir "Dataset" "Dataset root"
check_dir "Dataset/train" "Training dataset"
check_dir "Dataset/train/input" "Training input images"
check_dir "Dataset/train/target" "Training target images"

echo ""
echo "Checking validation datasets (reference)..."
echo "--------------------------------------------------------------------------------"

# Check reference validation sets
if check_dir "Dataset/testset(ref)" "Reference test sets"; then
    for subset in "Dataset/testset(ref)"/*; do
        if [[ -d "$subset" ]]; then
            subset_name=$(basename "$subset")
            check_dir "$subset/input" "  → $subset_name input"
            check_dir "$subset/target" "  → $subset_name target"
        fi
    done
fi

echo ""
echo "Checking validation datasets (non-reference)..."
echo "--------------------------------------------------------------------------------"

# Check non-reference validation sets
if check_dir "Dataset/testset(non-ref)" "Non-reference test sets"; then
    for subset in "Dataset/testset(non-ref)"/*; do
        if [[ -d "$subset" ]]; then
            subset_name=$(basename "$subset")
            if [[ -d "$subset/input" ]]; then
                check_dir "$subset/input" "  → $subset_name input"
            else
                check_dir "$subset" "  → $subset_name (no subfolder)"
            fi
        fi
    done
fi

echo ""
echo "Checking output directories..."
echo "--------------------------------------------------------------------------------"

# Check/create output directories
mkdir -p pseudo-labels experiments
check_dir "pseudo-labels" "Pseudo-labels directory (created if missing)"
check_dir "experiments" "Experiments directory (created if missing)"

echo ""
echo "Checking configuration files..."
echo "--------------------------------------------------------------------------------"

check_file "run_clip_training.sh" "Training script"
check_file "requirements.txt" "Python requirements"
check_file "prompts/degradation_prompts.json" "Degradation prompts"

echo ""
echo "================================================================================"
echo "Validation Summary"
echo "================================================================================"

if [[ $errors -eq 0 ]]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [[ $warnings -gt 0 ]]; then
        echo -e "${YELLOW}⚠ $warnings warning(s) (optional items missing)${NC}"
    fi
    echo ""
    echo "You can now run the training script:"
    echo "  bash run_clip_training.sh"
    exit 0
else
    echo -e "${RED}✗ $errors error(s) found!${NC}"
    if [[ $warnings -gt 0 ]]; then
        echo -e "${YELLOW}⚠ $warnings warning(s)${NC}"
    fi
    echo ""
    echo "Please fix the errors above before running the training script."
    exit 1
fi
