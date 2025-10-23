#!/bin/bash
# Quick script to detect dataset structure

echo "======================================"
echo "DATASET STRUCTURE DETECTION"
echo "======================================"
echo ""

DATA_BASE="${1:-Dataset}"

if [ ! -d "$DATA_BASE" ]; then
    echo "❌ Dataset directory not found: $DATA_BASE"
    exit 1
fi

echo "📁 Checking: $DATA_BASE"
echo ""

# Check for training set
echo "Training Data:"
if [ -d "$DATA_BASE/train" ]; then
    count=$(find "$DATA_BASE/train" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  ✅ $DATA_BASE/train/ ($count images)"
else
    echo "  ❌ $DATA_BASE/train/ (not found)"
fi
echo ""

# Check for validation set
echo "Validation Data:"
if [ -d "$DATA_BASE/val" ]; then
    count=$(find "$DATA_BASE/val" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  ✅ $DATA_BASE/val/ ($count images)"
fi
if [ -d "$DATA_BASE/testset(ref)" ]; then
    echo "  ✅ $DATA_BASE/testset(ref)/"
    for subset in "$DATA_BASE/testset(ref)"/*; do
        if [ -d "$subset" ]; then
            name=$(basename "$subset")
            count=$(find "$subset" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
            echo "     - $name ($count images)"
        fi
    done
fi
if [ -d "$DATA_BASE/testset_ref" ]; then
    echo "  ✅ $DATA_BASE/testset_ref/"
    for subset in "$DATA_BASE/testset_ref"/*; do
        if [ -d "$subset" ]; then
            name=$(basename "$subset")
            count=$(find "$subset" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
            echo "     - $name ($count images)"
        fi
    done
fi
echo ""

# Check for test sets (non-reference)
echo "Test Data (Non-Reference):"
found_tests=0

# Check direct under Dataset/
TEST_SETS=("test-EUVP-unpaired" "test-RUIE-unpaired" "test-UIEB-unpaired" "test-LSUI" "test-C60" "test-UCCS")
for test_set in "${TEST_SETS[@]}"; do
    if [ -d "$DATA_BASE/$test_set" ]; then
        count=$(find "$DATA_BASE/$test_set" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "  ✅ $DATA_BASE/$test_set/ ($count images)"
        found_tests=$((found_tests + 1))
    fi
done

# Check under testset(non-ref)/
if [ -d "$DATA_BASE/testset(non-ref)" ]; then
    echo "  ✅ $DATA_BASE/testset(non-ref)/"
    for subset in "$DATA_BASE/testset(non-ref)"/*; do
        if [ -d "$subset" ]; then
            name=$(basename "$subset")
            count=$(find "$subset" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
            echo "     - $name ($count images)"
            found_tests=$((found_tests + 1))
        fi
    done
fi

# Check under testset_nonref/
if [ -d "$DATA_BASE/testset_nonref" ]; then
    echo "  ✅ $DATA_BASE/testset_nonref/"
    for subset in "$DATA_BASE/testset_nonref"/*; do
        if [ -d "$subset" ]; then
            name=$(basename "$subset")
            count=$(find "$subset" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
            echo "     - $name ($count images)"
            found_tests=$((found_tests + 1))
        fi
    done
fi

if [ $found_tests -eq 0 ]; then
    echo "  ❌ No test sets found"
fi

echo ""
echo "======================================"
echo "SUMMARY"
echo "======================================"
echo ""

# Determine structure
if [ -d "$DATA_BASE/testset(non-ref)" ] || [ -d "$DATA_BASE/testset(ref)" ]; then
    echo "📂 Dataset Structure: OLD FORMAT"
    echo ""
    echo "   Dataset/"
    echo "   ├── train/"
    echo "   ├── testset(ref)/"
    echo "   │   └── <test-name>/"
    echo "   └── testset(non-ref)/"
    echo "       └── <test-name>/"
    echo ""
    echo "To export pseudo-labels:"
    echo "  bash export_all_siglip2.sh Dataset pseudo-labels/siglip2"
    echo ""
elif [ -d "$DATA_BASE/testset_nonref" ] || [ -d "$DATA_BASE/testset_ref" ]; then
    echo "📂 Dataset Structure: NEW FORMAT (underscore)"
    echo ""
    echo "   Dataset/"
    echo "   ├── train/"
    echo "   ├── testset_ref/"
    echo "   │   └── <test-name>/"
    echo "   └── testset_nonref/"
    echo "       └── <test-name>/"
    echo ""
    echo "To export pseudo-labels:"
    echo "  bash export_all_siglip2.sh Dataset pseudo-labels/siglip2"
    echo ""
else
    echo "📂 Dataset Structure: FLAT FORMAT"
    echo ""
    echo "   Dataset/"
    echo "   ├── train/"
    echo "   ├── test-EUVP-unpaired/"
    echo "   ├── test-RUIE-unpaired/"
    echo "   └── test-UIEB-unpaired/"
    echo ""
    echo "To export pseudo-labels:"
    echo "  bash export_all_siglip2.sh Dataset pseudo-labels/siglip2"
    echo ""
fi

echo "✅ All formats are supported by the export scripts!"
echo ""
