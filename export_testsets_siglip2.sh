#!/bin/bash
# Quick export for test sets only (non-reference)
# Use this if you already have training set pseudo-labels

echo "======================================"
echo "EXPORT TEST SETS (NON-REFERENCE)"
echo "======================================"
echo ""

DATA_BASE="${1:-Dataset}"
OUTPUT_BASE="${2:-pseudo-labels/siglip2/testset_nonref}"
MODEL_NAME="google/siglip2-large-patch16-512"

echo "📁 Data base: $DATA_BASE"
echo "📁 Output base: $OUTPUT_BASE"
echo ""

# Test sets to export
TEST_SETS=(
    "test-EUVP-unpaired"
    "test-RUIE-unpaired"
    "test-UIEB-unpaired"
)

SUCCESS=0
TOTAL=0

for test_set in "${TEST_SETS[@]}"; do
    # Try multiple possible locations
    data_path=""
    
    # Option 1: Direct under Dataset/ (e.g., Dataset/test-EUVP-unpaired/)
    if [ -d "$DATA_BASE/$test_set" ]; then
        data_path="$DATA_BASE/$test_set"
    # Option 2: Under testset(non-ref)/ (e.g., Dataset/testset(non-ref)/test-EUVP-unpaired/)
    elif [ -d "$DATA_BASE/testset(non-ref)/$test_set" ]; then
        data_path="$DATA_BASE/testset(non-ref)/$test_set"
    # Option 3: Under testset_nonref/ (e.g., Dataset/testset_nonref/test-EUVP-unpaired/)
    elif [ -d "$DATA_BASE/testset_nonref/$test_set" ]; then
        data_path="$DATA_BASE/testset_nonref/$test_set"
    fi
    
    if [ -z "$data_path" ]; then
        echo "⚠️  Skipping $test_set: not found in:"
        echo "     - $DATA_BASE/$test_set"
        echo "     - $DATA_BASE/testset(non-ref)/$test_set"
        echo "     - $DATA_BASE/testset_nonref/$test_set"
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    output_path="$OUTPUT_BASE/$test_set"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Exporting: $test_set"
    echo "   From: $data_path"
    echo "   To: $output_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python teacher/export_pseudolabels_siglip2.py \
        --data-root "$data_path" \
        --output-root "$output_path" \
        --model-name "$MODEL_NAME" \
        --device cuda \
        --batch-size 1
    
    if [ $? -eq 0 ]; then
        echo "✅ $test_set: Success"
        SUCCESS=$((SUCCESS + 1))
        
        # Quick validation
        sample_mask=$(find "$output_path" -name "*_masks.npy" | head -n 1)
        if [ -f "$sample_mask" ]; then
            python3 -c "
import numpy as np
mask = np.load('$sample_mask')
print(f'   Sample: {mask.shape}, range=[{mask.min():.4f}, {mask.max():.4f}]')
if mask.max() > 1.0:
    print('   ⚠️  WARNING: Masks exceed 1.0!')
elif mask.max() == 0:
    print('   ⚠️  WARNING: Masks are zero!')
else:
    print('   ✅ Masks look good')
" 2>/dev/null
        fi
    else
        echo "❌ $test_set: Failed"
    fi
done

echo ""
echo "======================================"
echo "SUMMARY"
echo "======================================"
echo "Exported $SUCCESS / $TOTAL test sets"
echo ""

if [ $SUCCESS -eq $TOTAL ] && [ $TOTAL -gt 0 ]; then
    echo "✅ All test sets exported successfully!"
    echo ""
    echo "Training can now evaluate on these test sets."
    exit 0
else
    echo "⚠️  Some test sets were not exported."
    echo "   Check if the dataset directories exist."
    exit 1
fi
