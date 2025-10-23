#!/bin/bash
# Batch export pseudo-labels for all datasets (train + test sets)

echo "======================================"
echo "BATCH EXPORT - ALL DATASETS"
echo "======================================"
echo ""

# Configuration
DATA_BASE="${1:-Dataset}"
OUTPUT_BASE="${2:-pseudo-labels/siglip2}"
MODEL_NAME="google/siglip2-large-patch16-512"

echo "📁 Data base: $DATA_BASE"
echo "📁 Output base: $OUTPUT_BASE"
echo "🤖 Model: $MODEL_NAME"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_BASE" ]; then
    echo "❌ Data base directory not found: $DATA_BASE"
    echo ""
    echo "Please specify the correct path:"
    echo "  bash export_all_siglip2.sh /path/to/Dataset output_base"
    exit 1
fi

# Function to export a single dataset
export_dataset() {
    local data_path=$1
    local output_path=$2
    local dataset_name=$3
    
    if [ ! -d "$data_path" ]; then
        echo "⚠️  Skipping $dataset_name: directory not found at $data_path"
        return 1
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Exporting: $dataset_name"
    echo "   From: $data_path"
    echo "   To: $output_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python teacher/export_pseudolabels_siglip2.py \
        --data-root "$data_path" \
        --output-root "$output_path" \
        --model-name "$MODEL_NAME" \
        --device cuda \
        --batch-size 1
    
    if [ $? -eq 0 ]; then
        echo "✅ $dataset_name: Export successful"
        return 0
    else
        echo "❌ $dataset_name: Export failed"
        return 1
    fi
}

# Track success/failure
TOTAL=0
SUCCESS=0
FAILED=0

echo "Starting batch export..."
echo ""

# ============================================
# TRAINING SET
# ============================================
if [ -d "$DATA_BASE/train" ]; then
    TOTAL=$((TOTAL + 1))
    export_dataset "$DATA_BASE/train" "$OUTPUT_BASE/train" "Training Set"
    [ $? -eq 0 ] && SUCCESS=$((SUCCESS + 1)) || FAILED=$((FAILED + 1))
fi

# ============================================
# VALIDATION SET (if exists)
# ============================================
if [ -d "$DATA_BASE/val" ]; then
    TOTAL=$((TOTAL + 1))
    export_dataset "$DATA_BASE/val" "$OUTPUT_BASE/testset_ref" "Validation Set (testset_ref)"
    [ $? -eq 0 ] && SUCCESS=$((SUCCESS + 1)) || FAILED=$((FAILED + 1))
fi

# ============================================
# TEST SETS (non-reference)
# ============================================
# Common underwater test sets
TEST_DATASETS=(
    "test-EUVP-unpaired"
    "test-RUIE-unpaired"
    "test-UIEB-unpaired"
    "test-LSUI"
    "test-C60"
    "test-UCCS"
)

for test_set in "${TEST_DATASETS[@]}"; do
    if [ -d "$DATA_BASE/$test_set" ]; then
        TOTAL=$((TOTAL + 1))
        export_dataset "$DATA_BASE/$test_set" "$OUTPUT_BASE/testset_nonref/$test_set" "Test Set: $test_set"
        [ $? -eq 0 ] && SUCCESS=$((SUCCESS + 1)) || FAILED=$((FAILED + 1))
    fi
done

# ============================================
# PAIRED TEST SETS (if exists)
# ============================================
PAIRED_TEST_DATASETS=(
    "test-EUVP-paired"
    "test-RUIE-paired"
    "test-UIEB-paired"
)

for test_set in "${PAIRED_TEST_DATASETS[@]}"; do
    if [ -d "$DATA_BASE/$test_set" ]; then
        TOTAL=$((TOTAL + 1))
        export_dataset "$DATA_BASE/$test_set" "$OUTPUT_BASE/testset_ref/$test_set" "Paired Test Set: $test_set"
        [ $? -eq 0 ] && SUCCESS=$((SUCCESS + 1)) || FAILED=$((FAILED + 1))
    fi
done

# ============================================
# SUMMARY
# ============================================
echo ""
echo "======================================"
echo "BATCH EXPORT COMPLETE"
echo "======================================"
echo ""
echo "📊 Summary:"
echo "   Total datasets: $TOTAL"
echo "   ✅ Successful: $SUCCESS"
echo "   ❌ Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "⚠️  Some datasets failed to export. Check logs above."
    echo ""
fi

# ============================================
# VALIDATION
# ============================================
echo "Running validation on exported pseudo-labels..."
echo ""

# Validate training set
if [ -d "$OUTPUT_BASE/train" ]; then
    echo "🔍 Validating training set..."
    python validate_teacher_export.py --pseudo-root "$OUTPUT_BASE/train"
    echo ""
fi

# Validate first test set found
for test_set in "${TEST_DATASETS[@]}"; do
    if [ -d "$OUTPUT_BASE/testset_nonref/$test_set" ]; then
        echo "🔍 Validating test set: $test_set..."
        python validate_teacher_export.py --pseudo-root "$OUTPUT_BASE/testset_nonref/$test_set"
        echo ""
        break  # Only validate one test set as example
    fi
done

echo "======================================"
echo "NEXT STEPS"
echo "======================================"
echo ""
echo "1. Check validation output above - all masks should be in [0, 1]"
echo ""
echo "2. Update training config if needed:"
echo "   --pseudo-root $OUTPUT_BASE"
echo ""
echo "3. Start training:"
echo "   bash train_ddp.sh"
echo ""
echo "4. Monitor metrics - PSNR should be > 15 dB after first epoch"
echo ""

if [ $SUCCESS -eq $TOTAL ]; then
    exit 0
else
    exit 1
fi
