#!/bin/bash
# Export pseudo-labels using SigLIP v2 instead of DACLiP

echo "======================================"
echo "SIGLIP V2 PSEUDO-LABEL EXPORT"
echo "======================================"
echo ""

# Configuration
DATA_ROOT="${1:-Dataset/train}"
OUTPUT_ROOT="${2:-pseudo-labels/siglip2/train}"
MODEL_NAME="google/siglip2-large-patch16-512"

echo "📁 Data root: $DATA_ROOT"
echo "📁 Output root: $OUTPUT_ROOT"
echo "🤖 Model: $MODEL_NAME"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Data directory not found: $DATA_ROOT"
    echo ""
    echo "Please specify the correct path:"
    echo "  bash export_siglip2.sh /path/to/images output_dir"
    exit 1
fi

# Install required packages if needed
echo "Checking dependencies..."
pip show transformers > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing transformers..."
    pip install transformers -q
fi

pip show torch > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  PyTorch not found! Please install PyTorch first."
    exit 1
fi

echo ""
echo "Starting export..."
echo ""

# Run export
python teacher/export_pseudolabels_siglip2.py \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --model-name "$MODEL_NAME" \
    --device cuda \
    --batch-size 1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ EXPORT SUCCESSFUL!"
    echo "======================================"
    echo ""
    echo "Pseudo-labels saved to: $OUTPUT_ROOT"
    echo ""
    echo "Next steps:"
    echo "  1. Validate pseudo-labels:"
    echo "     bash diagnose_pseudo_labels.sh"
    echo ""
    echo "  2. Update training config to use new pseudo-labels:"
    echo "     --pseudo-root $OUTPUT_ROOT"
    echo ""
    echo "  3. Start training:"
    echo "     bash train_ddp.sh"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ EXPORT FAILED!"
    echo "======================================"
    echo ""
    echo "Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - Out of GPU memory → Use smaller model or batch size"
    echo "  - transformers library not installed → pip install transformers"
    echo "  - Invalid data path → Check $DATA_ROOT exists"
    echo ""
    exit $EXIT_CODE
fi
