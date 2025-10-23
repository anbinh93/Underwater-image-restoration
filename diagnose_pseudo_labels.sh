#!/bin/bash
# Diagnostic script for pseudo-label issues

echo "======================================"
echo "PSEUDO-LABELS DIAGNOSTIC"
echo "======================================"
echo ""

# Primary location based on user's setup
PSEUDO_DIR="pseudo-labels/daclip/train"

echo "1️⃣  Checking primary location: $PSEUDO_DIR"
echo ""

if [ -d "$PSEUDO_DIR" ]; then
    echo "✅ Found pseudo-label directory: $PSEUDO_DIR"
    echo ""
    
    # Count files by type
    TOTAL_FILES=$(ls -1 "$PSEUDO_DIR" 2>/dev/null | wc -l)
    MASK_FILES=$(ls -1 "$PSEUDO_DIR"/*_masks.npy 2>/dev/null | wc -l)
    FEAT_FILES=$(ls -1 "$PSEUDO_DIR"/*_features.npy 2>/dev/null | wc -l)
    PROB_FILES=$(ls -1 "$PSEUDO_DIR"/*_probs.npy 2>/dev/null | wc -l)
    
    echo "📊 File counts:"
    echo "   Total files: $TOTAL_FILES"
    echo "   Mask files (*_masks.npy): $MASK_FILES"
    echo "   Feature files (*_features.npy): $FEAT_FILES"
    echo "   Probability files (*_probs.npy): $PROB_FILES"
    echo ""
    
    # Expected: each image should have 3 files (features, masks, probs)
    NUM_IMAGES=$MASK_FILES
    EXPECTED_TOTAL=$((NUM_IMAGES * 3))
    
    if [ $TOTAL_FILES -eq $EXPECTED_TOTAL ] && [ $MASK_FILES -eq $FEAT_FILES ] && [ $FEAT_FILES -eq $PROB_FILES ]; then
        echo "✅ File structure looks correct!"
        echo "   Found $NUM_IMAGES images with complete pseudo-labels"
    else
        echo "⚠️  File structure may be incomplete:"
        echo "   Expected $EXPECTED_TOTAL files (3 per image)"
        echo "   Found $TOTAL_FILES files"
    fi
    echo ""
    
    # Show sample files
    echo "📁 Sample files (first 15):"
    ls -lh "$PSEUDO_DIR" | head -n 16
    echo ""
    
    # Check if masks are all zeros (CRITICAL BUG CHECK)
    echo "🔬 Checking mask values (first 3 mask files)..."
    echo "   This checks if masks are all zeros (known bug in export script)"
    echo ""
    
    ZERO_MASKS=0
    CHECKED=0
    for mask_file in $(ls "$PSEUDO_DIR"/*_masks.npy 2>/dev/null | head -n 3); do
        echo "   📄 $(basename $mask_file)"
        python3 -c "
import numpy as np
try:
    mask = np.load('$mask_file')
    print(f'      Shape: {mask.shape}')
    print(f'      Min: {mask.min():.6f}, Max: {mask.max():.6f}, Mean: {mask.mean():.6f}')
    if mask.max() == 0.0:
        print('      ❌ WARNING: This mask is ALL ZEROS!')
        exit(1)
    else:
        print('      ✅ Mask has non-zero values')
        exit(0)
except Exception as e:
    print(f'      ❌ Error: {e}')
    exit(2)
" 2>&1
        result=$?
        if [ $result -eq 1 ]; then
            ZERO_MASKS=$((ZERO_MASKS + 1))
        fi
        CHECKED=$((CHECKED + 1))
        echo ""
    done
    
    if [ $ZERO_MASKS -gt 0 ]; then
        echo "======================================"
        echo "❌ CRITICAL BUG DETECTED!"
        echo "======================================"
        echo ""
        echo "Found $ZERO_MASKS zero masks out of $CHECKED checked!"
        echo ""
        echo "🐛 This is caused by a bug in export script:"
        echo "   teacher/export_pseudolabels_v2.py (or similar)"
        echo "   Line ~155-156 uses binary thresholding instead of softmax"
        echo ""
        echo "Current (BUGGY) code:"
        echo "   masks = (probs > threshold).float()  # threshold=0.5"
        echo ""
        echo "Should be:"
        echo "   masks = F.softmax(logits, dim=1)  # Use probabilities directly"
        echo ""
        echo "🔧 TO FIX:"
        echo "   1. Edit the export script to use softmax probabilities"
        echo "   2. Re-export ALL pseudo-labels"
        echo "   3. Resume training"
        echo ""
    else
        echo "======================================"
        echo "✅ MASKS LOOK GOOD!"
        echo "======================================"
        echo ""
        echo "All checked masks have non-zero values."
        echo "Your pseudo-labels should work for training."
        echo ""
    fi
    
else
    echo "❌ Pseudo-label directory not found: $PSEUDO_DIR"
    echo ""
    echo "🔍 Searching for alternative locations..."
    echo ""
    
    # Search in common alternative locations
    LOCATIONS=(
        "pseudo-labels/daclip/testset_ref"
        "pseudo-labels/daclip/val"
        "pseudo_labels/daclip/train"
        "pseudo_labels_v2/train"
        "../pseudo-labels/daclip/train"
        "/home/ec2-user/SageMaker/Underwater-image-restoration/pseudo-labels/daclip/train"
    )
    
    FOUND=0
    for loc in "${LOCATIONS[@]}"; do
        if [ -d "$loc" ]; then
            count=$(ls -1 "$loc"/*_masks.npy 2>/dev/null | wc -l)
            echo "   ✅ Found alternative: $loc"
            echo "      Mask files: $count"
            FOUND=1
        fi
    done
    
    if [ $FOUND -eq 0 ]; then
        echo "   ❌ No pseudo-label directories found anywhere"
        echo ""
        echo "======================================"
        echo "NEED TO EXPORT PSEUDO-LABELS"
        echo "======================================"
        echo ""
        echo "You need to run the export script first!"
        echo ""
        echo "Example command:"
        echo "  python teacher/export_pseudolabels_v2.py \\"
        echo "    --checkpoint path/to/daclip_teacher.pth \\"
        echo "    --data-root Dataset/train \\"
        echo "    --output-root pseudo-labels/daclip/train"
        echo ""
    else
        echo ""
        echo "💡 TIP: Update your training script to use the correct path"
        echo "   or move/symlink the pseudo-labels to: $PSEUDO_DIR"
        echo ""
    fi
fi

echo "======================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================"
