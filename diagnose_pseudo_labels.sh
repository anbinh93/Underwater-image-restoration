#!/bin/bash
# Script to diagnose and fix missing pseudo-labels issue

echo "======================================"
echo "PSEUDO-LABELS DIAGNOSTIC & FIX"
echo "======================================"
echo ""

# Check current directory
echo "1️⃣  Current directory:"
pwd
echo ""

# Look for pseudo-label directories
echo "2️⃣  Searching for pseudo-label directories..."
echo ""
find . -type d -name "*pseudo*" 2>/dev/null | head -20
echo ""

# Check common locations
echo "3️⃣  Checking common pseudo-label locations:"
locations=(
    "pseudo_labels_v2/train"
    "pseudo_labels/train"
    "outputs/pseudo_labels/train"
    "../pseudo_labels_v2/train"
    "~/pseudo_labels_v2/train"
    "/home/ec2-user/pseudo_labels_v2/train"
)

found=""
for loc in "${locations[@]}"; do
    expanded=$(eval echo "$loc")
    if [ -d "$expanded" ]; then
        echo "  ✅ Found: $expanded"
        count=$(find "$expanded" -type f \( -name "*.pt" -o -name "*.npy" \) 2>/dev/null | wc -l)
        echo "     Files: $count"
        found="$expanded"
    else
        echo "  ❌ Not found: $expanded"
    fi
done

echo ""

if [ -n "$found" ]; then
    echo "======================================"
    echo "✅ FOUND PSEUDO-LABELS AT: $found"
    echo "======================================"
    echo ""
    echo "Use this path in training:"
    echo "  --pseudo-root \"$found\""
    echo ""
    echo "Or update train_ddp.sh to use correct path"
else
    echo "======================================"
    echo "❌ NO PSEUDO-LABELS FOUND"
    echo "======================================"
    echo ""
    echo "You need to EXPORT pseudo-labels first!"
    echo ""
    echo "Steps to export pseudo-labels:"
    echo ""
    echo "Step 1: Verify you have teacher checkpoint"
    echo "  ls -lh teacher_checkpoint.pth  # or .pt"
    echo ""
    echo "Step 2: Run export script"
    echo "  # For V2 format (.npy files):"
    echo "  python underwater_ir/teacher/export_pseudolabels_v2.py \\"
    echo "    --teacher-ckpt path/to/teacher.pth \\"
    echo "    --data-root Dataset/train \\"
    echo "    --output-root pseudo_labels_v2/train"
    echo ""
    echo "  # Or for V1 format (.pt files):"
    echo "  python underwater_ir/teacher/export_pseudolabels.py \\"
    echo "    --teacher-ckpt path/to/teacher.pth \\"
    echo "    --data-root Dataset/train \\"
    echo "    --output-root pseudo_labels/train"
    echo ""
    echo "Step 3: Verify exported files"
    echo "  python check_pseudo_labels.py --pseudo-root pseudo_labels_v2/train"
    echo ""
    echo "Step 4: Start training with correct path"
    echo "  bash train_ddp.sh 8 2 20 128 128"
fi

echo ""
echo "======================================"
echo "QUICK FIXES"
echo "======================================"
echo ""

echo "Fix 1: If pseudo-labels are in different location"
echo "  # Update train_ddp.sh or use command line:"
echo "  python -m underwater_ir.student.train_student \\"
echo "    --pseudo-root /correct/path/to/pseudo_labels_v2 \\"
echo "    --train-root Dataset/train \\"
echo "    ..."
echo ""

echo "Fix 2: If you need to export from scratch"
echo "  # Make sure you have:"
echo "  #   1. Teacher model checkpoint"
echo "  #   2. Training dataset in Dataset/train/"
echo "  #   3. Enough disk space for pseudo-labels"
echo ""
echo "  # Then run export (example):"
echo "  python -m underwater_ir.teacher.export_pseudolabels_v2 \\"
echo "    --teacher-ckpt checkpoints/teacher_model.pth \\"
echo "    --data-root Dataset/train/input \\"
echo "    --output-root pseudo_labels_v2/train \\"
echo "    --batch-size 8 \\"
echo "    --device cuda"
echo ""

echo "Fix 3: Create symbolic link if pseudo-labels are elsewhere"
echo "  ln -s /actual/path/to/pseudo_labels_v2 ./pseudo_labels_v2"
echo ""

echo "======================================"
echo "For more help, check:"
echo "  - README.md for pseudo-label export instructions"
echo "  - validate_teacher_export.py to test teacher model"
echo "  - check_pseudo_labels.py to validate exported files"
echo "======================================"
