#!/bin/bash
# URGENT: Run this on the server to fix training!

echo "======================================"
echo "FIXING DDP TRAINING ISSUES"
echo "======================================"
echo ""

cd /home/ec2-user/SageMaker/Underwater-image-restoration

echo "1️⃣  Stopping any running training..."
pkill -9 -f train_student || true
pkill -9 -f torchrun || true
sleep 2

echo "2️⃣  Fetching latest code..."
git fetch origin

echo "3️⃣  Showing what will change..."
git log HEAD..origin/main --oneline

echo ""
echo "4️⃣  Applying fixes (hard reset)..."
git reset --hard origin/main

echo ""
echo "5️⃣  Verifying all fixes are applied..."
python3 verify_ddp_fixes.py

echo ""
if [ $? -eq 0 ]; then
    echo "======================================"
    echo "✅ ALL FIXES APPLIED SUCCESSFULLY!"
    echo "======================================"
    echo ""
    echo "Start training now:"
    echo "  bash train_ddp.sh 8 2 20 128 128"
    echo ""
else
    echo "======================================"
    echo "❌ VERIFICATION FAILED!"
    echo "======================================"
    echo ""
    echo "Contact support or check manually."
fi
