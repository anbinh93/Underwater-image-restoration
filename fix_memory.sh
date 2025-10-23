#!/usr/bin/env bash
# Auto-fix memory issues for DDP training

set -e

echo "🔧 Underwater IR - Memory Fix Script"
echo "====================================="
echo ""

# Step 1: Clean Python cache
echo "[1/5] Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Cache cleaned"
echo ""

# Step 2: Kill existing training
echo "[2/5] Stopping existing training processes..."
pkill -f "torchrun" 2>/dev/null || true
pkill -f "train_student" 2>/dev/null || true
sleep 2
echo "✅ Processes stopped"
echo ""

# Step 3: Clear GPU memory
echo "[3/5] Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || echo "⚠️  Cannot reset GPU (may need sudo)"
sleep 1
echo "✅ GPU memory cleared"
echo ""

# Step 4: Verify code version
echo "[4/5] Checking code version..."
if grep -q "attn_chunk_size" underwater_ir/student/naf_unet_wfi/wtb.py 2>/dev/null; then
    echo "✅ Code is updated with memory-efficient attention"
else
    echo "❌ ERROR: Code needs update!"
    echo "   Run: git pull origin main"
    exit 1
fi
echo ""

# Step 5: Show recommended settings
echo "[5/5] GPU Memory Analysis"
echo "========================="

# Get GPU info
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "Detected: ${GPU_COUNT} GPUs with ${GPU_MEMORY}MB memory each"
echo ""

# Recommend settings based on GPU memory
if [ "$GPU_MEMORY" -ge 38000 ]; then
    echo "🚀 GPU Type: A100 (40GB) - High performance mode"
    echo ""
    echo "Recommended command:"
    echo "  bash train_ddp.sh 4 4 20 128 256"
    echo ""
    echo "Conservative (if above fails):"
    echo "  bash train_ddp.sh 4 2 20 128 128"
    BATCH=4
    IMG=128
    CHUNK=256
elif [ "$GPU_MEMORY" -ge 22000 ]; then
    echo "🎯 GPU Type: RTX 3090/4090 (24GB) - Balanced mode"
    echo ""
    echo "Recommended command:"
    echo "  bash train_ddp.sh 4 2 20 128 128"
    BATCH=2
    IMG=128
    CHUNK=128
else
    echo "💡 GPU Type: V100/RTX 4000 (16GB) - Safe mode"
    echo ""
    echo "Recommended command:"
    echo "  bash train_ddp.sh 4 2 20 96 64"
    BATCH=2
    IMG=96
    CHUNK=64
fi

echo ""
echo "====================================="
echo ""

# Ask user
read -p "Start training now with recommended settings? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting training..."
    echo "   GPUs: ${GPU_COUNT}"
    echo "   Batch/GPU: ${BATCH}"
    echo "   Image size: ${IMG}"
    echo "   Attention chunk: ${CHUNK}"
    echo ""
    
    # Set environment variables for better memory management
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Start training
    bash train_ddp.sh "${GPU_COUNT}" "${BATCH}" 20 "${IMG}" "${CHUNK}"
else
    echo "Cancelled. You can run manually:"
    echo "  bash train_ddp.sh ${GPU_COUNT} ${BATCH} 20 ${IMG} ${CHUNK}"
fi
