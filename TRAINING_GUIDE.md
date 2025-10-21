# Running the Training Pipeline

## Quick Start

### 1. Validate Setup

First, ensure all modules and datasets are properly configured:

```bash
# Make scripts executable
chmod +x validate_setup.sh run_clip_training.sh

# Run validation
bash validate_setup.sh
```

This will check:
- ✅ Python module imports
- ✅ Dataset directory structure
- ✅ Required files and configurations

### 2. Run Training

Once validation passes, start the full training pipeline:

```bash
bash run_clip_training.sh
```

## What the Training Script Does

### Stage 1: Export Pseudo-Labels

Generates pseudo-labels using the VLM teacher (CLIP) for:

1. **Training set** (`Dataset/train`)
   - Input: `Dataset/train/input/*.jpg`
   - Target: `Dataset/train/target/*.jpg`
   - Output: `pseudo-labels/clip_hf/train/*.pt`

2. **Reference test sets** (`Dataset/testset(ref)/*/`)
   - Each subset with `input/` and `target/` folders
   - Output: `pseudo-labels/clip_hf/testset_ref/{subset}/*.pt`

3. **Non-reference test sets** (`Dataset/testset(non-ref)/*/`)
   - Each subset with only `input/` or direct images
   - Output: `pseudo-labels/clip_hf/testset_nonref/{subset}/*.pt`

### Stage 2: Train Student Model

Trains the NAFNet-WFI-Gate student model with:
- **Architecture**: WFI-Gate with multi-scale DW conv + UNet skip concatenation
- **Conditioning**: Degradation code (z_d) + masks from VLM teacher
- **Losses**: 
  - Reconstruction (Charbonnier + SSIM)
  - Frequency (HF + LF amplitude)
  - Perceptual (VGG features)
  - Distillation (KL divergence from teacher)
- **Evaluation**: Automatic benchmarking on reference (PSNR/SSIM) and non-reference (UIQM/UCIQE) sets

Output: `experiments/clip_hf_student.pt`

## Dataset Structure

Required directory layout:

```
Dataset/
├── train/
│   ├── input/          # Low-quality underwater images
│   └── target/         # Ground-truth restored images
├── testset(ref)/       # Reference benchmarks (with GT)
│   ├── UIEB/
│   │   ├── input/
│   │   └── target/
│   ├── EUVP/
│   │   ├── input/
│   │   └── target/
│   └── ...
└── testset(non-ref)/   # Non-reference benchmarks (no GT)
    ├── real_world/
    │   └── input/      # Or directly in folder
    └── ...
```

## Configuration

Edit `run_clip_training.sh` to customize:

```bash
HF_MODEL="hf-hub:openai/clip-vit-base-patch32"  # CLIP model
TRAIN_ROOT="Dataset/train"                       # Training data
VAL_REF_ROOT="Dataset/testset(ref)"             # Reference validation
VAL_NONREF_ROOT="Dataset/testset(non-ref)"      # Non-reference validation
PSEUDO_ROOT="pseudo-labels/clip_hf"             # Output pseudo-labels
SAVE_PATH="experiments/clip_hf_student.pt"      # Model checkpoint
EPOCHS=20                                        # Training epochs
BATCH=4                                          # Batch size
WORKERS=4                                        # Data loader workers
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:

1. You're running from the project root directory
2. `underwater_ir/` package is in `PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```
3. All `__init__.py` files exist

### Missing Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

For CLIP with Hugging Face support:
```bash
pip install open_clip_torch
```

### Dataset Not Found

Ensure your dataset follows the structure above. Check paths in `validate_setup.sh` output.

### GPU Memory Issues

Reduce batch size in `run_clip_training.sh`:
```bash
BATCH=2  # or even 1 for limited memory
```

## Module Usage (Alternative to Shell Script)

You can also run modules directly with Python:

### Export Pseudo-Labels
```bash
python -m underwater_ir.teacher.export_pseudolabels \
  --input-root Dataset/train/input \
  --target-root Dataset/train/target \
  --output pseudo-labels/train \
  --clip-model hf-hub:openai/clip-vit-base-patch32 \
  --use-crf \
  --num-workers 4
```

### Train Student
```bash
python -m underwater_ir.student.train_student \
  --train-root Dataset/train \
  --val-ref-root "Dataset/testset(ref)" \
  --val-nonref-root "Dataset/testset(non-ref)" \
  --pseudo-root pseudo-labels/clip_hf \
  --epochs 20 \
  --batch-size 4 \
  --save-path experiments/student.pt
```

## Expected Training Time

On a single GPU (e.g., V100):
- **Pseudo-label export**: ~2-5 min per 1000 images
- **Student training**: ~2-4 hours for 20 epochs (depends on dataset size)

## Output Files

After successful training:

```
pseudo-labels/
└── clip_hf/
    ├── train/*.pt                  # Training pseudo-labels
    ├── testset_ref/{subset}/*.pt   # Reference test pseudo-labels
    └── testset_nonref/{subset}/*.pt # Non-reference test pseudo-labels

experiments/
└── clip_hf_student.pt              # Trained student model checkpoint
```

Each `.pt` file contains:
- `masks`: Per-degradation binary masks
- `z_d`: Degradation code vector
- `global_prob`: Degradation class probabilities
- `confidence`: Mask confidence scores
- Metadata (prompts, hyperparameters)

## Next Steps

After training completes, use the model for inference:

```python
import torch
from underwater_ir.student.naf_unet_wfi import NAFNetWFIGate

# Load model
model = NAFNetWFIGate(...)
model.load_state_dict(torch.load('experiments/clip_hf_student.pt'))
model.eval()

# Inference
with torch.no_grad():
    restored, alphas, logits = model(input_img, z_d, masks)
```

See `underwater_ir/rl/infer_adapt.py` for inference with RL fine-tuning.
