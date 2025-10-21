# Hướng Dẫn Sử Dụng CLIP Model Loader Mới

## Tổng Quan

Đã tạo một hệ thống load CLIP model thống nhất cho `daclip-uir`, tham khảo từ cách tổ chức trong `guided-underwater-restoration`. Hệ thống này hỗ trợ:

1. **HuggingFace Transformers CLIP** (khuyến nghị) - Dễ sử dụng, không cần checkpoint phức tạp
2. **Custom DACLiP model** (legacy) - Sử dụng open_clip với checkpoint đã train

## Các File Mới Được Tạo

### 1. Config File: `configs/config_clip.yaml`
File cấu hình theo chuẩn của guided-underwater-restoration:
- Cấu hình cho cả HuggingFace CLIP và custom DACLiP
- Cấu hình dataset, training, evaluation
- Dễ dàng chuyển đổi giữa các model

### 2. Model Loader: `underwater_ir/model_loader.py`
Module load model thống nhất:
```python
from underwater_ir.model_loader import load_clip_model

# Load từ config file
model, processor, tokenizer = load_clip_model(config_path='configs/config_clip.yaml')

# Hoặc load từ dict
config = {'clip': {'model_type': 'openai_clip', ...}}
model, processor, tokenizer = load_clip_model(config_dict=config)
```

### 3. Export Pseudo-labels V2: `underwater_ir/teacher/export_pseudolabels_v2.py`
Phiên bản mới của export_pseudolabels sử dụng unified loader:
- Hỗ trợ cả HuggingFace CLIP và DACLiP
- Cấu hình thông qua YAML file
- API đơn giản hơn

### 4. Training Script V2: `run_clip_training_v2.sh`
Script training mới với kiểm tra dependencies và tự động fallback

### 5. Test Scripts:
- `test_new_loader.py` - Test model loader
- Các file test khác đã có

## Cài Đặt Dependencies

```bash
# Dependencies cơ bản
pip install torch torchvision
pip install transformers
pip install pyyaml pillow numpy

# Nếu muốn dùng custom DACLiP (optional)
pip install ftfy sentencepiece
```

## Cách Sử Dụng

### Option 1: Sử Dụng HuggingFace CLIP (Khuyến Nghị)

**Ưu điểm:**
- Không cần download checkpoint thủ công
- Tự động download từ HuggingFace Hub
- Dễ cài đặt và sử dụng
- Tương thích tốt với guided-underwater-restoration

**Cấu hình:** Trong `configs/config_clip.yaml`:
```yaml
clip:
  model_type: "openai_clip"  # Sử dụng HuggingFace
  hf_model:
    model_name: "openai/clip-vit-base-patch32"
    use_transformers: true
```

**Test:**
```bash
python test_new_loader.py
```

**Chạy training:**
```bash
bash run_clip_training_v2.sh
```

### Option 2: Sử Dụng Custom DACLiP (Legacy)

**Khi nào dùng:**
- Đã có checkpoint DACLiP được train sẵn
- Muốn tương thích với code cũ

**Cấu hình:** Trong `configs/config_clip.yaml`:
```yaml
clip:
  model_type: "daclip_custom"
  daclip_model:
    model_name: "daclip_ViT-B-32"
    checkpoint_path: "pretrained/daclip_ViT-B-32.pt"
```

**Yêu cầu:**
- Phải có file checkpoint tại `pretrained/daclip_ViT-B-32.pt`
- Phải cài ftfy và sentencepiece

## So Sánh với Code Cũ

### Code Cũ (`run_clip_training.sh`):
```bash
# Hardcoded model và checkpoint
CLIP_MODEL="daclip_ViT-B-32"
CLIP_CKPT="pretrained/daclip_ViT-B-32.pt"

# Phải có checkpoint file
if [[ ! -f "${CLIP_CKPT}" ]]; then
  echo "Error: checkpoint not found"
  exit 1
fi
```

### Code Mới (`run_clip_training_v2.sh`):
```bash
# Sử dụng config file
CONFIG_FILE="configs/config_clip.yaml"

# Tự động load model theo config
python -m underwater_ir.model_loader --config "${CONFIG_FILE}"

# Không cần checkpoint nếu dùng HuggingFace
```

## Khắc Phục Lỗi

### Lỗi: "Import torch could not be resolved"
```bash
pip install torch torchvision
```

### Lỗi: "Import transformers could not be resolved"
```bash
pip install transformers
```

### Lỗi: "Config file not found"
Đảm bảo file `configs/config_clip.yaml` tồn tại trong thư mục daclip-uir

### Lỗi: "DACLiP checkpoint not found"
**Solution 1 (Khuyến nghị):** Đổi sang dùng HuggingFace CLIP:
```yaml
clip:
  model_type: "openai_clip"  # Thay vì "daclip_custom"
```

**Solution 2:** Download checkpoint DACLiP theo hướng dẫn trong README gốc

## Test Từng Bước

### 1. Test dependencies:
```bash
python -c "import torch, transformers, yaml; print('OK')"
```

### 2. Test model loader:
```bash
python test_new_loader.py
```

### 3. Test pseudo-label export (với dataset nhỏ):
```bash
# Tạo test dataset
mkdir -p test_data/input test_data/target

# Export pseudo-labels
python -m underwater_ir.teacher.export_pseudolabels_v2 \
  --config configs/config_clip.yaml \
  --input-root test_data/input \
  --target-root test_data/target \
  --output pseudo-labels/test \
  --batch-size 2
```

### 4. Chạy full pipeline:
```bash
bash run_clip_training_v2.sh
```

## Tích Hợp với Guided-Underwater-Restoration

Cấu trúc config và code giờ tương thích với `guided-underwater-restoration`:

```python
# Trong guided-underwater-restoration
from models.degradation_analysis import DegradationAnalyzer

# Trong daclip-uir (style mới)
from underwater_ir.model_loader import load_clip_model

# Cả hai đều dùng:
# - YAML config
# - HuggingFace transformers
# - Cấu trúc tương tự
```

## Lợi Ích

1. **Dễ dàng chuyển đổi model:** Chỉ cần sửa config, không cần sửa code
2. **Không cần checkpoint phức tạp:** HuggingFace tự động download
3. **Tương thích cao:** Style giống guided-underwater-restoration
4. **Maintainable:** Code rõ ràng, dễ debug
5. **Flexible:** Vẫn hỗ trợ custom DACLiP nếu cần

## Next Steps

1. Test với dataset của bạn
2. Chạy training với HuggingFace CLIP
3. So sánh kết quả với DACLiP (nếu có checkpoint)
4. Tích hợp với guided-underwater-restoration nếu cần

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề, check:
1. Dependencies đã đủ chưa
2. Config file đúng format chưa
3. Dataset paths có đúng không
4. Device (GPU/CPU) có sẵn không
