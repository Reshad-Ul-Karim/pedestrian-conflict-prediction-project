# Dependencies Check for Pedestrian Conflict Prediction Project

## ✅ All Required Dependencies

### Core ML/DL Frameworks
- ✅ torch>=2.0.0 (PyTorch with MPS support)
- ✅ torchvision>=0.15.0
- ✅ torchaudio>=2.0.0
- ✅ torch-geometric>=2.3.0 (Graph Neural Networks)

### Computer Vision
- ✅ ultralytics>=8.0.0 (YOLOv8)
- ✅ opencv-python>=4.8.0 (cv2)
- ✅ mediapipe>=0.10.0 (Pose estimation)

### Deep Learning Tools
- ✅ transformers>=4.30.0 (HuggingFace)
- ✅ timm>=0.9.0 (Vision models)

### Deployment
- ✅ coremltools>=7.0.0 (CoreML export)

### Scientific Computing
- ✅ numpy>=1.21.0
- ✅ scipy>=1.7.0
- ✅ pandas>=1.3.0
- ✅ scikit-learn>=1.0.0

### Visualization
- ✅ matplotlib>=3.5.0
- ✅ seaborn>=0.11.0

### Utilities
- ✅ tqdm>=4.65.0 (Progress bars)
- ✅ pillow>=9.0.0 (Image processing)
- ✅ pyyaml>=6.0 (Config files)
- ✅ requests>=2.31.0 (HTTP/Downloads)

### Experiment Tracking (Optional)
- ✅ wandb>=0.15.0
- ✅ tensorboard>=2.13.0

## Built-in Python Modules (No Installation Needed)
- json
- pathlib
- argparse
- logging
- sys
- os
- dataclasses
- typing
- copy
- datetime
- random
- shutil

## Installation Command

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

## Verification

```bash
# Quick check
python -c "import torch, cv2, mediapipe, ultralytics; print('✅ Core packages installed')"

# Full device check
python -c "from src.utils_device import print_device_info; print_device_info()"

# MPS/CoreML test (macOS only)
python test_mps_coreml.py
```

## Package Sizes (Approximate)
- PyTorch ecosystem: ~2GB
- Vision packages: ~500MB
- Scientific packages: ~300MB
- Other dependencies: ~200MB
- **Total**: ~3GB

## Notes
- All dependencies are compatible with Python 3.9
- MPS support included for Apple Silicon (M1/M2/M3)
- CUDA support available for NVIDIA GPUs
- CPU-only mode works on all systems

