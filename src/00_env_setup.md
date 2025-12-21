# Environment Setup Guide

## System Requirements

### Hardware
- **Recommended**: Apple Silicon (M1/M2/M3) or NVIDIA GPU (RTX 3090/4090)
- **Minimum**: 16GB RAM, 50GB free disk space
- **GPU Memory**: 8GB+ recommended for training

### Software
- **Python**: 3.9.x (3.9.23 tested and working)
- **OS**: macOS 11+, Linux, or Windows 10+
- **Git**: For cloning repositories

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/pedestrian-conflict-prediction-project.git
cd pedestrian-conflict-prediction-project
```

### 2. Create Virtual Environment
```bash
# Using Python 3.9
python3.9 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

**For macOS (Apple Silicon with MPS):**
```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio  # Latest version with MPS support
pip install -r requirements.txt
```

**For Linux/Windows with NVIDIA GPU:**
```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CPU only:**
```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check device availability
python -c "from src.utils_device import print_device_info; print_device_info()"

# Test MPS and CoreML (macOS only)
python test_mps_coreml.py
```

## Package Versions (Tested)

```
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
ultralytics==8.3.240
opencv-python==4.9.0.80
mediapipe==0.10.30
torch-geometric==2.6.1
transformers==4.57.3
timm==1.0.22
coremltools==9.0
numpy==1.24.4
scipy==1.13.1
pandas==2.3.3
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
wandb==0.23.1
tensorboard==2.20.0
```

## Directory Structure

After setup, create the following directories:

```bash
mkdir -p data/{rsud20k,badodd,dashcam_videos,processed}
mkdir -p outputs/{detections,tracks,trajectories,predictions,clips,autolabels,models,reports}
mkdir -p configs
```

## Dataset Setup

### RSUD20K Dataset
1. Download from: https://github.com/hasibzunair/RSUD20K
2. Convert annotations to YOLO format
3. Create `data/rsud20k/dataset.yaml`:

```yaml
path: /absolute/path/to/data/rsud20k
train: images/train
val: images/val
test: images/test

nc: 13  # Number of classes
names: ['person', 'bicycle', 'motorcycle', 'car', 'bus', 'truck', 
        'rickshaw', 'CNG', 'van', 'pickup', 'ambulance', 'fire_truck', 'police']
```

### Dashcam Videos
- Place videos in `data/dashcam_videos/`
- Supported formats: .mp4, .avi, .mov
- Recommended: 30fps, 1280x720 or higher resolution

## Configuration

All configuration files are in `configs/`:
- `detector.yaml` - YOLOv8 training parameters
- `tracker.yaml` - Tracking parameters
- `trajectory.yaml` - Trajectory extraction/prediction
- `conflict.yaml` - Conflict detection parameters
- `model.yaml` - Fusion model architecture

Edit these files to customize:
- Batch sizes (based on GPU memory)
- Learning rates
- Model architectures
- Data augmentation parameters

## Device Selection

The project automatically detects the best available device:
1. **MPS** (Apple Silicon) - Fastest on M1/M2/M3
2. **CUDA** (NVIDIA GPU) - For Linux/Windows with GPU
3. **CPU** - Fallback (slower)

To manually set device, edit config files:
```yaml
training:
  device: mps  # or cuda or cpu
```

## Troubleshooting

### MPS Issues (macOS)
```bash
# If MPS not available
python -c "import torch; print(torch.backends.mps.is_available())"

# Clear MPS cache if needed
python -c "import torch; torch.mps.empty_cache()"
```

### CUDA Issues (Linux/Windows)
```bash
# Check CUDA version
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce batch size in config files
- Use smaller model variants (yolov8s instead of yolov8m)
- Enable gradient checkpointing

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Environment Variables

Optional environment variables:

```bash
# For Weights & Biases logging
export WANDB_API_KEY=your_key

# For distributed training
export CUDA_VISIBLE_DEVICES=0,1

# For debugging
export PYTHONUNBUFFERED=1
```

## Next Steps

After setup is complete:
1. Prepare RSUD20K dataset
2. Train detector: `python src/10_train_detector.py --data data/rsud20k/dataset.yaml`
3. Run tracking on videos
4. Continue through the pipeline

## Support

For issues:
1. Check this guide
2. Review configuration files
3. Check logs in `outputs/*/train.log`
4. Review documentation in README.md

