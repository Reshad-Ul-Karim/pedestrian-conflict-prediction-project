# Pedestrian Conflict Prediction - Quick Start Guide

## Complete Codebase Created ✅

### Configuration Files (5)
- `configs/detector.yaml` - YOLOv8 detector training config
- `configs/tracker.yaml` - ByteTrack tracking config  
- `configs/trajectory.yaml` - Trajectory extraction & prediction
- `configs/conflict.yaml` - Conflict detection parameters
- `configs/model.yaml` - Fusion model architecture

### Utility Modules (5)
- `src/utils_device.py` - MPS/CUDA/CPU device management
- `src/utils_config.py` - YAML configuration loader
- `src/utils_logger.py` - Logging utilities
- `src/utils_data_structures.py` - Data classes (BBox, Track, Trajectory, etc.)

### Pipeline Scripts (10)

**Phase 1 - Detection & Tracking:**
- `src/10_train_detector.py` - Train YOLOv8 detector
- `src/20_run_tracker.py` - Multi-object tracking with ByteTrack

**Phase 2 - Trajectory Extraction:**
- `src/30_extract_trajectories.py` - Extract trajectories with MediaPipe
- `src/35_compute_kinematics.py` - Compute velocity, acceleration, heading

**Phase 3 - Trajectory Prediction:**
- `src/40_predict_trajectories.py` - Train Transformer trajectory predictor

**Phase 4 - Conflict Detection:**
- `src/45_compute_conflict_metrics.py` - Compute TTC/PET, generate weak labels

**Phase 5 - Training Data:**
- `src/50_build_training_clips.py` - Extract visual & kinematic clips

**Phase 6 - Conflict Prediction:**
- `src/65_train_conflict_predictor.py` - Train fusion model (vision + kinematics)

**Phase 7-8 - Evaluation & Deployment:**
- `src/70_evaluate_system.py` - Comprehensive evaluation
- `src/80_distill_and_export.py` - Model distillation & ONNX/CoreML export

## Usage Pipeline

### 1. Train Detector
```bash
python src/10_train_detector.py \
  --config configs/detector.yaml \
  --data data/rsud20k/dataset.yaml
```

### 2. Run Tracking
```bash
python src/20_run_tracker.py \
  --video path/to/video.mp4 \
  --config configs/tracker.yaml \
  --output outputs/tracks \
  --visualize
```

### 3. Extract Trajectories
```bash
python src/30_extract_trajectories.py \
  --video path/to/video.mp4 \
  --tracks outputs/tracks/video.jsonl \
  --config configs/trajectory.yaml \
  --output outputs/trajectories/video_tracks.jsonl
```

### 4. Compute Kinematics
```bash
python src/35_compute_kinematics.py \
  --input outputs/trajectories/video_tracks.jsonl \
  --config configs/trajectory.yaml \
  --output outputs/trajectories/video_enhanced.jsonl
```

### 5. Train Trajectory Predictor
```bash
python src/40_predict_trajectories.py \
  --config configs/trajectory.yaml \
  --trajectories outputs/trajectories/video_enhanced.jsonl \
  --output outputs/models/trajectory_predictor
```

### 6. Compute Conflict Metrics
```bash
python src/45_compute_conflict_metrics.py \
  --config configs/conflict.yaml \
  --trajectories outputs/trajectories/video_enhanced.jsonl \
  --output outputs/autolabels/video_labels.jsonl
```

### 7. Build Training Clips
```bash
python src/50_build_training_clips.py \
  --video path/to/video.mp4 \
  --tracks outputs/tracks/video.jsonl \
  --trajectories outputs/trajectories/video_enhanced.jsonl \
  --labels outputs/autolabels/video_labels.jsonl \
  --output outputs/clips
```

### 8. Train Conflict Predictor
```bash
python src/65_train_conflict_predictor.py \
  --config configs/model.yaml \
  --clips outputs/clips \
  --output outputs/models/conflict_predictor
```

### 9. Evaluate System
```bash
python src/70_evaluate_system.py \
  --config configs/model.yaml \
  --model outputs/models/conflict_predictor/best_model.pt \
  --clips outputs/clips \
  --output outputs/reports/evaluation
```

### 10. Export Model
```bash
python src/80_distill_and_export.py \
  --teacher outputs/models/conflict_predictor/best_model.pt \
  --clips outputs/clips \
  --output outputs/models/distilled
```

## Key Features

✅ **Modular Architecture** - Each phase is independent
✅ **Configuration-Driven** - All parameters in YAML files
✅ **MPS/CUDA Support** - Automatic device detection
✅ **Comprehensive Logging** - Track progress and debug easily
✅ **Production-Ready** - Error handling, validation, checkpointing
✅ **Well-Documented** - Docstrings and inline comments

## Directory Structure After Running

```
project/
├── data/
│   └── rsud20k/                  # Your dataset
├── outputs/
│   ├── detections/
│   ├── tracks/                   # Tracking results
│   ├── trajectories/             # Extracted trajectories
│   ├── autolabels/               # Conflict labels
│   ├── clips/                    # Training clips
│   ├── models/                   # Trained models
│   └── reports/                  # Evaluation results
├── configs/                      # All configuration files
└── src/                          # All source code
```

## Next Steps

1. **Prepare Dataset**: Organize RSUD20K in YOLO format
2. **Train Detector**: Run step 1 above
3. **Process Videos**: Run steps 2-7 for each video
4. **Train Models**: Run steps 8-10
5. **Deploy**: Use exported ONNX/CoreML models

All scripts are executable and follow consistent patterns for easy extension and customization.

