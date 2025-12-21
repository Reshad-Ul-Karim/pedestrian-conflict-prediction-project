# Pedestrian Conflict Prediction Project - Complete Codebase Structure

## ✅ Created Configuration Files
- `configs/detector.yaml` - YOLOv8 detector configuration
- `configs/tracker.yaml` - ByteTrack tracking configuration
- `configs/trajectory.yaml` - Trajectory extraction & prediction config
- `configs/conflict.yaml` - Conflict detection configuration
- `configs/model.yaml` - Fusion model architecture configuration

## ✅ Created Utility Modules
- `src/utils_device.py` - Device management (MPS/CUDA/CPU)
- `src/utils_config.py` - Configuration loader
- `src/utils_logger.py` - Logging utilities
- `src/utils_data_structures.py` - Data structures (BBox, Track, Trajectory, etc.)

## ✅ Created Phase 1 Scripts
- `src/10_train_detector.py` - YOLOv8 detector training (Step 1.2)

## 📝 Scripts to Create Next

### Phase 1 - Step 1.3 (Tracking)
**File:** `src/20_run_tracker.py`
- Load trained YOLOv8 detector
- Run detection on video frames
- Apply ByteTrack for multi-object tracking
- Save tracks to JSONL format
- Visualize tracking results

### Phase 2 - Trajectory Extraction
**File:** `src/30_extract_trajectories.py`
- Extract trajectories from tracks
- Use MediaPipe Pose for pedestrians (foot/hip midpoints)
- Use bbox center for vehicles
- Apply smoothing (Savitzky-Golay filter)
- Save trajectory data

**File:** `src/35_compute_kinematics.py`
- Compute velocity from position
- Compute acceleration from velocity
- Compute heading angle
- Calculate trajectory quality metrics
- Save enhanced trajectory data

### Phase 3 - Trajectory Prediction
**File:** `src/40_predict_trajectories.py`
- Implement Transformer-based trajectory predictor
- Train on historical trajectories
- Predict multi-horizon futures (0.5s to 3.0s)
- Estimate uncertainty
- Save prediction model

### Phase 4 - Conflict Detection
**File:** `src/45_compute_conflict_metrics.py`
- Define dynamic conflict zones (grid-based)
- Compute TTC (Time-to-Conflict)
- Compute PET (Post-Encroachment Time)
- Calculate collision probability
- Generate weak labels with confidence weights
- Save conflict labels

### Phase 5 - Training Clip Generation
**File:** `src/50_build_training_clips.py`
- Extract 2-second temporal clips around conflict events
- Crop visual clips around tracked objects
- Extract kinematic sequences
- Normalize and augment data
- Save clips and metadata

### Phase 6 - Conflict Prediction Model
**File:** `src/60_train_trajectory_predictor.py`
- Define trajectory prediction model architecture
- Train on trajectory data (self-supervised)
- Evaluate ADE/FDE metrics
- Save trained model

**File:** `src/65_train_conflict_predictor.py`
- Define fusion model (vision + kinematics)
- Implement 3D ResNet or TimeSformer for vision
- Implement BiLSTM/Transformer for kinematics
- Train with weighted focal loss
- Apply curriculum learning
- Save trained fusion model

## Implementation Status

✅ **Completed:**
- All configuration files
- Core utility modules
- Detector training script
- Data structures

⏳ **In Progress:**
- Creating remaining scripts for Phases 1-6

## Next Steps

Due to token limitations, the complete implementation is being provided in modular chunks. The codebase structure is:

```
src/
├── utils_*.py          # Utility modules (✅ Complete)
├── 10_train_detector.py  # ✅ Complete
├── 20_run_tracker.py     # Next to create
├── 30_extract_trajectories.py
├── 35_compute_kinematics.py
├── 40_predict_trajectories.py
├── 45_compute_conflict_metrics.py
├── 50_build_training_clips.py
├── 60_train_trajectory_predictor.py
├── 65_train_conflict_predictor.py
├── 70_evaluate_system.py
└── 80_distill_and_export.py
```

## Usage Examples

### 1. Train Detector
```bash
python src/10_train_detector.py \
  --config configs/detector.yaml \
  --data data/rsud20k/dataset.yaml
```

### 2. Run Tracking (once created)
```bash
python src/20_run_tracker.py \
  --config configs/tracker.yaml \
  --video path/to/video.mp4 \
  --output outputs/tracks/
```

All scripts follow a consistent pattern:
- Load configuration from YAML
- Setup logging
- Process data/train models
- Save results with metadata
- Provide progress updates

The implementation is modular, well-documented, and follows best practices for research code.

