# Pedestrian Conflict Prediction: Real-Time Grid-Based Framework

## Overview

This research project implements a real-time framework for predicting pedestrian-vehicle conflicts using grid-based detection and pose analysis. The system combines:
- **YOLO12 multi-object detection** for reliable object detection
- **Multi-object tracking** for persistent object tracking across frames
- **MediaPipe pose estimation** for pedestrian pose analysis
- **Grid-based conflict detection** with two key rules:
  - **Rule 1:** Person pose inclination detection (leaning forward, crossing motion) → conflict probability
  - **Rule 2:** Vehicle proximity + grid coverage (too close to camera, covering conflict zone) → conflict probability
- **Real-time processing** optimized for live video streams

---

## Phase 0 — Setup and Data

### Step 0.1 – Project Structure

```text
project/
  data/
    rsud20k/              # RSUD20K dataset (detection training)
    badodd/               # BadODD dataset (optional domain adaptation)
    dashcam_videos/       # Public dashcam datasets:
                          #   - KITTI, nuScenes (extracted videos)
                          #   - A3D (Argoverse 3D)
                          #   - Waymo Open Dataset (videos)
                          #   - D2City, Cityscapes (urban scenes)
    processed/            # Preprocessed data cache
  outputs/
    detections/           # YOLO outputs per frame
    tracks/               # ByteTrack tracklets per video
    trajectories/         # Enhanced trajectory data with kinematics
    predictions/          # Multi-horizon trajectory predictions
    clips/                # Temporal clips for training (2-3s windows)
    autolabels/           # Weak labels: conflict scores, TTC, PET, confidence
    models/               # All trained models
    reports/              # Metrics, visualizations, analysis
  src/
    00_env_setup.md       # Environment configuration
    10_train_detector.py  # YOLO12 detector training (optional)
    20_run_tracker.py     # Multi-object tracking pipeline
    30_extract_trajectories.py  # Trajectory extraction with MediaPipe pose
    35_compute_kinematics.py    # Velocity, acceleration, heading estimation
    realtime_grid_conflict_detector.py  # Grid-based conflict detection
    realtime_conflict_pipeline.py       # Complete real-time pipeline
    45_compute_conflict_metrics.py  # Conflict metrics (legacy/optional)
    70_evaluate_system.py  # Comprehensive evaluation
  configs/
    detector.yaml         # YOLO12 detection configs
    tracker.yaml          # Tracking parameters
    trajectory.yaml       # Trajectory extraction configs
    realtime_conflict.yaml  # Real-time conflict detection parameters
```

### Step 0.2 – Environment Setup

**Core Dependencies:**
- Python 3.9+ (3.9 recommended for compatibility)
- PyTorch 2.0+ (with MPS support for Apple Silicon / CUDA for NVIDIA)
- **Ultralytics YOLO12** (latest YOLO version)
- OpenCV 4.8+
- **MediaPipe** (for pose estimation - required for pedestrian conflict detection)
- Tracking algorithm (ByteTrack or custom implementation)
- NumPy, SciPy (for signal processing)
- CoreMLTools (for model conversion and deployment on macOS/iOS)

**Setup Instructions:**

1. **Create virtual environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install torch torchvision torchaudio  # MPS support included for macOS
   pip install -r requirements.txt
   ```

   Or use the setup script:
   ```bash
   ./setup_venv.sh  # macOS/Linux
   # OR
   setup_venv.bat   # Windows
   ```

3. **Verify MPS support (macOS):**
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

4. **Test MPS and CoreML:**
   ```bash
   python test_mps_coreml.py
   ```

**Device Support:**
- **MPS (Metal Performance Shaders)**: Native support for Apple Silicon (M1/M2/M3) - automatically used when available
- **CUDA**: Support for NVIDIA GPUs (if CUDA is installed)
- **CPU**: Fallback for systems without GPU support
- **CoreML**: For model conversion and deployment on Apple devices

**Note:** The project automatically detects and uses the best available device (MPS > CUDA > CPU). Use `src/utils_device.py` for device management utilities.

**Recommended Datasets:**
1. **RSUD20K** / **BadODD** - For detector training (Bangladesh traffic scenes)
2. **KITTI** - Dashcam videos with rich urban scenarios
3. **nuScenes** - Diverse driving scenarios (extract videos)
4. **Argoverse 2** / **A3D** - High-quality trajectory data
5. **Waymo Open Dataset** - Large-scale dashcam videos
6. **D2City** / **Cityscapes** - Urban scene videos

---

## Phase 1 — Object Detection and Tracking Pipeline

### Step 1.1 – Dataset Preparation

1. **Download and organize datasets:**
   - RSUD20K: Convert annotations to YOLO format
   - Public dashcam datasets: Extract video clips, organize by scenario type
   
2. **Class mapping:**
   - Primary classes: `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`, `rickshaw`, `auto-rickshaw`
   - Map to consistent class IDs across datasets

3. **Data splits:**
   - RSUD20K: Train (70%) / Val (15%) / Test (15%)
   - Dashcam videos: Split by video/scenario (no temporal leakage)

### Step 1.2 – Detector Setup

**Model: YOLO12 (YOLOv12)**

YOLO12 is the latest version with improved accuracy and speed. You can either:
1. **Use pre-trained YOLO12** (recommended for quick start):
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo12n.pt')  # nano - fastest
   # or
   model = YOLO('yolo12s.pt')  # small - balanced
   # or
   model = YOLO('yolo12m.pt')  # medium - better accuracy
   ```

2. **Fine-tune on RSUD20K** (optional, for domain-specific improvement):
   ```bash
   yolo detect train \
     data=configs/rsud20k.yaml \
     model=yolo12m.pt \
     epochs=100 \
     imgsz=640 \
     batch=16 \
     name=detector_rsud20k \
     project=outputs/models
   ```

**Key configurations:**
- Use YOLO12 for best performance
- Monitor pedestrian class mAP during training
- Recommended: Start with pre-trained model, fine-tune if needed

**Deliverable:** `yolo12n.pt` (pre-trained) or `outputs/models/detector_rsud20k/weights/best.pt` (fine-tuned)

### Step 1.3 – Multi-Object Tracking

**Pipeline: `20_run_tracker.py`**

1. **Detection:**
   - Run YOLO12 on each frame (GPU acceleration)
   - Filter low-confidence detections (threshold: 0.5)
   - Apply NMS (IoU threshold: 0.45)

2. **Tracking with ByteTrack:**
   - High-score detections → track matching
   - Low-score detections → recovery matching
   - Kalman filter for motion prediction
   - Track lifecycle: new → tracked → lost → removed

3. **Output format:**
```json
{
  "video_id": "kitti_001",
  "frame_id": 120,
  "timestamp": 4.0,
  "tracks": [
    {
      "track_id": 5,
      "class": "person",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2],
      "kalman_state": {...}
    }
  ]
}
```

**Output:** `outputs/tracks/{video_id}.jsonl` (one JSON per frame)

---

## Phase 2 — Advanced Trajectory Extraction and Kinematics

### Step 2.1 – Trajectory Extraction with Pose Estimation

**Pipeline: `30_extract_trajectories.py`**

#### For Pedestrians (Humans):

1. **MediaPipe Pose extraction:**
   - Crop person bounding box (with padding: 20% on each side)
   - Run MediaPipe Pose on cropped image
   - Extract keypoints: 33 body landmarks

2. **Anchor point selection:**
   - **Primary:** Foot midpoint = (left_ankle + right_ankle) / 2
   - **Fallback:** Hip midpoint = (left_hip + right_hip) / 2
   - **Stability:** Use temporal median filter (window=3) to reduce jitter

3. **Additional features:**
   - Heading direction: Vector from hip to shoulder midpoint
   - Body orientation: Angle of torso
   - Pose confidence: Average MediaPipe confidence scores

#### For Non-Pedestrian Objects:

1. **Bounding box center:**
   - Use bbox center: `(x_center, y_center)` = `((x1+x2)/2, (y1+y2)/2)`
   - More stable than corner-based points

2. **Orientation estimation (for vehicles):**
   - Use bbox aspect ratio to infer orientation
   - Estimate heading from trajectory direction (using past 5 frames)

### Step 2.2 – Kinematics Computation

**Pipeline: `35_compute_kinematics.py`**

For each track, compute temporal derivatives:

1. **Velocity estimation:**
   ```python
   # Use Savitzky-Golay filter (window=5, poly_order=2) for smooth differentiation
   v_u = differentiate(u_t, method='savgol', window=5, order=2)
   v_v = differentiate(v_t, method='savgol', window=5, order=2)
   speed = sqrt(v_u^2 + v_v^2)
   ```

2. **Acceleration:**
   ```python
   a_u = differentiate(v_u, method='savgol')
   a_v = differentiate(v_v, method='savgol')
   acceleration = sqrt(a_u^2 + a_v^2)
   ```

3. **Heading angle:**
   ```python
   heading = atan2(v_v, v_u)  # Angle in image space
   ```

4. **Quality metrics:**
   - Trajectory smoothness: Variance of velocity changes
   - Tracking confidence: Average detection confidence over track lifetime
   - Occlusion indicator: Gaps in detection (interpolated frames)

**Output format:**
   ```json
   {
  "video_id": "kitti_001",
  "track_id": 5,
  "class": "person",
  "trajectory": [
    {
      "t": 4.0,
      "u": 450.2, "v": 610.5,
      "bbox": [x1, y1, x2, y2],
      "kinematics": {
        "v_u": 2.1, "v_v": -5.3,  # pixels per second
        "speed": 5.7,
        "a_u": 0.1, "a_v": -0.2,
        "acceleration": 0.22,
        "heading": -1.19,  # radians
        "heading_deg": -68.2
      },
      "pose": {
        "foot_midpoint": [450.2, 610.5],
        "hip_midpoint": [450.0, 580.0],
        "torso_angle": -1.15,
        "pose_confidence": 0.89
      },
      "quality": {
        "detection_conf": 0.92,
        "smoothness": 0.05,
        "occluded": false
      }
    }
     ]
   }
   ```

**Output:** `outputs/trajectories/{video_id}_tracks.json`

---

## Phase 3 — Advanced Trajectory Prediction

### Step 3.1 – Multi-Horizon Trajectory Prediction Model

**Pipeline: `40_predict_trajectories.py`**

**Goal:** Predict future trajectory points at multiple horizons (0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s) with uncertainty estimation.

#### Architecture Options:

**Option A: Attention-Based Temporal Encoder (Recommended)**

1. **Input encoding:**
   - Past trajectory: `T_past` frames (e.g., 20 frames ≈ 1 second @ 20fps)
   - Features per frame: `[u, v, v_u, v_v, a_u, a_v, heading, class_embedding]`

2. **Temporal encoder:**
   - Multi-head self-attention (Transformer encoder)
   - Position encoding for temporal order
   - Layer normalization and residual connections
   - Hidden dimension: 128-256

3. **Decoder for multi-horizon prediction:**
   - For each horizon Δt ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}s:
     - Predict: `(u_pred, v_pred)` and uncertainty `σ_u, σ_v`
     - Use MLP head: `encoded_features → [u_pred, v_pred, log_σ_u, log_σ_v]`

**Option B: Social-LSTM / Graph Neural Network (For multi-agent scenes)**

1. **Social interaction modeling:**
   - Build graph: nodes = agents, edges = spatial proximity
   - Use GCN or GAT (Graph Attention Network) to encode interactions
   - Combine with temporal LSTM/GRU

2. **Benefits:**
   - Captures pedestrian group behavior
   - Models avoidance behaviors
   - Better for crowded scenes

**Option C: Physics-Aware Models (For vehicles)**

1. **Constant acceleration model:**
   ```python
   u(t+Δt) = u(t) + v_u(t) * Δt + 0.5 * a_u(t) * Δt^2
   v(t+Δt) = v(t) + v_v(t) * Δt + 0.5 * a_v(t) * Δt^2
   ```
   - Use as baseline or as input feature to neural network

2. **Neural-physics hybrid:**
   - Use physics model as prior
   - Neural network learns residuals/deviations

#### Training Details:

**Loss function:**
```python
# Gaussian negative log-likelihood for uncertainty-aware learning
loss = -log N(u_true | u_pred, σ_u^2) - log N(v_true | v_pred, σ_v^2)
# Or simpler L1/L2 loss if not modeling uncertainty
loss = L1(u_pred, u_true) + L1(v_pred, v_true)
```

**Training data:**
- Use past `T_past` frames to predict future positions
- Sliding window over all trajectories
- Filter: Only use trajectories with length ≥ 2 seconds

**Deliverable:** `outputs/models/trajectory_predictor.pt`

### Step 3.2 – Prediction Output Format

For each track at time `t0`:

   ```json
   {
  "t0": 4.0,
  "track_id": 5,
  "past_trajectory": [...],  # Last T_past frames
  "predictions": {
    "0.5s": {"u": 452.1, "v": 608.2, "sigma_u": 1.2, "sigma_v": 1.5},
    "1.0s": {"u": 454.3, "v": 605.8, "sigma_u": 2.1, "sigma_v": 2.8},
    "1.5s": {"u": 456.5, "v": 603.4, "sigma_u": 3.5, "sigma_v": 4.2},
    "2.0s": {"u": 458.7, "v": 601.0, "sigma_u": 5.1, "sigma_v": 6.0},
    "2.5s": {"u": 460.9, "v": 598.6, "sigma_u": 7.2, "sigma_v": 8.5},
    "3.0s": {"u": 463.1, "v": 596.2, "sigma_u": 9.8, "sigma_v": 11.2}
  },
  "prediction_confidence": 0.87  # Overall confidence score
}
```

---

## Phase 4 — Real-Time Grid-Based Conflict Detection

### Step 4.1 – Grid-Based Conflict Zone Definition

**Pipeline: `realtime_grid_conflict_detector.py`**

The system uses a **grid-based approach** to define conflict zones in the camera feed:

1. **Grid Structure:**
   - Divide camera frame into grid (default: 8 rows × 6 columns)
   - Each cell represents a spatial region in the image
   - Configurable grid size based on camera resolution

2. **Conflict Zone (Ego Corridor):**
   - **Default:** Bottom-middle region (rows 6-7, columns 2-3)
   - Represents the area where the ego vehicle would be
   - Can be adjusted based on camera mounting position
   - Example: For 8×6 grid, conflict zone = bottom 2 rows, middle 2 columns

3. **Grid Cell Mapping:**
   ```python
   # Convert pixel coordinates to grid cell
   cell_width = frame_width / grid_cols
   cell_height = frame_height / grid_rows
   grid_row = int(center_y / cell_height)
   grid_col = int(center_x / cell_width)
   ```

### Step 4.2 – Conflict Detection Rules

The system implements **two primary conflict detection rules**:

#### Rule 1: Person Pose Inclination Detection

**For pedestrians (person class):**

Uses **MediaPipe pose estimation** to detect conflict-indicating poses:

1. **Torso Angle Analysis:**
   - Extract key landmarks: shoulders, hips, ankles
   - Compute torso vector (hip to shoulder)
   - Calculate angle from vertical
   - **Conflict indicator:** Forward lean > 15° (crossing motion)

2. **Leg Position Analysis:**
   - Check leg separation and height difference
   - **Conflict indicators:**
     - Legs apart (walking motion)
     - One leg raised (running motion)
     - Leg separation > 20 pixels or height diff > 15 pixels

3. **Position in Conflict Zone:**
   - Check if person is in conflict zone grid cells
   - Higher conflict probability if in ego corridor

4. **Temporal Consistency:**
   - Track pose angle over time
   - Increasing forward lean → higher conflict probability
   - Maintains history of last 30 frames for smoothing

**Conflict Score Calculation:**
```python
conflict_score = 0.0

# Torso inclination (0.3 weight)
if abs(torso_angle) > 15:
    conflict_score += 0.3

# Leg position (0.2 weight)
if leg_separation > 20 or leg_height_diff > 15:
    conflict_score += 0.2

# Position in conflict zone (0.3 weight)
if grid_cell in conflict_zone:
    conflict_score += 0.3

# Temporal trend (0.2 weight)
if angle_trend > 2:  # Increasing forward lean
    conflict_score += 0.2

# Threshold: conflict_prob > 0.5 → conflict detected
```

#### Rule 2: Vehicle Proximity + Grid Coverage

**For non-human objects (vehicles, bicycles, etc.):**

1. **Proximity Detection:**
   - Measure bbox height relative to frame height
   - **Conflict indicator:** bbox_height / frame_height > 0.4 (40% threshold)
   - Large bbox = object too close to camera

2. **Grid Coverage Analysis:**
   - Calculate which grid cells are covered by object bbox
   - Count conflict zone cells covered
   - **Conflict indicator:** coverage_ratio > 0.3 (30% of conflict zone)

3. **Position Check:**
   - Verify object center is in conflict zone
   - Additional weight if in ego corridor

4. **Temporal Consistency:**
   - Track conflict scores over time
   - Consistent high scores → confirmed conflict

**Conflict Score Calculation:**
```python
conflict_score = 0.0

# Proximity check (0.4 weight)
if bbox_height_ratio > 0.4:
    conflict_score += 0.4

# Grid coverage (0.4 weight)
if coverage_ratio > 0.3:
    conflict_score += 0.4

# Position in conflict zone (0.2 weight)
if grid_cell in conflict_zone:
    conflict_score += 0.2

# Temporal consistency (0.1 weight)
if mean(prev_conflicts) > 0.5:
    conflict_score += 0.1

# Threshold: conflict_prob > 0.5 → conflict detected
```

### Step 4.3 – Real-Time Processing Pipeline

**Pipeline: `realtime_conflict_pipeline.py`**

Complete end-to-end pipeline:

1. **Frame Input:**
   - Video stream (webcam, video file, or RTSP)
   - Process frame-by-frame in real-time

2. **Detection → Tracking → Conflict Detection:**
   ```python
   # Step 1: YOLO12 Detection
   detections = yolo12_model(frame)
   
   # Step 2: Tracking
   tracks = tracker.update(detections)
   
   # Step 3: Grid-based Conflict Detection
   conflicts = conflict_detector.detect_conflicts(frame, tracks)
   ```

3. **Output Format:**
   ```json
   {
     "frame_id": 120,
     "pedestrian_conflicts": [
       {
         "track_id": 5,
         "conflict_probability": 0.85,
         "grid_cell": [6, 3],
         "reason": "pose_inclination",
         "bbox": [x1, y1, x2, y2]
       }
     ],
     "vehicle_conflicts": [
       {
         "track_id": 12,
         "conflict_probability": 0.72,
         "grid_cells_covered": [[6,2], [6,3], [7,2], [7,3]],
         "reason": "proximity_and_coverage",
         "bbox": [x1, y1, x2, y2]
       }
     ],
     "grid_occupancy": [[...], [...]]  # 8x6 grid
   }
   ```

4. **Visualization:**
   - Draw grid overlay on frame
   - Highlight conflict zone in red
   - Draw bounding boxes with conflict alerts
   - Display conflict probability scores

**Output:** Real-time conflict alerts + visualization

---

## Phase 5 — Real-Time Usage and Integration

### Step 5.1 – Running Real-Time Conflict Detection

**Pipeline: `realtime_conflict_pipeline.py`**

1. **Initialize Pipeline:**
   ```python
   from src.realtime_conflict_pipeline import RealtimeConflictPipeline
   
   pipeline = RealtimeConflictPipeline(
       yolo_model_path="yolo12n.pt",  # or path to fine-tuned model
       grid_rows=8,
       grid_cols=6
   )
   ```

2. **Process Video Stream:**
   ```python
   # From webcam
   cap = cv2.VideoCapture(0)
   
   # Or from video file
   cap = cv2.VideoCapture("path/to/video.mp4")
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       # Process frame
       result = pipeline.process_frame(frame)
       
       # Visualize
       vis_frame = pipeline.visualize(frame, result)
       cv2.imshow('Conflict Detection', vis_frame)
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   ```

3. **Access Conflict Results:**
   ```python
   # Check for conflicts
   if result['conflicts']['pedestrian_conflicts']:
       for conflict in result['conflicts']['pedestrian_conflicts']:
           print(f"Pedestrian conflict detected! Track ID: {conflict['track_id']}, "
                 f"Probability: {conflict['conflict_probability']:.2f}")
   
   if result['conflicts']['vehicle_conflicts']:
       for conflict in result['conflicts']['vehicle_conflicts']:
           print(f"Vehicle conflict detected! Track ID: {conflict['track_id']}, "
                 f"Probability: {conflict['conflict_probability']:.2f}")
   ```

### Step 5.2 – Configuration

**Config file: `configs/realtime_conflict.yaml`**

```yaml
grid:
  rows: 8
  cols: 6
  conflict_zone_rows: [6, 7]  # Bottom 2 rows
  conflict_zone_cols: [2, 3]   # Middle 2 columns

detection:
  proximity_threshold: 0.4      # 40% of frame height = "too close"
  coverage_threshold: 0.3        # 30% of conflict zone cells

pose:
  torso_angle_threshold: 15     # Degrees for forward lean
  leg_separation_threshold: 20  # Pixels
  leg_height_diff_threshold: 15 # Pixels

yolo:
  model_path: "yolo12n.pt"
  conf_threshold: 0.5
  iou_threshold: 0.45
```

### Step 5.3 – Performance Optimization

**For Real-Time Processing:**

1. **Model Selection:**
   - Use `yolo12n.pt` (nano) for fastest inference
   - Use `yolo12s.pt` (small) for balanced speed/accuracy
   - Use `yolo12m.pt` (medium) for better accuracy (slower)

2. **Frame Processing:**
   - Process every Nth frame if needed (skip frames for speed)
   - Use GPU acceleration (MPS/CUDA)
   - Reduce input resolution if needed

3. **Grid Optimization:**
   - Smaller grid (6×4) = faster processing
   - Larger grid (10×8) = more precise detection

**Target Performance:**
- **Desktop GPU:** 30+ FPS
- **Edge Device:** 10-15 FPS (with optimizations)

---

## Phase 6 — Evaluation Framework

### Step 6.1 – Conflict Detection Metrics

**Pipeline: `70_evaluate_system.py`**

#### Primary Metrics:

1. **Conflict Detection Accuracy:**
   - True Positives (TP): Correctly detected conflicts
   - False Positives (FP): Incorrect conflict alerts
   - False Negatives (FN): Missed conflicts
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

2. **Per-Class Performance:**
   - Pedestrian conflict detection accuracy
   - Vehicle conflict detection accuracy
   - Separate metrics for each conflict rule

3. **Temporal Analysis:**
   - Average time before conflict (lead time)
   - Conflict duration accuracy
   - False alarm rate per minute

4. **Grid-Based Metrics:**
   - Conflict zone occupancy accuracy
   - Grid cell prediction accuracy
   - Spatial localization error

#### Scenario Breakdown:

1. **Lighting Conditions:**
   - Day vs. night performance
   - Low-light conflict detection

2. **Traffic Density:**
   - Sparse traffic (1-3 objects)
   - Dense traffic (10+ objects)
   - Crowded scenarios

3. **Object Types:**
   - Pedestrians (walking, running, standing)
   - Vehicles (cars, trucks, buses, motorcycles)
   - Two-wheelers (bicycles, motorcycles)

4. **Pose Variations:**
   - Different pedestrian poses
   - Occlusion scenarios
   - Partial visibility

### Step 6.2 – Qualitative Analysis

1. **Visualization:**
   - Overlay grid and conflict zones on video
   - Highlight detected conflicts with bounding boxes
   - Show conflict probability scores
   - Display pose landmarks for pedestrians

2. **Failure Case Analysis:**
   - False positives: Analyze why non-conflicts were flagged
   - False negatives: Analyze why conflicts were missed
   - Pose estimation failures
   - Tracking failures

**Output:** `outputs/reports/evaluation_report.pdf` with metrics, visualizations, and analysis

---

## Phase 7 — Research Outputs and Documentation

### Methodology Section:

1. **Detection and Tracking:**
   - YOLO12 object detection (pre-trained or fine-tuned on RSUD20K/BadODD)
   - Multi-object tracking algorithm (ByteTrack or custom)
   - Evaluation metrics (mAP, tracking accuracy)

2. **Pose Estimation:**
   - MediaPipe pose estimation for pedestrians
   - Key landmark extraction (shoulders, hips, ankles)
   - Torso angle and leg position analysis

3. **Grid-Based Conflict Detection:**
   - Spatial grid division of camera feed
   - Conflict zone definition (ego corridor)
   - Two-rule conflict detection system:
     - **Rule 1:** Person pose inclination → conflict probability
     - **Rule 2:** Vehicle proximity + grid coverage → conflict probability
   - Temporal smoothing for conflict scores

4. **Real-Time Processing:**
   - Frame-by-frame processing pipeline
   - GPU acceleration (MPS/CUDA)
   - Real-time visualization and alerts

### Results Section:

1. **Conflict Detection Performance:**
   - Precision/Recall/F1 for pedestrian conflicts
   - Precision/Recall/F1 for vehicle conflicts
   - Overall system accuracy
   - Per-scenario breakdown

2. **Pose Analysis Performance:**
   - Torso angle detection accuracy
   - Leg position classification accuracy
   - Pose estimation reliability

3. **Grid-Based Analysis:**
   - Conflict zone localization accuracy
   - Grid cell prediction accuracy
   - Spatial resolution impact

4. **Real-Time Performance:**
   - Processing speed (FPS)
   - Latency measurements
   - Resource usage (GPU/CPU)

5. **Qualitative Results:**
   - Visualization of conflict detections
   - Pose estimation examples
   - Failure case analysis
   - Real-time inference demonstrations

### Limitations and Future Work:

1. **Current limitations:**
   - Image-space conflict detection (not true metric space)
   - Assumes static camera (ego-vehicle perspective)
   - Limited to monocular vision
   - Pose estimation may fail in occlusion scenarios
   - Grid-based approach is resolution-dependent

2. **Future directions:**
   - Stereo/multi-camera fusion for depth estimation
   - Metric-space conflict detection
   - Integration with LiDAR (if available)
   - Improved pose estimation for occluded scenarios
   - Adaptive grid sizing based on scene complexity
   - Multi-agent interaction modeling
   - Machine learning refinement of conflict rules

---

## Key Implementation Notes

### Data Management:

- Use efficient storage formats (HDF5, LMDB) for large trajectory datasets
- Cache preprocessed clips to avoid recomputation
- Version control for data splits and configurations

### Reproducibility:

- Set random seeds (PyTorch, NumPy, Python)
- Log all hyperparameters and configurations
- Save model checkpoints at regular intervals
- Document dataset versions and preprocessing steps

### Computational Resources:

- **YOLO12 inference:** Real-time on GPU (30+ FPS), 10-15 FPS on CPU
- **Tracking:** Minimal overhead, real-time capable
- **MediaPipe pose:** Real-time on CPU/GPU
- **Conflict detection:** Real-time (negligible overhead)
- **Full pipeline:** Real-time capable (30+ FPS on desktop GPU)
- **Fine-tuning YOLO12 (optional):** 1-2 days on single GPU (RTX 3090/4090)

### Code Quality:

- Modular design: Each phase as separate script
- Configuration files (YAML) for all hyperparameters
- Logging and visualization utilities
- Unit tests for critical functions (trajectory prediction, conflict metrics)

### MPS and CoreML Usage:

**Using MPS for Training/Inference:**
```python
from src.utils_device import get_device, optimize_for_mps

device = get_device()  # Automatically selects MPS/CUDA/CPU
model = model.to(device)
data = data.to(device)
```

**Converting Models to CoreML:**
```python
from src.utils_device import convert_to_coreml

mlmodel = convert_to_coreml(
    model=your_model,
    example_input=example_tensor,
    output_path="outputs/models/model.mlmodel"
)
```

**Best Practices:**
- Use MPS for training on Apple Silicon (faster than CPU)
- Convert trained models to CoreML for deployment
- Test models on both MPS and CPU to ensure compatibility
- Use `torch.backends.mps.empty_cache()` to manage memory on MPS

---

## Quick Start Guide

### Step 1: Setup Environment
```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics opencv-python mediapipe numpy scipy
```

### Step 2: Download YOLO12 Model
```python
from ultralytics import YOLO
model = YOLO('yolo12n.pt')  # Automatically downloads if not present
```

### Step 3: Run Real-Time Conflict Detection
```python
from src.realtime_conflict_pipeline import RealtimeConflictPipeline

# Initialize
pipeline = RealtimeConflictPipeline(
    yolo_model_path="yolo12n.pt",
    grid_rows=8,
    grid_cols=6
)

# Process webcam
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = pipeline.process_frame(frame)
    vis_frame = pipeline.visualize(frame, result)
    cv2.imshow('Conflict Detection', vis_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Next Steps

1. **Phase 0-1:** Set up environment and configure YOLO12
2. **Phase 2:** Implement trajectory extraction with MediaPipe
3. **Phase 3:** Compute kinematics from trajectories (optional)
4. **Phase 4:** Implement grid-based conflict detection
5. **Phase 5:** Real-time pipeline integration
6. **Phase 6:** Evaluation and metrics
7. **Phase 7:** Documentation and results

For detailed implementation help on specific components, please specify which phase you'd like to focus on next.
