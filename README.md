# Pedestrian Conflict Prediction: Advanced Trajectory-Based Framework

## Overview

This research project implements a novel framework for predicting pedestrian-vehicle conflicts using advanced trajectory prediction and weak-supervised learning. The system combines:
- **Multi-object detection and tracking** on dashcam videos
- **Sophisticated trajectory prediction** using attention mechanisms and social interaction modeling
- **Physics-aware conflict detection** with time-to-conflict (TTC) and post-encroachment time (PET) metrics
- **Weakly-supervised fusion model** that learns from trajectory-based auto-labels

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
    10_train_detector.py  # YOLO detector training
    20_run_tracker.py     # Multi-object tracking pipeline
    30_extract_trajectories.py  # Trajectory extraction with pose estimation
    35_compute_kinematics.py    # Velocity, acceleration, heading estimation
    40_predict_trajectories.py  # Multi-horizon trajectory prediction
    45_compute_conflict_metrics.py  # TTC, PET, conflict zone detection
    50_build_training_clips.py  # Clip generation for model training
    60_train_trajectory_predictor.py  # Trajectory prediction model
    65_train_conflict_predictor.py    # Conflict prediction fusion model
    70_evaluate_system.py  # Comprehensive evaluation
    80_distill_and_export.py  # Model distillation and deployment
  configs/
    detector.yaml         # Detection model configs
    tracker.yaml          # Tracking parameters
    trajectory.yaml       # Trajectory prediction configs
    conflict.yaml         # Conflict detection parameters
    model.yaml            # Fusion model architecture
```

### Step 0.2 – Environment Setup

**Core Dependencies:**
- Python 3.9+ (3.9 recommended for compatibility)
- PyTorch 2.0+ (with MPS support for Apple Silicon / CUDA for NVIDIA)
- Ultralytics YOLOv8
- OpenCV 4.8+
- MediaPipe (for pose estimation)
- ByteTrack (tracking implementation)
- PyTorch Geometric (for graph-based trajectory models)
- Transformers (HuggingFace, for temporal encoding)
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

### Step 1.2 – Detector Training

**Model: YOLOv8m (or YOLOv8l for better accuracy)**

  ```bash
  yolo detect train \
    data=configs/rsud20k.yaml \
      model=yolov8m.pt \
    epochs=150 \
      imgsz=640 \
    batch=16 \
    name=detector_rsud20k \
    project=outputs/models
```

**Key configurations:**
- Augmentations: Mosaic, mixup, random affine, color jitter, motion blur
- Loss weights: Balance pedestrian class (increase weight if imbalanced)
- Validation: Monitor per-class mAP, especially `person` class

**Optional: Domain adaptation**
- Fine-tune on combined RSUD20K + BadODD
- Use test-time augmentation for robustness

**Deliverable:** `outputs/models/detector_rsud20k.pt`

### Step 1.3 – Multi-Object Tracking

**Pipeline: `20_run_tracker.py`**

1. **Detection:**
   - Run YOLOv8 on each frame (GPU acceleration)
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

## Phase 4 — Conflict Detection and Weak Label Generation

### Step 4.1 – Dynamic Conflict Zone Definition

**Pipeline: `45_compute_conflict_metrics.py`**

Instead of a static grid, define a **dynamic ego-vehicle conflict zone** based on:

1. **Ego corridor (image-space approximation):**
   - Bottom-middle region of image (e.g., rows 6-7 out of 8, cols 2-3 out of 6)
   - Can be adaptive based on ego speed (if available)

2. **Expanded conflict zone:**
   - Include nearby cells that could lead to conflict
   - Buffer: 1-2 grid cells around ego corridor

3. **Time-to-Conflict (TTC) approximation:**
   - For predicted trajectory point at horizon Δt:
     - Check if point is in conflict zone
     - TTC ≈ Δt (if predicted to be in zone at Δt)
   - More sophisticated: Linear interpolation to find exact TTC

### Step 4.2 – Conflict Metrics Computation

#### Primary Metrics:

1. **Time-to-Conflict (TTC):**
   ```python
   # For each predicted horizon, check if point enters conflict zone
   for Δt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
       pred_point = trajectory_predictor.predict(t0, Δt)
       if pred_point in conflict_zone:
           TTC = Δt  # Approximate
           break
   ```

2. **Post-Encroachment Time (PET):**
   - Time difference between when object leaves conflict zone and when ego would have passed
   - Requires ego speed estimation (can be approximated from video)

3. **Minimum Distance (MinD):**
   - Closest predicted distance between object and ego corridor center
   - Lower MinD → higher conflict risk

4. **Collision Probability:**
     ```python
   # Use predicted uncertainty (if available)
   if has_uncertainty:
       # Integrate over uncertainty distribution within conflict zone
       collision_prob = integrate_gaussian_over_zone(pred_mean, pred_cov, conflict_zone)
   else:
       collision_prob = 1.0 if pred_point in zone else 0.0
   ```

#### Conflict Label Generation:

For each `(video, track_id, t0)`:

```python
conflict_labels = {}
for Δt in [1.0, 2.0, 3.0]:  # Primary horizons
    pred_point = predictions[Δt]
    in_zone = check_conflict_zone(pred_point)
    
    # Binary label
    y_Δt = 1 if in_zone else 0
    
    # Confidence score (0-1)
    w_Δt = compute_confidence(
        current_distance=current_pos_to_zone,
        velocity_towards_zone=velocity_component,
        trajectory_stability=trajectory_smoothness,
        prediction_confidence=model_confidence,
        time_to_conflict=TTC
    )
    
    conflict_labels[f"y_{Δt}s"] = y_Δt
    conflict_labels[f"w_{Δt}s"] = w_Δt
    conflict_labels[f"ttc_{Δt}s"] = TTC if in_zone else None
```

**Confidence computation factors:**
- **High confidence (w > 0.8):**
  - Object is close to conflict zone (NEAR_ROWS)
  - Velocity directly towards zone
  - Stable trajectory (low variance in velocity direction)
  - High prediction model confidence
  - For pedestrians: motion perpendicular to lane (crossing behavior)
  
- **Low confidence (w < 0.5):**
  - Object far from zone
  - Velocity parallel or away from zone
  - Unstable trajectory (frequent direction changes)
  - Short track history
  - Occluded or partially detected

**Output format:**
```json
{
  "video_id": "kitti_001",
  "track_id": 5,
  "t0": 4.0,
  "current_state": {
    "u": 450.2, "v": 610.5,
  "grid_cell": [6, 3],
    "velocity": {"v_u": 2.1, "v_v": -5.3},
    "distance_to_zone": 15.2
  },
  "conflict_labels": {
    "y_1s": 1, "w_1s": 0.92, "ttc_1s": 0.85,
    "y_2s": 1, "w_2s": 0.88, "ttc_2s": 1.8,
    "y_3s": 0, "w_3s": 0.35, "ttc_3s": null
  },
  "metrics": {
    "min_distance": 12.5,
    "pet": null,
    "collision_prob_1s": 0.89,
    "collision_prob_2s": 0.76
  }
}
```

**Output:** `outputs/autolabels/{video_id}.jsonl`

---

## Phase 5 — Training Clip Generation

### Step 5.1 – Temporal Clip Extraction

**Pipeline: `50_build_training_clips.py`**

For each labeled timepoint `(video_id, track_id, t0)`:

1. **Time window:**
   - Extract `[t0 - 1.5s, t0 + 0.5s]` (2-second window, future-leaning)
   - Sample frames at fixed FPS (e.g., 10 fps → 20 frames)

2. **Visual clip:**
   - Crop around track bbox (with padding: 1.5× bbox size)
   - Resize to fixed size: 128×128 or 112×112
   - Apply normalization (ImageNet stats)

3. **Kinematic sequence:**
   - For each frame in window:
     ```python
     features = [
         u_normalized, v_normalized,  # Position (0-1)
         v_u_normalized, v_v_normalized,  # Velocity (normalized)
         a_u_normalized, a_v_normalized,  # Acceleration
         heading_sin, heading_cos,  # Heading (sine/cosine encoding)
         class_embedding,  # One-hot or learned embedding
         grid_row_normalized, grid_col_normalized,
         distance_to_zone_normalized,
         speed_normalized
     ]
     ```

4. **Context features:**
   - Number of nearby objects (within 100 pixels)
   - Average speed of nearby objects
   - Ego speed (if available, else approximate from video motion)

5. **Labels:**
   - Multi-horizon conflict labels: `y_1s, y_2s, y_3s`
   - Confidence weights: `w_1s, w_2s, w_3s`
   - TTC values (if conflict predicted)

**Output structure:**
```python
{
    "clip_id": "kitti_001_track5_t4.0",
    "video_id": "kitti_001",
    "track_id": 5,
    "t0": 4.0,
    "visual_clip": "clips/kitti_001_track5_t4.0_visual.npy",  # Shape: [T, C, H, W]
    "kinematic_seq": "clips/kitti_001_track5_t4.0_kin.npy",   # Shape: [T, F]
    "labels": {
        "y_1s": 1, "w_1s": 0.92,
        "y_2s": 1, "w_2s": 0.88,
        "y_3s": 0, "w_3s": 0.35
    },
    "metadata": {
        "class": "person",
        "num_frames": 20,
        "fps": 10
    }
}
```

**Metadata index:** `outputs/clips/metadata.csv` with all clip references

---

## Phase 6 — Conflict Prediction Model Training

### Step 6.1 – Model Architecture

**Pipeline: `65_train_conflict_predictor.py`**

**Two-stage approach:**

#### Stage 1: Trajectory Prediction Model (if not pre-trained)

Train the trajectory predictor from Phase 3 on all trajectory data (self-supervised).

#### Stage 2: Conflict Prediction Fusion Model

**Architecture:**

1. **Vision Encoder (Spatio-Temporal):**
   - Input: Video clip `[T, C, H, W]` (e.g., 20 frames × 3 channels × 128×128)
   - Backbone options:
     - **3D ResNet-18** or **R(2+1)D-18** (recommended for efficiency)
     - **TimeSformer-tiny** (transformer-based, more compute)
     - **I3D** (Inflated 3D ConvNet)
   - Output: Temporal feature sequence `[T, D_v]` or pooled `[D_v]`

2. **Kinematics Encoder (Temporal):**
   - Input: Kinematic sequence `[T, F]` (F = feature dimension)
   - Architecture:
     - **Option A:** 1-2 layer BiLSTM/GRU (hidden: 64-128)
     - **Option B:** Transformer encoder (small, 2-4 layers)
     - **Option C:** Temporal Convolutional Network (TCN)
   - Output: Encoded kinematics `[D_k]`

3. **Fusion Module:**
   ```python
   # Option A: Simple concatenation
   z = concat(f_vision, f_kinematics)  # [D_v + D_k]
   
   # Option B: Cross-attention (more expressive)
   f_fused = CrossAttention(f_vision, f_kinematics)
   z = concat(f_vision, f_fused, f_kinematics)
   
   # Option C: Late fusion with learned weights
   alpha = sigmoid(MLP([f_vision, f_kinematics]))
   z = alpha * f_vision + (1 - alpha) * f_kinematics
   ```

4. **Conflict Prediction Heads:**
   ```python
   # Shared backbone
   h = MLP(z, hidden=[256, 128], activation='ReLU', dropout=0.3)
   
   # Multi-horizon heads
   logits_1s = Linear(h, 1)
   logits_2s = Linear(h, 1)
   logits_3s = Linear(h, 1)
   
   # Optional: TTC regression head
   ttc_logits = Linear(h, 1)  # Predict TTC (sigmoid * max_ttc)
   ```

5. **Output:**
   - Probabilities: `p_1s = sigmoid(logits_1s)`, etc.
   - TTC: `ttc_pred = sigmoid(ttc_logits) * 3.0` (if using TTC head)

### Step 6.2 – Training Objective

**Loss function:**

```python
# Weighted Binary Cross-Entropy for each horizon
loss_conflict = 0
for Δt in [1.0, 2.0, 3.0]:
    w = labels[f'w_{Δt}s']  # Confidence weight
    y = labels[f'y_{Δt}s']   # Binary label
    p = predictions[f'p_{Δt}s']
    
    # Focal loss variant for hard examples
    bce = -[y * log(p) + (1-y) * log(1-p)]
    focal = alpha * (1 - p)^gamma * bce  # gamma=2, alpha=0.25
    
    loss_conflict += w * focal

# Optional: TTC regression loss
if predict_ttc:
    ttc_true = labels['ttc'] if labels['y_1s'] == 1 else 0.0
    loss_ttc = L1(ttc_pred, ttc_true) * mask  # Only on positive samples
    loss_total = loss_conflict + lambda_ttc * loss_ttc
else:
    loss_total = loss_conflict
```

**Training strategy:**

1. **Curriculum learning:**
   - Phase 1: Train only high-confidence samples (w ≥ 0.8)
   - Phase 2: Gradually include medium-confidence (w ≥ 0.6)
   - Phase 3: Full dataset

2. **Data augmentation:**
   - Temporal: Random frame dropping, temporal jittering
   - Spatial: Random crop, horizontal flip (adjust kinematics accordingly)
   - Kinematic noise: Add small Gaussian noise to velocity/acceleration

3. **Optimization:**
   - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
   - Learning rate schedule: Cosine annealing with warmup
   - Batch size: 16-32 (depending on GPU memory)
   - Epochs: 30-50 (with early stopping)

4. **Validation:**
   - Monitor: Val loss, Precision@Recall, Lead-time metrics
   - Early stopping: Patience=10 epochs on val loss

**Deliverable:** `outputs/models/conflict_predictor_fusion.pt`

---

## Phase 7 — Evaluation Framework

### Step 7.1 – Comprehensive Metrics

**Pipeline: `70_evaluate_system.py`**

#### Trajectory Prediction Metrics:

1. **Average Displacement Error (ADE):**
   - Mean L2 distance between predicted and true positions

2. **Final Displacement Error (FDE):**
   - L2 distance at final horizon (e.g., 3s)

3. **Per-horizon ADE:**
   - ADE at each prediction horizon

#### Conflict Prediction Metrics:

1. **Classification Metrics:**
   - Precision, Recall, F1-score (per horizon)
   - Precision-Recall curves
   - ROC-AUC

2. **Lead-time Metrics:**
   - **Lead-time@Precision:**
     - For each true conflict event, find earliest prediction ≥ threshold that occurred before event
     - Compute median/mean lead time
   - **Lead-time@Recall:**
     - Similar, but focus on recall of events

3. **Calibration:**
   - Expected Calibration Error (ECE)
   - Brier Score
   - Reliability diagrams

4. **TTC Estimation:**
   - Mean Absolute Error (MAE) of predicted vs. true TTC
   - Only on samples where conflict occurs

5. **Scenario Breakdown:**
   - Day vs. night
   - Dense vs. sparse traffic
   - Pedestrians vs. vehicles vs. two-wheelers
   - Urban vs. highway (if available)
   - Weather conditions (if available)

#### Ablation Studies:

1. **Component ablation:**
   - Vision only vs. Kinematics only vs. Fusion
   - Different fusion methods
   - With/without trajectory prediction module

2. **Architecture ablation:**
   - Different backbone choices (3D CNN vs. Transformer)
   - Different horizon combinations
   - Impact of confidence weighting

3. **Data ablation:**
   - Impact of trajectory prediction quality
   - Effect of confidence thresholds
   - Curriculum learning impact

### Step 7.2 – Qualitative Analysis

1. **Visualization tools:**
   - Overlay predicted trajectories on video frames
   - Show conflict zone and predicted conflict probabilities over time
   - Highlight true vs. predicted conflicts

2. **Failure case analysis:**
   - False positives: High prediction but no conflict
   - False negatives: Missed conflicts
   - Analyze patterns in failures

**Output:** `outputs/reports/evaluation_report.pdf` with metrics, plots, and analysis

---

## Phase 8 — Model Distillation and Deployment

### Step 8.1 – Knowledge Distillation

**Pipeline: `80_distill_and_export.py`**

**Student model architecture:**
- **Vision:** MobileNetV3-Small 2D CNN with temporal average pooling (or tiny 3D CNN)
- **Kinematics:** 1D CNN (3-5 layers) or small GRU (hidden: 32)
- **Fusion:** Simple concatenation + 2-layer MLP
- **Target:** < 10M parameters, < 50ms inference on edge device

**Distillation loss:**
```python
# Hard labels (from autolabels)
loss_hard = weighted_BCE(y_true, p_student)

# Soft labels (from teacher)
loss_soft = KL_divergence(p_teacher, p_student)

# Combined
loss = alpha * loss_hard + (1 - alpha) * T^2 * loss_soft
# T = temperature (typically 3-5)
```

**Training:**
- Lower resolution: 96×96 frames
- Fewer frames: 12-16 frames per clip
- Batch size: 32-64
- Learning rate: 1e-3 (higher for student)

**Deliverable:** `outputs/models/conflict_predictor_mobile.pt`

### Step 8.2 – Model Export

1. **ONNX export:**
  ```python
   torch.onnx.export(
       model, example_input,
       "conflict_predictor.onnx",
       opset_version=17,
       input_names=['visual', 'kinematics'],
       output_names=['p_1s', 'p_2s', 'p_3s', 'ttc']
   )
   ```

2. **TensorRT optimization (optional):**
   - For NVIDIA GPUs/Jetson devices
   - FP16 or INT8 quantization

3. **TensorFlow Lite (optional):**
   - For mobile/embedded devices
   - Quantization: INT8

### Step 8.3 – Real-time Inference Pipeline

**Inference workflow:**

1. **Frame processing:**
   - Run detector → tracker → trajectory extraction
   - Maintain trajectory buffers (sliding window)

2. **Prediction:**
   - For each active track:
     - Extract visual clip and kinematic sequence
     - Run conflict predictor
     - Get probabilities `p_1s, p_2s, p_3s`

3. **Alert logic:**
   ```python
   p_max = max(p_1s, p_2s, p_3s)
   most_dangerous_track = argmax(p_max over tracks)
   
   if p_max[most_dangerous_track] >= 0.8:
       # High alert
       trigger_alert(level='high', track_id=most_dangerous_track)
   elif p_max[most_dangerous_track] >= 0.6:
       # Warning
       trigger_alert(level='warning', track_id=most_dangerous_track)
   
   # Cooldown: 2 seconds between alerts
   # Suppress if trajectory unstable (low confidence)
   ```

**Target performance:**
- End-to-end latency: < 100ms per frame (10+ FPS)
- On edge device (Jetson Nano/Orin): Optimize for real-time

---

## Phase 9 — Research Outputs and Documentation

### Methodology Section:

1. **Detection and Tracking:**
   - YOLOv8 training on RSUD20K/BadODD
   - ByteTrack multi-object tracking
   - Evaluation metrics (mAP, tracking accuracy)

2. **Trajectory Extraction:**
   - MediaPipe pose estimation for pedestrians
   - Kinematics computation (velocity, acceleration, heading)
   - Trajectory smoothing and quality metrics

3. **Trajectory Prediction:**
   - Attention-based temporal encoder architecture
   - Multi-horizon prediction with uncertainty
   - Social interaction modeling (if using GNN)

4. **Conflict Detection:**
   - Dynamic conflict zone definition
   - TTC and PET computation
   - Weak label generation with confidence weighting

5. **Fusion Model:**
   - Spatio-temporal vision encoder
   - Kinematics encoder
   - Cross-modal fusion strategies
   - Multi-horizon prediction heads

6. **Training Strategy:**
   - Weakly-supervised learning with confidence weights
   - Curriculum learning
   - Data augmentation

### Results Section:

1. **Trajectory Prediction Performance:**
   - ADE/FDE across horizons
   - Comparison with baseline (linear extrapolation, constant velocity)

2. **Conflict Prediction Performance:**
   - Precision/Recall/F1 per horizon
   - Lead-time metrics
   - Calibration analysis

3. **Ablation Studies:**
   - Component contributions
   - Architecture choices
   - Impact of trajectory prediction quality

4. **Qualitative Results:**
   - Visualization of successful predictions
   - Failure case analysis
   - Real-time inference examples

### Limitations and Future Work:

1. **Current limitations:**
   - Image-space conflict detection (not true metric space)
   - Assumes static camera (ego-vehicle perspective)
   - Limited to monocular vision
   - Weak labels may contain noise

2. **Future directions:**
   - Stereo/multi-camera fusion for depth estimation
   - Metric-space TTC computation
   - Integration with LiDAR (if available)
   - Active learning to refine weak labels
   - Multi-agent scene understanding
   - Long-term trajectory forecasting (5-10 seconds)

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

- **Training detector:** 1-2 days on single GPU (RTX 3090/4090)
- **Trajectory extraction:** Depends on video length (can parallelize)
- **Trajectory predictor training:** 4-8 hours
- **Conflict predictor training:** 8-12 hours
- **Full pipeline:** ~1 week for initial experiments

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

## Next Steps

1. **Start with Phase 0-1:** Set up environment and train detector
2. **Phase 2-3:** Implement trajectory extraction and prediction
3. **Phase 4:** Build conflict detection pipeline
4. **Phase 5-6:** Train fusion model
5. **Phase 7:** Comprehensive evaluation
6. **Phase 8:** Distillation and deployment preparation
7. **Phase 9:** Documentation and paper writing

For detailed pseudocode or implementation help on specific phases, please specify which component you'd like to focus on next.
