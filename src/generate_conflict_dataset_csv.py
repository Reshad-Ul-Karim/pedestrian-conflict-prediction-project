#!/usr/bin/env python3
"""
Generate CSV dataset with conflict risk features for all images in RSUD20K
Processes all images and extracts features for each detected person
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import math
import warnings
import logging
import os
from tqdm import tqdm

# Suppress MediaPipe warnings more aggressively
# Must set BEFORE importing MediaPipe (done in visualize_conflict_risk.py)
os.environ['GLOG_minloglevel'] = '2'  # Only ERROR and FATAL (0=INFO, 1=WARNING, 2=ERROR)
os.environ['GLOG_logtostderr'] = '0'  # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '2'  # Only show ERROR and FATAL on stderr

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*NORM_RECT.*')
warnings.filterwarnings('ignore', message='.*IMAGE_DIMENSIONS.*')

# Import required modules
try:
    from visualize_conflict_risk import (
        RoadGrid, PoseAnalyzer, compute_conflict_risk, ROAD_DETECTOR_AVAILABLE,
        extract_spatial_relationships, extract_scene_context, extract_multiscale_features
    )
except ImportError as e:
    print(f"Error importing from visualize_conflict_risk: {e}")
    print("Make sure visualize_conflict_risk.py is in the src/ directory")
    exit(1)

try:
    from road_detector import RoadDetector
except ImportError:
    print("Warning: road_detector not available. Road detection will be disabled.")
    RoadDetector = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: YOLO not available. Install ultralytics: pip install ultralytics")
    exit(1)


def detect_available_device():
    """Detect available device (MPS, CUDA, or CPU)"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return '0'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def load_yolo_annotations(label_path, img_width, img_height, person_class_id=0):
    """
    Load YOLO format annotations from label file and extract ONLY person bounding boxes
    
    Args:
        label_path: Path to .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        person_class_id: Class ID for person (0 in RSUD20K - person is first class)
    
    Returns:
        List of (x1, y1, x2, y2) bounding boxes in absolute pixel coordinates (ONLY person class)
    """
    person_bboxes = []
    
    if not label_path.exists():
        return person_bboxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                # RSUD20K: person is class_id=0 (first class in classes.txt)
                # ONLY accept person class, reject all other classes (rickshaw, car, etc.)
                if class_id == person_class_id:
                    # YOLO format: class_id x_center y_center width height (normalized 0-1)
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    
                    # Convert to absolute pixel coordinates
                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    width = width_norm * img_width
                    height = height_norm * img_height
                    
                    # Convert to (x1, y1, x2, y2) format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    # Only add if bbox is valid (has area)
                    if x2 > x1 and y2 > y1:
                        person_bboxes.append((float(x1), float(y1), float(x2), float(y2)))
    except Exception as e:
        print(f"  Warning: Failed to load annotations from {label_path}: {e}")
    
    return person_bboxes


def process_image(image_path, yolo_model, road_detector, pose_analyzer, 
                  calibration_file=None, grid_rows=12, grid_cols=12, device='cpu',
                  use_ground_truth=True, rsud20k_dir=None):
    """
    Process a single image and extract features for all detected persons
    
    Args:
        image_path: Path to image file
        yolo_model: YOLO model instance (optional, used only if use_ground_truth=False)
        road_detector: RoadDetector instance (or None)
        pose_analyzer: PoseAnalyzer instance
        calibration_file: Path to calibration JSON file (optional)
        grid_rows: Number of grid rows
        grid_cols: Number of grid columns
        device: Device for YOLO inference
        use_ground_truth: If True, use RSUD20K ground truth annotations instead of YOLO
        rsud20k_dir: Path to rsud20k directory (needed to find label files)
    
    Returns:
        List of feature dictionaries (one per detected person)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    h, w = image.shape[:2]
    image_id = image_path.stem
    image_path_str = str(image_path)
    
    # Detect road with SegFormer (only if needed, but we'll skip for pure YOLO+pose approach)
    road_mask = None
    road_polygon = None
    sidewalk_mask = None
    
    # Skip road detection if we're using pure YOLO+pose features
    # if road_detector:
    #     try:
    #         road_mask, road_polygon, sidewalk_mask = road_detector.detect_road(image)
    #     except Exception as e:
    #         print(f"  Warning: Road detection failed for {image_id}: {e}")
    
    # Initialize RoadGrid (minimal, just for compatibility - won't use road features)
    try:
        road_grid = RoadGrid(
            img_width=w,
            img_height=h,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            road_mask=road_mask,
            road_polygon=road_polygon,
            sidewalk_mask=sidewalk_mask,
            calibration_file=calibration_file
        )
    except Exception as e:
        print(f"  Warning: RoadGrid initialization failed for {image_id}: {e}")
        return []
    
    # Load person bounding boxes from ground truth annotations or YOLO
    person_bboxes = []
    yolo_confidences = []
    all_detections = []  # Store ALL detections for spatial relationships
    
    if use_ground_truth and rsud20k_dir:
        # Use RSUD20K ground truth annotations
        # Find corresponding label file
        # Image path: rsud20k/images/train/train0.jpg
        # Label path: rsud20k/labels/train/train0.txt
        image_path_rel = image_path.relative_to(Path(rsud20k_dir) / 'images')
        label_path = Path(rsud20k_dir) / 'labels' / image_path_rel.with_suffix('.txt')
        
        # RSUD20K: person is class_id=0 (first class in classes.txt)
        # This will ONLY load person annotations, filtering out all other classes
        person_bboxes = load_yolo_annotations(label_path, w, h, person_class_id=0)
        # For ground truth, set confidence to 1.0 (perfect detection)
        yolo_confidences = [1.0] * len(person_bboxes)
        
        # For spatial relationships, we need ALL objects, so use YOLO for that
        if yolo_model:
            try:
                results = yolo_model(image, conf=0.25, iou=0.45, verbose=False, device=device)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = yolo_model.names[cls] if hasattr(yolo_model, 'names') else ''
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            if conf > 0.25:
                                all_detections.append({
                                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                    'class': class_name,
                                    'class_id': cls,
                                    'confidence': conf
                                })
            except Exception as e:
                print(f"  Warning: YOLO detection for all objects failed: {e}")
        
        if not person_bboxes:
            return []  # No persons in ground truth
    else:
        # Use YOLO detection for both persons and all objects
        if yolo_model:
            try:
                results = yolo_model(image, conf=0.25, iou=0.45, verbose=False, device=device)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = yolo_model.names[cls] if hasattr(yolo_model, 'names') else ''
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            if conf > 0.25:
                                detection = {
                                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                    'class': class_name,
                                    'class_id': cls,
                                    'confidence': conf
                                }
                                all_detections.append(detection)
                                
                                # Accept if class is 0 (person) or name contains 'person'
                                if cls == 0 or 'person' in class_name.lower():
                                    person_bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                                    yolo_confidences.append(conf)
            except Exception as e:
                print(f"  Warning: YOLO detection failed for {image_id}: {e}")
                return []
        
        if not person_bboxes:
            return []  # No persons detected
    
    # Extract features for each person
    features_list = []
    
    for person_id, bbox in enumerate(person_bboxes):
        x1, y1, x2, y2 = bbox
        yolo_conf = yolo_confidences[person_id]
        
        # Compute bbox features
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Extract pose
        pose_data = pose_analyzer.extract_pose(image, bbox) if pose_analyzer else None
        
        # Extract ADVANCED pose features
        advanced_pose_features = {}
        if pose_analyzer and pose_data:
            advanced_pose_features = pose_analyzer.extract_advanced_pose_features(pose_data)
        
        # Compute conflict risk
        person_center = (bbox_center_x, bbox_center_y)
        risk_data = compute_conflict_risk(
            person_center, pose_data, road_grid, pose_analyzer, bbox=bbox
        )
        
        # Extract SPATIAL RELATIONSHIP features
        spatial_features = extract_spatial_relationships(
            bbox, all_detections, (h, w)
        )
        
        # Extract SCENE CONTEXT features
        scene_features = extract_scene_context(
            image, road_grid, all_detections, bbox
        )
        
        # Extract MULTI-SCALE SPATIAL features
        multiscale_features = extract_multiscale_features(
            bbox, road_grid, (h, w)
        )
        
        # Check if bbox is inside trapezoid (more accurate check)
        bbox_inside_manual = False
        bbox_inside_segformer = False
        
        if road_grid.manual_trapezoid_mask is not None:
            bbox_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            corners_inside_manual = sum(1 for corner in bbox_corners 
                                       if road_grid.is_in_street(corner[0], corner[1], use_segformer=False))
            bbox_center = (bbox_center_x, bbox_center_y)
            bbox_inside_manual = (corners_inside_manual >= 2 or 
                                 road_grid.is_in_street(bbox_center[0], bbox_center[1], use_segformer=False))
        
        if road_grid.road_mask is not None:
            bbox_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            corners_inside_segformer = sum(1 for corner in bbox_corners 
                                          if road_grid.is_in_street(corner[0], corner[1], use_segformer=True))
            bbox_center = (bbox_center_x, bbox_center_y)
            bbox_inside_segformer = (corners_inside_segformer >= 2 or 
                                    road_grid.is_in_street(bbox_center[0], bbox_center[1], use_segformer=True))
        
        # Extract RAW pose features only (no computed scores)
        pose_detected = pose_data is not None
        pose_confidence = pose_data.get('confidence', 0.0) if pose_data else 0.0
        body_orientation_angle = None
        
        body_orientation = risk_data.get('body_orientation')
        if body_orientation:
            body_orientation_angle = body_orientation.get('angle_deg')
        
        # Get RAW angles (not computed scores)
        angle_to_manual_trapezoid = risk_data.get('angle_to_street_manual')
        angle_to_segformer_road = risk_data.get('angle_to_street_segformer')
        
        # ENHANCED: Improved thresholds for better category separation
        # Lowered thresholds to create more balanced distribution
        # These will be optimized during training, but start with better defaults
        conflict_score = risk_data.get('conflict_score', 0.0)
        if conflict_score > 0.65:  # Lowered from 0.7 for better HIGH detection
            risk_level = 'HIGH'
        elif conflict_score > 0.35:  # Lowered from 0.4 for better MED separation
            risk_level = 'MED'
        else:
            risk_level = 'LOW'
        
        # Compute raw position agreement (simple boolean logic, not a computed score)
        in_manual = risk_data.get('in_manual', False)
        in_segformer = risk_data.get('in_segformer', False)
        position_agreement = 1.0 if (in_manual == in_segformer) else 0.0  # Simple agreement boolean
        
        # Create feature dictionary - ENHANCED with new features
        features = {
            # Identifiers
            'image_id': image_id,
            'person_id': person_id,
            
            # Detection (RAW)
            'yolo_confidence': yolo_conf,
            'bbox_x1': x1,
            'bbox_y1': y1,
            'bbox_x2': x2,
            'bbox_y2': y2,
            'bbox_center_x': bbox_center_x,
            'bbox_center_y': bbox_center_y,
            'bbox_area': bbox_area,
            
            # Position (RAW booleans and types - no computed scores)
            'in_manual_trapezoid': in_manual,  # Raw boolean
            'bbox_inside_manual': bbox_inside_manual,  # Raw boolean
            'in_segformer_road': in_segformer,  # Raw boolean
            'bbox_inside_segformer': bbox_inside_segformer,  # Raw boolean
            'position_type': risk_data.get('position_type', 'unknown'),  # Raw category
            'position_agreement': position_agreement,  # Simple boolean agreement (not computed score)
            
            # Pose (RAW - no computed scores)
            'pose_detected': pose_detected,  # Raw boolean
            'pose_confidence': pose_confidence,  # Raw confidence from MediaPipe
            'angle_to_manual_trapezoid': angle_to_manual_trapezoid if angle_to_manual_trapezoid is not None else 0.0,  # Raw angle
            'angle_to_segformer_road': angle_to_segformer_road if angle_to_segformer_road is not None else 0.0,  # Raw angle
            'body_orientation_angle': body_orientation_angle if body_orientation_angle is not None else 0.0,  # Raw angle
            
            # NEW: Advanced Pose Features
            'torso_lean_angle': advanced_pose_features.get('torso_lean_angle', 0.0),
            'head_orientation_angle': advanced_pose_features.get('head_orientation_angle', 0.0),
            'leg_separation': advanced_pose_features.get('leg_separation', 0.0),
            'estimated_stride_ratio': advanced_pose_features.get('estimated_stride_ratio', 0.0),
            'arm_crossing_score': advanced_pose_features.get('arm_crossing_score', 0.0),
            
            # NEW: Spatial Relationship Features
            'min_distance_to_vehicle': spatial_features.get('min_distance_to_vehicle', float('inf')),
            'min_distance_to_vehicle_norm': spatial_features.get('min_distance_to_vehicle_norm', 1.0),
            'nearby_pedestrians_count': spatial_features.get('nearby_pedestrians_count', 0),
            'relative_x_to_vehicle': spatial_features.get('relative_x_to_vehicle', 0.0),
            'relative_y_to_vehicle': spatial_features.get('relative_y_to_vehicle', 0.0),
            
            # NEW: Scene Context Features
            'traffic_density': scene_features.get('traffic_density', 0.0),
            'pedestrian_density': scene_features.get('pedestrian_density', 0.0),
            'road_area_ratio': scene_features.get('road_area_ratio', 0.0),
            'distance_to_road_center': scene_features.get('distance_to_road_center', 1.0),
            'road_segments_count': scene_features.get('road_segments_count', 0),
            'is_intersection': scene_features.get('is_intersection', 0.0),
            'image_blur_score': scene_features.get('image_blur_score', 0.0),
            'image_brightness': scene_features.get('image_brightness', 0.5),
            
            # NEW: Multi-scale Spatial Features
            'local_road_ratio': multiscale_features.get('local_road_ratio', 0.0),
            'regional_road_ratio': multiscale_features.get('regional_road_ratio', 0.0),
            'global_road_ratio': multiscale_features.get('global_road_ratio', 0.0),
            'distance_to_left_edge': multiscale_features.get('distance_to_left_edge', 0.5),
            'distance_to_right_edge': multiscale_features.get('distance_to_right_edge', 0.5),
            'distance_to_top_edge': multiscale_features.get('distance_to_top_edge', 0.5),
            'distance_to_bottom_edge': multiscale_features.get('distance_to_bottom_edge', 0.5),
            'position_x_norm': multiscale_features.get('position_x_norm', 0.5),
            'position_y_norm': multiscale_features.get('position_y_norm', 0.5),
            
            # Target variable (conflict_score) - NOT a feature, just the target
            'conflict_score': conflict_score,
            'risk_level': risk_level,
        }
        
        # Handle infinite values (replace with large finite value)
        for key, value in features.items():
            if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                if math.isinf(value) and value > 0:
                    features[key] = 10000.0  # Large finite value
                elif math.isinf(value) and value < 0:
                    features[key] = -10000.0
                else:
                    features[key] = 0.0
        
        features_list.append(features)
    
    return features_list


def generate_csv_dataset(rsud20k_dir, output_csv, calibration_file=None, 
                         yolo_model_path=None, grid_rows=12, grid_cols=12,
                         splits=['train', 'val', 'test'], use_ground_truth=True):
    """
    Generate CSV dataset from RSUD20K images
    
    Args:
        rsud20k_dir: Path to rsud20k directory
        output_csv: Path to output CSV file
        calibration_file: Path to calibration JSON file (optional, auto-detected from train/ if not provided)
        yolo_model_path: Path to YOLO model (optional, uses yolo12n.pt by default)
        grid_rows: Number of grid rows (will be overridden by calibration file if available)
        grid_cols: Number of grid columns (will be overridden by calibration file if available)
        splits: List of splits to process ['train', 'val', 'test']
    """
    rsud20k_path = Path(rsud20k_dir)
    images_dir = rsud20k_path / 'images'
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Auto-detect calibration file if not provided
    if calibration_file is None:
        calibration_file = images_dir / 'train' / 'grid_calibration.json'
        if not calibration_file.exists():
            calibration_file = None
    
    # Load grid dimensions from calibration file if available
    if calibration_file and Path(calibration_file).exists():
        try:
            with open(calibration_file, 'r') as f:
                cal_data = json.load(f)
                grid_rows = cal_data.get('grid_rows', grid_rows)
                grid_cols = cal_data.get('grid_cols', grid_cols)
                print(f"✓ Grid dimensions from calibration: {grid_rows}x{grid_cols}")
        except Exception as e:
            print(f"⚠ Warning: Could not load grid dimensions from calibration: {e}")
    
    # Initialize models
    print("Initializing models...")
    
    # YOLO model (only needed if not using ground truth)
    yolo_model = None
    if not use_ground_truth:
        if yolo_model_path:
            yolo_path = Path(yolo_model_path)
            if yolo_path.exists():
                yolo_model = YOLO(str(yolo_path))
                print(f"✓ YOLO model loaded from: {yolo_path}")
            else:
                print(f"⚠ YOLO model not found: {yolo_path}, using default YOLO12n")
                yolo_model = YOLO('yolo12n.pt')
        else:
            yolo_model = YOLO('yolo12n.pt')
            print("✓ Using default YOLO12n model")
    else:
        print("✓ Using RSUD20K ground truth annotations (YOLO model not needed)")
    
    # Road detector (SegFormer)
    road_detector = None
    if ROAD_DETECTOR_AVAILABLE and RoadDetector is not None:
        try:
            road_detector = RoadDetector()
            print("✓ SegFormer road detector loaded")
        except Exception as e:
            print(f"⚠ Road detector initialization failed: {e}")
            road_detector = None
    else:
        print("⚠ Road detector not available (SegFormer will be skipped)")
    
    # Pose analyzer
    pose_analyzer = PoseAnalyzer()
    if pose_analyzer.pose is None:
        print("⚠ MediaPipe pose not available")
    else:
        print("✓ MediaPipe pose analyzer loaded")
    
    # Detect device
    device = detect_available_device()
    print(f"✓ Using device: {device}")
    
    # Print calibration status (already loaded grid dimensions above)
    if calibration_file and Path(calibration_file).exists():
        print(f"✓ Calibration file will be used: {calibration_file}")
    else:
        print("⚠ No calibration file - using default grid dimensions")
    
    # Collect all image paths
    all_image_paths = []
    for split in splits:
        split_dir = images_dir / split
        if split_dir.exists():
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            for ext in image_extensions:
                all_image_paths.extend(split_dir.glob(ext))
    
    all_image_paths = sorted(all_image_paths)
    print(f"\nFound {len(all_image_paths)} images to process")
    
    if not all_image_paths:
        print("No images found!")
        return
    
    # Process all images with batch writing to CSV (memory-efficient for 20k+ images)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write header first - ALL RAW FEATURES (no deterministic/computed scores)
    essential_columns = [
        # Identifiers
        'image_id', 'person_id',
        
        # Detection (RAW)
        'yolo_confidence',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_center_x', 'bbox_center_y', 'bbox_area',
        
        # Position (RAW booleans and types)
        'in_manual_trapezoid', 'bbox_inside_manual',
        'in_segformer_road', 'bbox_inside_segformer',
        'position_type', 'position_agreement',
        
        # Pose (RAW)
        'pose_detected', 'pose_confidence',
        'angle_to_manual_trapezoid', 'angle_to_segformer_road',
        'body_orientation_angle',
        
        # Advanced Pose Features
        'torso_lean_angle',
        'head_orientation_angle',
        'leg_separation',
        'estimated_stride_ratio',
        'arm_crossing_score',
        
        # Spatial Relationship Features
        'min_distance_to_vehicle',
        'min_distance_to_vehicle_norm',
        'nearby_pedestrians_count',
        'relative_x_to_vehicle',
        'relative_y_to_vehicle',
        
        # Scene Context Features
        'traffic_density',
        'pedestrian_density',
        'road_area_ratio',
        'distance_to_road_center',
        'road_segments_count',
        'is_intersection',
        'image_blur_score',
        'image_brightness',
        
        # Multi-scale Spatial Features
        'local_road_ratio',
        'regional_road_ratio',
        'global_road_ratio',
        'distance_to_left_edge',
        'distance_to_right_edge',
        'distance_to_top_edge',
        'distance_to_bottom_edge',
        'position_x_norm',
        'position_y_norm',
        
        # Target variable (NOT a feature, just the target)
        'conflict_score', 'risk_level',
        
        # REMOVED deterministic features:
        # 'pose_score', 'position_score', 'distance_score_segformer', 'agreement_score'
    ]
    
    # Initialize CSV with header
    df_header = pd.DataFrame(columns=essential_columns)
    df_header.to_csv(output_path, index=False, mode='w')
    
    # Process images in batches and append to CSV (memory-efficient for 20k+ images)
    write_batch_size = 1000  # Write to CSV every 1000 person records
    all_features = []
    total_persons = 0
    processed_images = 0
    failed_images = 0
    
    print(f"\nProcessing {len(all_image_paths)} images...")
    print("This may take a while for 20k+ images. Progress will be saved incrementally.\n")
    
    for image_path in tqdm(all_image_paths, desc="Processing images"):
        try:
            features_list = process_image(
                image_path, yolo_model, road_detector, pose_analyzer,
                calibration_file=calibration_file,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                device=device,
                use_ground_truth=use_ground_truth,
                rsud20k_dir=rsud20k_dir
            )
            
            if features_list:
                all_features.extend(features_list)
                total_persons += len(features_list)
                processed_images += 1
            
            # Write batch to CSV periodically (memory-efficient)
            if len(all_features) >= write_batch_size:
                df_batch = pd.DataFrame(all_features)
                # Ensure all columns exist
                for col in essential_columns:
                    if col not in df_batch.columns:
                        df_batch[col] = None
                df_batch = df_batch[essential_columns]
                df_batch.to_csv(output_path, index=False, mode='a', header=False)
                all_features = []  # Clear batch from memory
                
        except Exception as e:
            failed_images += 1
            print(f"\n⚠ Error processing {image_path.name}: {e}")
            # Continue processing other images
            continue
    
    # Write remaining features
    if all_features:
        df_batch = pd.DataFrame(all_features)
        # Ensure all columns exist
        for col in essential_columns:
            if col not in df_batch.columns:
                df_batch[col] = None
        df_batch = df_batch[essential_columns]
        df_batch.to_csv(output_path, index=False, mode='a', header=False)
    
    # Load final CSV to get statistics
    print(f"\n✓ Processing complete!")
    print(f"  Processed images: {processed_images}/{len(all_image_paths)}")
    print(f"  Failed images: {failed_images}")
    print(f"  Total person records: {total_persons}")
    
    # Read final CSV for statistics
    try:
        df = pd.read_csv(output_path)
    
        print(f"\n✓ CSV saved to: {output_path}")
        print(f"  Total records: {len(df)}")
        print(f"  Unique images: {df['image_id'].nunique()}")
        if df['image_id'].nunique() > 0:
            print(f"  Average persons per image: {len(df) / df['image_id'].nunique():.2f}")
    except Exception as e:
        print(f"⚠ Could not read final CSV for statistics: {e}")
        print(f"  CSV file should be at: {output_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Conflict scores - HIGH (>0.7): {(df['conflict_score'] > 0.7).sum()}")
    print(f"  Conflict scores - MED (0.4-0.7): {((df['conflict_score'] > 0.4) & (df['conflict_score'] <= 0.7)).sum()}")
    print(f"  Conflict scores - LOW (<0.4): {(df['conflict_score'] <= 0.4).sum()}")
    print(f"  Persons in manual trapezoid: {df['in_manual_trapezoid'].sum()}")
    print(f"  Persons in SegFormer road: {df['in_segformer_road'].sum()}")
    print(f"  Pose detected: {df['pose_detected'].sum()}")


if __name__ == "__main__":
    # Configuration - no argument parser needed
    rsud20k_dir = 'rsud20k'
    output_csv = 'outputs/conflict_dataset.csv'
    splits = ['train', 'val', 'test']
    
    # Use ground truth annotations from RSUD20K (recommended)
    use_ground_truth = True
    
    # Auto-detect calibration file from train directory (saved by calibrate_grid.py)
    # Note: Not needed if using pure YOLO+pose features, but kept for compatibility
    calibration_file = Path(rsud20k_dir) / 'images' / 'train' / 'grid_calibration.json'
    if not calibration_file.exists():
        print(f"⚠ Calibration file not found: {calibration_file}")
        print("  Run calibrate_grid.py first to create calibration, or set calibration_file=None")
        calibration_file = None
    else:
        print(f"✓ Using calibration file: {calibration_file}")
    
    # YOLO model (only used if use_ground_truth=False)
    yolo_model_path = None  # Set to "best_500 image.pt" if you want to use custom model
    
    # Grid configuration (will be loaded from calibration file if available)
    grid_rows = 12
    grid_cols = 12
    
    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("CSV Dataset Generation Configuration")
    print("=" * 60)
    print(f"Using ground truth annotations: {use_ground_truth}")
    if use_ground_truth:
        print("✓ Will use RSUD20K ground truth person bounding boxes")
        print("  (from labels/train/, labels/val/, labels/test/ directories)")
        print("  Class ID for person: 0 (RSUD20K format - person is first class)")
        print("  ⚠ Only person class annotations will be used (all other classes filtered out)")
    else:
        print("⚠ Will use YOLO12n detections (not recommended)")
    print("=" * 60 + "\n")
    
    # Generate dataset
    generate_csv_dataset(
        rsud20k_dir=rsud20k_dir,
        output_csv=output_csv,
        calibration_file=str(calibration_file) if calibration_file else None,
        yolo_model_path=yolo_model_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        splits=splits,
        use_ground_truth=use_ground_truth
    )

