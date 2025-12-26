#!/usr/bin/env python3
"""
Pedestrian Conflict Risk Inference Script
Supports both image and video input with comprehensive feature extraction
and performance logging. Compatible with Apple MPS and CUDA.

COLAB SETUP (Simple - No imports needed!):
1. Upload these files directly to /content/ directory:
   - inference_conflict_risk.py (this file)
   - visualize_conflict_risk.py
   - road_detector.py (optional, for road detection)
   
2. Then use:
   from inference_conflict_risk import predict_image_colab
   result = predict_image_colab('/content/your_image.jpg')
"""

import cv2
import numpy as np
import pandas as pd
import pickle
import json
import math
import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Colab setup helper - add /content to path (for direct file uploads)
def setup_colab_paths():
    """Helper function to set up Colab paths for direct file uploads to /content/"""
    # Priority 1: Add /content to path if it exists (for direct file uploads)
    content_path = '/content'
    if os.path.exists(content_path) and content_path not in sys.path:
        sys.path.insert(0, content_path)
        print(f"✓ Added {content_path} to Python path (for direct file uploads)")
    
    # Priority 2: Add current directory
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Priority 3: Try src directory if it exists
    src_paths = [
        '/content/src',
        str(Path.cwd() / 'src'),
    ]
    for src_path in src_paths:
        if os.path.exists(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"✓ Added {src_path} to Python path")
    
    return True

# For Colab visualization
try:
    from IPython.display import display, Image, HTML
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    print("⚠ Matplotlib/IPython not available - visualization will be limited")

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_stderrthreshold'] = '2'
logging.getLogger().setLevel(logging.ERROR)

# Import dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Error: ultralytics not found. Install with: pip install ultralytics")
    sys.exit(1)

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not available. Pose features will be disabled.")
    MP_AVAILABLE = False
    mp = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Error: PyTorch not found. Install with: pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Error: scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

# Try to auto-setup Colab paths
try:
    setup_colab_paths()
except:
    pass

# Import feature extraction functions
# Try multiple import paths for Colab compatibility
try:
    # Try direct import (if in same directory)
    from visualize_conflict_risk import (
        RoadGrid, PoseAnalyzer, extract_spatial_relationships,
        extract_scene_context, extract_multiscale_features
    )
except ImportError:
    try:
        # Try src.visualize_conflict_risk (if src is a package)
        from src.visualize_conflict_risk import (
            RoadGrid, PoseAnalyzer, extract_spatial_relationships,
            extract_scene_context, extract_multiscale_features
        )
    except ImportError:
        try:
            # Try adding src to path and importing
            from pathlib import Path
            src_path = Path(__file__).parent if '__file__' in globals() else Path('.')
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from visualize_conflict_risk import (
                RoadGrid, PoseAnalyzer, extract_spatial_relationships,
                extract_scene_context, extract_multiscale_features
            )
        except ImportError:
            print("\n" + "="*60)
            print("IMPORT ERROR: visualize_conflict_risk.py not found")
            print("="*60)
            print("\nSOLUTION: Upload visualize_conflict_risk.py to /content/ directory")
            print("\nSteps in Colab:")
            print("  1. Upload visualize_conflict_risk.py to /content/")
            print("  2. Upload road_detector.py to /content/ (optional)")
            print("  3. Re-run this cell")
            print("\nNo need to modify sys.path - it's done automatically!")
            print("="*60)
            raise ImportError("visualize_conflict_risk module not found. Upload it to /content/ directory.")

# Try to import road detector (optional)
try:
    from road_detector import RoadDetector
    ROAD_DETECTOR_AVAILABLE = True
    print("✓ Road detector loaded")
except ImportError:
    try:
        from src.road_detector import RoadDetector
        ROAD_DETECTOR_AVAILABLE = True
        print("✓ Road detector loaded from src")
    except ImportError:
        ROAD_DETECTOR_AVAILABLE = False
        RoadDetector = None
        print("⚠ Road detector not available (optional - upload road_detector.py to /content/ if needed)")


class ConflictRiskInference:
    """Main inference class for pedestrian conflict risk prediction"""
    
    def __init__(self, model_dir: str, calibration_file: Optional[str] = None):
        """
        Initialize inference system
        
        Args:
            model_dir: Directory containing trained models (.pkl files)
            calibration_file: Optional path to calibration JSON file
        """
        self.model_dir = Path(model_dir)
        self.calibration_file = calibration_file
        self.device = self._detect_device()
        
        # Initialize models
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        self.thresholds = {'low': 0.360, 'high': 0.750}  # Default thresholds
        
        # Performance logging
        self.metrics = {
            'total_inferences': 0,
            'total_time': 0.0,
            'feature_extraction_time': 0.0,
            'model_inference_time': 0.0,
            'per_model_times': defaultdict(list),
            'latencies': []
        }
        
        # Load models
        self._load_models()
        
        # Initialize YOLO
        print("Loading YOLO12n model...")
        self.yolo_model = YOLO('yolo12n.pt')
        print(f"✓ YOLO12n loaded (device: {self.device})")
        
        # Initialize pose analyzer
        if MP_AVAILABLE:
            self.pose_analyzer = PoseAnalyzer()
            print("✓ MediaPipe pose analyzer loaded")
        else:
            self.pose_analyzer = None
            print("⚠ MediaPipe not available - pose features will be disabled")
        
        # Initialize road detector (optional)
        self.road_detector = None
        if ROAD_DETECTOR_AVAILABLE:
            try:
                self.road_detector = RoadDetector()
                print("✓ SegFormer road detector loaded")
            except Exception as e:
                print(f"⚠ Road detector initialization failed: {e}")
        
        print(f"\n✓ Inference system initialized (device: {self.device})")
    
    def _detect_device(self) -> str:
        """Detect available device (MPS, CUDA, or CPU)"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_models(self):
        """Load all trained models"""
        print("\n" + "="*60)
        print("Loading Models")
        print("="*60)
        
        # Load FT-Transformer (check multiple possible locations)
        ft_model_paths = [
            self.model_dir / 'ft_transformer_model.pkl',
            self.model_dir / 'ensemble' / 'ft_transformer_model_1.pkl',  # Ensemble member
            Path('/content/ft_transformer_model/ft_transformer_model.pkl'),  # Colab default
        ]
        
        ft_loaded = False
        for ft_model_path in ft_model_paths:
            if ft_model_path.exists():
                try:
                    with open(ft_model_path, 'rb') as f:
                        self.models['FT-Transformer'] = pickle.load(f)
                    print(f"✓ FT-Transformer loaded from {ft_model_path}")
                    ft_loaded = True
                    break
                except Exception as e:
                    print(f"⚠ Error loading FT-Transformer from {ft_model_path}: {e}")
                    continue
        
        if not ft_loaded:
            print(f"⚠ FT-Transformer model not found. Checked: {[str(p) for p in ft_model_paths]}")
        
        # Load XGBoost
        xgb_model_paths = [
            self.model_dir / 'xgboost_model.pkl',
            Path('/content/ft_transformer_model/xgboost_model.pkl'),  # Colab default
        ]
        
        xgb_loaded = False
        for xgb_model_path in xgb_model_paths:
            if xgb_model_path.exists():
                try:
                    with open(xgb_model_path, 'rb') as f:
                        self.models['XGBoost'] = pickle.load(f)
                    print(f"✓ XGBoost loaded from {xgb_model_path}")
                    xgb_loaded = True
                    break
                except Exception as e:
                    print(f"⚠ Error loading XGBoost from {xgb_model_path}: {e}")
                    continue
        
        if not xgb_loaded:
            print(f"⚠ XGBoost model not found (optional)")
        
        # Load CatBoost
        cb_model_paths = [
            self.model_dir / 'catboost_model.pkl',
            Path('/content/ft_transformer_model/catboost_model.pkl'),  # Colab default
        ]
        
        cb_loaded = False
        for cb_model_path in cb_model_paths:
            if cb_model_path.exists():
                try:
                    with open(cb_model_path, 'rb') as f:
                        self.models['CatBoost'] = pickle.load(f)
                    print(f"✓ CatBoost loaded from {cb_model_path}")
                    cb_loaded = True
                    break
                except Exception as e:
                    print(f"⚠ Error loading CatBoost from {cb_model_path}: {e}")
                    continue
        
        if not cb_loaded:
            print(f"⚠ CatBoost model not found (optional)")
        
        if not self.models:
            raise ValueError("No models loaded! Check model directory.")
        
        # Load thresholds from metrics (if available) - Colab-friendly paths
        metrics_paths = [
            self.model_dir.parent / 'metrics.json',
            Path('/content/metrics.json'),  # Colab default
            self.model_dir / 'metrics.json'
        ]
        
        thresholds_loaded = False
        for metrics_path in metrics_paths:
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        if 'optimized_thresholds' in metrics:
                            self.thresholds = {
                                'low': metrics['optimized_thresholds'].get('low', 0.360),
                                'high': metrics['optimized_thresholds'].get('high', 0.750)
                            }
                            thresholds_loaded = True
                            print(f"✓ Loaded thresholds from {metrics_path}")
                            break
                except Exception as e:
                    continue
        
        if not thresholds_loaded:
            print(f"⚠ Using default thresholds: LOW≤{self.thresholds['low']:.3f}, HIGH≥{self.thresholds['high']:.3f}")
        
        # Define feature columns (EXACTLY matching training script - 44 safe features, no leaking features)
        # These match the feature_columns in train_ft_transformer_conflict.py
        self.feature_columns = [
            # Normalized bbox features (10)
            'bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm',
            'bbox_center_x_norm', 'bbox_center_y_norm', 'bbox_area_norm',
            'bbox_width_norm', 'bbox_height_norm', 'bbox_aspect_ratio',
            # Basic pose features (2) - safe, not used in conflict_score formula
            'pose_detected', 'pose_confidence',
            # Advanced pose features (5)
            'torso_lean_angle', 'head_orientation_angle', 'leg_separation',
            'estimated_stride_ratio', 'arm_crossing_score',
            # Spatial relationship features (5)
            'min_distance_to_vehicle', 'min_distance_to_vehicle_norm', 'nearby_pedestrians_count',
            'relative_x_to_vehicle', 'relative_y_to_vehicle',
            # Scene context features (8)
            'traffic_density', 'pedestrian_density', 'road_area_ratio', 'distance_to_road_center',
            'road_segments_count', 'is_intersection', 'image_blur_score', 'image_brightness',
            # Multi-scale spatial features (9)
            'local_road_ratio', 'regional_road_ratio', 'global_road_ratio',
            'distance_to_left_edge', 'distance_to_right_edge', 'distance_to_top_edge',
            'distance_to_bottom_edge', 'position_x_norm', 'position_y_norm',
            # Interaction features (5) - using safe base features only
            'area_pose_confidence_interaction', 'position_pose_confidence_interaction',
            'orientation_agreement_interaction', 'area_orientation_interaction',
            'pose_orientation_interaction'
        ]
        # Total: 44 features (matching training script)
        
        print(f"\n✓ Loaded {len(self.models)} model(s)")
        print(f"✓ Using {len(self.feature_columns)} features")
        print(f"✓ Thresholds: LOW≤{self.thresholds['low']:.3f}, MED={self.thresholds['low']:.3f}-{self.thresholds['high']:.3f}, HIGH≥{self.thresholds['high']:.3f}")
    
    def _extract_features(self, image: np.ndarray, person_bbox: Tuple[float, float, float, float],
                          yolo_confidence: float = 1.0) -> Dict:
        """
        Extract all features for a person bounding box
        
        Args:
            image: Image array (BGR format)
            person_bbox: (x1, y1, x2, y2) bounding box
            yolo_confidence: Detection confidence
        
        Returns:
            Dictionary of features
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = person_bbox
        
        # Calculate bbox features
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Normalize bbox features
        bbox_x1_norm = x1 / w
        bbox_y1_norm = y1 / h
        bbox_x2_norm = x2 / w
        bbox_y2_norm = y2 / h
        bbox_center_x_norm = bbox_center_x / w
        bbox_center_y_norm = bbox_center_y / h
        bbox_area_norm = bbox_area / (w * h)
        bbox_width_norm = bbox_width / w
        bbox_height_norm = bbox_height / h
        bbox_aspect_ratio = bbox_width / (bbox_height + 1e-6)
        
        # Initialize RoadGrid (minimal - for feature extraction)
        road_mask = None
        road_polygon = None
        sidewalk_mask = None
        
        if self.road_detector:
            try:
                road_mask, road_polygon, sidewalk_mask = self.road_detector.detect_road(image)
            except Exception as e:
                pass
        
        try:
            road_grid = RoadGrid(
                img_width=w,
                img_height=h,
                grid_rows=12,
                grid_cols=12,
                road_mask=road_mask,
                road_polygon=road_polygon,
                sidewalk_mask=sidewalk_mask,
                calibration_file=self.calibration_file
            )
        except Exception:
            # Fallback: create minimal RoadGrid
            road_grid = RoadGrid(
                img_width=w,
                img_height=h,
                grid_rows=12,
                grid_cols=12,
                road_mask=None,
                road_polygon=None,
                sidewalk_mask=None,
                calibration_file=None
            )
        
        # Detect all objects for spatial relationships
        all_detections = []
        try:
            results = self.yolo_model(image, conf=0.25, iou=0.45, verbose=False, device=self.device)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else ''
                        x1_d, y1_d, x2_d, y2_d = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        if conf > 0.25:
                            all_detections.append({
                                'bbox': (float(x1_d), float(y1_d), float(x2_d), float(y2_d)),
                                'class': class_name,
                                'class_id': cls,
                                'confidence': conf
                            })
        except Exception as e:
            pass
        
        # Extract pose
        pose_data = None
        if self.pose_analyzer:
            try:
                pose_data = self.pose_analyzer.extract_pose(image, person_bbox)
            except Exception:
                pass
        
        # Extract advanced pose features
        advanced_pose_features = {}
        if self.pose_analyzer and pose_data:
            try:
                advanced_pose_features = self.pose_analyzer.extract_advanced_pose_features(pose_data)
            except Exception:
                pass
        
        # Extract spatial relationships
        spatial_features = extract_spatial_relationships(person_bbox, all_detections, (h, w))
        
        # Extract scene context
        scene_features = extract_scene_context(image, road_grid, all_detections, person_bbox)
        
        # Extract multi-scale features
        multiscale_features = extract_multiscale_features(person_bbox, road_grid, (h, w))
        
        # Compile features
        pose_detected = 1.0 if pose_data is not None else 0.0
        pose_confidence = pose_data.get('confidence', 0.0) if pose_data else 0.0
        
        features = {
            'bbox_x1_norm': bbox_x1_norm,
            'bbox_y1_norm': bbox_y1_norm,
            'bbox_x2_norm': bbox_x2_norm,
            'bbox_y2_norm': bbox_y2_norm,
            'bbox_center_x_norm': bbox_center_x_norm,
            'bbox_center_y_norm': bbox_center_y_norm,
            'bbox_area_norm': bbox_area_norm,
            'bbox_width_norm': bbox_width_norm,
            'bbox_height_norm': bbox_height_norm,
            'bbox_aspect_ratio': bbox_aspect_ratio,
            'pose_detected': pose_detected,
            'pose_confidence': pose_confidence,
            'torso_lean_angle': advanced_pose_features.get('torso_lean_angle', 0.0),
            'head_orientation_angle': advanced_pose_features.get('head_orientation_angle', 0.0),
            'leg_separation': advanced_pose_features.get('leg_separation', 0.0),
            'estimated_stride_ratio': advanced_pose_features.get('estimated_stride_ratio', 0.0),
            'arm_crossing_score': advanced_pose_features.get('arm_crossing_score', 0.0),
            'min_distance_to_vehicle': spatial_features.get('min_distance_to_vehicle', float('inf')),
            'min_distance_to_vehicle_norm': spatial_features.get('min_distance_to_vehicle_norm', 1.0),
            'nearby_pedestrians_count': spatial_features.get('nearby_pedestrians_count', 0),
            'relative_x_to_vehicle': spatial_features.get('relative_x_to_vehicle', 0.0),
            'relative_y_to_vehicle': spatial_features.get('relative_y_to_vehicle', 0.0),
            'traffic_density': scene_features.get('traffic_density', 0.0),
            'pedestrian_density': scene_features.get('pedestrian_density', 0.0),
            'road_area_ratio': scene_features.get('road_area_ratio', 0.0),
            'distance_to_road_center': scene_features.get('distance_to_road_center', 1.0),
            'road_segments_count': scene_features.get('road_segments_count', 0),
            'is_intersection': scene_features.get('is_intersection', 0.0),
            'image_blur_score': scene_features.get('image_blur_score', 0.0),
            'image_brightness': scene_features.get('image_brightness', 0.5),
            'local_road_ratio': multiscale_features.get('local_road_ratio', 0.0),
            'regional_road_ratio': multiscale_features.get('regional_road_ratio', 0.0),
            'global_road_ratio': multiscale_features.get('global_road_ratio', 0.0),
            'distance_to_left_edge': multiscale_features.get('distance_to_left_edge', 0.5),
            'distance_to_right_edge': multiscale_features.get('distance_to_right_edge', 0.5),
            'distance_to_top_edge': multiscale_features.get('distance_to_top_edge', 0.5),
            'distance_to_bottom_edge': multiscale_features.get('distance_to_bottom_edge', 0.5),
            'position_x_norm': multiscale_features.get('position_x_norm', 0.5),
            'position_y_norm': multiscale_features.get('position_y_norm', 0.5),
        }
        
        # Handle infinite values
        for key, value in features.items():
            if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                if math.isinf(value) and value > 0:
                    features[key] = 10000.0
                elif math.isinf(value) and value < 0:
                    features[key] = -10000.0
                else:
                    features[key] = 0.0
        
        # Add interaction features
        if 'bbox_area_norm' in features and 'pose_confidence' in features:
            features['area_pose_confidence_interaction'] = features['bbox_area_norm'] * features['pose_confidence']
        else:
            features['area_pose_confidence_interaction'] = 0.0
        
        if 'bbox_center_y_norm' in features and 'pose_confidence' in features:
            features['position_pose_confidence_interaction'] = features['bbox_center_y_norm'] * features['pose_confidence']
        else:
            features['position_pose_confidence_interaction'] = 0.0
        
        if 'torso_lean_angle' in features and 'head_orientation_angle' in features:
            features['orientation_agreement_interaction'] = (abs(features['torso_lean_angle']) / 180.0) * (abs(features['head_orientation_angle']) / 180.0)
        else:
            features['orientation_agreement_interaction'] = 0.0
        
        if 'bbox_area_norm' in features and 'torso_lean_angle' in features:
            features['area_orientation_interaction'] = features['bbox_area_norm'] * (abs(features['torso_lean_angle']) / 180.0)
        else:
            features['area_orientation_interaction'] = 0.0
        
        if 'pose_confidence' in features and 'torso_lean_angle' in features:
            features['pose_orientation_interaction'] = features['pose_confidence'] * (abs(features['torso_lean_angle']) / 180.0)
        else:
            features['pose_orientation_interaction'] = 0.0
        
        return features
    
    def _predict_single(self, features: Dict) -> Dict:
        """
        Run inference on a single feature vector
        
        Args:
            features: Dictionary of features
        
        Returns:
            Dictionary with predictions from all models
        """
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        # Select only required features
        X = feature_df[self.feature_columns].values
        
        predictions = {}
        
        # FT-Transformer
        if 'FT-Transformer' in self.models:
            try:
                start_time = time.time()
                pred_df = pd.DataFrame(X, columns=self.feature_columns)
                pred_result = self.models['FT-Transformer'].predict(pred_df)
                
                if isinstance(pred_result, pd.DataFrame):
                    if 'conflict_score_prediction' in pred_result.columns:
                        score = pred_result['conflict_score_prediction'].values[0]
                    else:
                        score = pred_result.iloc[0, 0]
                else:
                    score = pred_result[0] if isinstance(pred_result, (list, np.ndarray)) else pred_result
                
                inference_time = time.time() - start_time
                predictions['FT-Transformer'] = {
                    'score': float(score),
                    'latency_ms': inference_time * 1000
                }
                self.metrics['per_model_times']['FT-Transformer'].append(inference_time)
            except Exception as e:
                print(f"⚠ FT-Transformer prediction error: {e}")
        
        # XGBoost (needs DMatrix for prediction)
        if 'XGBoost' in self.models:
            try:
                start_time = time.time()
                # XGBoost models trained with xgb.train() require DMatrix
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                score = self.models['XGBoost'].predict(dtest)[0]
                inference_time = time.time() - start_time
                predictions['XGBoost'] = {
                    'score': float(score),
                    'latency_ms': inference_time * 1000
                }
                self.metrics['per_model_times']['XGBoost'].append(inference_time)
            except Exception as e:
                print(f"⚠ XGBoost prediction error: {e}")
        
        # CatBoost
        if 'CatBoost' in self.models:
            try:
                start_time = time.time()
                score = self.models['CatBoost'].predict(X)[0]
                inference_time = time.time() - start_time
                predictions['CatBoost'] = {
                    'score': float(score),
                    'latency_ms': inference_time * 1000
                }
                self.metrics['per_model_times']['CatBoost'].append(inference_time)
            except Exception as e:
                print(f"⚠ CatBoost prediction error: {e}")
        
        return predictions
    
    def _score_to_level(self, score: float) -> str:
        """Convert conflict score to risk level"""
        if score > self.thresholds['high']:
            return 'HIGH'
        elif score > self.thresholds['low']:
            return 'MED'
        else:
            return 'LOW'
    
    def predict_image(self, image_path: Union[str, Path], visualize: bool = True) -> Dict:
        """
        Predict conflict risk for all persons in an image
        
        Args:
            image_path: Path to image file
            visualize: Whether to create visualization
        
        Returns:
            Dictionary with predictions and metrics
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Detect persons
        feat_start = time.time()
        results = self.yolo_model(image, conf=0.25, iou=0.45, verbose=False, device=self.device)
        
        person_detections = []
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else ''
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    if conf > 0.25:
                        all_detections.append({
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'class': class_name,
                            'class_id': cls,
                            'confidence': conf
                        })
                        
                        # Filter for person class (class 0)
                        if cls == 0 or 'person' in class_name.lower():
                            person_detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'confidence': conf
                            })
        
        if not person_detections:
            return {
                'image_path': str(image_path),
                'persons_detected': 0,
                'predictions': [],
                'total_time_ms': (time.time() - start_time) * 1000,
                'feature_extraction_time_ms': 0,
                'model_inference_time_ms': 0
            }
        
        # Extract features and predict for each person
        predictions_list = []
        model_inf_time = 0.0
        
        for i, person in enumerate(person_detections):
            person_start = time.time()
            
            # Extract features
            features = self._extract_features(image, person['bbox'], person['confidence'])
            feat_time = time.time() - person_start
            
            # Predict
            pred_start = time.time()
            model_predictions = self._predict_single(features)
            model_inf_time += time.time() - pred_start
            
            # Aggregate predictions (use ensemble average if multiple models)
            if model_predictions:
                scores = [p['score'] for p in model_predictions.values()]
                avg_score = np.mean(scores)
                risk_level = self._score_to_level(avg_score)
                
                predictions_list.append({
                    'person_id': i,
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'conflict_score': float(avg_score),
                    'risk_level': risk_level,
                    'model_predictions': {k: v['score'] for k, v in model_predictions.items()},
                    'model_latencies_ms': {k: v['latency_ms'] for k, v in model_predictions.items()}
                })
        
        total_time = time.time() - start_time
        feat_extraction_time = (time.time() - feat_start) - model_inf_time
        
        # Update metrics
        self.metrics['total_inferences'] += len(predictions_list)
        self.metrics['total_time'] += total_time
        self.metrics['feature_extraction_time'] += feat_extraction_time
        self.metrics['model_inference_time'] += model_inf_time
        self.metrics['latencies'].append(total_time * 1000)
        
        result = {
            'image_path': str(image_path),
            'image_size': (w, h),
            'persons_detected': len(predictions_list),
            'predictions': predictions_list,
            'total_time_ms': total_time * 1000,
            'feature_extraction_time_ms': feat_extraction_time * 1000,
            'model_inference_time_ms': model_inf_time * 1000,
            'avg_time_per_person_ms': (total_time * 1000) / len(predictions_list) if predictions_list else 0
        }
        
        # Visualization
        if visualize and predictions_list:
            # Determine save path (Colab-friendly) - avoid double "_prediction"
            image_path_obj = Path(image_path)
            # Remove existing "_prediction" suffix if present
            stem = image_path_obj.stem
            if stem.endswith('_prediction'):
                stem = stem[:-11]  # Remove "_prediction" suffix (11 chars)
            save_path = str(image_path_obj.parent / f"{stem}_prediction.png")
            
            self._visualize_predictions(image, predictions_list, image_path, 
                                      save_path=save_path, show_in_colab=True)
            result['visualization_path'] = save_path
        
        return result
    
    def predict_video(self, video_path: Union[str, Path], output_path: Optional[str] = None,
                     frame_skip: int = 1, max_frames: Optional[int] = None) -> Dict:
        """
        Predict conflict risk for all persons in a video
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save output video
            frame_skip: Process every Nth frame
            max_frames: Maximum number of frames to process
        
        Returns:
            Dictionary with predictions and metrics
        """
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_predictions = []
        frame_count = 0
        processed_frames = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print(f"Frame skip: {frame_skip}, Max frames: {max_frames or 'all'}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % frame_skip != 0:
                if writer:
                    writer.write(frame)
                continue
            
            # Check max frames
            if max_frames and processed_frames >= max_frames:
                break
            
            # Process frame
            frame_start = time.time()
            
            # Detect persons
            results = self.yolo_model(frame, conf=0.25, iou=0.45, verbose=False, device=self.device)
            
            person_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = self.yolo_model.names[cls] if hasattr(self.yolo_model, 'names') else ''
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        if conf > 0.25 and (cls == 0 or 'person' in class_name.lower()):
                            person_detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'confidence': conf
                            })
            
            # Predict for each person
            frame_predictions_frame = []
            for person in person_detections:
                features = self._extract_features(frame, person['bbox'], person['confidence'])
                model_predictions = self._predict_single(features)
                
                if model_predictions:
                    scores = [p['score'] for p in model_predictions.values()]
                    avg_score = np.mean(scores)
                    risk_level = self._score_to_level(avg_score)
                    
                    frame_predictions_frame.append({
                        'bbox': person['bbox'],
                        'conflict_score': float(avg_score),
                        'risk_level': risk_level
                    })
                    
                    # Draw on frame
                    x1, y1, x2, y2 = [int(v) for v in person['bbox']]
                    color = (0, 0, 255) if risk_level == 'HIGH' else (0, 165, 255) if risk_level == 'MED' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{risk_level} ({avg_score:.2f})", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_predictions.append({
                'frame': frame_count,
                'time_sec': frame_count / fps,
                'persons': frame_predictions_frame,
                'processing_time_ms': (time.time() - frame_start) * 1000
            })
            
            processed_frames += 1
            
            if writer:
                writer.write(frame)
        
        cap.release()
        if writer:
            writer.release()
        
        total_time = time.time() - start_time
        
        result = {
            'video_path': str(video_path),
            'output_path': str(output_path) if output_path else None,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'frame_predictions': frame_predictions,
            'total_time_sec': total_time,
            'avg_time_per_frame_ms': (total_time * 1000) / processed_frames if processed_frames > 0 else 0,
            'processing_fps': processed_frames / total_time if total_time > 0 else 0
        }
        
        return result
    
    def _visualize_predictions(self, image: np.ndarray, predictions: List[Dict], image_path: Union[str, Path], 
                              save_path: Optional[str] = None, show_in_colab: bool = True):
        """
        Visualize predictions on image with enhanced Colab support
        
        Args:
            image: Image array (BGR format)
            predictions: List of prediction dictionaries
            image_path: Original image path
            save_path: Optional path to save visualization
            show_in_colab: Whether to display in Colab notebook
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # Color scheme matching training visualization
        color_map = {
            'HIGH': '#FF0000',  # Red
            'MED': '#FF8C00',   # Orange
            'LOW': '#00FF00'    # Green
        }
        
        # Draw predictions
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2 = [int(v) for v in pred['bbox']]
            risk_level = pred['risk_level']
            score = pred['conflict_score']
            confidence = pred.get('confidence', 1.0)
            
            # Get color
            color = color_map.get(risk_level, '#808080')
            
            # Draw bounding box with thickness based on risk
            thickness = 4 if risk_level == 'HIGH' else 3 if risk_level == 'MED' else 2
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=thickness, edgecolor=color, 
                                    facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            
            # Draw label with background
            label = f"{risk_level} {score:.3f}"
            if 'model_predictions' in pred:
                # Show individual model scores if available
                model_scores = pred['model_predictions']
                label += f"\nFT:{model_scores.get('FT-Transformer', 0):.2f}"
                if 'XGBoost' in model_scores:
                    label += f" XGB:{model_scores['XGBoost']:.2f}"
                if 'CatBoost' in model_scores:
                    label += f" CB:{model_scores['CatBoost']:.2f}"
            
            # Text with background
            ax.text(x1, y1-5, label, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='white'),
                   fontsize=10, fontweight='bold', color='white',
                   verticalalignment='bottom')
        
        # Add title with summary
        title = f"Pedestrian Conflict Risk Prediction\n"
        title += f"Persons Detected: {len(predictions)} | "
        risk_counts = {}
        for pred in predictions:
            risk = pred['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        title += " | ".join([f"{k}: {v}" for k, v in sorted(risk_counts.items())])
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"✓ Visualization saved to {save_path}")
        
        # Display in Colab
        if show_in_colab and COLAB_AVAILABLE:
            plt.show()
        else:
            plt.close()
        
        # Also save OpenCV version for compatibility
        vis_image_cv = image.copy()
        for pred in predictions:
            x1, y1, x2, y2 = [int(v) for v in pred['bbox']]
            risk_level = pred['risk_level']
            score = pred['conflict_score']
            
            # BGR color coding
            if risk_level == 'HIGH':
                color_cv = (0, 0, 255)  # Red
            elif risk_level == 'MED':
                color_cv = (0, 165, 255)  # Orange
            else:
                color_cv = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(vis_image_cv, (x1, y1), (x2, y2), color_cv, 3)
            
            # Draw label
            label = f"{risk_level} {score:.3f}"
            cv2.putText(vis_image_cv, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cv, 2)
        
        # Save OpenCV version
        if save_path:
            cv_path = str(save_path).replace('.png', '_cv.jpg').replace('.jpg', '_cv.jpg')
            cv2.imwrite(cv_path, vis_image_cv)
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if self.metrics['total_inferences'] == 0:
            return self.metrics
        
        avg_latency = np.mean(self.metrics['latencies']) if self.metrics['latencies'] else 0
        avg_feat_time = (self.metrics['feature_extraction_time'] / self.metrics['total_inferences']) * 1000
        avg_model_time = (self.metrics['model_inference_time'] / self.metrics['total_inferences']) * 1000
        
        per_model_avg = {}
        for model_name, times in self.metrics['per_model_times'].items():
            if times:
                per_model_avg[model_name] = {
                    'avg_ms': np.mean(times) * 1000,
                    'min_ms': np.min(times) * 1000,
                    'max_ms': np.max(times) * 1000
                }
        
        return {
            'total_inferences': self.metrics['total_inferences'],
            'total_time_sec': self.metrics['total_time'],
            'avg_latency_ms': avg_latency,
            'avg_feature_extraction_time_ms': avg_feat_time,
            'avg_model_inference_time_ms': avg_model_time,
            'per_model_latencies_ms': per_model_avg,
            'throughput_inferences_per_sec': self.metrics['total_inferences'] / self.metrics['total_time'] if self.metrics['total_time'] > 0 else 0
        }
    
    def print_metrics(self):
        """Print performance metrics"""
        metrics = self.get_metrics()
        print("\n" + "="*60)
        print("Performance Metrics")
        print("="*60)
        print(f"Total Inferences: {metrics['total_inferences']}")
        print(f"Total Time: {metrics['total_time_sec']:.2f} sec")
        print(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"Average Feature Extraction: {metrics['avg_feature_extraction_time_ms']:.2f} ms")
        print(f"Average Model Inference: {metrics['avg_model_inference_time_ms']:.2f} ms")
        print(f"Throughput: {metrics['throughput_inferences_per_sec']:.2f} inferences/sec")
        
        if metrics['per_model_latencies_ms']:
            print("\nPer-Model Latencies:")
            for model_name, lat in metrics['per_model_latencies_ms'].items():
                print(f"  {model_name}:")
                print(f"    Avg: {lat['avg_ms']:.2f} ms")
                print(f"    Min: {lat['min_ms']:.2f} ms")
                print(f"    Max: {lat['max_ms']:.2f} ms")
        print("="*60)


def predict_image_colab(image_path: Union[str, Path], model_dir: str = '/content/ft_transformer_model',
                        calibration_file: Optional[str] = None, visualize: bool = True) -> Dict:
    """
    Colab-friendly function for image inference with visualization
    
    Args:
        image_path: Path to image file (can be uploaded to /content/)
        model_dir: Directory containing trained models (default: /content/ft_transformer_model)
        calibration_file: Optional calibration JSON file path
        visualize: Whether to show visualization in Colab
    
    Returns:
        Dictionary with predictions and metrics
    
    Example:
        # Upload image to Colab first, then:
        result = predict_image_colab('/content/uploaded_image.jpg')
        print(f"Found {result['persons_detected']} persons")
    """
    print("="*60)
    print("Pedestrian Conflict Risk Inference (Colab)")
    print("="*60)
    
    # Initialize inference system
    print("\nInitializing inference system...")
    inference = ConflictRiskInference(model_dir, calibration_file)
    
    # Process image
    print(f"\nProcessing image: {image_path}")
    result = inference.predict_image(image_path, visualize=visualize)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Inference Results")
    print(f"{'='*60}")
    print(f"✓ Persons Detected: {result['persons_detected']}")
    print(f"✓ Total Processing Time: {result['total_time_ms']:.2f} ms")
    print(f"  - Feature Extraction: {result['feature_extraction_time_ms']:.2f} ms")
    print(f"  - Model Inference: {result['model_inference_time_ms']:.2f} ms")
    if result['persons_detected'] > 0:
        print(f"  - Average per Person: {result['avg_time_per_person_ms']:.2f} ms")
    
    # Print per-person details
    if result['predictions']:
        print(f"\n{'='*60}")
        print("Per-Person Predictions")
        print(f"{'='*60}")
        for pred in result['predictions']:
            print(f"\nPerson {pred['person_id'] + 1}:")
            print(f"  Risk Level: {pred['risk_level']}")
            print(f"  Conflict Score: {pred['conflict_score']:.4f}")
            print(f"  BBox: ({pred['bbox'][0]:.0f}, {pred['bbox'][1]:.0f}, {pred['bbox'][2]:.0f}, {pred['bbox'][3]:.0f})")
            print(f"  Detection Confidence: {pred['confidence']:.3f}")
            if pred.get('model_predictions'):
                print(f"  Model Scores:")
                for model, score in pred['model_predictions'].items():
                    latency = pred['model_latencies_ms'].get(model, 0)
                    print(f"    {model}: {score:.4f} ({latency:.2f} ms)")
        
        # Risk level summary
        risk_counts = {}
        for pred in result['predictions']:
            risk = pred['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print(f"\n{'='*60}")
        print("Risk Level Summary")
        print(f"{'='*60}")
        for risk in ['LOW', 'MED', 'HIGH']:
            count = risk_counts.get(risk, 0)
            percentage = (count / result['persons_detected']) * 100 if result['persons_detected'] > 0 else 0
            print(f"  {risk}: {count} ({percentage:.1f}%)")
    
    # Print performance metrics
    inference.print_metrics()
    
    return result


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pedestrian Conflict Risk Inference')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained models (.pkl files)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or video file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (for video) or visualization (for image)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Optional calibration JSON file path')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame for video (default: 1)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process for video')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization for images')
    
    args = parser.parse_args()
    
    # Initialize inference system
    print("Initializing inference system...")
    inference = ConflictRiskInference(args.model_dir, args.calibration)
    
    # Determine input type
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Check if image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if input_path.suffix.lower() in image_extensions:
        # Process image
        print(f"\nProcessing image: {args.input}")
        result = inference.predict_image(args.input, visualize=not args.no_visualize)
        
        print(f"\n✓ Processed {result['persons_detected']} person(s)")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Feature extraction: {result['feature_extraction_time_ms']:.2f} ms")
        print(f"  Model inference: {result['model_inference_time_ms']:.2f} ms")
        
        for pred in result['predictions']:
            print(f"\n  Person {pred['person_id']}:")
            print(f"    Risk Level: {pred['risk_level']}")
            print(f"    Conflict Score: {pred['conflict_score']:.4f}")
            print(f"    BBox: {pred['bbox']}")
            if pred['model_predictions']:
                print(f"    Model Scores:")
                for model, score in pred['model_predictions'].items():
                    print(f"      {model}: {score:.4f} ({pred['model_latencies_ms'][model]:.2f} ms)")
    
    elif input_path.suffix.lower() in video_extensions:
        # Process video
        print(f"\nProcessing video: {args.input}")
        result = inference.predict_video(
            args.input,
            output_path=args.output,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
        
        print(f"\n✓ Processed {result['processed_frames']}/{result['total_frames']} frames")
        print(f"  Total time: {result['total_time_sec']:.2f} sec")
        print(f"  Average per frame: {result['avg_time_per_frame_ms']:.2f} ms")
        print(f"  Processing FPS: {result['processing_fps']:.2f}")
        if result['output_path']:
            print(f"  Output saved to: {result['output_path']}")
    
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")
        print(f"Supported image formats: {', '.join(image_extensions)}")
        print(f"Supported video formats: {', '.join(video_extensions)}")
        return
    
    # Print metrics
    inference.print_metrics()


if __name__ == "__main__":
    main()

