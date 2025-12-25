#!/usr/bin/env python3
"""
Visualize Conflict Risk Assessment on a Single Image
- Defines trapezoidal road grid with pavement extensions
- Extracts pose using MediaPipe
- Computes conflict risk based on position and pose inclination
- Visualizes grid, pose landmarks, and conflict scores
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import math
import json

# Try to import MediaPipe with error handling
MP_AVAILABLE = False
mp = None
try:
    import mediapipe as mp
    # Try different import paths for solutions
    try:
        # Try direct access
        _ = mp.solutions
        MP_AVAILABLE = True
    except AttributeError:
        try:
            # Try from mediapipe.python
            from mediapipe.python import solutions
            mp.solutions = solutions
            MP_AVAILABLE = True
        except (ImportError, AttributeError):
            try:
                # Try direct import
                from mediapipe import solutions
                mp.solutions = solutions
                MP_AVAILABLE = True
            except ImportError:
                MP_AVAILABLE = False
except (ImportError, AttributeError, Exception) as e:
    MP_AVAILABLE = False
    mp = None
    # Only print warning if it's a real import error
    error_str = str(e)
    if "cannot import name" not in error_str or ("solutions" not in error_str and "model_ckpt_util" not in error_str):
        print(f"Warning: MediaPipe not available: {e}")
        print("Pose analysis will be disabled. Install with: pip install mediapipe")

# Try to import road detector
try:
    from road_detector import RoadDetector
    ROAD_DETECTOR_AVAILABLE = True
except ImportError:
    ROAD_DETECTOR_AVAILABLE = False
    print("Note: Road detector not available. Install transformers: pip install transformers")


class RoadGrid:
    """SegFormer-based road grid with polygonal road region"""
    
    def __init__(self, img_width=640, img_height=640, 
                 grid_rows=8, grid_cols=6,
                 road_mask=None,
                 road_polygon=None,
                 sidewalk_mask=None,
                 pavement_width=50,
                 calibration_file=None):
        """
        Initialize road grid based on SegFormer road detection
        
        Args:
            img_width: Image width
            img_height: Image height
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            road_mask: Binary road mask from SegFormer (numpy array, 0-255)
            road_polygon: List of points defining road polygon from SegFormer
            sidewalk_mask: Binary sidewalk mask from SegFormer (optional)
            pavement_width: Width of pavement extensions (if not using sidewalk_mask)
            calibration_file: Path to calibration JSON file (optional, for backward compatibility)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # SegFormer-based road region (separate feature)
        self.road_mask = road_mask  # SegFormer road mask
        self.road_polygon = road_polygon  # SegFormer road polygon
        self.sidewalk_mask = sidewalk_mask  # SegFormer sidewalk mask
        
        # Manual trapezoid (separate feature, loaded from calibration)
        self.manual_trapezoid = None
        self.manual_trapezoid_mask = None  # Manual trapezoid as mask
        
        # Load from calibration file if provided (for manual trapezoid)
        if calibration_file and Path(calibration_file).exists():
            self.load_from_calibration(calibration_file)
        
        # Keep SegFormer detection even if calibration is loaded (both are separate features)
        if road_mask is not None:
            self.road_mask = road_mask
            self.road_polygon = road_polygon
            self.sidewalk_mask = sidewalk_mask
        
        # If no road_mask provided, create default (fallback)
        if self.road_mask is None:
            # Create default rectangular road region
            self.road_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            center_x = img_width / 2
            road_width = img_width * 0.6
            cv2.rectangle(self.road_mask, 
                         (int(center_x - road_width/2), 0),
                         (int(center_x + road_width/2), img_height),
                         255, -1)
        
        # Derive pavement mask from SegFormer sidewalk or adjacent areas
        self.pavement_mask = self._derive_pavement_mask(adjacent_threshold=50)
        
        # Compute road bounding box for grid
        self._compute_road_bbox()
        
        # Grid cell dimensions (based on road bounding box)
        self.cell_width = (self.road_bbox[2] - self.road_bbox[0]) / grid_cols if grid_cols > 0 else img_width / grid_cols
        self.cell_height = (self.road_bbox[3] - self.road_bbox[1]) / grid_rows if grid_rows > 0 else img_height / grid_rows
    
    def _derive_pavement_mask(self, adjacent_threshold=50):
        """
        Derive pavement mask from SegFormer sidewalk mask or adjacent areas to road
        
        Args:
            adjacent_threshold: Maximum distance (in pixels) from road edge to consider as pavement
        
        Returns:
            pavement_mask: Binary mask of pavement region
        """
        pavement_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        # Priority 1: Use SegFormer sidewalk mask if available
        if self.sidewalk_mask is not None:
            pavement_mask = (self.sidewalk_mask > 0).astype(np.uint8) * 255
            return pavement_mask
        
        # Priority 2: Derive from adjacent areas to road mask
        if self.road_mask is not None:
            # Create distance transform from road edge
            # Invert road mask (road=0, non-road=255)
            inverted_road = 255 - self.road_mask
            dist_transform = cv2.distanceTransform(inverted_road, cv2.DIST_L2, 5)
            
            # Pavement = areas within threshold distance from road, but not road itself
            pavement_mask = ((dist_transform > 0) & (dist_transform <= adjacent_threshold)).astype(np.uint8) * 255
            
            # Exclude road region from pavement
            pavement_mask[self.road_mask > 0] = 0
        
        return pavement_mask
    
    def _compute_road_bbox(self):
        """Compute bounding box of road region"""
        if self.road_mask is not None:
            # Get bounding box from mask
            coords = np.column_stack(np.where(self.road_mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                self.road_bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
            else:
                self.road_bbox = (0, 0, self.img_width, self.img_height)
        elif self.road_polygon:
            # Get bounding box from polygon
            xs = [p[0] for p in self.road_polygon]
            ys = [p[1] for p in self.road_polygon]
            self.road_bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        else:
            self.road_bbox = (0, 0, self.img_width, self.img_height)
    
    def load_from_calibration(self, calibration_file):
        """Load grid parameters from calibration JSON file (backward compatibility)"""
        with open(calibration_file, 'r') as f:
            cal = json.load(f)
        
        self.grid_rows = cal.get('grid_rows', 8)
        self.grid_cols = cal.get('grid_cols', 6)
        
        # Load manual trapezoid from calibration (separate feature, not replacing SegFormer)
        if 'street_trapezoid' in cal:
            trapezoid = cal['street_trapezoid']
            self.manual_trapezoid = [
                tuple(trapezoid['top_left']),
                tuple(trapezoid['top_right']),
                tuple(trapezoid['bottom_right']),
                tuple(trapezoid['bottom_left'])
            ]
            
            # Create separate mask for manual trapezoid (keep it separate from SegFormer)
            self.manual_trapezoid_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            pts = np.array(self.manual_trapezoid, np.int32)
            cv2.fillPoly(self.manual_trapezoid_mask, [pts], 255)
            
            # Only use manual trapezoid as fallback if no SegFormer detection
            if self.road_mask is None:
                self.road_mask = self.manual_trapezoid_mask.copy()
                self.road_polygon = self.manual_trapezoid
        
        # Scale if image size differs
        cal_width = cal.get('image_width', self.img_width)
        cal_height = cal.get('image_height', self.img_height)
        
        if cal_width != self.img_width or cal_height != self.img_height:
            scale_x = self.img_width / cal_width
            scale_y = self.img_height / cal_height
            if self.manual_trapezoid:
                self.manual_trapezoid = [(p[0] * scale_x, p[1] * scale_y) for p in self.manual_trapezoid]
                # Recreate manual trapezoid mask (keep separate)
                self.manual_trapezoid_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
                pts = np.array(self.manual_trapezoid, np.int32)
                cv2.fillPoly(self.manual_trapezoid_mask, [pts], 255)
                
                # Only use manual trapezoid as fallback if no SegFormer
                if self.road_mask is None:
                    self.road_mask = self.manual_trapezoid_mask.copy()
                    self.road_polygon = self.manual_trapezoid
        
        # Derive pavement mask (no sidewalk mask in old calibration)
        self.pavement_mask = self._derive_pavement_mask(adjacent_threshold=50)
        
        # Compute road bounding box for grid
        self._compute_road_bbox()
        
        # Grid cell dimensions (based on road bounding box)
        self.cell_width = (self.road_bbox[2] - self.road_bbox[0]) / self.grid_cols if self.grid_cols > 0 else self.img_width / self.grid_cols
        self.cell_height = (self.road_bbox[3] - self.road_bbox[1]) / self.grid_rows if self.grid_rows > 0 else self.img_height / self.grid_rows
    
    def get_grid_cell(self, x, y):
        """Get grid cell coordinates for a point (relative to road bounding box)"""
        x_min, y_min, x_max, y_max = self.road_bbox
        
        # Convert to road-relative coordinates
        rel_x = x - x_min
        rel_y = y - y_min
        
        col = int(rel_x / self.cell_width) if self.cell_width > 0 else 0
        row = int(rel_y / self.cell_height) if self.cell_height > 0 else 0
        
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return (row, col)
    
    def is_in_street(self, x, y, use_segformer=True):
        """
        Check if point is in road region
        
        Args:
            x, y: Point coordinates
            use_segformer: If True, check SegFormer road; if False, check manual trapezoid
        
        Returns:
            True if point is in the specified road region
        """
        px, py = int(x), int(y)
        if not (0 <= px < self.img_width and 0 <= py < self.img_height):
            return False
        
        if use_segformer:
            # Check SegFormer road mask
            if self.road_mask is not None:
                return self.road_mask[py, px] > 0
            # Fallback to polygon check
            if self.road_polygon:
                return self._point_in_polygon(x, y, self.road_polygon)
        else:
            # Check manual trapezoid
            if self.manual_trapezoid_mask is not None:
                return self.manual_trapezoid_mask[py, px] > 0
            if self.manual_trapezoid:
                return self._point_in_polygon(x, y, self.manual_trapezoid)
        
        return False
    
    def is_in_street_both(self, x, y):
        """Check if point is in both SegFormer road AND manual trapezoid"""
        in_segformer = self.is_in_street(x, y, use_segformer=True)
        in_manual = self.is_in_street(x, y, use_segformer=False) if self.manual_trapezoid_mask is not None else False
        return in_segformer, in_manual
    
    def _point_in_polygon(self, px, py, polygon):
        """Point-in-polygon test using ray casting algorithm"""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def is_in_pavement(self, x, y):
        """Check if point is in pavement/sidewalk region using SegFormer-derived mask"""
        if not hasattr(self, 'pavement_mask') or self.pavement_mask is None:
            return False
        
        px, py = int(x), int(y)
        if 0 <= px < self.img_width and 0 <= py < self.img_height:
            return self.pavement_mask[py, px] > 0
        
        return False
    
    def get_street_center(self):
        """Get center point of SegFormer road region"""
        if self.road_mask is not None:
            # Use centroid of road mask
            coords = np.column_stack(np.where(self.road_mask > 0))
            if len(coords) > 0:
                y_center, x_center = coords.mean(axis=0)
                return (float(x_center), float(y_center))
        
        # Fallback to polygon centroid
        if self.road_polygon:
            xs = [p[0] for p in self.road_polygon]
            ys = [p[1] for p in self.road_polygon]
            return (float(np.mean(xs)), float(np.mean(ys)))
        
        # Default center
        return (self.img_width / 2, self.img_height / 2)
    
    def _get_trapezoid_bbox(self, trapezoid):
        """Get bounding box of a trapezoid"""
        xs = [p[0] for p in trapezoid]
        ys = [p[1] for p in trapezoid]
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
    
    def _get_trapezoid_center(self, trapezoid):
        """Get center point of a trapezoid"""
        xs = [p[0] for p in trapezoid]
        ys = [p[1] for p in trapezoid]
        return (float(np.mean(xs)), float(np.mean(ys)))
    
    def get_distance_to_road(self, x, y):
        """Get distance from point to nearest road pixel (for proximity scoring)"""
        if self.road_mask is None:
            return float('inf')
        
        px, py = int(x), int(y)
        if 0 <= px < self.img_width and 0 <= py < self.img_height:
            # If inside road, compute distance to bottom (camera proximity)
            if self.road_mask[py, px] > 0:
                # Distance from top of road bbox = proximity to camera
                _, y_min, _, y_max = self.road_bbox
                road_height = y_max - y_min
                if road_height > 0:
                    rel_y = (py - y_min) / road_height  # 0 = top (far), 1 = bottom (near)
                    return rel_y
            else:
                # If outside road, compute distance to road boundary
                dist_transform = cv2.distanceTransform(255 - self.road_mask, cv2.DIST_L2, 5)
                return float(dist_transform[py, px])
        
        return float('inf')


class HumanSegmenter:
    """Human segmentation using MediaPipe, aligned with pose landmarks"""
    
    def __init__(self):
        if not MP_AVAILABLE or mp is None:
            self.mp_selfie_segmentation = None
            self.selfie_segmentation = None
            return
        
        try:
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 0=general, 1=landscape (better for dashcam)
            )
        except Exception as e:
            print(f"Warning: MediaPipe selfie segmentation initialization failed: {e}")
            self.mp_selfie_segmentation = None
            self.selfie_segmentation = None
    
    def segment_person(self, image, bbox, pose_data=None):
        """
        Segment person from image using MediaPipe, aligned with pose landmarks
        
        Args:
            image: Full image (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
            pose_data: Optional pose data for alignment
        
        Returns:
            segmentation_mask: Binary mask of person (numpy array, 0-255)
        """
        if self.selfie_segmentation is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding for better segmentation
        padding = 0.3
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1_crop = max(0, int(x1) - pad_x)
        y1_crop = max(0, int(y1) - pad_y)
        x2_crop = min(w, int(x2) + pad_x)
        y2_crop = min(h, int(y2) + pad_y)
        
        person_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if person_crop.size == 0:
            return None
        
        # Convert to RGB
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Get segmentation
        results = self.selfie_segmentation.process(person_rgb)
        
        if not results.segmentation_mask:
            return None
        
        # Get segmentation mask
        segmentation_mask = results.segmentation_mask
        
        # Convert to binary mask (threshold at 0.5)
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Refine with pose landmarks if available
        if pose_data and pose_data.get('landmarks'):
            binary_mask = self._align_with_pose(binary_mask, pose_data, x1_crop, y1_crop, x2_crop, y2_crop)
        
        # Create full image mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        crop_h, crop_w = binary_mask.shape
        full_mask[y1_crop:y1_crop+crop_h, x1_crop:x1_crop+crop_w] = binary_mask
        
        return full_mask
    
    def _align_with_pose(self, mask, pose_data, x1, y1, x2, y2):
        """
        Refine segmentation mask using pose landmarks
        
        Args:
            mask: Binary segmentation mask (crop coordinates)
            pose_data: Pose data with landmarks
            x1, y1, x2, y2: Crop coordinates in full image
        
        Returns:
            Refined mask
        """
        if not pose_data or 'landmarks' not in pose_data or not MP_AVAILABLE or mp is None:
            return mask
        
        landmarks = pose_data['landmarks']
        mp_pose = mp.solutions.pose
        
        # Key body parts to include in segmentation
        key_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.NOSE
        ]
        
        # Create a refined mask based on pose landmarks
        refined_mask = mask.copy()
        h, w = mask.shape
        
        # Get landmark points in crop coordinates
        landmark_points = []
        for landmark in key_landmarks:
            if landmark in landmarks:
                lm = landmarks[landmark]
                if lm['visibility'] > 0.3:
                    # Convert to crop coordinates
                    crop_x = int((lm['x'] - x1) * w / (x2 - x1))
                    crop_y = int((lm['y'] - y1) * h / (y2 - y1))
                    if 0 <= crop_x < w and 0 <= crop_y < h:
                        landmark_points.append((crop_x, crop_y))
        
        # If we have enough landmarks, create a convex hull and fill it
        if len(landmark_points) >= 4:
            try:
                points = np.array(landmark_points, dtype=np.int32)
                hull = cv2.convexHull(points)
                
                # Create a mask from the hull
                hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(hull_mask, [hull], 255)
                
                # Combine with original segmentation (intersection)
                refined_mask = cv2.bitwise_and(mask, hull_mask)
                
                # Also include areas near landmarks (dilate)
                kernel = np.ones((5, 5), np.uint8)
                refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
            except:
                pass  # Fallback to original mask
        
        return refined_mask


class PoseAnalyzer:
    """Analyze pose using MediaPipe"""
    
    def __init__(self):
        if not MP_AVAILABLE or mp is None:
            print("MediaPipe not available. Pose analysis disabled.")
            self.mp_pose = None
            self.pose = None
            return
        
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                smooth_landmarks=True
            )
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            self.mp_pose = None
            self.pose = None
    
    def extract_pose(self, image, bbox):
        """Extract pose landmarks from person crop"""
        if self.pose is None:
            return None
            
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
        padding = 0.2
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1_crop = max(0, int(x1) - pad_x)
        y1_crop = max(0, int(y1) - pad_y)
        x2_crop = min(w, int(x2) + pad_x)
        y2_crop = min(h, int(y2) + pad_y)
        
        person_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if person_crop.size == 0:
            return None
        
        # Convert to RGB
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(person_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to image coordinates
        landmarks = {}
        for landmark in self.mp_pose.PoseLandmark:
            lm = results.pose_landmarks.landmark[landmark.value]
            # Convert from crop coordinates to full image coordinates
            x = lm.x * (x2_crop - x1_crop) + x1_crop
            y = lm.y * (y2_crop - y1_crop) + y1_crop
            landmarks[landmark] = {
                'x': x,
                'y': y,
                'z': lm.z,
                'visibility': lm.visibility
            }
        
        return {
            'landmarks': landmarks,
            'confidence': np.mean([lm['visibility'] for lm in landmarks.values()])
        }
    
    def compute_body_orientation(self, pose_data):
        """Compute body orientation vector"""
        if self.mp_pose is None or pose_data is None:
            return None
            
        landmarks = pose_data['landmarks']
        
        # Get key points
        left_shoulder = landmarks.get(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = landmarks.get(self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_HIP)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        # Compute midpoints
        shoulder_mid = (
            (left_shoulder['x'] + right_shoulder['x']) / 2,
            (left_shoulder['y'] + right_shoulder['y']) / 2
        )
        hip_mid = (
            (left_hip['x'] + right_hip['x']) / 2,
            (left_hip['y'] + right_hip['y']) / 2
        )
        
        # Body orientation vector (from hip to shoulder)
        body_vector = (
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1]
        )
        
        # Normalize
        magnitude = math.sqrt(body_vector[0]**2 + body_vector[1]**2)
        if magnitude > 0:
            body_vector = (body_vector[0] / magnitude, body_vector[1] / magnitude)
        
        return {
            'vector': body_vector,
            'shoulder_mid': shoulder_mid,
            'hip_mid': hip_mid,
            'angle_deg': math.degrees(math.atan2(body_vector[1], body_vector[0]))
        }
    
    def compute_leg_features(self, pose_data):
        """Compute leg position features"""
        if self.mp_pose is None or pose_data is None:
            return None
            
        landmarks = pose_data['landmarks']
        
        left_ankle = landmarks.get(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        if not left_ankle or not right_ankle:
            return None
        
        separation = abs(left_ankle['x'] - right_ankle['x'])
        height_diff = abs(left_ankle['y'] - right_ankle['y'])
        
        return {
            'separation': separation,
            'height_diff': height_diff,
            'left_ankle': (left_ankle['x'], left_ankle['y']),
            'right_ankle': (right_ankle['x'], right_ankle['y'])
        }


def compute_conflict_risk(person_center, pose_data, road_grid, pose_analyzer):
    """
    Compute conflict risk score (0-1) using BOTH SegFormer and manual trapezoid parameters
    
    Factors:
    1. Position: In SegFormer road? In manual trapezoid? Proximity? (0.4 weight)
    2. Agreement: Both SegFormer and manual agree on road? (0.1 weight)
    3. Pose: Body inclining towards road? (0.5 weight)
    """
    x, y = person_center
    conflict_score = 0.0
    
    # Check position in both SegFormer and manual trapezoid (separate features)
    in_segformer, in_manual = road_grid.is_in_street_both(x, y)
    
    # Factor 1: Position-based risk (0.4 weight)
    position_score = 0.0
    distance_score_segformer = 0.0
    distance_score_manual = 0.0
    
    if in_segformer:
        # In SegFormer-detected road
        distance_to_road = road_grid.get_distance_to_road(x, y)
        if distance_to_road != float('inf'):
            distance_score_segformer = distance_to_road
        else:
            grid_row, grid_col = road_grid.get_grid_cell(x, y)
            distance_score_segformer = grid_row / road_grid.grid_rows if road_grid.grid_rows > 0 else 0.5
        position_score += 0.2 * distance_score_segformer  # 0.2 weight for SegFormer
    
    if in_manual:
        # In manual trapezoid
        if road_grid.manual_trapezoid_mask is not None:
            # Compute distance in manual trapezoid
            _, y_min, _, y_max = road_grid._get_trapezoid_bbox(road_grid.manual_trapezoid)
            road_height = y_max - y_min
            if road_height > 0:
                distance_score_manual = (y - y_min) / road_height  # 0=top, 1=bottom
        else:
            grid_row, grid_col = road_grid.get_grid_cell(x, y)
            distance_score_manual = grid_row / road_grid.grid_rows if road_grid.grid_rows > 0 else 0.5
        position_score += 0.2 * distance_score_manual  # 0.2 weight for manual
    
    # Factor 2: Agreement between SegFormer and manual (0.1 weight)
    agreement_score = 0.0
    if in_segformer and in_manual:
        agreement_score = 1.0  # Both agree person is in road
    elif not in_segformer and not in_manual:
        agreement_score = 0.0  # Both agree person is not in road
    else:
        agreement_score = 0.5  # Disagreement
    
    conflict_score += 0.4 * position_score + 0.1 * agreement_score
    
    # Determine position type
    if in_segformer and in_manual:
        position_type = "street_both"
    elif in_segformer:
        position_type = "street_segformer"
    elif in_manual:
        position_type = "street_manual"
    elif road_grid.is_in_pavement(x, y):
        position_type = "pavement"
    else:
        position_type = "outside"
    
    # Factor 3: Pose inclination (0.5 weight)
    pose_score = 0.0
    angle_to_street_segformer = None
    angle_to_street_manual = None
    body_orientation = None
    
    if pose_data and pose_data.get('confidence', 0) > 0.5:
        body_orientation = pose_analyzer.compute_body_orientation(pose_data)
        
        if body_orientation:
            # Compute angle to SegFormer road center
            if road_grid.road_mask is not None:
                segformer_center = road_grid.get_street_center()
                to_segformer = (segformer_center[0] - x, segformer_center[1] - y)
                to_segformer_mag = math.sqrt(to_segformer[0]**2 + to_segformer[1]**2)
                if to_segformer_mag > 0:
                    to_segformer = (to_segformer[0] / to_segformer_mag, to_segformer[1] / to_segformer_mag)
                    body_vec = body_orientation['vector']
                    dot_product = body_vec[0] * to_segformer[0] + body_vec[1] * to_segformer[1]
                    angle_rad = math.acos(max(-1, min(1, dot_product)))
                    angle_to_street_segformer = math.degrees(angle_rad)
            
            # Compute angle to manual trapezoid center
            if road_grid.manual_trapezoid:
                manual_center = road_grid._get_trapezoid_center(road_grid.manual_trapezoid)
                to_manual = (manual_center[0] - x, manual_center[1] - y)
                to_manual_mag = math.sqrt(to_manual[0]**2 + to_manual[1]**2)
                if to_manual_mag > 0:
                    to_manual = (to_manual[0] / to_manual_mag, to_manual[1] / to_manual_mag)
                    body_vec = body_orientation['vector']
                    dot_product = body_vec[0] * to_manual[0] + body_vec[1] * to_manual[1]
                    angle_rad = math.acos(max(-1, min(1, dot_product)))
                    angle_to_street_manual = math.degrees(angle_rad)
            
            # Use average angle if both available, otherwise use available one
            if angle_to_street_segformer is not None and angle_to_street_manual is not None:
                angle_to_street = (angle_to_street_segformer + angle_to_street_manual) / 2
            elif angle_to_street_segformer is not None:
                angle_to_street = angle_to_street_segformer
            elif angle_to_street_manual is not None:
                angle_to_street = angle_to_street_manual
            else:
                angle_to_street = None
            
            if angle_to_street is not None:
                if angle_to_street < 45:
                    pose_score = 1.0 - (angle_to_street / 45.0)
                elif angle_to_street > 135:
                    pose_score = 0.0
                else:
                    pose_score = 0.5 - ((angle_to_street - 45) / 90.0) * 0.5
        
        conflict_score += 0.5 * pose_score
    
    return {
        'conflict_score': min(1.0, conflict_score),
        'position_type': position_type,
        'in_segformer': in_segformer,
        'in_manual': in_manual,
        'distance_score_segformer': distance_score_segformer,
        'distance_score_manual': distance_score_manual,
        'agreement_score': agreement_score,
        'pose_score': pose_score,
        'angle_to_street_segformer': angle_to_street_segformer,
        'angle_to_street_manual': angle_to_street_manual,
        'body_orientation': body_orientation
    }


def load_yolo_bboxes(label_path, image_width=640, image_height=640, class_id=0):
    """
    Load person bounding boxes from YOLO label file
    
    Args:
        label_path: Path to YOLO label file (.txt)
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_id: Class ID for person (usually 0)
    
    Returns:
        List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    bboxes = []
    label_path = Path(label_path)
    
    if not label_path.exists():
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                if cls == class_id:  # Person class
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height
                    
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    bboxes.append((x1, y1, x2, y2))
    
    return bboxes


def visualize_conflict_risk(image_path, person_bboxes, 
                           calibration_file=None, use_road_detector=True):
    """
    Visualize conflict risk assessment on a single image using SegFormer road detection
    
    Args:
        image_path: Path to image file
        person_bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        calibration_file: Path to calibration JSON file (optional, for backward compatibility)
        use_road_detector: Whether to use SegFormer for automatic road detection (default: True)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Initialize road grid with SegFormer detection
    road_mask = None
    road_polygon = None
    sidewalk_mask = None
    segformer_detected = False
    
    if use_road_detector and ROAD_DETECTOR_AVAILABLE:
        try:
            from road_detector import RoadDetector
            print("Detecting road region with SegFormer...")
            detector = RoadDetector()
            road_mask, road_polygon, sidewalk_mask = detector.detect_road(image)
            
            if road_mask is not None and road_polygon:
                print(f"✓ SegFormer road region detected: {len(road_polygon)} polygon points")
                print(f"  Road pixels: {(road_mask > 0).sum()}")
                if sidewalk_mask is not None:
                    print(f"  Sidewalk pixels: {(sidewalk_mask > 0).sum()}")
                segformer_detected = True
            else:
                print("⚠ SegFormer road detection failed, using calibration/default")
                use_road_detector = False
        except Exception as e:
            print(f"Warning: Road detection failed: {e}")
            import traceback
            traceback.print_exc()
            use_road_detector = False
    
    # Initialize road grid with SegFormer results AND calibration (both for comparison)
    if segformer_detected and road_mask is not None:
        road_grid = RoadGrid(
            img_width=w, 
            img_height=h,
            road_mask=road_mask,
            road_polygon=road_polygon,
            sidewalk_mask=sidewalk_mask,
            calibration_file=calibration_file  # Also load manual trapezoid if available
        )
    else:
        # Fallback to calibration file or default
        road_grid = RoadGrid(img_width=w, img_height=h, calibration_file=calibration_file)
    
    # Initialize pose analyzer and human segmenter
    pose_analyzer = PoseAnalyzer()
    human_segmenter = HumanSegmenter()
    
    # Process each person
    results = []
    for i, bbox in enumerate(person_bboxes):
        x1, y1, x2, y2 = bbox
        person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Extract pose
        pose_data = pose_analyzer.extract_pose(image, bbox)
        
        # Segment person (aligned with pose)
        segmentation_mask = human_segmenter.segment_person(image, bbox, pose_data)
        
        # Compute conflict risk
        risk_data = compute_conflict_risk(person_center, pose_data, road_grid, pose_analyzer)
        
        results.append({
            'person_id': i,
            'bbox': bbox,
            'center': person_center,
            'pose_data': pose_data,
            'segmentation_mask': segmentation_mask,
            'risk_data': risk_data
        })
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image_rgb)
    title = 'Conflict Risk Assessment - SegFormer + Manual Calibration'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Draw SegFormer-detected road polygon (yellow, dashed) - ALWAYS show if available
    if road_grid.road_polygon:
        street_corners = np.array(road_grid.road_polygon)
        street_poly = Polygon(street_corners, closed=True, 
                             fill=False, edgecolor='yellow', linewidth=2.5, linestyle='--', alpha=0.9)
        ax.add_patch(street_poly)
        # Add label at top of road region
        if len(street_corners) > 0:
            top_point = min(street_corners, key=lambda p: p[1])
            ax.text(top_point[0], top_point[1] - 10,
                    'SEGFORMER ROAD', color='yellow', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.5, edgecolor='yellow'))
    
    # Draw manual trapezoid overlay (magenta, solid) - ALWAYS show if available (separate feature)
    if road_grid.manual_trapezoid:
        manual_corners = np.array(road_grid.manual_trapezoid)
        manual_poly = Polygon(manual_corners, closed=True,
                             fill=False, edgecolor='magenta', linewidth=2.5, linestyle='-', alpha=0.9)
        ax.add_patch(manual_poly)
        # Add label
        if len(manual_corners) > 0:
            top_point = min(manual_corners, key=lambda p: p[1])
            ax.text(top_point[0], top_point[1] - 30,
                    'MANUAL TRAPEZOID', color='magenta', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='magenta', alpha=0.5, edgecolor='magenta'))
    
    # Draw SegFormer-derived sidewalk/pavement mask (cyan overlay)
    if hasattr(road_grid, 'pavement_mask') and road_grid.pavement_mask is not None and road_grid.pavement_mask.sum() > 0:
        # Create overlay for pavement visualization (cyan semi-transparent)
        pavement_overlay = np.zeros((h, w, 4), dtype=np.float32)
        pavement_mask_bool = (road_grid.pavement_mask > 0)
        pavement_overlay[:, :, 0] = 1.0  # Cyan (RGB: cyan = 0, 1, 1)
        pavement_overlay[:, :, 1] = 1.0
        pavement_overlay[:, :, 3] = pavement_mask_bool.astype(np.float32) * 0.3  # Alpha
        
        # Draw pavement overlay
        ax.imshow(pavement_overlay, extent=[0, w, h, 0], alpha=0.3)
        
        # Draw pavement boundary
        pavement_contours, _ = cv2.findContours(road_grid.pavement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in pavement_contours:
            if len(contour) > 2:
                contour_points = contour.reshape(-1, 2)
                ax.plot(contour_points[:, 0], contour_points[:, 1], 'c-', linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Add label
        if len(pavement_contours) > 0:
            # Find a good position for label (top-left of pavement region)
            for contour in pavement_contours:
                if len(contour) > 0:
                    top_point = contour[contour[:, :, 1].argmin()][0]
                    ax.text(top_point[0], top_point[1] - 10, 'SEGFORMER PAVEMENT', 
                           color='cyan', fontsize=9, fontweight='bold')
                    break
    
    # Add legend in top-right corner
    legend_y = 20
    if road_grid.road_polygon and not road_grid.use_manual_trapezoid:
        ax.text(w - 200, legend_y, 'Yellow (dashed): SegFormer Road', 
               fontsize=8, color='yellow', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        legend_y += 25
    if road_grid.manual_trapezoid:
        ax.text(w - 200, legend_y, 'Magenta (solid): Manual Trapezoid', 
               fontsize=8, color='magenta', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        legend_y += 25
    if hasattr(road_grid, 'pavement_mask') and road_grid.pavement_mask is not None and road_grid.pavement_mask.sum() > 0:
        ax.text(w - 200, legend_y, 'Cyan: SegFormer Sidewalk/Pavement', 
               fontsize=8, color='cyan', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    # Draw grid lines (based on road bounding box)
    x_min, y_min, x_max, y_max = road_grid.road_bbox
    for i in range(road_grid.grid_rows + 1):
        y = y_min + i * road_grid.cell_height
        ax.axhline(y, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    for j in range(road_grid.grid_cols + 1):
        x = x_min + j * road_grid.cell_width
        ax.axvline(x, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Draw each person
    for result in results:
        bbox = result['bbox']
        person_center = result['center']
        pose_data = result['pose_data']
        risk_data = result['risk_data']
        
        x1, y1, x2, y2 = bbox
        conflict_score = risk_data['conflict_score']
        
        # Color based on conflict score
        if conflict_score > 0.7:
            color = 'red'
            risk_level = 'HIGH'
        elif conflict_score > 0.4:
            color = 'orange'
            risk_level = 'MEDIUM'
        else:
            color = 'green'
            risk_level = 'LOW'
        
        # Draw human segmentation mask if available
        if result.get('segmentation_mask') is not None:
            seg_mask = result['segmentation_mask']
            # Create overlay for segmentation
            seg_overlay = np.zeros((h, w, 4), dtype=np.float32)
            seg_mask_bool = (seg_mask > 0)
            # Color based on conflict risk
            if conflict_score > 0.7:
                seg_overlay[:, :, 0] = 1.0  # Red
            elif conflict_score > 0.4:
                seg_overlay[:, :, 0] = 1.0  # Orange
                seg_overlay[:, :, 1] = 0.65
            else:
                seg_overlay[:, :, 1] = 1.0  # Green
            seg_overlay[:, :, 3] = seg_mask_bool.astype(np.float32) * 0.4  # Alpha
            ax.imshow(seg_overlay, extent=[0, w, h, 0], alpha=0.4)
        
        # Draw bounding box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        # Draw pose landmarks if available
        if pose_data and pose_analyzer.mp_pose:
            landmarks = pose_data['landmarks']
            mp_pose = pose_analyzer.mp_pose
            
            # Draw key points
            key_points = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.NOSE
            ]
            
            for landmark in key_points:
                if landmark in landmarks:
                    lm = landmarks[landmark]
                    if lm['visibility'] > 0.5:
                        ax.plot(lm['x'], lm['y'], 'o', color=color, markersize=6)
            
            # Draw body orientation vector
            body_orientation = risk_data.get('body_orientation')
            if body_orientation:
                hip_mid = body_orientation['hip_mid']
                shoulder_mid = body_orientation['shoulder_mid']
                ax.arrow(hip_mid[0], hip_mid[1],
                        (shoulder_mid[0] - hip_mid[0]) * 2,
                        (shoulder_mid[1] - hip_mid[1]) * 2,
                        head_width=10, head_length=10, fc=color, ec=color, alpha=0.7)
                
                # Draw direction to street
                street_center = road_grid.get_street_center()
                ax.arrow(person_center[0], person_center[1],
                        (street_center[0] - person_center[0]) * 0.3,
                        (street_center[1] - person_center[1]) * 0.3,
                        head_width=8, head_length=8, fc='yellow', ec='yellow', 
                        linestyle='--', alpha=0.5)
        
        # Add text annotation with both SegFormer and manual parameters
        grid_row, grid_col = road_grid.get_grid_cell(person_center[0], person_center[1])
        
        # Build angle string
        angle_parts = []
        if risk_data.get('angle_to_street_segformer') is not None:
            angle_parts.append(f"SegF: {risk_data['angle_to_street_segformer']:.1f}°")
        if risk_data.get('angle_to_street_manual') is not None:
            angle_parts.append(f"Man: {risk_data['angle_to_street_manual']:.1f}°")
        angle_str = " | ".join(angle_parts) if angle_parts else "N/A"
        
        # Position info
        pos_info = risk_data['position_type']
        if risk_data.get('in_segformer') is not None and risk_data.get('in_manual') is not None:
            pos_info += f" (SegF:{risk_data['in_segformer']}, Man:{risk_data['in_manual']})"
        
        info_text = (
            f"Person {result['person_id']+1}\n"
            f"Risk: {risk_level} ({conflict_score:.2f})\n"
            f"Position: {pos_info}\n"
            f"Grid: ({grid_row}, {grid_col})\n"
            f"Angles: {angle_str}\n"
            f"Agreement: {risk_data.get('agreement_score', 0):.2f}"
        )
        
        ax.text(x1, y1 - 5, info_text, color=color, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_conflict_risk.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CONFLICT RISK ASSESSMENT SUMMARY")
    print("="*60)
    for result in results:
        risk = result['risk_data']
        print(f"\nPerson {result['person_id']+1}:")
        print(f"  Conflict Score: {risk['conflict_score']:.3f}")
        print(f"  Position: {risk['position_type']}")
        if risk['position_type'] == 'street':
            print(f"  Distance Score: {risk['distance_score']:.3f}")
        print(f"  Pose Score: {risk['pose_score']:.3f}")
        if risk['angle_to_street']:
            print(f"  Angle to Street: {risk['angle_to_street']:.1f}°")
    print("="*60)
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Set image path (change this to your image)
    image_path = Path("rsud20k_person2000_resized/images/train/train14961.jpg")
    
    # SegFormer is now the default method for road detection
    # Set use_road_detector=True to use SegFormer (recommended)
    # Set use_road_detector=False and provide calibration_file for manual calibration
    use_road_detector = True  # Use SegFormer for automatic road detection
    
    # Optional: Load calibration from JSON file (for backward compatibility)
    calibration_file = image_path.parent.parent / "grid_calibration.json"
    if not calibration_file.exists():
        calibration_file = None
    
    # Load bounding boxes from YOLO label file (recommended)
    label_path = image_path.parent.parent.parent / "labels" / "train" / f"{image_path.stem}.txt"
    person_bboxes = load_yolo_bboxes(label_path, image_width=640, image_height=640, class_id=0)
    
    # Alternative: Define bounding boxes manually
    # person_bboxes = [
    #     (200, 300, 350, 550),  # Person 1
    #     (400, 250, 550, 500),  # Person 2
    # ]
    
    if not person_bboxes:
        print("No person bounding boxes found. Please check the label file or define manually.")
    else:
        print(f"Found {len(person_bboxes)} person(s) in the image")
        print("Using SegFormer for road detection...")
        visualize_conflict_risk(image_path, person_bboxes, 
                               calibration_file=calibration_file,
                               use_road_detector=use_road_detector)

