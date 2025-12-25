#!/usr/bin/env python3
"""
Interactive Grid Calibration Tool with Image Navigation
Navigate through multiple images to validate grid calibration parameters
"""

import cv2
import numpy as np
from pathlib import Path
import json

# Try to import MediaPipe
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
        print(f"Note: MediaPipe not available: {e}")

# Import road detector
try:
    from road_detector import RoadDetector
    ROAD_DETECTOR_AVAILABLE = True
except ImportError:
    ROAD_DETECTOR_AVAILABLE = False
    print("Warning: Road detector not available. Install transformers: pip install transformers")


class HumanSegmenter:
    """Human segmentation using MediaPipe, aligned with pose landmarks"""
    
    def __init__(self):
        if not MP_AVAILABLE or mp is None:
            self.mp_selfie_segmentation = None
            self.selfie_segmentation = None
            self.mp_pose = None
            self.pose = None
            return
        
        try:
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # Landscape mode for dashcam
            )
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            self.mp_selfie_segmentation = None
            self.selfie_segmentation = None
            self.mp_pose = None
            self.pose = None
    
    def segment_person(self, image, bbox):
        """Segment person from image using MediaPipe"""
        if self.selfie_segmentation is None:
            return None
        
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Add padding
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
        
        # Get pose for alignment
        pose_results = self.pose.process(person_rgb) if self.pose else None
        
        # Convert to binary mask
        segmentation_mask = results.segmentation_mask
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Align with pose if available
        if pose_results and pose_results.pose_landmarks:
            binary_mask = self._align_with_pose(binary_mask, pose_results, x1_crop, y1_crop, x2_crop, y2_crop)
        
        # Create full image mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        crop_h, crop_w = binary_mask.shape
        full_mask[y1_crop:y1_crop+crop_h, x1_crop:x1_crop+crop_w] = binary_mask
        
        return full_mask
    
    def _align_with_pose(self, mask, pose_results, x1, y1, x2, y2):
        """Refine segmentation mask using pose landmarks"""
        if not pose_results or not pose_results.pose_landmarks:
            return mask
        
        h, w = mask.shape
        landmark_points = []
        
        # Get key landmarks
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.NOSE
        ]
        
        for landmark in key_landmarks:
            lm = pose_results.pose_landmarks.landmark[landmark.value]
            if lm.visibility > 0.3:
                crop_x = int(lm.x * w)
                crop_y = int(lm.y * h)
                if 0 <= crop_x < w and 0 <= crop_y < h:
                    landmark_points.append((crop_x, crop_y))
        
        if len(landmark_points) >= 4:
            try:
                points = np.array(landmark_points, dtype=np.int32)
                hull = cv2.convexHull(points)
                hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(hull_mask, [hull], 255)
                mask = cv2.bitwise_and(mask, hull_mask)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            except:
                pass
        
        return mask


class GridCalibrator:
    """Interactive grid calibration with multi-image navigation"""
    
    def __init__(self, image_dir, grid_rows=8, grid_cols=6, calibration_file=None, use_road_detector=True):
        """
        Initialize calibrator with image directory
        
        Args:
            image_dir: Directory containing images
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            calibration_file: Path to existing calibration JSON (optional)
            use_road_detector: Whether to use SegFormer for automatic road detection
        """
        self.image_dir = Path(image_dir)
        
        # Load all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        self.image_paths = sorted(self.image_paths)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nFound {len(self.image_paths)} images")
        
        self.current_image_idx = 0
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Initialize road detector
        self.use_road_detector = use_road_detector and ROAD_DETECTOR_AVAILABLE
        self.road_detector = None
        if self.use_road_detector:
            try:
                self.road_detector = RoadDetector()
                print("✓ SegFormer road detector loaded")
            except Exception as e:
                print(f"⚠ Road detector initialization failed: {e}")
                self.use_road_detector = False
        
        # Load calibration if provided, otherwise use defaults
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)
        else:
            self.init_default_calibration()
        
        # Mouse interaction state
        self.dragging = False
        self.selected_point = None
        self.show_grid = True
        self.edit_mode = True  # Toggle between edit and view mode
        self.show_road_mask = False  # Toggle road mask overlay
        self.show_full_segmentation = False  # Toggle full SegFormer segmentation
        self.show_human_segmentation = False  # Toggle human segmentation overlay
        self.show_pose = False  # Toggle pose visualization
        
        # Initialize human segmenter (has pose detection built-in)
        self.human_segmenter = HumanSegmenter()
        
        # Store detected poses
        self.detected_poses = []
        
        # Load first image
        self.load_image(0)
        
        # Create window
        cv2.namedWindow('Grid Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Grid Calibration', self.mouse_callback)
        
        self.print_instructions()
    
    def init_default_calibration(self):
        """Initialize with default calibration values"""
        # These will be set when first image loads
        self.trapezoid_points = None
        self.pavement_mask = None  # SegFormer-derived pavement mask
        self.image_width = 640
        self.image_height = 640
    
    def load_calibration(self, calibration_file):
        """Load calibration from JSON file"""
        with open(calibration_file, 'r') as f:
            cal = json.load(f)
        
        self.grid_rows = cal.get('grid_rows', 8)
        self.grid_cols = cal.get('grid_cols', 6)
        # Note: pavement_width is deprecated, using pavement_mask instead
        
        # Load trapezoid points
        if 'street_trapezoid' in cal:
            trapezoid = cal['street_trapezoid']
            self.trapezoid_points = [
                tuple(trapezoid['top_left']),
                tuple(trapezoid['top_right']),
                tuple(trapezoid['bottom_right']),
                tuple(trapezoid['bottom_left'])
            ]
        else:
            self.trapezoid_points = None
        
        print(f"✓ Loaded calibration from {calibration_file}")
    
    def load_image(self, idx):
        """Load image at given index"""
        if idx < 0 or idx >= len(self.image_paths):
            return False
        
        self.current_image_idx = idx
        self.image_path = self.image_paths[idx]
        self.image = cv2.imread(str(self.image_path))
        
        if self.image is None:
            print(f"Warning: Could not load {self.image_path}")
            return False
        
        self.h, self.w = self.image.shape[:2]
        
        # Initialize trapezoid if not set
        if self.trapezoid_points is None:
            center_x = self.w / 2
            self.trapezoid_points = [
                (center_x - 200, 50),      # Top-left
                (center_x + 200, 50),      # Top-right
                (self.w, self.h),          # Bottom-right
                (0, self.h)                # Bottom-left
            ]
        else:
            # Scale trapezoid points if image size differs
            if hasattr(self, 'image_width') and self.image_width != self.w:
                scale_x = self.w / self.image_width
                scale_y = self.h / self.image_height
                self.trapezoid_points = [
                    (p[0] * scale_x, p[1] * scale_y) for p in self.trapezoid_points
                ]
        
        self.image_width = self.w
        self.image_height = self.h
        
        # Store road detection results
        self.road_mask = None
        self.sidewalk_mask = None
        self.pavement_mask = None
        self.full_segmentation = None
        self.segmentation_info = None
        
        # Detect poses in the image
        self.detect_poses()
        
        self.update_display()
        return True
    
    def detect_poses(self):
        """Detect all poses in the current image using MediaPipe"""
        self.detected_poses = []
        
        if not MP_AVAILABLE or not self.human_segmenter or not self.human_segmenter.pose:
            return
        
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Process full image for pose detection
            results = self.human_segmenter.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Convert landmarks to image coordinates
                h, w = self.image.shape[:2]
                landmarks = {}
                for landmark in self.human_segmenter.mp_pose.PoseLandmark:
                    lm = results.pose_landmarks.landmark[landmark.value]
                    if lm.visibility > 0.3:  # Only include visible landmarks
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        landmarks[landmark] = {
                            'x': x,
                            'y': y,
                            'z': lm.z,
                            'visibility': lm.visibility
                        }
                
                if landmarks:
                    self.detected_poses.append({
                        'landmarks': landmarks,
                        'confidence': np.mean([lm['visibility'] for lm in landmarks.values()])
                    })
        except Exception as e:
            # Silently fail if pose detection fails
            pass
    
    def auto_detect_road(self):
        """Automatically detect road region using SegFormer"""
        if not self.road_detector:
            print("Road detector not available")
            return False
        
        print("Detecting road region with SegFormer...")
        try:
            # Get full segmentation for visualization
            result = self.road_detector.detect_road(self.image, return_full_segmentation=True)
            road_mask, road_polygon, sidewalk_mask, full_segmentation, class_info = result
            
            if road_polygon and len(road_polygon) >= 4:
                # Update trapezoid points
                self.trapezoid_points = road_polygon
                
                # Store masks and full segmentation for visualization
                self.road_mask = road_mask
                self.sidewalk_mask = sidewalk_mask
                self.full_segmentation = full_segmentation
                self.segmentation_info = class_info
                
                # Derive pavement mask from SegFormer sidewalk or adjacent areas
                self.pavement_mask = self._derive_pavement_mask(adjacent_threshold=50)
                
                # Print detection statistics
                if class_info:
                    print(f"\n✓ SegFormer Detection Results:")
                    print(f"  Road pixels: {class_info['road_pixels']} ({class_info['road_pixels']/class_info['total_pixels']*100:.1f}%)")
                    print(f"  Sidewalk pixels: {class_info['sidewalk_pixels']} ({class_info['sidewalk_pixels']/class_info['total_pixels']*100:.1f}%)")
                    print(f"  Average confidence: {class_info['avg_confidence']:.3f}")
                    print(f"  Classes detected: {len(class_info['classes_detected'])}")
                
                self.update_display()
                print("✓ Road region auto-detected")
                return True
            else:
                print("⚠ Could not detect road region")
                return False
        except Exception as e:
            print(f"Error during road detection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if not self.edit_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near a trapezoid corner
            for i, point in enumerate(self.trapezoid_points):
                px, py = point
                if abs(x - px) < 10 and abs(y - py) < 10:
                    self.dragging = True
                    self.selected_point = i
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_point is not None:
                # Update selected point
                self.trapezoid_points[self.selected_point] = (x, y)
                self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_point = None
    
    def update_display(self):
        """Update the display image with grid overlay"""
        self.display_image = self.image.copy()
        
        # Show full SegFormer segmentation if enabled
        if self.show_full_segmentation and self.full_segmentation is not None and self.road_detector:
            # Use road detector's visualization method
            self.display_image = self.road_detector.visualize_segmentation(
                self.display_image, 
                self.full_segmentation,
                self.segmentation_info
            )
        # Show road mask overlay if available and enabled (simpler view)
        elif self.show_road_mask and self.road_mask is not None:
            # Overlay road mask in green
            road_mask_colored = np.zeros_like(self.display_image)
            road_mask_colored[:, :, 1] = self.road_mask  # Green channel
            self.display_image = cv2.addWeighted(self.display_image, 0.7, road_mask_colored, 0.3, 0)
            
            # Overlay sidewalk mask in blue
            if self.sidewalk_mask is not None:
                sidewalk_mask_colored = np.zeros_like(self.display_image)
                sidewalk_mask_colored[:, :, 0] = self.sidewalk_mask  # Blue channel
                self.display_image = cv2.addWeighted(self.display_image, 0.7, sidewalk_mask_colored, 0.3, 0)
        
        # Draw manual trapezoid (street region) - Magenta for manual
        pts = np.array(self.trapezoid_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Create overlay for semi-transparent fill (magenta for manual)
        overlay = self.display_image.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 255))  # Magenta (BGR)
        cv2.addWeighted(overlay, 0.15, self.display_image, 0.85, 0, self.display_image)
        cv2.polylines(self.display_image, [pts], True, (255, 0, 255), 2)  # Magenta border
        
        # Draw SegFormer road polygon overlay (yellow, dashed) if available
        if self.road_mask is not None and self.road_polygon and len(self.road_polygon) >= 4:
            seg_pts = np.array(self.road_polygon, np.int32)
            seg_pts = seg_pts.reshape((-1, 1, 2))
            # Draw as dashed line (approximate with small segments)
            for i in range(len(seg_pts) - 1):
                pt1 = tuple(seg_pts[i][0])
                pt2 = tuple(seg_pts[i+1][0])
                cv2.line(self.display_image, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
            # Close the polygon
            cv2.line(self.display_image, tuple(seg_pts[-1][0]), tuple(seg_pts[0][0]), 
                    (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw trapezoid corners (only in edit mode)
        if self.edit_mode:
            for i, (px, py) in enumerate(self.trapezoid_points):
                color = (0, 0, 255) if i == self.selected_point else (0, 255, 255)
                cv2.circle(self.display_image, (int(px), int(py)), 8, color, -1)
                cv2.putText(self.display_image, f"P{i+1}", (int(px)+10, int(py)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw SegFormer-derived pavement mask
        if self.pavement_mask is not None and self.pavement_mask.sum() > 0:
            overlay = self.display_image.copy()
            # Create cyan overlay for pavement
            pavement_colored = np.zeros_like(overlay)
            pavement_colored[:, :, 0] = 255  # Cyan (BGR: blue channel)
            pavement_colored[:, :, 1] = 255  # Green channel
            pavement_colored = cv2.bitwise_and(pavement_colored, pavement_colored, mask=self.pavement_mask)
            cv2.addWeighted(overlay, 0.8, pavement_colored, 0.2, 0, self.display_image)
            
            # Draw pavement boundary
            pavement_contours, _ = cv2.findContours(self.pavement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in pavement_contours:
                if len(contour) > 2:
                    cv2.drawContours(self.display_image, [contour], -1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Draw pose visualization if enabled
        if self.show_pose and self.detected_poses:
            self._draw_poses()
        
        # Draw human segmentation if enabled
        if self.show_human_segmentation and self.human_segmenter.selfie_segmentation:
            # Try to detect and segment people in the image
            # For now, we'll use a simple approach - segment the center region
            # In a full implementation, you'd use a person detector first
            h_center, w_center = self.h // 2, self.w // 2
            test_bbox = (w_center - 100, h_center - 150, w_center + 100, h_center + 50)
            seg_mask = self.human_segmenter.segment_person(self.image, test_bbox)
            if seg_mask is not None and seg_mask.sum() > 0:
                overlay = self.display_image.copy()
                # Create green overlay for human segmentation
                human_colored = np.zeros_like(overlay)
                human_colored[:, :, 1] = 255  # Green channel
                human_colored = cv2.bitwise_and(human_colored, human_colored, mask=seg_mask)
                cv2.addWeighted(overlay, 0.7, human_colored, 0.3, 0, self.display_image)
        
        # Draw grid if enabled
        if self.show_grid:
            cell_width = self.w / self.grid_cols
            cell_height = self.h / self.grid_rows
            
            # Vertical lines
            for i in range(self.grid_cols + 1):
                x = int(i * cell_width)
                cv2.line(self.display_image, (x, 0), (x, self.h), (128, 128, 128), 1)
            
            # Horizontal lines
            for i in range(self.grid_rows + 1):
                y = int(i * cell_height)
                cv2.line(self.display_image, (0, y), (self.w, y), (128, 128, 128), 1)
        
        # Add labels
        mode_text = "EDIT MODE" if self.edit_mode else "VIEW MODE"
        mode_color = (0, 255, 0) if self.edit_mode else (0, 165, 255)
        cv2.putText(self.display_image, mode_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(self.display_image, "MANUAL TRAPEZOID (Magenta)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if self.road_mask is not None:
            cv2.putText(self.display_image, "SEGFORMER ROAD (Yellow)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(self.display_image, "PAVEMENT/SIDEWALK (Cyan)", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if self.show_pose:
            cv2.putText(self.display_image, f"POSES: {len(self.detected_poses)}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show image info
        image_name = self.image_path.name
        cv2.putText(self.display_image, f"Image: {image_name}", 
                   (10, self.h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.display_image, f"{self.current_image_idx + 1}/{len(self.image_paths)}", 
                   (10, self.h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show current parameters
        detector_status = "ON" if self.use_road_detector else "OFF"
        seg_view = ""
        if self.show_full_segmentation:
            seg_view = " | SegView: FULL"
        elif self.show_road_mask:
            seg_view = " | SegView: MASK"
        
        pavement_status = "Detected" if (self.pavement_mask is not None and self.pavement_mask.sum() > 0) else "None"
        
        pose_status = f"{len(self.detected_poses)} detected" if self.show_pose else "OFF"
        info_text = [
            f"Grid: {self.grid_rows}x{self.grid_cols} | Pavement: {pavement_status} | Detector: {detector_status}{seg_view}",
            f"A/D: Navigate | E: Edit/View | F: Auto-detect | M: Mask | V: Full Seg | H: Human Seg | P: Pose | G: Grid | S: Save | Q: Quit"
        ]
        y_offset = self.h - 50
        for i, text in enumerate(info_text):
            cv2.putText(self.display_image, text, (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def adjust_pavement_threshold(self, delta):
        """Adjust pavement threshold (distance from road edge)"""
        if self.road_mask is not None:
            # Re-derive pavement with new threshold
            current_threshold = 50  # Default, could be stored as attribute
            new_threshold = max(10, min(200, current_threshold + delta))
            self.pavement_mask = self._derive_pavement_mask(adjacent_threshold=new_threshold)
            self.update_display()
            print(f"Pavement threshold: {new_threshold}px")
    
    def _draw_poses(self):
        """Draw MediaPipe pose landmarks and connections"""
        if not self.detected_poses or not self.human_segmenter.mp_pose:
            return
        
        mp_pose = self.human_segmenter.mp_pose
        mp_drawing = mp.solutions.drawing_utils
        
        for pose_data in self.detected_poses:
            landmarks = pose_data['landmarks']
            
            # Draw pose connections
            # Define pose connections (skeleton structure)
            pose_connections = [
                # Face
                (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE),
                (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.NOSE),
                (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.NOSE),
                # Torso
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                # Left arm
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                # Right arm
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                # Left leg
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                # Right leg
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]
            
            # Draw connections
            for connection in pose_connections:
                start_landmark, end_landmark = connection
                if start_landmark in landmarks and end_landmark in landmarks:
                    start = landmarks[start_landmark]
                    end = landmarks[end_landmark]
                    if start['visibility'] > 0.3 and end['visibility'] > 0.3:
                        cv2.line(self.display_image,
                               (start['x'], start['y']),
                               (end['x'], end['y']),
                               (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw key landmarks (important points)
            key_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.NOSE
            ]
            
            for landmark in key_landmarks:
                if landmark in landmarks:
                    lm = landmarks[landmark]
                    if lm['visibility'] > 0.3:
                        # Draw circle for key point
                        cv2.circle(self.display_image, (lm['x'], lm['y']), 5, (0, 255, 0), -1)
                        cv2.circle(self.display_image, (lm['x'], lm['y']), 5, (0, 0, 0), 1)
            
            # Draw body orientation vector (shoulder to hip)
            if (mp_pose.PoseLandmark.LEFT_SHOULDER in landmarks and
                mp_pose.PoseLandmark.RIGHT_SHOULDER in landmarks and
                mp_pose.PoseLandmark.LEFT_HIP in landmarks and
                mp_pose.PoseLandmark.RIGHT_HIP in landmarks):
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Compute midpoints
                shoulder_mid = ((left_shoulder['x'] + right_shoulder['x']) // 2,
                               (left_shoulder['y'] + right_shoulder['y']) // 2)
                hip_mid = ((left_hip['x'] + right_hip['x']) // 2,
                          (left_hip['y'] + right_hip['y']) // 2)
                
                # Draw orientation arrow
                cv2.arrowedLine(self.display_image, hip_mid, shoulder_mid,
                              (255, 0, 0), 3, cv2.LINE_AA, tipLength=0.3)
    
    def _derive_pavement_mask(self, adjacent_threshold=50):
        """
        Derive pavement mask from SegFormer sidewalk mask or adjacent areas to road
        
        Args:
            adjacent_threshold: Maximum distance (in pixels) from road edge to consider as pavement
        
        Returns:
            pavement_mask: Binary mask of pavement region
        """
        pavement_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
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
    
    def reset(self):
        """Reset to default values"""
        center_x = self.w / 2
        self.trapezoid_points = [
            (center_x - 200, 50),
            (center_x + 200, 50),
            (self.w, self.h),
            (0, self.h)
        ]
        # Reset pavement mask
        if self.road_mask is not None:
            self.pavement_mask = self._derive_pavement_mask(adjacent_threshold=50)
        else:
            self.pavement_mask = None
        self.update_display()
    
    def save_calibration(self):
        """Save calibration parameters to JSON file"""
        # Calculate trapezoid parameters
        top_left, top_right, bottom_right, bottom_left = self.trapezoid_points
        
        street_top_width = top_right[0] - top_left[0]
        street_bottom_width = bottom_right[0] - bottom_left[0]
        
        calibration = {
            'image_width': self.w,
            'image_height': self.h,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'street_trapezoid': {
                'top_left': list(top_left),
                'top_right': list(top_right),
                'bottom_left': list(bottom_left),
                'bottom_right': list(bottom_right),
                'top_width': street_top_width,
                'bottom_width': street_bottom_width
            },
            # Note: pavement_width deprecated, using SegFormer-derived pavement_mask
            'pavement_detection_method': 'segformer_derived'
        }
        
        # Save to file in image directory
        output_path = self.image_dir / "grid_calibration.json"
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✓ Calibration saved to: {output_path}")
        print("\nCalibration parameters:")
        print(f"  Grid: {self.grid_rows}x{self.grid_cols}")
        print(f"  Street top width: {street_top_width:.1f}px")
        print(f"  Street bottom width: {street_bottom_width:.1f}px")
        if self.pavement_mask is not None:
            pavement_pixels = (self.pavement_mask > 0).sum()
            print(f"  Pavement: {pavement_pixels} pixels (SegFormer-derived)")
        else:
            print(f"  Pavement: Not detected")
        
        return output_path
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("GRID CALIBRATION TOOL - Multi-Image Navigation")
        print("="*60)
        print("Navigation:")
        print("  - 'A' / 'D' or 'P' / 'N': Previous/Next image")
        print("  - 'E': Toggle Edit/View mode")
        print("\nRoad Detection (SegFormer):")
        print("  - 'F': Auto-detect road region using SegFormer")
        print("  - 'M': Toggle simple road/sidewalk mask overlay")
        print("  - 'V': Toggle full SegFormer segmentation visualization (all classes)")
        print("\nHuman Segmentation (MediaPipe):")
        print("  - 'H': Toggle human segmentation overlay (aligned with pose)")
        print("\nPose Visualization (MediaPipe):")
        print("  - 'P': Toggle pose landmarks and skeleton visualization")
        print("\nEditing (Edit Mode):")
        print("  - Click and drag trapezoid corners (P1-P4) to adjust street region")
        print("  - Press '+' or '-' to adjust pavement width")
        print("  - Press 'R' to reset to default")
        print("\nDisplay:")
        print("  - Press 'G' to toggle grid overlay")
        print("  - Press 'S' to save calibration")
        print("  - Press 'Q' or ESC to quit")
        print("="*60 + "\n")
    
    def run(self):
        """Run the calibration tool"""
        self.update_display()
        
        while True:
            cv2.imshow('Grid Calibration', self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Navigation
            # Arrow keys: 81=Left, 83=Right (on some systems)
            # Alternative: Use 'A'/'D' or 'P'/'N' for navigation
            if key == ord('a') or key == ord('A') or key == ord('p') or key == ord('P'):  # Previous
                if self.load_image(self.current_image_idx - 1):
                    print(f"Image {self.current_image_idx + 1}/{len(self.image_paths)}: {self.image_path.name}")
            elif key == ord('d') or key == ord('D') or key == ord('n') or key == ord('N'):  # Next
                if self.load_image(self.current_image_idx + 1):
                    print(f"Image {self.current_image_idx + 1}/{len(self.image_paths)}: {self.image_path.name}")
            
            # Edit/View toggle
            elif key == ord('e') or key == ord('E'):
                self.edit_mode = not self.edit_mode
                self.update_display()
                mode = "EDIT" if self.edit_mode else "VIEW"
                print(f"Switched to {mode} mode")
            
            # Quit
            elif key == ord('q') or key == 27:  # Q or ESC
                break
            
            # Road detection
            elif key == ord('f') or key == ord('F'):
                if self.use_road_detector:
                    self.auto_detect_road()
                else:
                    print("Road detector not available. Install transformers: pip install transformers")
            
            # Toggle road mask overlay (simple)
            elif key == ord('m') or key == ord('M'):
                if self.show_full_segmentation:
                    self.show_full_segmentation = False
                self.show_road_mask = not self.show_road_mask
                self.update_display()
                status = "ON" if self.show_road_mask else "OFF"
                print(f"Road mask overlay: {status}")
            
            # Toggle full segmentation visualization
            elif key == ord('v') or key == ord('V'):
                if self.show_road_mask:
                    self.show_road_mask = False
                self.show_full_segmentation = not self.show_full_segmentation
                self.update_display()
                status = "ON" if self.show_full_segmentation else "OFF"
                print(f"Full SegFormer segmentation view: {status}")
                if self.show_full_segmentation and self.segmentation_info:
                    print(f"  Detected {len(self.segmentation_info['classes_detected'])} classes")
            
            # Toggle human segmentation
            elif key == ord('h') or key == ord('H'):
                self.show_human_segmentation = not self.show_human_segmentation
                self.update_display()
                status = "ON" if self.show_human_segmentation else "OFF"
                print(f"Human segmentation overlay: {status}")
            
            # Toggle pose visualization
            elif key == ord('p') or key == ord('P'):
                self.show_pose = not self.show_pose
                # Re-detect poses if enabling
                if self.show_pose:
                    self.detect_poses()
                self.update_display()
                status = "ON" if self.show_pose else "OFF"
                print(f"Pose visualization: {status} ({len(self.detected_poses)} pose(s) detected)")
            
            # Editing controls (only in edit mode)
            elif self.edit_mode:
                if key == ord('g') or key == ord('G'):
                    self.show_grid = not self.show_grid
                    self.update_display()
                elif key == ord('r') or key == ord('R'):
                    self.reset()
                elif key == ord('s') or key == ord('S'):
                    self.save_calibration()
                elif key == ord('+') or key == ord('='):
                    self.adjust_pavement_threshold(5)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_pavement_threshold(-5)
        
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Set image directory
    image_dir = Path("rsud20k_person2000_resized/images/train")
    
    # Optional: Load existing calibration
    calibration_file = None  # Set to path if you have existing calibration
    
    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        print("Please update the image_dir variable in the script.")
    else:
        calibrator = GridCalibrator(
            image_dir, 
            grid_rows=8, 
            grid_cols=6,
            calibration_file=calibration_file
        )
        calibrator.run()
