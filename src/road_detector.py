#!/usr/bin/env python3
"""
Road Detection using SegFormer
Automatic road segmentation from dashcam images for irregular streets
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path


class RoadDetector:
    """Road detection using SegFormer semantic segmentation"""
    
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-640-1280"):
        """
        Initialize SegFormer road detector
        
        Args:
            model_name: HuggingFace model name for SegFormer
                       Options:
                       - "nvidia/segformer-b0-finetuned-cityscapes-640-1280" (lightweight)
                       - "nvidia/segformer-b1-finetuned-cityscapes-640-1280" (balanced)
                       - "nvidia/segformer-b2-finetuned-cityscapes-640-1280" (better accuracy)
        """
        print(f"Loading SegFormer model: {model_name}...")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.eval()
            
            # Cityscapes class mapping: 0=road, 1=sidewalk, etc.
            # We'll use road (0) and optionally sidewalk (1) for pavement
            self.road_class_id = 0
            self.sidewalk_class_id = 1
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if torch.backends.mps.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f"✓ SegFormer loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading SegFormer: {e}")
            print("Falling back to simple color-based detection")
            self.model = None
            self.processor = None
    
    def detect_road(self, image, return_full_segmentation=False):
        """
        Detect road region in image using SegFormer
        
        Args:
            image: Input image (numpy array BGR or PIL Image)
            return_full_segmentation: If True, returns full segmentation map and class info
        
        Returns:
            road_mask: Binary mask of road region (numpy array, 0-255)
            road_polygon: Approximate polygon of road region (list of points)
            sidewalk_mask: Binary mask of sidewalk/pavement (optional)
            full_segmentation: Full semantic segmentation map (if return_full_segmentation=True)
            class_info: Dictionary with class statistics (if return_full_segmentation=True)
        """
        if self.model is None:
            result = self._detect_road_simple(image)
            if return_full_segmentation:
                return result + (None, {})
            return result
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            else:
                image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        original_size = image_pil.size  # (width, height)
        
        # Preprocess
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=original_size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Get predictions (class IDs)
        predictions = upsampled_logits[0].argmax(dim=0).cpu().numpy()
        
        # Get confidence scores (softmax probabilities)
        probs = torch.nn.functional.softmax(upsampled_logits[0], dim=0).cpu().numpy()
        max_probs = probs.max(axis=0)  # Maximum probability for each pixel
        
        # Extract road mask
        road_mask = (predictions == self.road_class_id).astype(np.uint8) * 255
        
        # Extract sidewalk/pavement mask
        sidewalk_mask = (predictions == self.sidewalk_class_id).astype(np.uint8) * 255
        
        # Convert road mask to polygon (simplified for backward compatibility)
        road_polygon = self._mask_to_trapezoid(road_mask)
        
        # Prepare full segmentation info if requested
        full_segmentation = None
        class_info = {}
        if return_full_segmentation:
            full_segmentation = predictions
            # Get class statistics
            unique_classes, counts = np.unique(predictions, return_counts=True)
            class_info = {
                'segmentation_map': predictions,
                'confidence_map': max_probs,
                'classes_detected': unique_classes.tolist(),
                'class_counts': dict(zip(unique_classes.tolist(), counts.tolist())),
                'road_pixels': int((predictions == self.road_class_id).sum()),
                'sidewalk_pixels': int((predictions == self.sidewalk_class_id).sum()),
                'total_pixels': int(predictions.size),
                'avg_confidence': float(max_probs.mean())
            }
        
        if return_full_segmentation:
            return road_mask, road_polygon, sidewalk_mask, full_segmentation, class_info
        else:
            return road_mask, road_polygon, sidewalk_mask
    
    def get_road_polygon_and_mask(self, image):
        """
        Get detailed road polygon and mask for grid-based analysis
        
        Args:
            image: Input image
        
        Returns:
            road_polygon: List of points defining road polygon
            road_mask: Binary road mask
            sidewalk_mask: Binary sidewalk mask
        """
        road_mask, _, sidewalk_mask = self.detect_road(image, return_full_segmentation=False)
        road_polygon, _ = self.mask_to_polygon(road_mask)
        return road_polygon, road_mask, sidewalk_mask
    
    def visualize_segmentation(self, image, segmentation_map, class_info=None):
        """
        Create a color-coded visualization of the full semantic segmentation
        
        Args:
            image: Original image (BGR)
            segmentation_map: Full segmentation map (class IDs)
            class_info: Optional class information dictionary
        
        Returns:
            vis_image: Visualization image with color-coded classes
        """
        # Cityscapes color palette (19 classes)
        # Format: (B, G, R) for OpenCV
        cityscapes_colors = {
            0: (128, 64, 128),    # road - purple
            1: (244, 35, 232),    # sidewalk - pink
            2: (70, 70, 70),      # building - dark gray
            3: (102, 102, 156),   # wall - blue-gray
            4: (190, 153, 153),   # fence - light brown
            5: (153, 153, 153),   # pole - gray
            6: (250, 170, 30),    # traffic light - orange
            7: (220, 220, 0),     # traffic sign - yellow
            8: (107, 142, 35),    # vegetation - green
            9: (152, 251, 152),   # terrain - light green
            10: (70, 130, 180),   # sky - blue
            11: (220, 20, 60),    # person - red
            12: (255, 0, 0),      # rider - bright red
            13: (0, 0, 142),      # car - dark blue
            14: (0, 0, 70),       # truck - darker blue
            15: (0, 60, 100),     # bus - blue-green
            16: (0, 80, 100),     # train - teal
            17: (0, 0, 230),      # motorcycle - bright blue
            18: (119, 11, 32),    # bicycle - dark red
        }
        
        # Create color-coded segmentation
        h, w = segmentation_map.shape
        vis_image = image.copy()
        overlay = np.zeros_like(vis_image)
        
        # Color each pixel based on its class
        for class_id, color in cityscapes_colors.items():
            mask = (segmentation_map == class_id)
            overlay[mask] = color
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 0.5, overlay, 0.5, 0)
        
        # Add legend if class_info provided
        if class_info:
            y_offset = 20
            for class_id in sorted(class_info.get('classes_detected', [])):
                if class_id in cityscapes_colors:
                    color = cityscapes_colors[class_id]
                    class_name = self._get_class_name(class_id)
                    count = class_info['class_counts'].get(class_id, 0)
                    percentage = (count / class_info['total_pixels']) * 100
                    
                    # Draw colored box
                    cv2.rectangle(vis_image, (10, y_offset - 15), (30, y_offset - 5), color, -1)
                    # Draw text
                    text = f"{class_name}: {percentage:.1f}%"
                    cv2.putText(vis_image, text, (35, y_offset - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
        
        return vis_image
    
    def _get_class_name(self, class_id):
        """Get human-readable class name"""
        class_names = {
            0: "Road", 1: "Sidewalk", 2: "Building", 3: "Wall", 4: "Fence",
            5: "Pole", 6: "Traffic Light", 7: "Traffic Sign", 8: "Vegetation",
            9: "Terrain", 10: "Sky", 11: "Person", 12: "Rider", 13: "Car",
            14: "Truck", 15: "Bus", 16: "Train", 17: "Motorcycle", 18: "Bicycle"
        }
        return class_names.get(class_id, f"Class_{class_id}")
    
    def _mask_to_trapezoid(self, mask):
        """
        Convert road mask to polygon (simplified for backward compatibility)
        Returns a simplified polygon approximation
        
        Args:
            mask: Binary road mask
        
        Returns:
            polygon_points: List of points defining the road polygon
        """
        h, w = mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (main road region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce points (Douglas-Peucker algorithm)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of tuples
        polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified_contour]
        
        # Ensure minimum 4 points for backward compatibility
        if len(polygon_points) < 4:
            # Fallback to bounding box approach
            x, y, width, height = cv2.boundingRect(largest_contour)
            top_y = y
            bottom_y = y + height
            polygon_points = [
                (float(x), float(top_y)),
                (float(x + width), float(top_y)),
                (float(x + width), float(bottom_y)),
                (float(x), float(bottom_y))
            ]
        
        return polygon_points
    
    def mask_to_polygon(self, mask):
        """
        Convert road mask to detailed polygon
        
        Args:
            mask: Binary road mask
        
        Returns:
            polygon_points: List of points defining the road polygon
            road_mask: The original mask (for point-in-polygon checks)
        """
        h, w = mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # Get largest contour (main road region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce points
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of tuples
        polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified_contour]
        
        return polygon_points, mask
    
    def _detect_road_simple(self, image):
        """Fallback: Simple color-based road detection"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w = image.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Road typically has gray/brown tones
        # Adjust these ranges for Dhaka streets (concrete/asphalt)
        lower_road = np.array([0, 0, 40])   # Dark gray/brown
        upper_road = np.array([180, 50, 200])  # Light gray
        
        mask = cv2.inRange(hsv, lower_road, upper_road)
        
        # Focus on bottom portion (where road usually is in dashcam)
        mask[:h//3, :] = 0  # Ignore top third
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert to trapezoid
        road_polygon = self._mask_to_trapezoid(mask)
        
        return mask, road_polygon, None
    
    def estimate_pavement_width(self, sidewalk_mask, image_width):
        """
        Estimate pavement width from sidewalk mask
        
        Args:
            sidewalk_mask: Sidewalk segmentation mask
            image_width: Image width
        
        Returns:
            pavement_width: Estimated pavement width in pixels
        """
        if sidewalk_mask is None or sidewalk_mask.sum() == 0:
            # Default if no sidewalk detected
            return 50
        
        h, w = sidewalk_mask.shape
        
        # Find left and right sidewalk regions
        left_pavement = sidewalk_mask[:, :w//2].sum()
        right_pavement = sidewalk_mask[:, w//2:].sum()
        
        # Estimate width based on sidewalk area
        total_sidewalk_area = (sidewalk_mask > 0).sum()
        avg_sidewalk_height = total_sidewalk_area / w if w > 0 else 0
        
        # Estimate width (average of left and right)
        if left_pavement > 0 or right_pavement > 0:
            # Find actual width by checking columns
            left_width = 0
            right_width = 0
            
            # Check left side
            for x in range(w//2):
                if sidewalk_mask[:, x].sum() > 0:
                    left_width = max(left_width, x)
            
            # Check right side
            for x in range(w-1, w//2, -1):
                if sidewalk_mask[:, x].sum() > 0:
                    right_width = max(right_width, w - x)
            
            pavement_width = max(left_width, right_width, 30)  # Minimum 30px
        else:
            pavement_width = 50  # Default
        
        return min(pavement_width, w // 4)  # Cap at 25% of image width


def test_road_detector():
    """Test the road detector on a sample image"""
    detector = RoadDetector()
    
    # Test image
    image_path = Path("rsud20k_person2000_resized/images/train/train14961.jpg")
    if image_path.exists():
        image = cv2.imread(str(image_path))
        road_mask, road_polygon, sidewalk_mask = detector.detect_road(image)
        
        print(f"Road polygon: {road_polygon}")
        print(f"Road mask shape: {road_mask.shape}")
        
        # Visualize
        vis = image.copy()
        if road_polygon:
            pts = np.array(road_polygon, np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
        
        cv2.imshow("Road Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_road_detector()

