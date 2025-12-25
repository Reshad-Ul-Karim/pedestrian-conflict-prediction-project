# SegFormer Road Detection Integration

This document explains how to use SegFormer for automatic road segmentation in the conflict risk assessment pipeline.

## Overview

SegFormer is a transformer-based semantic segmentation model that can automatically detect road regions from dashcam images. This is particularly useful for:
- Irregular street shapes that don't fit standard trapezoids
- Automatic calibration without manual annotation
- Better adaptation to different camera perspectives

## Installation

The required dependencies are already in `requirements.txt`:
- `transformers>=4.30.0` (for SegFormer)
- `torch>=2.0.0` (for model inference)
- `pillow>=9.0.0` (for image processing)

Install with:
```bash
pip install transformers torch pillow
```

## Usage

### 1. Grid Calibrator (`calibrate_grid.py`)

The calibrator now supports automatic road detection:

```python
from pathlib import Path
from calibrate_grid import GridCalibrator

# Initialize with road detector enabled (default)
calibrator = GridCalibrator(
    image_dir="rsud20k_person2000_resized/images/train",
    grid_rows=8,
    grid_cols=6,
    use_road_detector=True  # Enable SegFormer
)

calibrator.run()
```

**Keyboard Controls:**
- **'F'**: Auto-detect road region using SegFormer
- **'M'**: Toggle road mask overlay (green = road, blue = sidewalk)
- Other controls remain the same (A/D for navigation, E for edit/view, etc.)

**Workflow:**
1. Navigate to an image
2. Press **'F'** to auto-detect the road region
3. Review the detected trapezoid (yellow outline)
4. Fine-tune manually if needed (drag corners)
5. Press **'M'** to see the raw segmentation mask
6. Save calibration with **'S'**

### 2. Conflict Risk Visualizer (`visualize_conflict_risk.py`)

The visualizer can use SegFormer for automatic road detection:

```python
from pathlib import Path
from visualize_conflict_risk import visualize_conflict_risk, load_yolo_bboxes

image_path = Path("rsud20k_person2000_resized/images/train/train14961.jpg")
label_path = image_path.parent.parent.parent / "labels" / "train" / f"{image_path.stem}.txt"

# Load bounding boxes
person_bboxes = load_yolo_bboxes(label_path, image_width=640, image_height=640, class_id=0)

# Use SegFormer for automatic road detection
visualize_conflict_risk(
    image_path, 
    person_bboxes,
    calibration_file=None,  # Or path to calibration JSON
    use_road_detector=True  # Enable SegFormer
)
```

**Options:**
- `calibration_file`: Path to saved calibration JSON (from calibrator)
- `use_road_detector`: If True, uses SegFormer to auto-detect road region
- If both are provided, calibration file takes precedence

## Road Detector Module (`road_detector.py`)

The `RoadDetector` class provides the core functionality:

```python
from road_detector import RoadDetector
import cv2

# Initialize detector
detector = RoadDetector(model_name="nvidia/segformer-b0-finetuned-cityscapes-640-1280")

# Detect road in image
image = cv2.imread("image.jpg")
road_mask, road_polygon, sidewalk_mask = detector.detect_road(image)

# road_mask: Binary mask of road region (numpy array, 0-255)
# road_polygon: List of 4 points defining trapezoid [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
# sidewalk_mask: Binary mask of sidewalk/pavement (optional)
```

**Available Models:**
- `nvidia/segformer-b0-finetuned-cityscapes-640-1280` (lightweight, fastest)
- `nvidia/segformer-b1-finetuned-cityscapes-640-1280` (balanced)
- `nvidia/segformer-b2-finetuned-cityscapes-640-1280` (better accuracy, slower)

## How It Works

1. **Segmentation**: SegFormer segments the image into semantic classes (road, sidewalk, vehicles, etc.)
2. **Road Extraction**: The road class (ID 0 in Cityscapes) is extracted as a binary mask
3. **Trapezoid Approximation**: The road mask is converted to a trapezoid by:
   - Finding the largest road contour
   - Identifying top and bottom edges
   - Computing left/right boundaries at top and bottom
4. **Pavement Estimation**: Sidewalk regions are used to estimate pavement width

## Tips

1. **First Run**: The model will download (~50-100MB) on first use
2. **GPU Acceleration**: Automatically uses CUDA/MPS if available
3. **Fallback**: If SegFormer fails, the system falls back to color-based detection
4. **Manual Refinement**: Always review and refine auto-detected regions
5. **Calibration Persistence**: Save calibrations to JSON for consistent use across images

## Troubleshooting

**Error: "transformers not found"**
```bash
pip install transformers
```

**Error: "CUDA out of memory"**
- Use a smaller model (b0 instead of b1/b2)
- Process images in smaller batches

**Poor road detection:**
- Try different models (b0, b1, b2)
- Manually refine the detected trapezoid
- Check if image perspective matches Cityscapes training data

**Slow inference:**
- Use GPU if available (CUDA/MPS)
- Use smaller model (b0)
- Reduce image resolution before detection

