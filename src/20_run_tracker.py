#!/usr/bin/env python3
"""
Step 1.3 - Run Multi-Object Tracking with ByteTrack
Applies detector + tracker on video to generate tracklets
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils_config import load_config
from utils_logger import setup_logger
from utils_device import get_device
from utils_data_structures import BBox, Track


class ByteTracker:
    """Simple ByteTrack implementation"""
    
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id_count = 0
        
    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracks with new detections"""
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_dets = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # First association with high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._match(
            self.tracked_tracks, high_dets
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracked_tracks[track_idx].update(high_dets[det_idx], self.frame_id)
        
        # Second association with low confidence detections
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            matched_low, unmatched_tracks, _ = self._match(
                [self.tracked_tracks[i] for i in unmatched_tracks],
                low_dets
            )
            for track_idx, det_idx in matched_low:
                self.tracked_tracks[unmatched_tracks[track_idx]].update(
                    low_dets[det_idx], self.frame_id
                )
        
        # Handle unmatched tracks
        for idx in unmatched_tracks:
            track = self.tracked_tracks[idx]
            if track.end_frame() - track.frame_id < self.track_buffer:
                self.lost_tracks.append(track)
        
        # Create new tracks from unmatched high conf detections
        for det in [high_dets[i] for i in unmatched_dets]:
            new_track = STrack(det, self.frame_id, self.track_id_count)
            self.track_id_count += 1
            self.tracked_tracks.append(new_track)
        
        # Remove old lost tracks
        self.tracked_tracks = [t for t in self.tracked_tracks 
                              if t.end_frame() == self.frame_id]
        
        return self._get_current_tracks()
    
    def _match(self, tracks, detections):
        """Simple IoU-based matching"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det['bbox'])
        
        matched, unmatched_tracks, unmatched_dets = [], [], []
        
        # Greedy matching
        while iou_matrix.size > 0:
            max_iou_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[max_iou_idx] >= self.match_thresh:
                matched.append(max_iou_idx)
                iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
                iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)
            else:
                break
        
        unmatched_tracks = [i for i in range(len(tracks)) 
                           if i not in [m[0] for m in matched]]
        unmatched_dets = [i for i in range(len(detections)) 
                         if i not in [m[1] for m in matched]]
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _get_current_tracks(self):
        """Get current active tracks"""
        return [t.to_track(self.frame_id) for t in self.tracked_tracks]


class STrack:
    """Single track"""
    
    def __init__(self, detection, frame_id, track_id):
        self.track_id = track_id
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.frame_id = frame_id
        self._end_frame = frame_id
        
    def update(self, detection, frame_id):
        """Update track with new detection"""
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self._end_frame = frame_id
        
    def end_frame(self):
        return self._end_frame
    
    def to_track(self, frame_id):
        """Convert to Track object"""
        return Track(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            bbox=BBox(*self.bbox),
            confidence=self.confidence,
            frame_id=frame_id,
            timestamp=frame_id / 30.0  # Assuming 30 fps
        )


def run_tracking(
    video_path: str,
    config_path: str,
    output_dir: str,
    visualize: bool = False
):
    """
    Run detection and tracking on video
    
    Args:
        video_path: Path to input video
        config_path: Path to tracker configuration
        output_dir: Directory to save outputs
        visualize: Whether to save visualization
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        'tracking',
        log_file=output_dir / 'tracking.log'
    )
    
    logger.info("=" * 60)
    logger.info("Multi-Object Tracking")
    logger.info("=" * 60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load detector
    device = get_device()
    model_path = config.detection.model_path
    logger.info(f"Loading detector: {model_path}")
    logger.info(f"Device: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video info: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Initialize tracker
    tracker = ByteTracker(
        track_thresh=config.tracker.track_high_thresh,
        track_buffer=config.tracker.track_buffer,
        match_thresh=config.tracker.match_thresh
    )
    
    # Prepare output
    video_id = Path(video_path).stem
    tracks_file = output_dir / f"{video_id}.jsonl"
    
    if visualize and config.output.save_video:
        viz_dir = Path(config.output.video_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"{video_id}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(viz_path), fourcc, fps, (width, height))
    else:
        out_video = None
    
    # Process video
    logger.info("Processing video...")
    frame_id = 0
    
    with open(tracks_file, 'w') as f:
        pbar = tqdm(total=total_frames, desc="Tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(
                frame,
                conf=config.detection.conf_threshold,
                iou=config.detection.iou_threshold,
                max_det=config.detection.max_det,
                verbose=False
            )[0]
            
            # Convert to detections
            detections = []
            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': model.names[cls]
                    })
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # Save tracks
            frame_data = {
                'frame_id': frame_id,
                'timestamp': frame_id / fps,
                'tracks': [t.to_dict() for t in tracks]
            }
            f.write(json.dumps(frame_data) + '\n')
            
            # Visualize
            if out_video:
                vis_frame = frame.copy()
                for track in tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    
                    # Draw bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"ID:{track.track_id} {track.class_name} {track.confidence:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                out_video.write(vis_frame)
            
            frame_id += 1
            pbar.update(1)
        
        pbar.close()
    
    cap.release()
    if out_video:
        out_video.release()
    
    logger.info(f"Tracking complete! Processed {frame_id} frames")
    logger.info(f"Tracks saved to: {tracks_file}")
    if out_video:
        logger.info(f"Visualization saved to: {viz_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run multi-object tracking")
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--config', type=str, default='configs/tracker.yaml',
                       help='Tracker configuration')
    parser.add_argument('--output', type=str, default='outputs/tracks',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Save visualization')
    
    args = parser.parse_args()
    
    run_tracking(args.video, args.config, args.output, args.visualize)


if __name__ == "__main__":
    main()

