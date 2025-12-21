#!/usr/bin/env python3
"""
Phase 2 - Step 2.1: Extract Trajectories from Tracks
Uses MediaPipe Pose for pedestrians, bbox center for vehicles
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy.signal import medfilt

from utils_config import load_config
from utils_logger import setup_logger
from utils_data_structures import Trajectory, TrajectoryPoint, PoseInfo, BBox


class TrajectoryExtractor:
    """Extract trajectories from tracking data"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=config.mediapipe.model_complexity,
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            smooth_landmarks=config.mediapipe.smooth_landmarks
        )
        
        self.logger = setup_logger('trajectory_extraction')
    
    def extract_from_video(
        self,
        video_path: str,
        tracks_path: str,
        output_path: str
    ):
        """
        Extract trajectories from video and tracking data
        
        Args:
            video_path: Path to video file
            tracks_path: Path to tracking JSONL file
            output_path: Path to save trajectories
        """
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Tracks: {tracks_path}")
        
        # Load tracks
        tracks_by_id = self._load_tracks(tracks_path)
        self.logger.info(f"Loaded {len(tracks_by_id)} unique tracks")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames
        trajectories = {tid: [] for tid in tracks_by_id.keys()}
        frame_id = 0
        
        self.logger.info("Extracting trajectories...")
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get tracks for this frame
            if frame_id in tracks_by_id.get('frame_map', {}):
                for track in tracks_by_id['frame_map'][frame_id]:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    class_name = track['class_name']
                    
                    # Extract trajectory point
                    if class_name == 'person':
                        point, pose_info = self._extract_person_point(
                            frame, bbox
                        )
                    else:
                        point, pose_info = self._extract_object_point(bbox)
                    
                    if point is not None:
                        traj_point = TrajectoryPoint(
                            t=frame_id / fps,
                            u=point[0],
                            v=point[1],
                            bbox=BBox(*bbox),
                            confidence=track['confidence']
                        )
                        
                        trajectories[track_id].append({
                            'point': traj_point,
                            'pose': pose_info,
                            'class_name': class_name,
                            'class_id': track['class_id']
                        })
            
            frame_id += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Smooth trajectories and convert to Trajectory objects
        self.logger.info("Smoothing trajectories...")
        final_trajectories = []
        video_id = Path(video_path).stem
        
        for track_id, traj_data in trajectories.items():
            if len(traj_data) < self.config.kinematics.min_track_length:
                continue
            
            # Smooth coordinates
            u_coords = [d['point'].u for d in traj_data]
            v_coords = [d['point'].v for d in traj_data]
            
            u_smooth = self._smooth_signal(u_coords)
            v_smooth = self._smooth_signal(v_coords)
            
            # Create trajectory
            points = []
            poses = []
            for i, data in enumerate(traj_data):
                point = TrajectoryPoint(
                    t=data['point'].t,
                    u=u_smooth[i],
                    v=v_smooth[i],
                    bbox=data['point'].bbox,
                    confidence=data['point'].confidence
                )
                points.append(point)
                poses.append(data['pose'])
            
            trajectory = Trajectory(
                track_id=track_id,
                video_id=video_id,
                class_id=traj_data[0]['class_id'],
                class_name=traj_data[0]['class_name'],
                points=points,
                pose_info=poses
            )
            
            final_trajectories.append(trajectory)
        
        # Save trajectories
        self.logger.info(f"Saving {len(final_trajectories)} trajectories...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for traj in final_trajectories:
                f.write(json.dumps(traj.to_dict()) + '\n')
        
        self.logger.info(f"Trajectories saved to: {output_path}")
        return final_trajectories
    
    def _load_tracks(self, tracks_path: str) -> Dict:
        """Load tracks from JSONL file"""
        tracks_by_id = {}
        frame_map = {}
        
        with open(tracks_path, 'r') as f:
            for line in f:
                frame_data = json.loads(line)
                frame_id = frame_data['frame_id']
                tracks = frame_data['tracks']
                
                frame_map[frame_id] = tracks
                
                for track in tracks:
                    track_id = track['track_id']
                    if track_id not in tracks_by_id:
                        tracks_by_id[track_id] = []
                    tracks_by_id[track_id].append({
                        'frame_id': frame_id,
                        **track
                    })
        
        tracks_by_id['frame_map'] = frame_map
        return tracks_by_id
    
    def _extract_person_point(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> Tuple[Optional[Tuple[float, float]], PoseInfo]:
        """Extract anchor point for pedestrian using MediaPipe"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        h, w = frame.shape[:2]
        padding = 0.2
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None, PoseInfo()
        
        # Run MediaPipe Pose
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(person_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get foot midpoint (ankles)
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:
                foot_u = (left_ankle.x + right_ankle.x) / 2 * (x2 - x1) + x1
                foot_v = (left_ankle.y + right_ankle.y) / 2 * (y2 - y1) + y1
                anchor = (foot_u, foot_v)
            else:
                # Fallback to hip midpoint
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                hip_u = (left_hip.x + right_hip.x) / 2 * (x2 - x1) + x1
                hip_v = (left_hip.y + right_hip.y) / 2 * (y2 - y1) + y1
                anchor = (hip_u, hip_v)
            
            # Compute pose info
            pose_info = PoseInfo(
                foot_midpoint=(foot_u, foot_v) if 'foot_u' in locals() else (0, 0),
                hip_midpoint=(hip_u, hip_v) if 'hip_u' in locals() else (0, 0),
                pose_confidence=np.mean([lm.visibility for lm in landmarks])
            )
            
            return anchor, pose_info
        else:
            # No pose detected, use bbox center
            center_u = (x1 + x2) / 2
            center_v = (y1 + y2) / 2
            return (center_u, center_v), PoseInfo()
    
    def _extract_object_point(
        self,
        bbox: List[float]
    ) -> Tuple[Tuple[float, float], PoseInfo]:
        """Extract center point for non-person objects"""
        x1, y1, x2, y2 = bbox
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        return (center_u, center_v), PoseInfo()
    
    def _smooth_signal(self, signal: List[float]) -> np.ndarray:
        """Smooth signal using configured method"""
        signal = np.array(signal)
        
        if self.config.smoothing.method == 'moving_average':
            window = self.config.smoothing.window_size
            return np.convolve(signal, np.ones(window)/window, mode='same')
        elif self.config.smoothing.method == 'median':
            return medfilt(signal, kernel_size=self.config.smoothing.window_size)
        else:  # savgol
            from scipy.signal import savgol_filter
            window = self.config.smoothing.window_size
            if window % 2 == 0:
                window += 1
            order = min(self.config.smoothing.poly_order, window - 1)
            return savgol_filter(signal, window, order)


def main():
    parser = argparse.ArgumentParser(description="Extract trajectories from tracks")
    parser.add_argument('--video', type=str, required=True, help='Input video')
    parser.add_argument('--tracks', type=str, required=True, help='Tracks JSONL file')
    parser.add_argument('--config', type=str, default='configs/trajectory.yaml')
    parser.add_argument('--output', type=str, required=True, help='Output trajectories file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    extractor = TrajectoryExtractor(config)
    extractor.extract_from_video(args.video, args.tracks, args.output)


if __name__ == "__main__":
    main()

