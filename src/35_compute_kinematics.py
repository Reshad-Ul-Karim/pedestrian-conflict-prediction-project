#!/usr/bin/env python3
"""
Phase 2 - Step 2.2: Compute Kinematics (Velocity, Acceleration, Heading)
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

from utils_config import load_config
from utils_logger import setup_logger
from utils_data_structures import Trajectory, Kinematics, QualityMetrics


class KinematicsComputer:
    """Compute kinematic properties of trajectories"""
    
    def __init__(self, config):
        self.config = config
        self.fps = config.kinematics.fps
        self.dt = 1.0 / self.fps
        self.logger = setup_logger('kinematics')
    
    def compute_kinematics(self, trajectories_path: str, output_path: str):
        """
        Compute kinematics for all trajectories
        
        Args:
            trajectories_path: Input trajectories JSONL file
            output_path: Output path for enhanced trajectories
        """
        self.logger.info(f"Loading trajectories from: {trajectories_path}")
        
        # Load trajectories
        trajectories = []
        with open(trajectories_path, 'r') as f:
            for line in f:
                traj_dict = json.loads(line)
                trajectories.append(self._dict_to_trajectory(traj_dict))
        
        self.logger.info(f"Loaded {len(trajectories)} trajectories")
        
        # Compute kinematics
        enhanced_trajectories = []
        self.logger.info("Computing kinematics...")
        
        for traj in tqdm(trajectories, desc="Processing"):
            if len(traj.points) < self.config.kinematics.min_track_length:
                continue
            
            # Extract coordinates
            t = np.array([p.t for p in traj.points])
            u = np.array([p.u for p in traj.points])
            v = np.array([p.v for p in traj.points])
            
            # Compute velocity
            v_u = self._differentiate(u, t, method=self.config.kinematics.velocity_method)
            v_v = self._differentiate(v, t, method=self.config.kinematics.velocity_method)
            speed = np.sqrt(v_u**2 + v_v**2)
            
            # Compute acceleration
            a_u = self._differentiate(v_u, t, method=self.config.kinematics.acceleration_method)
            a_v = self._differentiate(v_v, t, method=self.config.kinematics.acceleration_method)
            acceleration = np.sqrt(a_u**2 + a_v**2)
            
            # Compute heading
            heading = np.arctan2(v_v, v_u)
            heading_deg = np.degrees(heading)
            
            # Remove outliers if configured
            if self.config.kinematics.remove_outliers:
                v_u, v_v, speed = self._remove_outliers(v_u, v_v, speed)
                a_u, a_v, acceleration = self._remove_outliers(a_u, a_v, acceleration)
            
            # Create kinematics objects
            kinematics_list = []
            for i in range(len(traj.points)):
                kin = Kinematics(
                    v_u=float(v_u[i]),
                    v_v=float(v_v[i]),
                    speed=float(speed[i]),
                    a_u=float(a_u[i]) if i < len(a_u) else 0.0,
                    a_v=float(a_v[i]) if i < len(a_v) else 0.0,
                    acceleration=float(acceleration[i]) if i < len(acceleration) else 0.0,
                    heading=float(heading[i]),
                    heading_deg=float(heading_deg[i])
                )
                kinematics_list.append(kin)
            
            # Compute quality metrics
            avg_conf = np.mean([p.confidence for p in traj.points])
            smoothness = self._compute_smoothness(v_u, v_v)
            
            quality = QualityMetrics(
                detection_conf=float(avg_conf),
                smoothness=float(smoothness),
                occluded=any(p.occluded for p in traj.points),
                track_length=len(traj.points)
            )
            
            # Create enhanced trajectory
            traj.kinematics = kinematics_list
            traj.quality = quality
            
            # Filter by quality
            if (quality.smoothness >= self.config.quality.min_smoothness and
                quality.detection_conf >= self.config.quality.min_confidence):
                enhanced_trajectories.append(traj)
        
        self.logger.info(f"Enhanced {len(enhanced_trajectories)} trajectories")
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for traj in enhanced_trajectories:
                f.write(json.dumps(traj.to_dict()) + '\n')
        
        self.logger.info(f"Saved to: {output_path}")
    
    def _differentiate(self, signal: np.ndarray, t: np.ndarray, method: str = 'savgol'):
        """Compute derivative of signal"""
        if method == 'savgol':
            window = min(5, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
            if window < 3:
                return np.gradient(signal, t)
            order = min(2, window - 1)
            try:
                smoothed = savgol_filter(signal, window, order)
                return np.gradient(smoothed, t)
            except:
                return np.gradient(signal, t)
        else:
            return np.gradient(signal, t)
    
    def _remove_outliers(self, *signals, threshold=3.0):
        """Remove outliers using z-score"""
        cleaned = []
        for signal in signals:
            signal = np.array(signal)
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 0:
                z_scores = np.abs((signal - mean) / std)
                signal_clean = np.where(z_scores < threshold, signal, mean)
            else:
                signal_clean = signal
            cleaned.append(signal_clean)
        return cleaned
    
    def _compute_smoothness(self, v_u: np.ndarray, v_v: np.ndarray) -> float:
        """Compute trajectory smoothness (lower = smoother)"""
        if len(v_u) < 2:
            return 0.0
        
        # Compute variance of velocity changes
        v_u_diff = np.diff(v_u)
        v_v_diff = np.diff(v_v)
        variance = np.var(v_u_diff) + np.var(v_v_diff)
        
        # Normalize by mean speed
        mean_speed = np.mean(np.sqrt(v_u**2 + v_v**2))
        if mean_speed > 0:
            smoothness = variance / (mean_speed ** 2)
        else:
            smoothness = 0.0
        
        return min(1.0, smoothness)  # Cap at 1.0
    
    def _dict_to_trajectory(self, traj_dict: dict) -> Trajectory:
        """Convert dictionary to Trajectory object"""
        from utils_data_structures import TrajectoryPoint, PoseInfo, BBox
        
        points = []
        poses = []
        
        for point_data in traj_dict['trajectory']:
            point = TrajectoryPoint(
                t=point_data['t'],
                u=point_data['u'],
                v=point_data['v'],
                bbox=BBox(*point_data['bbox']) if point_data.get('bbox') else None,
                confidence=point_data.get('confidence', 1.0),
                occluded=point_data.get('occluded', False)
            )
            points.append(point)
            
            if point_data.get('pose'):
                pose_data = point_data['pose']
                pose = PoseInfo(
                    foot_midpoint=tuple(pose_data.get('foot_midpoint', [0, 0])),
                    hip_midpoint=tuple(pose_data.get('hip_midpoint', [0, 0])),
                    torso_angle=pose_data.get('torso_angle', 0.0),
                    pose_confidence=pose_data.get('pose_confidence', 0.0)
                )
                poses.append(pose)
        
        return Trajectory(
            track_id=traj_dict['track_id'],
            video_id=traj_dict['video_id'],
            class_id=traj_dict['class_id'],
            class_name=traj_dict['class_name'],
            points=points,
            pose_info=poses
        )


def main():
    parser = argparse.ArgumentParser(description="Compute trajectory kinematics")
    parser.add_argument('--input', type=str, required=True,
                       help='Input trajectories JSONL file')
    parser.add_argument('--config', type=str, default='configs/trajectory.yaml')
    parser.add_argument('--output', type=str, required=True,
                       help='Output enhanced trajectories file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    computer = KinematicsComputer(config)
    computer.compute_kinematics(args.input, args.output)


if __name__ == "__main__":
    main()

