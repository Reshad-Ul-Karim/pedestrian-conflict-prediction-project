#!/usr/bin/env python3
"""
Phase 4: Conflict Detection and Weak Label Generation
Computes TTC, PET, and generates weak labels
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils_config import load_config
from utils_logger import setup_logger
from utils_data_structures import ConflictMetrics, ConflictLabel


class ConflictDetector:
    """Detect conflicts and generate weak labels"""
    
    def __init__(self, config):
        self.config = config
        self.grid_rows = config.grid.rows
        self.grid_cols = config.grid.cols
        self.ego_corridor = set(tuple(cell) for cell in config.grid.ego_corridor)
        self.logger = setup_logger('conflict_detection')
    
    def process_trajectories(self, trajectories_path, output_path):
        """Process trajectories and generate conflict labels"""
        self.logger.info(f"Processing: {trajectories_path}")
        
        conflict_labels = []
        
        with open(trajectories_path, 'r') as f:
            for line in tqdm(f, desc="Generating labels"):
                traj = json.loads(line)
                labels = self._process_single_trajectory(traj)
                conflict_labels.extend(labels)
        
        self.logger.info(f"Generated {len(conflict_labels)} conflict labels")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for label in conflict_labels:
                f.write(json.dumps(label.to_dict()) + '\n')
        
        self.logger.info(f"Saved to: {output_path}")
    
    def _process_single_trajectory(self, traj):
        """Generate conflict labels for a single trajectory"""
        labels = []
        trajectory_data = traj['trajectory']
        
        for i, point in enumerate(trajectory_data):
            if i < 10:  # Need history
                continue
            
            u, v = point['u'], point['v']
            grid_cell = self._get_grid_cell(u, v, 1280, 720)
            
            # Get velocity
            if point.get('kinematics'):
                v_u = point['kinematics']['v_u']
                v_v = point['kinematics']['v_v']
            else:
                continue
            
            # Predict future positions (simple linear)
            future_positions = {}
            for horizon in self.config.metrics.horizons:
                u_future = u + v_u * horizon
                v_future = v + v_v * horizon
                future_cell = self._get_grid_cell(u_future, v_future, 1280, 720)
                future_positions[horizon] = (u_future, v_future, future_cell)
            
            # Compute conflict labels
            conflict_dict = {}
            for horizon in self.config.metrics.horizons:
                _, _, fut_cell = future_positions[horizon]
                in_conflict = tuple(fut_cell) in self.ego_corridor
                
                # Compute confidence
                distance = self._distance_to_zone(u, v)
                velocity_towards = self._velocity_towards_zone(v_u, v_v, u, v)
                confidence = self._compute_confidence(distance, velocity_towards, i)
                
                conflict_dict[f'y_{int(horizon)}s'] = 1 if in_conflict else 0
                conflict_dict[f'w_{int(horizon)}s'] = confidence
                conflict_dict[f'ttc_{int(horizon)}s'] = horizon if in_conflict else None
            
            # Create label
            label = ConflictLabel(
                video_id=traj['video_id'],
                track_id=traj['track_id'],
                t0=point['t'],
                grid_cell=grid_cell,
                current_state={'u': u, 'v': v, 'v_u': v_u, 'v_v': v_v},
                conflict_labels=conflict_dict,
                metrics=ConflictMetrics()
            )
            labels.append(label)
        
        return labels
    
    def _get_grid_cell(self, u, v, width, height):
        """Convert pixel coordinates to grid cell"""
        row = int(v / (height / self.grid_rows))
        col = int(u / (width / self.grid_cols))
        row = max(0, min(self.grid_rows - 1, row))
        col = max(0, min(self.grid_cols - 1, col))
        return [row, col]
    
    def _distance_to_zone(self, u, v):
        """Compute distance to ego corridor"""
        # Simplified: distance to bottom-middle
        ego_center_u = 1280 / 2
        ego_center_v = 720 * 0.85
        return np.sqrt((u - ego_center_u)**2 + (v - ego_center_v)**2)
    
    def _velocity_towards_zone(self, v_u, v_v, u, v):
        """Check if velocity is towards zone"""
        ego_center_u = 1280 / 2
        ego_center_v = 720 * 0.85
        to_zone_u = ego_center_u - u
        to_zone_v = ego_center_v - v
        dot = v_u * to_zone_u + v_v * to_zone_v
        return max(0, dot / (np.sqrt(v_u**2 + v_v**2 + 1e-6) * np.sqrt(to_zone_u**2 + to_zone_v**2 + 1e-6)))
    
    def _compute_confidence(self, distance, velocity_towards, track_length):
        """Compute label confidence"""
        dist_conf = 1.0 - min(1.0, distance / 300.0)
        vel_conf = velocity_towards
        length_conf = min(1.0, track_length / 30.0)
        
        confidence = (
            self.config.confidence.distance_weight * dist_conf +
            self.config.confidence.velocity_weight * vel_conf +
            self.config.confidence.trajectory_length_weight * length_conf
        )
        return float(np.clip(confidence, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/conflict.yaml')
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    detector = ConflictDetector(config)
    detector.process_trajectories(args.trajectories, args.output)


if __name__ == "__main__":
    main()

