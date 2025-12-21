#!/usr/bin/env python3
"""
Phase 5: Build Training Clips
Extracts visual and kinematic clips for conflict prediction training
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from utils_config import load_config
from utils_logger import setup_logger


class ClipBuilder:
    """Build training clips from videos and conflict labels"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger('clip_builder')
    
    def build_clips(self, video_path, tracks_path, trajectories_path, labels_path, output_dir):
        """Build training clips"""
        self.logger.info("Building training clips...")
        
        # Load data
        labels = self._load_labels(labels_path)
        trajectories = self._load_trajectories(trajectories_path)
        
        self.logger.info(f"Loaded {len(labels)} labels")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for label_idx, label in enumerate(tqdm(labels, desc="Extracting clips")):
            t0 = label['t0']
            track_id = label['track_id']
            
            # Find trajectory
            traj = next((t for t in trajectories if t['track_id'] == track_id), None)
            if not traj:
                continue
            
            # Extract clip window: [t0 - 1.5s, t0 + 0.5s]
            start_frame = int((t0 - 1.5) * fps)
            end_frame = int((t0 + 0.5) * fps)
            
            if start_frame < 0:
                continue
            
            # Extract visual frames
            visual_clip = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Crop around bbox (simplified)
                visual_clip.append(cv2.resize(frame, (128, 128)))
            
            if len(visual_clip) < 10:
                continue
            
            # Extract kinematic sequence
            traj_points = traj['trajectory']
            kin_seq = []
            for point in traj_points:
                if start_frame / fps <= point['t'] <= end_frame / fps:
                    if point.get('kinematics'):
                        feat = [
                            point['u'] / 1280.0,
                            point['v'] / 720.0,
                            point['kinematics']['v_u'],
                            point['kinematics']['v_v'],
                            point['kinematics']['speed'],
                            point['kinematics']['heading']
                        ]
                        kin_seq.append(feat)
            
            if len(kin_seq) < 10:
                continue
            
            # Save clip
            clip_id = f"{label['video_id']}_track{track_id}_t{t0:.2f}"
            
            visual_array = np.array(visual_clip)
            kin_array = np.array(kin_seq)
            
            np.save(output_dir / f"{clip_id}_visual.npy", visual_array)
            np.save(output_dir / f"{clip_id}_kin.npy", kin_array)
            
            # Metadata
            metadata.append({
                'clip_id': clip_id,
                'video_id': label['video_id'],
                'track_id': track_id,
                't0': t0,
                'labels': label['conflict_labels'],
                'visual_path': f"{clip_id}_visual.npy",
                'kinematic_path': f"{clip_id}_kin.npy"
            })
        
        cap.release()
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Built {len(metadata)} clips")
    
    def _load_labels(self, path):
        """Load conflict labels"""
        labels = []
        with open(path, 'r') as f:
            for line in f:
                labels.append(json.loads(line))
        return labels
    
    def _load_trajectories(self, path):
        """Load trajectories"""
        trajs = []
        with open(path, 'r') as f:
            for line in f:
                trajs.append(json.loads(line))
        return trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--tracks', required=True)
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', default='outputs/clips')
    parser.add_argument('--config', default='configs/model.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    builder = ClipBuilder(config)
    builder.build_clips(args.video, args.tracks, args.trajectories, args.labels, args.output)


if __name__ == "__main__":
    main()

