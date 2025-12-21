"""
Common data structures and types for the project
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class BBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bbox center"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_dict(self) -> Dict:
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        }


@dataclass
class Detection:
    """Single object detection"""
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox.to_list(),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence
        }


@dataclass
class Track:
    """Object track over time"""
    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float
    frame_id: int
    timestamp: float
    kalman_state: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        data = {
            'track_id': self.track_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'bbox': self.bbox.to_list(),
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp
        }
        if self.kalman_state:
            data['kalman_state'] = self.kalman_state
        return data


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    t: float  # timestamp
    u: float  # x coordinate
    v: float  # y coordinate
    bbox: Optional[BBox] = None
    confidence: float = 1.0
    occluded: bool = False


@dataclass
class Kinematics:
    """Kinematic properties at a trajectory point"""
    v_u: float = 0.0  # velocity in x
    v_v: float = 0.0  # velocity in y
    speed: float = 0.0
    a_u: float = 0.0  # acceleration in x
    a_v: float = 0.0  # acceleration in y
    acceleration: float = 0.0
    heading: float = 0.0  # radians
    heading_deg: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'v_u': self.v_u,
            'v_v': self.v_v,
            'speed': self.speed,
            'a_u': self.a_u,
            'a_v': self.a_v,
            'acceleration': self.acceleration,
            'heading': self.heading,
            'heading_deg': self.heading_deg
        }


@dataclass
class PoseInfo:
    """Pose information from MediaPipe"""
    foot_midpoint: Tuple[float, float] = (0.0, 0.0)
    hip_midpoint: Tuple[float, float] = (0.0, 0.0)
    torso_angle: float = 0.0
    pose_confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'foot_midpoint': list(self.foot_midpoint),
            'hip_midpoint': list(self.hip_midpoint),
            'torso_angle': self.torso_angle,
            'pose_confidence': self.pose_confidence
        }


@dataclass
class QualityMetrics:
    """Trajectory quality metrics"""
    detection_conf: float = 0.0
    smoothness: float = 0.0
    occluded: bool = False
    track_length: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'detection_conf': self.detection_conf,
            'smoothness': self.smoothness,
            'occluded': self.occluded,
            'track_length': self.track_length
        }


@dataclass
class Trajectory:
    """Complete trajectory for a track"""
    track_id: int
    video_id: str
    class_id: int
    class_name: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    kinematics: List[Kinematics] = field(default_factory=list)
    pose_info: List[PoseInfo] = field(default_factory=list)
    quality: Optional[QualityMetrics] = None
    
    def __len__(self) -> int:
        return len(self.points)
    
    def to_dict(self) -> Dict:
        data = {
            'track_id': self.track_id,
            'video_id': self.video_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'trajectory': [
                {
                    't': p.t,
                    'u': p.u,
                    'v': p.v,
                    'bbox': p.bbox.to_list() if p.bbox else None,
                    'confidence': p.confidence,
                    'occluded': p.occluded,
                    'kinematics': k.to_dict() if i < len(self.kinematics) else None,
                    'pose': self.pose_info[i].to_dict() if i < len(self.pose_info) else None
                }
                for i, (p, k) in enumerate(zip(self.points, self.kinematics + [None] * len(self.points)))
            ]
        }
        if self.quality:
            data['quality'] = self.quality.to_dict()
        return data


@dataclass
class ConflictMetrics:
    """Conflict metrics for a trajectory point"""
    ttc: Optional[float] = None  # Time to conflict
    pet: Optional[float] = None  # Post-encroachment time
    min_distance: Optional[float] = None
    collision_prob: float = 0.0
    in_conflict_zone: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'ttc': self.ttc,
            'pet': self.pet,
            'min_distance': self.min_distance,
            'collision_prob': self.collision_prob,
            'in_conflict_zone': self.in_conflict_zone
        }


@dataclass
class ConflictLabel:
    """Conflict label for training"""
    video_id: str
    track_id: int
    t0: float
    grid_cell: Tuple[int, int]
    current_state: Dict
    conflict_labels: Dict[str, float]  # y_1s, w_1s, etc.
    metrics: ConflictMetrics
    
    def to_dict(self) -> Dict:
        return {
            'video_id': self.video_id,
            'track_id': self.track_id,
            't0': self.t0,
            'grid_cell': list(self.grid_cell),
            'current_state': self.current_state,
            'conflict_labels': self.conflict_labels,
            'metrics': self.metrics.to_dict()
        }

