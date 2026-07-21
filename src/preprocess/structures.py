"""Unified data structures for pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """2D detection result."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: Optional[int] = None

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)


@dataclass
class Pose:
    """2D pose result for a single person."""

    keypoints: np.ndarray  # [K, 3] x, y, confidence
    bbox: Optional[np.ndarray] = None  # associated bbox [x1, y1, x2, y2]
    score: Optional[float] = None
    track_id: Optional[int] = None
    frame_id: Optional[int] = None

    def __post_init__(self):
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.bbox is not None:
            self.bbox = np.asarray(self.bbox, dtype=np.float32)


@dataclass
class Track:
    """Tracking result for a single object."""

    track_id: int
    detection: Detection
    pose: Optional[Pose] = None


@dataclass
class FrameResult:
    """All results for a single video frame."""

    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    poses: List[Pose] = field(default_factory=list)
