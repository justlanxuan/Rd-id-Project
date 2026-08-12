"""Typed detector, tracker and pose-estimator interchange records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    class_id: int | None = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        if self.bbox.shape != (4,):
            raise ValueError(f"Detection bbox must have shape (4,), got {self.bbox.shape}")


@dataclass
class Pose:
    keypoints: np.ndarray
    bbox: np.ndarray | None = None
    score: float | None = None
    track_id: int | None = None
    frame_id: int | None = None

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.keypoints.ndim != 2 or self.keypoints.shape[-1] != 3:
            raise ValueError(f"Pose keypoints must have shape (K, 3), got {self.keypoints.shape}")
        if self.bbox is not None:
            self.bbox = np.asarray(self.bbox, dtype=np.float32)
            if self.bbox.shape != (4,):
                raise ValueError(f"Pose bbox must have shape (4,), got {self.bbox.shape}")


@dataclass
class Track:
    track_id: int
    detection: Detection
    pose: Pose | None = None


@dataclass
class FrameResult:
    frame_id: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    poses: list[Pose] = field(default_factory=list)
