"""Abstract pose-estimator interface for Re-id-Project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.data import Pose, Track


class BasePoseEstimator(ABC):
    """Common interface for pose-estimation backends (decoupled SPPE)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any sequence-local state."""
        raise NotImplementedError

    @abstractmethod
    def estimate(self, frame: np.ndarray, tracks: List[Track]) -> List[Pose]:
        """Estimate poses for tracked bounding boxes in a single frame.

        Args:
            frame: Input image in HWC format.
            tracks: List of Track objects with detections.

        Returns:
            List of Pose objects aligned with input tracks.
        """
        raise NotImplementedError
