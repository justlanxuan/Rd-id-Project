"""Abstract tracker interface for Autism-project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.data import Detection, Track


class BaseTracker(ABC):
    """Common interface for tracker backends."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state."""
        raise NotImplementedError

    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker state with new detections.

        Args:
            detections: List of Detection objects for the current frame.
            frame: Input image in HWC format (may be used for appearance features).

        Returns:
            List of Track objects with assigned track_ids.
        """
        raise NotImplementedError
