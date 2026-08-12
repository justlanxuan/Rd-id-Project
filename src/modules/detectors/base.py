"""Abstract detector interface for Autism-project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.data import Detection


class BaseDetector(ABC):
    """Common interface for detector backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame.

        Args:
            frame: Input image in HWC format.

        Returns:
            List of Detection objects.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any sequence-local state."""
        raise NotImplementedError
