"""YOLOX detector stub (embedded inside ByteTrack full pipeline for now)."""

from __future__ import annotations

from typing import List

import numpy as np

from src.data import Detection
from src.modules.detectors.base import BaseDetector


class YOLOXDetector(BaseDetector):
    """Placeholder YOLOX detector.

    Full in-process implementation will be added when ComposedExtractor
    switches from subprocess to pure Python pipeline.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def reset(self) -> None:
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError(
            "YOLOXDetector in-process mode is not yet implemented. "
            "Use ByteTrack full subprocess path instead."
        )
