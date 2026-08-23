"""Stable contracts shared by embedding-based evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingBundle:
    """Aligned metadata and modality embeddings consumed by metrics."""

    rows: list[dict[str, str]]
    imu: np.ndarray
    video: np.ndarray
    orientation: np.ndarray | None = None
    imu_sequences: np.ndarray | None = None


class EvaluationMetric(ABC):
    """Metric operating on a canonical embedding bundle."""

    @abstractmethod
    def evaluate(self, bundle: EmbeddingBundle) -> dict[str, Any]:
        raise NotImplementedError
