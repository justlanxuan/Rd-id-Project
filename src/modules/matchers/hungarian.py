"""Hungarian matcher implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseMatcher


@dataclass
class HungarianConfig:
    threshold: float = 0.0


class HungarianMatcher(BaseMatcher):
    """Match using Hungarian algorithm on similarity matrix."""

    def __init__(self, config_dict: Dict):
        self.config = HungarianConfig(**config_dict)

    def match(
        self,
        similarity_matrix: Any,
        imu_ids: Optional[List[Any]] = None,
        person_ids: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        from scipy.optimize import linear_sum_assignment

        sim = np.asarray(similarity_matrix, dtype=np.float32)
        if sim.ndim != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {sim.shape}")

        cost = -sim
        rows, cols = linear_sum_assignment(cost)

        imu_ids = imu_ids or list(range(sim.shape[0]))
        person_ids = person_ids or list(range(sim.shape[1]))

        assignments = []
        scores = []
        confidences = []

        for r, c in zip(rows, cols, strict=True):
            score = float(sim[r, c])
            if score < self.config.threshold:
                continue
            row_scores = sim[r]
            best = float(np.max(row_scores)) if row_scores.size else 0.0
            conf = score / best if best > 0 else 0.0
            assignments.append((imu_ids[r], person_ids[c]))
            scores.append(score)
            confidences.append(conf)

        return {
            "assignments": assignments,
            "scores": scores,
            "confidences": confidences,
        }


def build_hungarian_matcher(config_dict: Dict) -> HungarianMatcher:
    return HungarianMatcher(config_dict)
