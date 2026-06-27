"""Temporal matcher with tracker ID consideration.

This matcher extends Hungarian matching by incorporating information from
previous window matches (tracker continuity). It helps stabilize matching
when current-window embedding confidence is low by falling back to
tracker ID continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.modules.matchers.base import BaseMatcher


@dataclass
class TemporalMatcherConfig:
    """Configuration for temporal matching."""
    threshold: float = 0.0
    confidence_threshold: float = 0.8  # Switch to tracker-based when current conf < this
    alpha: float = 0.7  # Weight of current confidence in fusion (1-alpha = weight of history)


class TemporalMatcher(BaseMatcher):
    """Match using Hungarian algorithm with temporal continuity from tracker IDs.
    
    This matcher maintains a tracker state across windows to provide continuity.
    When current window embedding confidence is low, it preferentially assigns
    to skeleton IDs that were matched to the same IMU in the previous window.
    """

    def __init__(self, config_dict: Dict):
        self.config = TemporalMatcherConfig(**config_dict)
        self.tracker_state: Dict[Any, Dict[str, Any]] = {}  # {imu_id -> {skel_id, confidence, window_idx}}
        self.prev_assignments: Dict[Any, Any] = {}  # {imu_id -> skel_id}
        self.prev_confidences: Dict[Any, float] = {}  # {imu_id -> confidence}

    def _compute_confidence(self, similarity_row: np.ndarray, match_score: float) -> float:
        """Compute confidence as normalized score within max."""
        best = float(np.max(similarity_row)) if similarity_row.size > 0 else 0.0
        conf = match_score / best if best > 0 else 0.0
        return min(1.0, max(0.0, conf))

    def _fuse_with_history(
        self,
        similarity_matrix: np.ndarray,
        imu_ids: List[Any],
        person_ids: List[Any],
    ) -> Tuple[List[Tuple], List[float], List[float]]:
        """Fuse current matching with historical tracker information.
        
        If current window confidence is low, prefer assignments aligned with
        previous window's tracker IDs.
        """
        from scipy.optimize import linear_sum_assignment

        sim = np.asarray(similarity_matrix, dtype=np.float32)
        if sim.ndim != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {sim.shape}")

        # Compute Hungarian matching cost
        cost = -sim
        rows, cols = linear_sum_assignment(cost)

        assignments = []
        scores = []
        confidences = []

        for r, c in zip(rows, cols):
            imu_id = imu_ids[r]
            skel_id = person_ids[c]
            score = float(sim[r, c])

            if score < self.config.threshold:
                continue

            # Compute confidence
            row_scores = sim[r]
            conf = self._compute_confidence(row_scores, score)

            # Decide: use current or prefer tracker history?
            final_skel_id = skel_id
            final_conf = conf

            if conf < self.config.confidence_threshold and imu_id in self.prev_assignments:
                # Current confidence low; check if previous match is available
                prev_skel = self.prev_assignments[imu_id]
                prev_conf = self.prev_confidences.get(imu_id, 0.0)

                # If previous skeleton is still in valid IDs, prefer it
                if prev_skel in person_ids:
                    prev_col = person_ids.index(prev_skel)
                    prev_score = float(sim[r, prev_col])
                    # Fuse confidences: weight current vs history
                    fused_conf = (
                        conf * self.config.alpha +
                        self._compute_confidence(row_scores, prev_score) * (1 - self.config.alpha)
                    )
                    # Prefer history if it's reasonably good
                    if fused_conf >= conf * 0.95:  # Don't switch if it hurts too much
                        final_skel_id = prev_skel
                        final_conf = fused_conf

            assignments.append((imu_id, final_skel_id))
            scores.append(score)
            confidences.append(final_conf)

        return assignments, scores, confidences

    def match(
        self,
        similarity_matrix: Any,
        imu_ids: Optional[List[Any]] = None,
        person_ids: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Single-window matching with optional tracker history fusion."""
        from scipy.optimize import linear_sum_assignment

        sim = np.asarray(similarity_matrix, dtype=np.float32)
        if sim.ndim != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {sim.shape}")

        imu_ids = imu_ids or list(range(sim.shape[0]))
        person_ids = person_ids or list(range(sim.shape[1]))

        # Try fusion-based matching if we have history
        if self.prev_assignments:
            assignments, scores, confidences = self._fuse_with_history(sim, imu_ids, person_ids)
        else:
            # First window: pure Hungarian matching
            cost = -sim
            rows, cols = linear_sum_assignment(cost)

            assignments = []
            scores = []
            confidences = []

            for r, c in zip(rows, cols):
                score = float(sim[r, c])
                if score < self.config.threshold:
                    continue
                row_scores = sim[r]
                conf = self._compute_confidence(row_scores, score)
                assignments.append((imu_ids[r], person_ids[c]))
                scores.append(score)
                confidences.append(conf)

        # Update tracker state for next window
        self.prev_assignments = {imu_id: skel_id for imu_id, skel_id in assignments}
        self.prev_confidences = {
            imu_id: conf
            for (imu_id, _), conf in zip(assignments, confidences)
        }

        return {
            "assignments": assignments,
            "scores": scores,
            "confidences": confidences,
        }

    def reset(self) -> None:
        """Reset tracker state (call between different test sequences)."""
        self.tracker_state.clear()
        self.prev_assignments.clear()
        self.prev_confidences.clear()


def build_temporal_matcher(config_dict: Dict) -> TemporalMatcher:
    """Factory function to build temporal matcher."""
    return TemporalMatcher(config_dict)
