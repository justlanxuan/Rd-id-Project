"""Causal matching with independent state for each observed tracklet.

This module deliberately treats a tracklet identifier as opaque.  It never
links, merges, or transfers history between identifiers: seeing a new ID
always creates a fresh state vector from that observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from src.modules.matchers.assignment import solve_assignment


@dataclass(frozen=True)
class TrackletHistoryResult:
    """One matching update with enough provenance for raw prediction output."""

    assignment: np.ndarray
    accumulated_similarity: np.ndarray
    confidences: np.ndarray
    initialized_tracklets: tuple[Hashable, ...]
    updated_tracklets: tuple[Hashable, ...]
    preserved_tracklets: tuple[Hashable, ...]

    def prediction_fields(self) -> dict[str, object]:
        return {
            "accumulated_similarity": self.accumulated_similarity.tolist(),
            "confidence": self.confidences.tolist(),
            "initialized_tracklets": list(self.initialized_tracklets),
            "updated_tracklets": list(self.updated_tracklets),
            "preserved_tracklets": list(self.preserved_tracklets),
        }


class PerTrackletHistoryMatcher:
    """Accumulate similarity independently for each opaque tracklet ID.

    ``decay`` is the fraction forgotten at every observation of that same
    tracklet.  Therefore ``decay=0`` is a cumulative signed vote and
    ``decay=1`` uses only the current observation.  A missing tracklet is not
    updated and its state is never assigned to another identifier.
    """

    def __init__(
        self,
        *,
        decay: float = 0.0,
        confidence_threshold: float = 0.0,
        confidence_mode: str = "margin",
        confidence_scale: float = 3.0,
        low_confidence_action: str = "preserve",
        assigner: str = "hungarian",
    ) -> None:
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if float(confidence_threshold) < 0.0:
            raise ValueError("confidence_threshold must be non-negative")
        action = str(low_confidence_action).strip().lower()
        if action not in {"preserve", "update"}:
            raise ValueError(
                "low_confidence_action must be 'preserve' or 'update', "
                f"got {low_confidence_action!r}"
            )
        mode = str(confidence_mode).strip().lower()
        if mode not in {"margin", "sigmoid_margin"}:
            raise ValueError("confidence_mode must be 'margin' or 'sigmoid_margin'")
        if float(confidence_scale) <= 0.0:
            raise ValueError("confidence_scale must be positive")
        assignment_method = str(assigner).strip().lower()
        if assignment_method not in {"hungarian", "greedy"}:
            raise ValueError("assigner must be 'hungarian' or 'greedy'")
        self.decay = float(decay)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_mode = mode
        self.confidence_scale = float(confidence_scale)
        self.low_confidence_action = action
        self.assigner = assignment_method
        self._history: dict[Hashable, np.ndarray] = {}
        self._num_imu: int | None = None

    def reset(self) -> None:
        """Discard all tracklet state, normally at a session boundary."""
        self._history.clear()
        self._num_imu = None

    def history_for(self, tracklet_id: Hashable) -> np.ndarray:
        """Return a copy so callers cannot mutate internal history."""
        if tracklet_id not in self._history:
            raise KeyError(tracklet_id)
        return self._history[tracklet_id].copy()

    def _column_confidences(self, similarity: np.ndarray) -> np.ndarray:
        if similarity.shape[0] <= 1:
            margins = np.abs(similarity[0]) if similarity.shape[0] else np.empty(0)
        else:
            ordered = np.sort(similarity, axis=0)
            margins = ordered[-1] - ordered[-2]
        if self.confidence_mode == "sigmoid_margin":
            return 1.0 / (1.0 + np.exp(-self.confidence_scale * margins))
        return margins

    def update(
        self,
        similarity: np.ndarray,
        active_tracklet_ids: Sequence[Hashable],
    ) -> TrackletHistoryResult:
        """Update active IDs and solve the current rectangular assignment."""
        current = np.asarray(similarity, dtype=np.float64)
        if current.ndim != 2:
            raise ValueError(f"similarity must be 2D, got shape {current.shape}")
        if not np.isfinite(current).all():
            raise ValueError("similarity must contain only finite values")
        tracklets = tuple(active_tracklet_ids)
        if current.shape[1] != len(tracklets):
            raise ValueError(
                "similarity columns must equal active tracklet IDs: "
                f"{current.shape[1]} != {len(tracklets)}"
            )
        if len(set(tracklets)) != len(tracklets):
            raise ValueError("active tracklet IDs must be unique")
        if self._num_imu is None:
            self._num_imu = int(current.shape[0])
        elif current.shape[0] != self._num_imu:
            raise ValueError(
                f"IMU candidate count changed from {self._num_imu} to {current.shape[0]}"
            )

        confidences = self._column_confidences(current)
        initialized: list[Hashable] = []
        updated: list[Hashable] = []
        preserved: list[Hashable] = []
        accumulated = np.empty_like(current)
        retention = 1.0 - self.decay
        for column, tracklet_id in enumerate(tracklets):
            observation = current[:, column]
            if tracklet_id not in self._history:
                # Initialization is unconditional: preserving an all-zero state
                # would make a newly seen low-confidence tracklet meaningless.
                history = observation.copy()
                initialized.append(tracklet_id)
            elif (
                confidences[column] <= self.confidence_threshold
                and self.low_confidence_action == "preserve"
            ):
                history = self._history[tracklet_id].copy()
                preserved.append(tracklet_id)
            else:
                history = retention * self._history[tracklet_id] + observation
                updated.append(tracklet_id)
            self._history[tracklet_id] = history
            accumulated[:, column] = history

        assignment = solve_assignment(accumulated, self.assigner)
        return TrackletHistoryResult(
            assignment=assignment,
            accumulated_similarity=accumulated,
            confidences=confidences,
            initialized_tracklets=tuple(initialized),
            updated_tracklets=tuple(updated),
            preserved_tracklets=tuple(preserved),
        )


__all__ = ["PerTrackletHistoryMatcher", "TrackletHistoryResult"]
