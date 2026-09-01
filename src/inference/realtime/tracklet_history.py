"""Stateful realtime inference backed by per-tracklet history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

import numpy as np

from src.inference.contracts import InferenceDecision
from src.modules.matchers.tracklet_history import PerTrackletHistoryMatcher


@dataclass
class TrackletHistoryRealtimePolicy:
    """Apply one-to-one matching while preserving history per tracklet ID.

    The policy is deliberately stateful: callers must call :meth:`reset` at a
    session boundary. A missing tracklet is not reassigned internally; a new
    identifier starts a new history entry.
    """

    decay: float = 0.0
    confidence_threshold: float = 0.0
    confidence_mode: str = "margin"
    confidence_scale: float = 3.0
    low_confidence_action: str = "preserve"
    assigner: str = "hungarian"
    mode: str = field(default="realtime", init=False)
    policy_name: str = "tracklet_history"
    _matcher: PerTrackletHistoryMatcher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._matcher = PerTrackletHistoryMatcher(
            decay=self.decay,
            confidence_threshold=self.confidence_threshold,
            confidence_mode=self.confidence_mode,
            confidence_scale=self.confidence_scale,
            low_confidence_action=self.low_confidence_action,
            assigner=self.assigner,
        )

    def reset(self) -> None:
        """Discard state before starting a new capture session."""
        self._matcher.reset()

    def infer(
        self,
        similarity: np.ndarray,
        active_tracklet_ids: Sequence[Hashable],
    ) -> InferenceDecision:
        """Consume one streaming similarity matrix and return a decision."""
        tracklets = tuple(active_tracklet_ids)
        result = self._matcher.update(similarity, tracklets)
        confidence_values = np.asarray(result.confidences, dtype=np.float64)
        confidence = float(np.mean(confidence_values)) if len(confidence_values) else None
        metadata: dict[str, Any] = result.prediction_fields()
        metadata["active_tracklet_ids"] = list(tracklets)
        return InferenceDecision(
            mode="realtime",
            policy=self.policy_name,
            assignment=result.assignment.copy(),
            confidence=confidence,
            metadata=metadata,
        )


__all__ = ["TrackletHistoryRealtimePolicy"]
