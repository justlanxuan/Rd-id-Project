"""Inference-only builder for temporal tracklet decision strategies."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from src.modules.matchers.tracklet_history import PerTrackletHistoryMatcher


class TemporalMatchResult(Protocol):
    assignment: np.ndarray

    def prediction_fields(self) -> dict[str, object]: ...


class TemporalTrackletMatcher(Protocol):
    def update(
        self,
        similarity: np.ndarray,
        active_tracklet_ids: Sequence[object],
    ) -> TemporalMatchResult: ...


def build_temporal_tracklet_matcher(frame_cfg: Any) -> TemporalTrackletMatcher:
    """Construct the configured strategy without exposing it to training."""
    name = str(frame_cfg.MATCHER).strip().lower()
    if name == "tracklet_history":
        history_cfg = frame_cfg.HISTORY
        return PerTrackletHistoryMatcher(
            decay=float(history_cfg.DECAY),
            confidence_threshold=float(history_cfg.CONFIDENCE_THRESHOLD),
            confidence_mode=str(history_cfg.CONFIDENCE_MODE),
            confidence_scale=float(history_cfg.CONFIDENCE_SCALE),
            low_confidence_action=str(history_cfg.LOW_CONFIDENCE_ACTION),
            assigner=str(frame_cfg.ASSIGNER),
        )
    raise ValueError(
        "Full-session unmerged evaluation requires MATCHER to be "
        f"'tracklet_history'; got {name!r}"
    )


def temporal_matcher_metadata(frame_cfg: Any) -> dict[str, object]:
    name = str(frame_cfg.MATCHER).strip().lower()
    common: dict[str, object] = {
        "name": name,
        "assigner": str(frame_cfg.ASSIGNER),
    }
    if name == "tracklet_history":
        history_cfg = frame_cfg.HISTORY
        common["parameters"] = {
            "decay": float(history_cfg.DECAY),
            "confidence_threshold": float(history_cfg.CONFIDENCE_THRESHOLD),
            "confidence_mode": str(history_cfg.CONFIDENCE_MODE),
            "confidence_scale": float(history_cfg.CONFIDENCE_SCALE),
            "low_confidence_action": str(history_cfg.LOW_CONFIDENCE_ACTION),
        }
    return common


__all__ = [
    "TemporalMatchResult",
    "TemporalTrackletMatcher",
    "build_temporal_tracklet_matcher",
    "temporal_matcher_metadata",
]
