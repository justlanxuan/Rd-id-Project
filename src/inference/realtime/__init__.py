"""Realtime inference policies with explicit streaming state."""

from __future__ import annotations

from typing import Any

from .tracklet_history import TrackletHistoryRealtimePolicy


def build_realtime_policy(
    name: str = "tracklet_history", **kwargs: Any
) -> TrackletHistoryRealtimePolicy:
    """Build a stateful realtime policy by name."""
    policy = str(name).strip().lower().replace("-", "_")
    if policy in {"tracklet_history", "history", "g8", "g8_full_session"}:
        return TrackletHistoryRealtimePolicy(**kwargs)
    raise ValueError(f"Unknown realtime inference policy: {name!r}")


__all__ = ["TrackletHistoryRealtimePolicy", "build_realtime_policy"]
