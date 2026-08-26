"""Application-time inference policies.

Offline policies can inspect complete sequences before deciding. Realtime
policies preserve state across incoming windows and may use a bounded
context.
"""

from __future__ import annotations

from typing import Any

from .contracts import InferenceDecision, InferenceMode, InferencePolicy, RealtimeScope


def build_inference_policy(mode: str = "offline", policy: str = "multi_person", **kwargs: Any):
    """Build an inference policy by mode and policy name."""
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode == "offline":
        from .offline import build_offline_policy

        return build_offline_policy(policy, **kwargs)
    if normalized_mode == "realtime":
        from .realtime import build_realtime_policy

        return build_realtime_policy(policy, **kwargs)
    raise ValueError(f"Unknown inference mode: {mode!r}")


__all__ = [
    "InferenceDecision",
    "InferenceMode",
    "InferencePolicy",
    "RealtimeScope",
    "build_inference_policy",
]
