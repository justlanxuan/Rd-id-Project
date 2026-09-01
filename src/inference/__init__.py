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


def build_configured_inference_policy(cfg: Any):
    """Build the configured policy without coupling inference to YACS.

    The helper reads the INFERENCE section but accepts any object exposing the
    same attributes, which keeps application code independent of the config
    implementation.
    """
    inference = cfg.INFERENCE
    mode = str(inference.MODE)
    if mode.strip().lower().replace("-", "_") == "offline":
        options = inference.OFFLINE
        return build_inference_policy(
            mode,
            str(inference.POLICY),
            method=str(options.METHOD),
            segment_frames=int(options.SEGMENT_FRAMES),
            min_windows=int(options.MIN_WINDOWS),
            top_k=int(options.TOP_K),
        )

    realtime = inference.REALTIME
    return build_inference_policy(
        mode,
        str(realtime.POLICY),
        decay=float(realtime.DECAY),
        confidence_threshold=float(realtime.CONFIDENCE_THRESHOLD),
        confidence_mode=str(realtime.CONFIDENCE_MODE),
        confidence_scale=float(realtime.CONFIDENCE_SCALE),
        low_confidence_action=str(realtime.LOW_CONFIDENCE_ACTION),
        assigner=str(realtime.ASSIGNER),
    )


__all__ = [
    "InferenceDecision",
    "InferenceMode",
    "InferencePolicy",
    "RealtimeScope",
    "build_configured_inference_policy",
    "build_inference_policy",
]
