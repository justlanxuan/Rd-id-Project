from __future__ import annotations

import numpy as np

from src.config import get_cfg_defaults
from src.inference import build_configured_inference_policy, build_inference_policy
from src.inference.offline import MultiPersonOfflinePolicy
from src.inference.realtime import TrackletHistoryRealtimePolicy


def test_configured_default_remains_offline_multi_person_policy():
    policy = build_configured_inference_policy(get_cfg_defaults())

    assert isinstance(policy, MultiPersonOfflinePolicy)
    assert policy.method == "global_best_segment"


def test_realtime_policy_builder_returns_tracklet_history_policy():
    policy = build_inference_policy(
        "realtime",
        "tracklet_history",
        decay=0.0,
        assigner="hungarian",
    )

    assert isinstance(policy, TrackletHistoryRealtimePolicy)


def test_configured_realtime_policy_uses_realtime_section():
    cfg = get_cfg_defaults()
    cfg.INFERENCE.MODE = "realtime"
    policy = build_configured_inference_policy(cfg)

    assert isinstance(policy, TrackletHistoryRealtimePolicy)
    assert policy.mode == "realtime"


def test_realtime_policy_preserves_history_and_resets_at_session_boundary():
    policy = TrackletHistoryRealtimePolicy(decay=0.0)
    first = policy.infer(
        np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        ["track-a", "track-b"],
    )
    second = policy.infer(
        np.asarray([[0.1, 0.8], [0.8, 0.1]], dtype=np.float32),
        ["track-a", "track-b"],
    )

    assert first.assignment.tolist() == [0, 1]
    assert second.assignment.tolist() == [0, 1]
    assert second.metadata["updated_tracklets"] == ["track-a", "track-b"]

    policy.reset()
    restarted = policy.infer(
        np.asarray([[0.1, 0.8], [0.8, 0.1]], dtype=np.float32),
        ["track-a", "track-b"],
    )
    assert restarted.assignment.tolist() == [1, 0]
