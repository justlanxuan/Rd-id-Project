from __future__ import annotations

import numpy as np
import pytest

from src.config import get_cfg_defaults
from src.inference import build_inference_policy
from src.inference.offline import MultiPersonOfflinePolicy, evaluate_scores


def test_inference_defaults_include_offline_mode():
    cfg = get_cfg_defaults()

    assert cfg.INFERENCE.MODE == "offline"
    assert cfg.INFERENCE.POLICY == "multi_person"
    assert cfg.INFERENCE.REALTIME.SCOPE == "global"
    assert cfg.INFERENCE.REALTIME.CONTEXT_SECONDS == 10.0


def test_inference_policy_builder_returns_multi_person_offline_policy():
    policy = build_inference_policy(
        "offline",
        "multi_person",
        method="global_best_segment",
        segment_frames=20,
        min_windows=2,
    )

    assert isinstance(policy, MultiPersonOfflinePolicy)


def test_global_best_segment_prefers_the_larger_gap_segment():
    scores = np.asarray(
        [
            [[5.0, 0.0], [0.0, 5.0]],
            [[4.0, 1.0], [1.0, 4.0]],
            [[2.0, 3.0], [3.0, 2.0]],
            [[1.0, 4.0], [4.0, 1.0]],
        ],
        dtype=np.float32,
    )
    centers = np.asarray([5, 15, 25, 35], dtype=np.int64)

    result = evaluate_scores(
        scores,
        centers,
        method="global_best_segment",
        segment_frames=20,
        min_windows=2,
        top_k=2,
    )

    assert result["mode"] == "offline"
    assert result["policy"] == "multi_person"
    assert result["selected_segments"] == [0]
    assert result["assignment"] == [0, 1]
    assert result["selected_score"] == pytest.approx(9.0)
    assert result["segments"][0]["global_gap"] > result["segments"][1]["global_gap"]


def test_local_top1_segment_uses_rowwise_evidence():
    scores = np.asarray(
        [
            [[5.0, 1.0], [2.0, 1.0]],
            [[4.0, 1.0], [2.0, 1.0]],
            [[2.0, 1.0], [1.0, 5.0]],
            [[1.0, 1.0], [1.0, 4.0]],
        ],
        dtype=np.float32,
    )
    centers = np.asarray([5, 15, 25, 35], dtype=np.int64)
    policy = MultiPersonOfflinePolicy(
        method="local_top1_segment",
        segment_frames=20,
        min_windows=2,
        top_k=2,
    )

    decision = policy.infer(scores, centers)

    assert decision.assignment.tolist() == [0, 1]
    assert decision.selected_segments == (0, 1)
    assert decision.metadata["row_selected_segments"] == [[0], [1]]
