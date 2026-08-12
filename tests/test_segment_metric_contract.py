from __future__ import annotations

import pytest

from src.engine.evaluate import aggregate_segment_sessions


def test_segment_results_include_weighted_and_clip_mean_per_session():
    clips = [
        {"sequence_id": "custom_s1_seg0", "frame_acc": 0.5, "correct": 5, "total": 10},
        {"sequence_id": "custom_s1_seg1", "frame_acc": 1.0, "correct": 1, "total": 1},
        {"sequence_id": "custom_s2_seg0", "frame_acc": 0.25, "correct": 2, "total": 8},
    ]

    result = aggregate_segment_sessions(clips, ["s1", "s2"])

    assert result["s1"]["correct"] == 6
    assert result["s1"]["total"] == 11
    assert result["s1"]["frame_acc"] == pytest.approx(6 / 11)
    assert result["s1"]["mean_clip_frame_acc"] == pytest.approx(0.75)
    assert result["s2"]["frame_acc"] == pytest.approx(0.25)
