from __future__ import annotations

import numpy as np
import pytest

from src.modules.matchers.tracklet_history import PerTrackletHistoryMatcher


def test_new_tracklet_starts_from_its_own_similarity_without_inheriting_history():
    matcher = PerTrackletHistoryMatcher(decay=0.0)

    first = matcher.update(
        np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        ["track-a", "track-b"],
    )
    second = matcher.update(
        np.asarray([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
        ["track-a", "track-c"],
    )

    assert first.initialized_tracklets == ("track-a", "track-b")
    assert second.initialized_tracklets == ("track-c",)
    assert second.accumulated_similarity[:, 0] == pytest.approx([1.1, 0.9])
    assert second.accumulated_similarity[:, 1] == pytest.approx([0.8, 0.2])
    assert matcher.history_for("track-b") == pytest.approx([0.1, 0.9])


def test_low_confidence_preserves_existing_tracklet_but_initializes_a_new_one():
    matcher = PerTrackletHistoryMatcher(
        decay=0.0,
        confidence_threshold=0.25,
        confidence_mode="margin",
        low_confidence_action="preserve",
    )
    matcher.update(np.asarray([[0.9], [0.1]], dtype=np.float32), ["known"])

    result = matcher.update(
        np.asarray([[0.51, 0.52], [0.49, 0.48]], dtype=np.float32),
        ["known", "new"],
    )

    assert result.updated_tracklets == ()
    assert result.preserved_tracklets == ("known",)
    assert result.initialized_tracklets == ("new",)
    assert result.accumulated_similarity[:, 0] == pytest.approx([0.9, 0.1])
    assert result.accumulated_similarity[:, 1] == pytest.approx([0.52, 0.48])


def test_matcher_rejects_duplicate_active_tracklet_ids():
    matcher = PerTrackletHistoryMatcher()

    with pytest.raises(ValueError, match="unique"):
        matcher.update(np.ones((2, 2), dtype=np.float32), ["same", "same"])


def test_historical_sigmoid_margin_confidence_is_available_per_tracklet():
    matcher = PerTrackletHistoryMatcher(
        confidence_threshold=0.7,
        confidence_mode="sigmoid_margin",
        confidence_scale=3.0,
    )

    result = matcher.update(
        np.asarray([[0.9, 0.55], [0.1, 0.45]], dtype=np.float32),
        ["confident", "ambiguous"],
    )

    expected = 1.0 / (1.0 + np.exp(-3.0 * np.asarray([0.8, 0.1])))
    assert result.confidences == pytest.approx(expected)


def test_greedy_assigner_can_be_selected_for_historical_compatibility():
    similarity = np.asarray(
        [[3.0, 8.0, 2.0], [2.0, 7.0, 6.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )

    tracklets = ["a", "b", "c"]
    hungarian = PerTrackletHistoryMatcher(assigner="hungarian").update(similarity, tracklets)
    greedy = PerTrackletHistoryMatcher(assigner="greedy").update(similarity, tracklets)

    assert hungarian.assignment.tolist() == [1, 2, 0]
    assert greedy.assignment.tolist() == [1, 0, 2]
