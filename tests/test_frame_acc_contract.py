from __future__ import annotations

import numpy as np
import pytest

from preprocess.common.slice import assign_frame_acc_candidate_groups
from src.metrics import EmbeddingBundle, FrameAccEvaluator


def _row(npz_path: str, subject: str, candidate_group_id: str = "") -> dict[str, str]:
    return {
        "npz_path": npz_path,
        "session": "session",
        "window_start": "0",
        "window_end": "24",
        "subject": subject,
        "candidate_group_id": candidate_group_id,
    }


def test_frame_acc_rejects_non_discriminative_singletons_by_default():
    bundle = EmbeddingBundle(
        rows=[_row("single.npz", "S1")],
        imu=np.asarray([[1.0, 0.0]], dtype=np.float32),
        video=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="only one item"):
        FrameAccEvaluator().evaluate(bundle)


def test_candidate_group_id_can_form_cross_sequence_source_frame_acc_groups():
    bundle = EmbeddingBundle(
        rows=[
            _row("sequence_a.npz", "S1", "group-0"),
            _row("sequence_b.npz", "S2", "group-0"),
        ],
        imu=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        video=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    result = FrameAccEvaluator(shuffle_match=True, seed=0).evaluate(bundle)

    assert result["frame_acc"] == 1.0
    assert result["correct_assignments"] == 2
    assert result["num_assignments"] == 2
    assert result["candidate_group_size_min"] == 2
    assert result["candidate_group_size_mean"] == 2.0
    assert result["singleton_rate"] == 0.0
    assert result["prediction_schema_version"] == "1.0"
    assert result["assignments"] == [
        {
            "candidate_group": ["candidate_group_id", "group-0", 0, 0],
            "status": "evaluated",
            "row_indices": [0, 1],
            "similarity": [[1.0, 0.0], [0.0, 1.0]],
            "imu_permutation": [0, 1],
            "hungarian_rows": [0, 1],
            "hungarian_columns": [0, 1],
            "matched_imu_row_indices": [0, 1],
            "matched_video_row_indices": [0, 1],
            "correct": 2,
            "total": 2,
        }
    ]


def test_single_person_source_rows_are_grouped_deterministically_by_window_position():
    rows = [
        {
            "split": "test",
            "npz_path": f"sequence_{index}.npz",
            "window_start": 0,
            "window_end": 24,
            "person_idx": 0,
            "imu_idx": 0,
        }
        for index in range(3)
    ]

    stats = assign_frame_acc_candidate_groups(rows, cross_sequence_group_size=2)

    assert stats == {"candidate_groups": 1, "candidate_rows": 2, "singleton_rows": 1}
    assert rows[0]["candidate_group_id"] == rows[1]["candidate_group_id"]
    assert rows[0]["candidate_index"] == 0
    assert rows[1]["candidate_index"] == 1
    assert rows[2]["candidate_group_id"] == ""
