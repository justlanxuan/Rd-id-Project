from __future__ import annotations

import json

import numpy as np

from src.datasets.custom_session import load_custom_tracklet_session


def _keypoints(x: float) -> list[float]:
    values: list[float] = []
    for joint in range(17):
        values.extend([x + joint, 10.0 + joint, 0.9])
    return values


def test_raw_list_tracklet_id_is_atomic_and_does_not_expand(tmp_path):
    aligned = tmp_path / "custom_session-a.npz"
    np.savez_compressed(
        aligned,
        sequence_id=np.asarray("custom_session-a", dtype=object),
        frame_ids=np.asarray([0, 1], dtype=np.int64),
        imu=np.zeros((2, 2, 7), dtype=np.float32),
        imu_ids=np.asarray([10, 20], dtype=np.int64),
        gt_person_ids=np.asarray([10, 20], dtype=np.int64),
        gt_bboxes=np.asarray(
            [[[0, 0, 20, 40], [100, 0, 120, 40]], [[0, 0, 20, 40], [100, 0, 120, 40]]],
            dtype=np.float32,
        ),
        gt_visibility=np.ones((2, 2), dtype=bool),
    )
    raw = tmp_path / "skeleton_unmerged.json"
    raw.write_text(
        json.dumps(
            [
                {"image_id": "0.jpg", "idx": [1, 2], "box": [0, 0, 20, 40], "keypoints": _keypoints(0), "score": 1.0},
                {"image_id": "0.jpg", "idx": 3, "box": [100, 0, 20, 40], "keypoints": _keypoints(100), "score": 1.0},
                {"image_id": "1.jpg", "idx": 4, "box": [0, 0, 20, 40], "keypoints": _keypoints(0), "score": 1.0},
            ]
        )
    )

    session = load_custom_tracklet_session(aligned, raw)

    assert session.tracklet_labels == ("[1,2]", "3", "4")
    assert session.extract_visibility.tolist() == [[True, True, False], [False, False, True]]
    assert session.gt_to_extract_map.tolist() == [[0, 1], [2, -1]]
    assert session.imu.shape == (2, 2, 7)


def test_full_session_loader_clamps_longer_ground_truth_timeline(tmp_path):
    aligned = tmp_path / "custom_session-a.npz"
    np.savez_compressed(
        aligned,
        sequence_id=np.asarray("custom_session-a", dtype=object),
        frame_ids=np.asarray([0], dtype=np.int64),
        imu=np.zeros((1, 1, 7), dtype=np.float32),
        imu_ids=np.asarray([0], dtype=np.int64),
        gt_person_ids=np.asarray([0], dtype=np.int64),
        gt_bboxes=np.zeros((2, 1, 4), dtype=np.float32),
        gt_visibility=np.ones((2, 1), dtype=bool),
    )
    raw = tmp_path / "skeleton_unmerged.json"
    raw.write_text("[]")

    session = load_custom_tracklet_session(aligned, raw)

    assert session.gt_bboxes.shape == (1, 1, 4)
    assert session.gt_visibility.shape == (1, 1)
