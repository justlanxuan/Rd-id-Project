from __future__ import annotations

import csv

import numpy as np
import pytest

from preprocess.adapters import validate_prepared_dataset, validate_preprocess_output


def _write_sequence(path, *, imu, skeleton, frame_ids=None):
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        schema_version=np.array("1.0", dtype=object),
        dataset=np.array("custom", dtype=object),
        sequence_id=np.array("custom_fixture", dtype=object),
        frame_ids=np.arange(len(imu)) if frame_ids is None else frame_ids,
        imu=imu,
        gt_skeleton=skeleton,
    )


def test_adapter_validation_rejects_empty_output(tmp_path):
    with pytest.raises(ValueError, match="produced no sequence"):
        validate_preprocess_output("custom", tmp_path)


@pytest.mark.parametrize("placeholder", ["imu", "skeleton"])
def test_adapter_validation_rejects_placeholder_arrays(tmp_path, placeholder):
    imu = np.ones((4, 1, 7), dtype=np.float32)
    skeleton = np.ones((4, 1, 17, 3), dtype=np.float32)
    if placeholder == "imu":
        imu.fill(0)
    else:
        skeleton.fill(0)
    _write_sequence(tmp_path / "sequences" / "fixture.npz", imu=imu, skeleton=skeleton)

    with pytest.raises(ValueError, match=f"(?i)all-zero {placeholder}"):
        validate_preprocess_output("custom", tmp_path)


def test_adapter_validation_accepts_minimal_canonical_sequence(tmp_path):
    _write_sequence(
        tmp_path / "sequences" / "fixture.npz",
        imu=np.ones((4, 1, 7), dtype=np.float32),
        skeleton=np.ones((4, 1, 17, 3), dtype=np.float32),
    )

    assert validate_preprocess_output("custom", tmp_path) == tmp_path.resolve()


def test_adapter_validation_rejects_non_monotonic_frame_ids(tmp_path):
    _write_sequence(
        tmp_path / "sequences" / "fixture.npz",
        imu=np.ones((4, 1, 7), dtype=np.float32),
        skeleton=np.ones((4, 1, 17, 3), dtype=np.float32),
        frame_ids=np.asarray([0, 2, 2, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="non-monotonic or duplicate frame_ids"):
        validate_preprocess_output("custom", tmp_path)


def _write_prepared_split(root, split, session, *, candidates=2):
    rows = []
    for candidate_index in range(candidates):
        npz_name = f"{split}_{candidate_index}.npz"
        np.savez(
            root / npz_name,
            imu=np.ones((4, 7), dtype=np.float32),
            skeleton=np.ones((4, 17, 3), dtype=np.float32),
        )
        rows.append(
            {
                "npz_path": npz_name,
                "session": session,
                "source_sequence": f"{split}_candidate_group",
                "source_window_start": "0",
                "window_start": "0",
                "window_end": "4",
                "candidate_index": str(candidate_index),
            }
        )
    with (root / f"windows_{split}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prepared_cache_validator_accepts_disjoint_sessions(tmp_path):
    _write_prepared_split(tmp_path, "train", "session_train")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(tmp_path, "test", "session_test")

    assert validate_prepared_dataset(
        tmp_path,
        expected_test_sessions={"session_test"},
    ) == tmp_path.resolve()


def test_prepared_cache_validator_rejects_session_leakage(tmp_path):
    _write_prepared_split(tmp_path, "train", "shared_session")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(tmp_path, "test", "shared_session")

    with pytest.raises(ValueError, match="session leakage"):
        validate_prepared_dataset(tmp_path)


def test_prepared_cache_validator_requires_explicit_singleton_policy(tmp_path):
    _write_prepared_split(tmp_path, "train", "session_train")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(tmp_path, "test", "session_test", candidates=1)

    with pytest.raises(ValueError, match="singleton FrameAcc groups"):
        validate_prepared_dataset(tmp_path)
    assert validate_prepared_dataset(
        tmp_path,
        allow_singleton_test_groups=True,
    ) == tmp_path.resolve()
