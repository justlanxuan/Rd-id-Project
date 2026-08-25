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


def _write_prepared_split(
    root,
    split,
    session,
    *,
    candidates=2,
    subject="",
    explicit_candidate_group=False,
    starts=(0,),
):
    rows = []
    for source_start in starts:
        for candidate_index in range(candidates):
            npz_name = f"{split}_{source_start}_{candidate_index}.npz"
            np.savez(
                root / npz_name,
                imu=np.ones((4, 7), dtype=np.float32),
                skeleton=np.ones((4, 17, 3), dtype=np.float32),
            )
            rows.append(
                {
                    "npz_path": npz_name,
                    "session": session,
                    "subject": subject,
                    "source_sequence": f"{split}_candidate_group",
                    "source_person": str(candidate_index),
                    "source_window_start": str(source_start),
                    "window_start": "0",
                    "window_end": "4",
                    "window_len": "4",
                    "candidate_index": str(candidate_index),
                    "candidate_group_id": (
                        f"{split}_explicit_group_{source_start}"
                        if explicit_candidate_group
                        else ""
                    ),
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


def test_prepared_cache_validator_allows_explicitly_empty_validation(tmp_path):
    _write_prepared_split(tmp_path, "train", "session_train")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(tmp_path, "test", "session_test")
    val_csv = tmp_path / "windows_val.csv"
    header = val_csv.read_text().splitlines()[0]
    val_csv.write_text(header + "\n")

    with pytest.raises(ValueError, match="split is empty"):
        validate_prepared_dataset(tmp_path)
    assert validate_prepared_dataset(
        tmp_path,
        expected_test_sessions={"session_test"},
        allow_empty_validation=True,
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


def test_prepared_cache_validator_supports_subject_split_identity(tmp_path):
    _write_prepared_split(tmp_path, "train", "shared_session", subject="S1")
    _write_prepared_split(tmp_path, "val", "shared_session", subject="S2")
    _write_prepared_split(tmp_path, "test", "shared_session", subject="S3")

    assert validate_prepared_dataset(
        tmp_path,
        split_identity="subject",
        expected_test_values={"S3"},
    ) == tmp_path.resolve()


def test_prepared_cache_validator_rejects_source_sequence_leakage(tmp_path):
    _write_prepared_split(tmp_path, "train", "session_train")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(tmp_path, "test", "session_test")
    val_csv = tmp_path / "windows_val.csv"
    rows = list(csv.DictReader(val_csv.open(newline="")))
    rows[0]["source_sequence"] = "train_candidate_group"
    with val_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="source_sequence leakage"):
        validate_prepared_dataset(tmp_path)


def test_prepared_cache_validator_prefers_explicit_candidate_group(tmp_path):
    _write_prepared_split(tmp_path, "train", "session_train")
    _write_prepared_split(tmp_path, "val", "session_val")
    _write_prepared_split(
        tmp_path,
        "test",
        "session_test",
        explicit_candidate_group=True,
    )

    assert validate_prepared_dataset(tmp_path) == tmp_path.resolve()


def test_prepared_cache_validator_checks_actual_window_len_and_stride(tmp_path):
    for split, session in (
        ("train", "session_train"),
        ("val", "session_val"),
        ("test", "session_test"),
    ):
        _write_prepared_split(tmp_path, split, session, starts=(0, 2, 4))

    assert validate_prepared_dataset(
        tmp_path,
        expected_window_len=4,
        expected_stride=2,
    ) == tmp_path.resolve()
    with pytest.raises(ValueError, match="stride mismatch"):
        validate_prepared_dataset(tmp_path, expected_window_len=4, expected_stride=4)
    with pytest.raises(ValueError, match="window_len mismatch"):
        validate_prepared_dataset(tmp_path, expected_window_len=3, expected_stride=2)
