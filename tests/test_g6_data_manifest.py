from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tools.g6.data_manifest import build_prepared_data_manifest


def _write_split(root: Path, split: str, session: str, rows: int = 2) -> None:
    sequence_dir = root / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"windows_{split}.csv"
    fieldnames = [
        "subject",
        "session",
        "split",
        "npz_path",
        "window_start",
        "window_end",
        "window_len",
        "source_sequence",
        "source_person",
        "source_window_start",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for person in range(rows):
            name = f"{split}_p{person}.npz"
            np.savez(
                sequence_dir / name,
                imu=np.ones((24, 7), dtype=np.float32) * (person + 1),
                skeleton=np.ones((24, 17, 2), dtype=np.float32),
            )
            writer.writerow(
                {
                    "subject": f"P{person}",
                    "session": session,
                    "split": split,
                    "npz_path": f"sequences/{name}",
                    "window_start": 0,
                    "window_end": 24,
                    "window_len": 24,
                    "source_sequence": f"sequence_{session}",
                    "source_person": person,
                    "source_window_start": 0,
                }
            )


def test_prepared_manifest_is_deterministic_and_reports_candidate_groups(tmp_path):
    _write_split(tmp_path, "train", "train_session")
    _write_split(tmp_path, "val", "val_session")
    _write_split(tmp_path, "test", "test_session")

    first = build_prepared_data_manifest(tmp_path, dataset="custom", fold_id=1)
    second = build_prepared_data_manifest(tmp_path, dataset="custom", fold_id=1)

    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["candidate_groups_test"]["size_distribution"] == {"2": 1}
    assert first["candidate_groups_test"]["singleton_rate"] == 0.0
    assert first["content_summary"]["zero_imu_files"] == 0
    assert first["split_identity"] == "session"
    assert first["split_overlaps"]["session"] == {
        "train_test": [],
        "train_val": [],
        "val_test": [],
    }


def test_custom_manifest_hash_binds_external_evaluation_inputs(tmp_path):
    _write_split(tmp_path, "train", "train_session")
    _write_split(tmp_path, "val", "val_session")
    _write_split(tmp_path, "test", "test_session")
    segment = tmp_path / "segment.npz"
    segment.write_bytes(b"segment-v1")
    imu_csv = tmp_path / "imu.csv"
    imu_csv.write_text("timestamp,ax\n0,1\n", encoding="utf-8")

    first = build_prepared_data_manifest(
        tmp_path,
        dataset="custom",
        fold_id=1,
        evaluation_artifacts={"segment/segment.npz": segment, "raw_imu/imu.csv": imu_csv},
    )
    segment.write_bytes(b"segment-v2")
    second = build_prepared_data_manifest(
        tmp_path,
        dataset="custom",
        fold_id=1,
        evaluation_artifacts={"segment/segment.npz": segment, "raw_imu/imu.csv": imu_csv},
    )

    assert first["manifest_hash"] != second["manifest_hash"]
    assert first["content_summary"]["evaluation_artifact_files"] == 2
    assert set(first["evaluation_artifact_sha256"]) == {
        "raw_imu/imu.csv",
        "segment/segment.npz",
    }
