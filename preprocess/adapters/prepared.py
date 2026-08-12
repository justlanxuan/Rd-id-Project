"""Validation for reusable train/validation/test window caches."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def validate_prepared_dataset(
    root_dir: str | Path,
    *,
    expected_test_sessions: set[str] | None = None,
    allow_singleton_test_groups: bool = False,
) -> Path:
    root = Path(root_dir).expanduser().resolve()
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        csv_path = root / f"windows_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Prepared dataset is missing {csv_path}")
        with csv_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Prepared dataset split is empty: {csv_path}")
        split_rows[split] = rows

    sessions = {split: {str(row.get("session", "")) for row in rows} for split, rows in split_rows.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sessions[left] & sessions[right]
        if overlap:
            raise ValueError(f"Prepared dataset session leakage between {left}/{right}: {sorted(overlap)}")
    if expected_test_sessions is not None and sessions["test"] != expected_test_sessions:
        raise ValueError(
            f"Prepared test sessions {sorted(sessions['test'])} do not match expected {sorted(expected_test_sessions)}"
        )

    unique_npz = sorted({str(row["npz_path"]) for rows in split_rows.values() for row in rows})
    for relative_path in unique_npz:
        npz_path = (root / relative_path).resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"Prepared dataset references missing NPZ: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            if "imu" not in data.files:
                raise ValueError(f"Prepared window has no IMU: {npz_path}")
            skeleton_key = "gt_skeleton" if "gt_skeleton" in data.files else "skeleton"
            if skeleton_key not in data.files:
                raise ValueError(f"Prepared window has no skeleton: {npz_path}")
            imu = np.asarray(data["imu"])
            skeleton = np.asarray(data[skeleton_key])
            if imu.shape[0] <= 0 or skeleton.shape[0] != imu.shape[0]:
                raise ValueError(f"Prepared window is temporally misaligned: {npz_path}, {imu.shape}, {skeleton.shape}")
            if imu.shape[-1] < 7:
                raise ValueError(f"Prepared window IMU has fewer than 7 channels: {npz_path}, {imu.shape}")
            if not np.isfinite(imu).all() or not np.isfinite(skeleton).all():
                raise ValueError(f"Prepared window contains non-finite values: {npz_path}")

    test_groups: dict[tuple[str, str, str], int] = {}
    for row in split_rows["test"]:
        source_sequence = str(row.get("source_sequence") or row.get("npz_path", ""))
        source_start = str(row.get("source_window_start") or row.get("window_start", ""))
        key = (source_sequence, source_start, str(row.get("window_end", "")))
        test_groups[key] = test_groups.get(key, 0) + 1
    singleton_groups = [key for key, size in test_groups.items() if size < 2]
    if singleton_groups and not allow_singleton_test_groups:
        raise ValueError(
            f"Prepared test split has {len(singleton_groups)} singleton FrameAcc groups; first={singleton_groups[0]}"
        )
    return root
