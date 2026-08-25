"""Validation for reusable train/validation/test window caches."""

from __future__ import annotations

import csv
import math
from functools import reduce
from pathlib import Path

import numpy as np


def validate_prepared_dataset(
    root_dir: str | Path,
    *,
    expected_test_sessions: set[str] | None = None,
    split_identity: str = "session",
    expected_test_values: set[str] | None = None,
    expected_window_len: int | None = None,
    expected_stride: int | None = None,
    allow_singleton_test_groups: bool = False,
    allow_empty_validation: bool = False,
) -> Path:
    root = Path(root_dir).expanduser().resolve()
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        csv_path = root / f"windows_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Prepared dataset is missing {csv_path}")
        with csv_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows and not (split == "val" and allow_empty_validation):
            raise ValueError(f"Prepared dataset split is empty: {csv_path}")
        split_rows[split] = rows

    _validate_window_contract(
        split_rows,
        expected_window_len=expected_window_len,
        expected_stride=expected_stride,
    )

    if expected_test_sessions is not None and expected_test_values is not None:
        raise ValueError("Use expected_test_sessions or expected_test_values, not both")
    expected = expected_test_values if expected_test_values is not None else expected_test_sessions
    identities = {
        split: {str(row.get(split_identity, "")) for row in rows if str(row.get(split_identity, ""))}
        for split, rows in split_rows.items()
    }
    if any(not values for split, values in identities.items() if not (split == "val" and allow_empty_validation)):
        raise ValueError(f"Prepared dataset has an empty {split_identity} split identity")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(
                f"Prepared dataset {split_identity} leakage between {left}/{right}: {sorted(overlap)}"
            )
    if expected is not None and identities["test"] != expected:
        raise ValueError(
            f"Prepared test {split_identity} values {sorted(identities['test'])} "
            f"do not match expected {sorted(expected)}"
        )

    source_sequences = {
        split: {str(row.get("source_sequence", "")) for row in rows if str(row.get("source_sequence", ""))}
        for split, rows in split_rows.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = source_sequences[left] & source_sequences[right]
        if overlap:
            raise ValueError(
                f"Prepared dataset source_sequence leakage between {left}/{right}: {sorted(overlap)}"
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

    test_groups: dict[tuple[str, ...], int] = {}
    for row in split_rows["test"]:
        explicit_group = str(row.get("candidate_group_id", "")).strip()
        if explicit_group:
            key = ("candidate_group_id", explicit_group)
        else:
            source_sequence = str(row.get("source_sequence") or row.get("npz_path", ""))
            source_start = str(row.get("source_window_start") or row.get("window_start", ""))
            key = ("derived", source_sequence, source_start, str(row.get("window_end", "")))
        test_groups[key] = test_groups.get(key, 0) + 1
    singleton_groups = [key for key, size in test_groups.items() if size < 2]
    if singleton_groups and not allow_singleton_test_groups:
        raise ValueError(
            f"Prepared test split has {len(singleton_groups)} singleton FrameAcc groups; first={singleton_groups[0]}"
        )
    return root


def _validate_window_contract(
    split_rows: dict[str, list[dict[str, str]]],
    *,
    expected_window_len: int | None,
    expected_stride: int | None,
) -> None:
    """Check the actual CSV window geometry, not only the requested config."""
    for split, rows in split_rows.items():
        if not rows:
            continue
        if expected_window_len is not None:
            observed_lengths = {
                int(row["window_end"]) - int(row["window_start"])
                for row in rows
            }
            declared_lengths = {
                int(row["window_len"])
                for row in rows
                if str(row.get("window_len", "")).strip()
            }
            if observed_lengths != {expected_window_len} or (
                declared_lengths and declared_lengths != {expected_window_len}
            ):
                raise ValueError(
                    f"Prepared {split} window_len mismatch: expected={expected_window_len}, "
                    f"observed={sorted(observed_lengths)}, declared={sorted(declared_lengths)}"
                )

        if expected_stride is None:
            continue
        starts_by_stream: dict[str, set[int]] = {}
        for row in rows:
            source_sequence = str(row.get("source_sequence") or row.get("npz_path", ""))
            source_start = int(row.get("source_window_start") or row.get("window_start", ""))
            starts_by_stream.setdefault(source_sequence, set()).add(source_start)
        observed_deltas = {
            right - left
            for starts in starts_by_stream.values()
            for left, right in zip(sorted(starts), sorted(starts)[1:], strict=False)
        }
        inferred_stride = reduce(math.gcd, observed_deltas) if observed_deltas else None
        if inferred_stride is not None and inferred_stride != expected_stride:
            raise ValueError(
                f"Prepared {split} stride mismatch: expected={expected_stride}, "
                f"inferred={inferred_stride}, observed_deltas={sorted(observed_deltas)}"
            )
