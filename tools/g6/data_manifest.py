"""Build deterministic manifests for prepared alignment-window datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_key(array: np.ndarray) -> str:
    return "x".join(str(size) for size in array.shape)


def _candidate_group_key(row: dict[str, str]) -> str:
    explicit = str(row.get("candidate_group_id", "")).strip()
    if explicit:
        return explicit
    source_sequence = str(row.get("source_sequence") or row.get("npz_path", ""))
    source_start = str(row.get("source_window_start") or row.get("window_start", ""))
    return ":".join((source_sequence, source_start, str(row.get("window_end", ""))))


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_prepared_data_manifest(
    root: str | Path,
    *,
    dataset: str,
    fold_id: int | None = None,
    evaluation_artifacts: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Inspect and fingerprint a prepared train/val/test cache.

    The manifest hash excludes the absolute root so relocating byte-identical
    artifacts does not change their identity.
    """

    prepared_root = Path(root).expanduser().resolve()
    if not prepared_root.is_dir():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")

    split_rows: dict[str, list[dict[str, str]]] = {}
    csv_hashes: dict[str, str] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        csv_path = prepared_root / f"windows_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Prepared split is empty: {csv_path}")
        split_rows[split] = rows
        csv_hashes[csv_path.name] = sha256_file(csv_path)
        split_stats[split] = {
            "rows": len(rows),
            "sessions": sorted({str(row.get("session", "")) for row in rows}),
            "subjects": sorted({str(row.get("subject", "")) for row in rows}),
            "source_sequences": len({str(row.get("source_sequence", "")) for row in rows}),
            "source_persons": len(
                {
                    (str(row.get("source_sequence", "")), str(row.get("source_person", "")))
                    for row in rows
                }
            ),
            "window_lengths": _counter_dict(Counter(str(row.get("window_len", "")) for row in rows)),
        }

    def overlap_by(field: str) -> dict[str, list[str]]:
        values = {
            split: {str(row.get(field, "")) for row in rows if str(row.get(field, ""))}
            for split, rows in split_rows.items()
        }
        return {
            f"{left}_{right}": sorted(values[left] & values[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }

    overlaps = {
        "session": overlap_by("session"),
        "subject": overlap_by("subject"),
        "source_sequence": overlap_by("source_sequence"),
    }
    split_identity = "subject" if dataset == "totalcapture" else "session"
    invalid_overlaps = overlaps[split_identity]
    if any(invalid_overlaps.values()):
        raise ValueError(
            f"Prepared {dataset} cache has {split_identity} leakage: {invalid_overlaps}"
        )
    if any(overlaps["source_sequence"].values()):
        raise ValueError(
            f"Prepared {dataset} cache has source_sequence leakage: {overlaps['source_sequence']}"
        )

    relative_npz_paths = sorted(
        {str(row["npz_path"]) for rows in split_rows.values() for row in rows}
    )
    npz_hashes: dict[str, str] = {}
    imu_shapes: Counter[str] = Counter()
    skeleton_shapes: Counter[str] = Counter()
    zero_imu_files = 0
    zero_skeleton_files = 0
    for relative in relative_npz_paths:
        npz_path = (prepared_root / relative).resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"Prepared cache references missing NPZ: {npz_path}")
        npz_hashes[relative] = sha256_file(npz_path)
        with np.load(npz_path, allow_pickle=True) as payload:
            if "imu" not in payload.files:
                raise ValueError(f"Prepared window has no IMU: {npz_path}")
            skeleton_key = "gt_skeleton" if "gt_skeleton" in payload.files else "skeleton"
            if skeleton_key not in payload.files:
                raise ValueError(f"Prepared window has no skeleton: {npz_path}")
            imu = np.asarray(payload["imu"])
            skeleton = np.asarray(payload[skeleton_key])
            if not np.isfinite(imu).all() or not np.isfinite(skeleton).all():
                raise ValueError(f"Prepared window contains non-finite values: {npz_path}")
            imu_shapes[_shape_key(imu)] += 1
            skeleton_shapes[_shape_key(skeleton)] += 1
            zero_imu_files += int(not np.any(imu))
            zero_skeleton_files += int(not np.any(skeleton))

    test_group_sizes = Counter(_candidate_group_key(row) for row in split_rows["test"])
    group_size_distribution = Counter(test_group_sizes.values())
    singleton_groups = sum(size == 1 for size in test_group_sizes.values())
    candidate_groups = {
        "count": len(test_group_sizes),
        "size_distribution": _counter_dict(group_size_distribution),
        "singleton_groups": singleton_groups,
        "singleton_rate": singleton_groups / len(test_group_sizes) if test_group_sizes else 0.0,
    }

    evaluation_hashes: dict[str, str] = {}
    for name, artifact in sorted((evaluation_artifacts or {}).items()):
        artifact_path = Path(artifact).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Evaluation artifact does not exist: {artifact_path}")
        evaluation_hashes[str(name)] = sha256_file(artifact_path)

    identity = {
        "schema_version": "1.0",
        "dataset": dataset,
        "fold_id": fold_id,
        "csv_sha256": csv_hashes,
        "npz_sha256": npz_hashes,
        "evaluation_artifact_sha256": evaluation_hashes,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    return {
        **identity,
        "manifest_hash": manifest_hash,
        "prepared_root": str(prepared_root),
        "split_stats": split_stats,
        "split_identity": split_identity,
        "split_overlaps": overlaps,
        "candidate_groups_test": candidate_groups,
        "content_summary": {
            "npz_files": len(relative_npz_paths),
            "imu_shapes": _counter_dict(imu_shapes),
            "skeleton_shapes": _counter_dict(skeleton_shapes),
            "all_finite": True,
            "zero_imu_files": zero_imu_files,
            "zero_skeleton_files": zero_skeleton_files,
            "evaluation_artifact_files": len(evaluation_hashes),
        },
    }
