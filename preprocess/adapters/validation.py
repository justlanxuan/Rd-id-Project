"""Validation at the raw-adapter/canonical-sequence boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED_ARRAYS = (
    "schema_version",
    "dataset",
    "sequence_id",
    "frame_ids",
    "imu",
    "gt_skeleton",
)


def validate_preprocess_output(dataset: str, output_dir: str | Path) -> Path:
    """Reject empty, malformed or obvious placeholder adapter output."""
    output = Path(output_dir).expanduser().resolve()
    sequence_dir = output / "sequences"
    paths = sorted(sequence_dir.glob("*.npz")) if sequence_dir.is_dir() else []
    if not paths:
        raise ValueError(f"{dataset} adapter produced no sequence NPZ files under {sequence_dir}")

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in REQUIRED_ARRAYS if key not in data.files]
            if missing:
                raise ValueError(f"Canonical sequence {path} is missing fields: {missing}")
            declared_dataset = str(data["dataset"].item())
            if declared_dataset != dataset:
                raise ValueError(f"Canonical sequence {path} declares dataset={declared_dataset!r}, expected {dataset!r}")
            frame_ids = np.asarray(data["frame_ids"])
            imu = np.asarray(data["imu"])
            skeleton = np.asarray(data["gt_skeleton"])
            if frame_ids.ndim != 1 or len(frame_ids) == 0:
                raise ValueError(f"Canonical sequence {path} has invalid frame_ids shape {frame_ids.shape}")
            if len(frame_ids) > 1 and not np.all(np.diff(frame_ids.astype(np.float64)) > 0):
                raise ValueError(
                    f"Canonical sequence {path} has non-monotonic or duplicate frame_ids"
                )
            if imu.shape[0] != len(frame_ids) or skeleton.shape[0] != len(frame_ids):
                raise ValueError(
                    f"Canonical sequence {path} is temporally misaligned: "
                    f"frames={len(frame_ids)}, imu={imu.shape}, skeleton={skeleton.shape}"
                )
            if not np.isfinite(imu).all() or not np.isfinite(skeleton).all():
                raise ValueError(f"Canonical sequence {path} contains non-finite IMU or skeleton values")
            if not np.any(np.abs(imu) > 0):
                raise ValueError(f"Canonical sequence {path} contains placeholder all-zero IMU")
            skeleton_candidates = [skeleton]
            if "extract_skeleton" in data.files:
                skeleton_candidates.append(np.asarray(data["extract_skeleton"]))
            if not any(np.any(np.abs(candidate) > 0) for candidate in skeleton_candidates):
                raise ValueError(f"Canonical sequence {path} contains placeholder all-zero skeletons")
    return output
