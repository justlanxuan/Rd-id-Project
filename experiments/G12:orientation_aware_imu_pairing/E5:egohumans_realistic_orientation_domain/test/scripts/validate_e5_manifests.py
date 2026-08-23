# Experiment Note: E5-manifest-and-orientation-gate
"""Validate E5 session splits, native duration and orientation feature contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.g12.orientation_motion import OrientationMotionDataset


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_split(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    train = _rows(Path(data["train"]["csv"]))
    validation = _rows(Path(data["validation"]["csv"]))
    train_sessions = {row["session"] for row in train}
    validation_sessions = {row["session"] for row in validation}
    if train_sessions & validation_sessions:
        raise AssertionError("Ego train/validation sessions overlap")
    rows = train + validation
    if any(int(row["window_end"]) - int(row["window_start"]) != 16 for row in rows):
        raise AssertionError("Ego E5 manifest contains a non-16-frame native window")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
    }


def _check_orientation(spec: dict[str, str], mode: str, profile: str) -> dict[str, object]:
    dataset = OrientationMotionDataset([spec], orientation_mode=mode, orientation_profile=profile, target_len=24, window_seconds=0.8)
    activity_values = []
    for index in range(min(len(dataset), 64)):
        orientation = dataset[index]["orientation"].numpy()
        if orientation.shape != (24, 5) or not np.isfinite(orientation).all():
            raise AssertionError(f"Invalid orientation feature shape/value: {orientation.shape}")
        if profile == "rate" and not np.allclose(orientation[:, :2], 0.0):
            raise AssertionError("rate profile leaked absolute sin/cos heading")
        activity_values.append(float(orientation[:, 4].mean()))
    return {
        "rows": len(dataset),
        "mode": mode,
        "profile": profile,
        "orientation_shape": list(orientation.shape),
        "activity_fraction_first64_mean": float(np.mean(activity_values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--aligned-train", default=None, help="Optional source-aligned Ego train CSV")
    parser.add_argument("--aligned-root", default="/data/fzliang/reid-project/g12/e4_1_source_aligned/motionbert_alphapose_cache_v2")
    parser.add_argument("--ego-root", default="/data/fzliang/reid-project/egohumans/preprocessed/egohumans_realistic_hybrid_source")
    parser.add_argument("--ego-gyro", default="/data/fzliang/reid-project/g10/e1_global_features/gyro_sidecar_egohumans")
    args = parser.parse_args()
    splits = {str(Path(path)): _check_split(Path(path)) for path in args.split}
    spec = {
        "dataset": "egohumans_e5",
        "csv": args.split[0].replace("egohumans_e5_session_split.json", "egohumans_train.csv"),
        "root": args.ego_root,
        "fps_hz": 20.0,
        "gyro_sidecar_root": args.ego_gyro,
    }
    checks = {
        "splits": splits,
        "canonical_3d_full": _check_orientation(spec, "3d_heading", "full"),
        "canonical_3d_rate": _check_orientation(spec, "3d_heading", "rate"),
    }
    if args.aligned_train:
        aligned_spec = {
            "dataset": "egohumans_alphapose_e5",
            "csv": args.aligned_train,
            "root": args.aligned_root,
            "fps_hz": 20.0,
            "gyro_sidecar_root": args.ego_gyro,
        }
        checks["source_aligned_2d_proxy"] = _check_orientation(aligned_spec, "proxy", "full")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
