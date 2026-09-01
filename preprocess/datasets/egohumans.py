"""Local EgoHumans preprocess entrypoint.

Creates canonical sequence-level NPZ files from the prepared EgoHumans cache
and optionally writes a video manifest for downstream extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.config import load_config
from preprocess.common.sequence import write_sequence_meta, write_sequence_npz
from preprocess.common.video import write_video_manifest

__all__ = ["main", "run_preprocess"]


def _session_from_cache_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Unexpected EgoHumans cache filename: {path.name}")
    return f"{parts[0]}_{parts[1]}"


def _pose_bboxes(pose2d: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    t_len, n_people = pose2d.shape[:2]
    bboxes = np.zeros((t_len, n_people, 4), dtype=np.float32)
    for frame_idx in range(t_len):
        for person_idx in range(n_people):
            if not visibility[frame_idx, person_idx]:
                continue
            xy = pose2d[frame_idx, person_idx]
            valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 0)
            if not valid.any():
                continue
            points = xy[valid]
            bboxes[frame_idx, person_idx] = [
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            ]
    return bboxes


def run_preprocess(config_path: str | Path | None, output_dir: str | Path | None = None, manifest_csv: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess"), dict) else {}
    raw_root = Path(preprocess_cfg.get("raw_root", "/data/lyxie/ReID/Data/egohumans/data")).expanduser().resolve()
    extracted_root = Path(
        preprocess_cfg.get(
            "extracted_root",
            "/data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data",
        )
    ).expanduser().resolve()
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"EgoHumans realistic IMU cache not found: {extracted_root}")
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(preprocess_cfg.get("output", str(raw_root / "preprocessed"))).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = resolved_output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_csv).expanduser().resolve() if manifest_csv is not None else resolved_output_dir / "video_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    del raw_root
    slice_cfg = cfg.get("slice", {}) if isinstance(cfg.get("slice"), dict) else {}
    selected_sessions = {
        str(session)
        for key in ("train_sessions", "val_sessions", "test_sessions")
        for session in _as_list(slice_cfg.get(key, []))
    }
    grouped: dict[str, list[Path]] = {}
    for cache_path in sorted(extracted_root.glob("*.npy")):
        session = _session_from_cache_path(cache_path)
        if selected_sessions and session not in selected_sessions:
            continue
        grouped.setdefault(session, []).append(cache_path)
    if not grouped:
        raise ValueError(f"No EgoHumans cache files matched configured sessions under {extracted_root}")

    sensor_name = str(preprocess_cfg.get("imu", {}).get("sensor", "LeftWrist"))
    conditioner = str(preprocess_cfg.get("imu", {}).get("conditioner", "identity") or "identity").strip().lower()
    if conditioner not in {"", "identity"}:
        raise ValueError(
            "EgoHumans preprocessing currently supports only the identity IMU conditioner; "
            "RG23 requires raw native-rate gyro data, which is not exposed by this cache adapter."
        )
    for session, person_paths in sorted(grouped.items()):
        person_records = [np.load(path, allow_pickle=True).item() for path in person_paths]
        tlen = min(int(record["pose2d"].shape[0]) for record in person_records)
        if tlen <= 0:
            raise ValueError(f"EgoHumans session {session} has no aligned frames")
        imu_people = []
        pose_people = []
        visibility_people = []
        person_ids = []
        for person_path, record in zip(person_paths, person_records, strict=True):
            metadata = record.get("metadata", {})
            sensor_names = [str(name) for name in metadata.get("sensor_names", [])]
            if sensor_name not in sensor_names:
                raise ValueError(f"Sensor {sensor_name!r} not found in {person_path}; available={sensor_names}")
            sensor_idx = sensor_names.index(sensor_name)
            acc = np.asarray(record["acc"][:tlen, sensor_idx], dtype=np.float32)
            quat = np.asarray(record["quat"][:tlen, sensor_idx], dtype=np.float32)
            imu_people.append(np.concatenate([acc, quat], axis=-1))
            pose_people.append(np.asarray(record["pose2d"][:tlen], dtype=np.float32))
            visibility_people.append(np.asarray(record["pose2d_mask"][:tlen] > 0, dtype=bool))
            person_ids.append(int(person_path.stem.rsplit("aria", 1)[1]))

        imu = np.stack(imu_people, axis=1)
        pose2d = np.stack(pose_people, axis=1)
        visibility = np.stack(visibility_people, axis=1)
        skeleton = np.zeros((*pose2d.shape[:-1], 3), dtype=np.float32)
        skeleton[..., :2] = pose2d
        skeleton[..., 2] = visibility[:, :, None]
        bboxes = _pose_bboxes(pose2d, visibility)
        sequence_id = f"egohumans_{session}"
        payload = {
            "schema_version": np.array("1.0", dtype=object),
            "video_path": np.array("", dtype=object),
            "dataset": np.array("egohumans", dtype=object),
            "sequence_id": np.array(sequence_id, dtype=object),
            "frame_ids": np.arange(tlen, dtype=np.int64),
            "imu": imu.astype(np.float32),
            "imu_channels": np.asarray(
                ["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"],
                dtype=object,
            ),
            "imu_location": np.array(sensor_name, dtype=object),
            "imu_ids": np.asarray(person_ids, dtype=np.int64),
            "gt_person_ids": np.asarray(person_ids, dtype=np.int64),
            "gt_bboxes": bboxes,
            "gt_visibility": visibility,
            "gt_skeleton": skeleton,
        }
        write_sequence_npz(sequences_dir / f"{sequence_id}.npz", payload)
        write_sequence_meta(
            sequences_dir / f"{sequence_id}.json",
            {
                "dataset": "egohumans",
                "sequence_id": sequence_id,
                "source_files": [str(path) for path in person_paths],
                "imu_location": sensor_name,
                "n_frames": tlen,
                "n_people": len(person_ids),
            },
        )

    write_video_manifest(manifest_path, [])
    print(f"Wrote {len(grouped)} egohumans sequence NPZ files to {sequences_dir}")
    print(f"Manifest: {manifest_path}")
    return resolved_output_dir


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the EgoHumans dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--manifest_csv", default=None)
    args = parser.parse_args(argv)
    run_preprocess(args.config, output_dir=args.output_dir, manifest_csv=args.manifest_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
