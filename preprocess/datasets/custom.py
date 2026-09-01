"""Local custom-dataset preprocess entrypoint.

This mirrors the Autism-project preprocessing flow for the custom dataset:
1. parse raw IMU CSVs,
2. convert them to 7D or 48D features,
3. align to the available video/annotation timeline,
4. write per-sequence NPZ files compatible with the later slice stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.config import load_config
from preprocess.common.extract import resolve_extract_config, run_extraction_if_enabled
from preprocess.common.imu import (
    convert_single_imu_to_7d,
    convert_single_imu_to_48,
    lowpass_filter_fft,
    parse_imu_csv,
    parse_imu_csv_with_gyro,
    resample_imu_to_target,
    rotmat_to_quat_wxyz,
)
from preprocess.common.imu_conditioning import condition_imu
from preprocess.common.sequence import write_sequence_meta, write_sequence_npz
from preprocess.common.video import find_video_for_sequence, get_video_resolution, write_video_manifest

__all__ = [
    "legacy_imu48_sensor_to_7d",
    "load_custom_rawcsv_7d_sequence",
    "load_custom_split_7d_sequence",
    "main",
    "run_preprocess",
]


def legacy_imu48_sensor_to_7d(imu48: np.ndarray, sensor_name: str = "L_LowArm") -> np.ndarray:
    """Extract ``acc(3) + quaternion-wxyz(4)`` from legacy custom 48D IMU."""
    order = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
    if sensor_name not in order:
        raise ValueError(f"Unsupported legacy sensor {sensor_name!r}; expected one of {order}")
    values = np.asarray(imu48, dtype=np.float32)
    if values.shape[-1] < 48:
        raise ValueError(f"Expected legacy 48D IMU, got shape {values.shape}")
    sensor_idx = order.index(sensor_name)
    rotation = values[..., sensor_idx * 9 : (sensor_idx + 1) * 9].reshape(*values.shape[:-1], 3, 3)
    acceleration = values[..., 36 + sensor_idx * 3 : 36 + (sensor_idx + 1) * 3]
    return np.concatenate([acceleration, rotmat_to_quat_wxyz(rotation)], axis=-1).astype(np.float32)


def load_custom_split_7d_sequence(
    root: Path,
    session: str,
    seg_idx: int,
    person: int,
    target_len: int | None = None,
) -> np.ndarray:
    """Reconstruct a chronological custom segment from historical split files."""
    parts: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        path = root / f"{session}_seg{seg_idx}_person{person}_{split}.npy"
        if not path.exists():
            parts = []
            break
        values = np.load(path, allow_pickle=True).item()["imu"].astype(np.float32)
        if values.ndim == 3:
            values = values[:, 0, :]
        parts.append(values[:, :7])
    if not parts:
        path = root / f"{session}_seg{seg_idx}_person{person}_test.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        values = np.load(path, allow_pickle=True).item()["imu"].astype(np.float32)
        if values.ndim == 3:
            values = values[:, 0, :]
        parts = [values[:, :7]]

    full = np.concatenate(parts, axis=0).astype(np.float32)
    if target_len is None:
        return full
    if len(full) >= target_len:
        return full[:target_len]
    return np.pad(full, ((0, target_len - len(full)), (0, 0))).astype(np.float32)


def load_custom_rawcsv_7d_sequence(
    root: Path,
    session: str,
    frame_ids: np.ndarray,
    imu_person_map: str | dict | None = None,
    n_persons: int = 2,
) -> np.ndarray:
    """Load a raw Custom session and align each person's IMU to video frames."""
    session_dir = Path(root) / session
    timestamp_path = session_dir / "video" / f"{session}_frame_timestamps_retimed.csv"
    if not timestamp_path.exists():
        timestamp_path = session_dir / "video" / f"{session}_frame_timestamps.csv"
    if not timestamp_path.exists():
        raise FileNotFoundError(timestamp_path)

    with timestamp_path.open("r", newline="", encoding="utf-8-sig") as handle:
        timestamp_rows = list(csv.DictReader(handle))
    if not timestamp_rows:
        raise ValueError(f"Empty timestamp file: {timestamp_path}")
    frame_to_timestamp = {
        int(row["frame_index"]) - 1: float(row["timestamp_ms"])
        for row in timestamp_rows
    }
    missing_frames = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in frame_to_timestamp]
    if missing_frames:
        raise KeyError(f"Frame ids missing from {timestamp_path}: {missing_frames[:10]}")
    target_timestamps = np.asarray([frame_to_timestamp[int(frame_id)] for frame_id in frame_ids], dtype=np.float64)

    mapping: dict[str, int] = {}
    if isinstance(imu_person_map, str) and imu_person_map.strip():
        mapping = {str(key): int(value) for key, value in json.loads(imu_person_map).items()}
    elif isinstance(imu_person_map, dict):
        mapping = {str(key): int(value) for key, value in imu_person_map.items()}

    imu_dir = session_dir / "imu"
    imu_paths = sorted(imu_dir.glob(f"{session}_*.csv"))
    if not imu_paths:
        raise FileNotFoundError(f"No IMU CSV files found under {imu_dir}")

    output = np.zeros((len(frame_ids), n_persons, 7), dtype=np.float32)
    used = np.zeros(n_persons, dtype=bool)
    fallback_person = 0
    for imu_path in imu_paths:
        mac = imu_path.stem.replace(f"{session}_", "")
        if mac in mapping:
            person = mapping[mac]
        else:
            while fallback_person < n_persons and used[fallback_person]:
                fallback_person += 1
            if fallback_person >= n_persons:
                continue
            person = fallback_person
        if not 0 <= person < n_persons:
            raise ValueError(f"Invalid person index {person} for IMU {imu_path}")
        imu_timestamps, quat4, acc3 = parse_imu_csv(imu_path)
        imu7 = convert_single_imu_to_7d(quat4, acc3)
        aligned = resample_imu_to_target(imu_timestamps, imu7, target_timestamps)
        output[:, person] = np.nan_to_num(aligned, nan=0.0)
        used[person] = True

    if not used.all():
        missing = np.where(~used)[0].tolist()
        raise ValueError(f"Missing raw CSV IMU for session={session}, person indices={missing}")
    return output


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _infer_imu_files(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    return sorted([p for p in raw_root.rglob("*.csv") if p.is_file() and any(token in p.name.lower() for token in ["imu", "sensor", "motion"])])


def run_preprocess(config_path: str | Path | None, output_dir: str | Path | None = None, manifest_csv: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess"), dict) else {}

    raw_root = Path(preprocess_cfg.get("raw_root", "/data/fzliang/custom")).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(preprocess_cfg.get("output", str(Path(raw_root) / "preprocessed"))).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = resolved_output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_csv).expanduser().resolve() if manifest_csv is not None else resolved_output_dir / "video_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    imu_cfg = preprocess_cfg.get("imu", {}) if isinstance(preprocess_cfg.get("imu"), dict) else {}
    lowpass_cutoff = float(imu_cfg.get("lowpass_cutoff_hz", 0.0) or 0.0)
    lowpass_fs_hz = float(imu_cfg.get("lowpass_fs_hz", 100.0) or 100.0)
    use_48d = bool(imu_cfg.get("use_48d", True))
    conditioner = str(imu_cfg.get("conditioner", "identity") or "identity").strip().lower()
    conditioner = {"rg23": "madgwick6", "madgwick": "madgwick6"}.get(conditioner, conditioner)
    if conditioner not in {"identity", "madgwick6"}:
        raise ValueError(f"Unsupported preprocess.imu.conditioner={conditioner!r}")
    conditioner_beta = float(imu_cfg.get("conditioner_beta", 0.033))
    skeleton_normalize = _parse_bool(preprocess_cfg.get("skeleton_normalize"), default=False)
    extract_cfg = resolve_extract_config(cfg)

    imu_files = _infer_imu_files(raw_root)
    if not imu_files:
        raise FileNotFoundError(f"No IMU CSV files found under {raw_root}")

    video_paths: list[str] = []
    sequence_payloads: list[dict[str, Any]] = []

    for imu_path in imu_files:
        stem = imu_path.stem
        sequence_id = f"custom_{stem}"
        if not imu_path.exists():
            continue
        try:
            if conditioner == "madgwick6":
                timestamps_ms, quat4, acc3, gyro3 = parse_imu_csv_with_gyro(imu_path)
                quat4 = condition_imu(conditioner, timestamps_ms, quat4, acc3, gyro3, conditioner_beta)
            else:
                timestamps_ms, quat4, acc3 = parse_imu_csv(imu_path)
                quat4 = condition_imu(conditioner, timestamps_ms, quat4, acc3, beta=conditioner_beta)
        except Exception as exc:
            print(f"[WARN] Failed to parse {imu_path}: {exc}")
            continue

        output_acc = acc3
        imu_7d = convert_single_imu_to_7d(quat4, output_acc)
        if lowpass_cutoff > 0:
            if conditioner == "identity":
                # Keep the historical identity orientation while filtering
                # only the acceleration channels.
                imu_7d = lowpass_filter_fft(imu_7d, lowpass_cutoff, lowpass_fs_hz)
                output_acc = imu_7d[:, :3]
            else:
                output_acc = lowpass_filter_fft(output_acc, lowpass_cutoff, lowpass_fs_hz)
                imu_7d = convert_single_imu_to_7d(quat4, output_acc)
        imu_feat = convert_single_imu_to_48(quat4, output_acc) if use_48d else imu_7d

        tlen = imu_feat.shape[0]
        frame_ids = np.arange(tlen, dtype=np.int64)
        imu = imu_feat[:, np.newaxis, :].astype(np.float32)
        imu_ids = np.array([0], dtype=np.int64)
        gt_person_ids = np.array([0], dtype=np.int64)
        gt_bboxes = np.zeros((tlen, 1, 4), dtype=np.float32)
        gt_visibility = np.ones((tlen, 1), dtype=bool)
        gt_skeleton = np.zeros((tlen, 1, 17, 3), dtype=np.float32)
        extracted_path: Path | None = None

        video_path = find_video_for_sequence(raw_root, stem, stem, "cam1")
        if video_path is not None and video_path.exists():
            video_paths.append(str(video_path))
            try:
                w, h = get_video_resolution(video_path)
                gt_bboxes = np.tile(np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32), (tlen, 1, 1))
            except Exception:
                pass
            if extract_cfg:
                extracted_path = run_extraction_if_enabled(
                    video_path,
                    resolved_output_dir / "extracts" / sequence_id,
                    extract_cfg,
                )

        payload = {
            "schema_version": np.array("1.0", dtype=object),
            "video_path": np.array(str(video_path) if video_path is not None else "", dtype=object),
            "dataset": np.array("custom", dtype=object),
            "sequence_id": np.array(sequence_id, dtype=object),
            "frame_ids": frame_ids,
            "imu": imu,
            "imu_channels": np.asarray(
                ["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"],
                dtype=object,
            ) if not use_48d else np.asarray([f"legacy_{idx}" for idx in range(48)], dtype=object),
            "imu_location": np.array("custom_single_sensor", dtype=object),
            "imu_ids": imu_ids,
            "gt_person_ids": gt_person_ids,
            "gt_bboxes": gt_bboxes,
            "gt_visibility": gt_visibility,
            "gt_skeleton": gt_skeleton,
            "gt_skeleton_meters": gt_skeleton.copy(),
            "extract_person_ids": np.array([0], dtype=np.int64),
            "extract_bboxes": gt_bboxes.copy(),
            "extract_visibility": np.ones((tlen, 1), dtype=bool),
            "extract_skeleton": gt_skeleton.copy(),
            "gt_to_extract_map": np.zeros((tlen, 1), dtype=np.int64),
            "imu_person_map": np.array(json.dumps({"default": 0}), dtype=object),
        }
        if extracted_path is not None:
            payload["extract_skeleton_path"] = np.array(str(extracted_path), dtype=object)
        if skeleton_normalize:
            payload["gt_skeleton"] = payload["gt_skeleton"].astype(np.float32)
        out_path = sequences_dir / f"{sequence_id}.npz"
        write_sequence_npz(out_path, payload)
        meta = {
            "dataset": "custom",
            "sequence_id": sequence_id,
            "source_imu_csv": str(imu_path),
            "video_path": str(video_path) if video_path is not None else "",
            "n_frames": int(tlen),
            "imu_dim": int(imu.shape[-1]),
            "imu_conditioner": conditioner,
            "imu_conditioner_beta": conditioner_beta,
        }
        write_sequence_meta(out_path.with_suffix(".json"), meta)
        sequence_payloads.append(payload)

    write_video_manifest(manifest_path, video_paths)
    print(f"Wrote {len(sequence_payloads)} custom sequence NPZ files to {sequences_dir}")
    print(f"Manifest: {manifest_path}")
    return resolved_output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the custom dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--manifest_csv", default=None)
    args = parser.parse_args(argv)
    run_preprocess(args.config, output_dir=args.output_dir, manifest_csv=args.manifest_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
