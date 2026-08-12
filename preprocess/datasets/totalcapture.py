"""Local TotalCapture preprocess entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from preprocess.common.config import resolve_config
from preprocess.common.extract import resolve_extract_config, run_extraction_if_enabled
from preprocess.common.sequence import write_sequence_meta, write_sequence_npz
from preprocess.common.slice import (
    convert_imu_to_48,
    map_totalcapture21_to_h36m17,
    normalize_skeleton,
    parse_sensor_order,
    parse_vicon_pos,
    parse_xsens_sensors,
)
from preprocess.common.video import find_video_for_sequence, get_video_resolution, write_video_manifest

__all__ = ["run_preprocess", "main"]


def run_preprocess(config_path: str | Path | None, output_dir: str | Path | None = None, manifest_csv: str | Path | None = None) -> Path:
    cfg = resolve_config(config_path)
    preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess"), dict) else {}
    raw_root = Path(preprocess_cfg.get("raw_root", "/data/fzliang/totalcapture")).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(preprocess_cfg.get("output", str(raw_root / "preprocessed"))).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = resolved_output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_csv).expanduser().resolve() if manifest_csv is not None else resolved_output_dir / "video_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    extract_cfg = resolve_extract_config(cfg)

    camera = str(preprocess_cfg.get("camera", "cam1"))
    sensor_order = parse_sensor_order(preprocess_cfg.get("sensor_order"))
    imu_cfg = preprocess_cfg.get("imu", {}) if isinstance(preprocess_cfg.get("imu"), dict) else {}
    imu_output_format = str(imu_cfg.get("output_format", "legacy_48d")).strip().lower()
    imu_sensor = str(imu_cfg.get("sensor", "L_LowArm")).strip()
    if imu_output_format not in {"7d", "legacy_48d"}:
        raise ValueError(f"Unsupported preprocess.imu.output_format={imu_output_format!r}")
    if imu_output_format == "7d" and imu_sensor not in sensor_order:
        raise ValueError(f"IMU sensor {imu_sensor!r} is not in sensor_order={sensor_order}")
    sequences: list[tuple[str, str, Path, Path]] = []
    for subject_dir in sorted(raw_root.glob("S[1-5]")):
        subject = subject_dir.name
        imu_subject = subject.lower()
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name
            vicon_path = session_dir / "gt_skel_gbl_pos.txt"
            imu_path = raw_root / imu_subject / f"{imu_subject}_{session}_Xsens.sensors"
            if vicon_path.exists() and imu_path.exists():
                sequences.append((subject, session, vicon_path, imu_path))

    video_paths: list[str] = []
    for subject, session, vicon_path, imu_path in sequences:
        sequence_id = f"totalcapture_{subject}_{session}_{camera}"
        quat4, acc3 = parse_xsens_sensors(imu_path, sensor_order)
        joint_names, xyz21 = parse_vicon_pos(vicon_path)
        skel17 = map_totalcapture21_to_h36m17(joint_names, xyz21)
        tlen = min(skel17.shape[0], quat4.shape[0])
        skel17 = skel17[:tlen]
        quat4 = quat4[:tlen]
        acc3 = acc3[:tlen]
        if imu_output_format == "7d":
            sensor_idx = sensor_order.index(imu_sensor)
            imu_values = np.concatenate(
                [acc3[:, sensor_idx, :], quat4[:, sensor_idx, :]],
                axis=-1,
            ).astype(np.float32)
            imu_channels = ["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"]
        else:
            imu_values = convert_imu_to_48(quat4, acc3)
            imu_channels = [f"legacy_{idx}" for idx in range(48)]
        skel17_meters = skel17.copy().astype(np.float32)
        skel17 = normalize_skeleton(skel17)

        video_path = find_video_for_sequence(raw_root, subject, session, camera)
        if video_path is not None and video_path.exists():
            video_paths.append(str(video_path))
            w, h = get_video_resolution(video_path)
            gt_bboxes = np.tile(np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32), (tlen, 1, 1))
            if extract_cfg:
                run_extraction_if_enabled(video_path, resolved_output_dir / "extracts" / sequence_id, extract_cfg)
        else:
            gt_bboxes = np.zeros((tlen, 1, 4), dtype=np.float32)

        frame_ids = np.arange(tlen, dtype=np.int64)
        imu = imu_values[:, np.newaxis, :].astype(np.float32)
        imu_ids = np.array([0], dtype=np.int64)
        gt_person_ids = np.array([0], dtype=np.int64)
        gt_skeleton = skel17[:, np.newaxis, :, :].astype(np.float32)
        gt_visibility = np.ones((tlen, 1), dtype=bool)
        payload = {
            "schema_version": np.array("1.0", dtype=object),
            "video_path": np.array(str(video_path) if video_path is not None else "", dtype=object),
            "dataset": np.array("totalcapture", dtype=object),
            "sequence_id": np.array(sequence_id, dtype=object),
            "frame_ids": frame_ids,
            "imu": imu,
            "imu_channels": np.asarray(imu_channels, dtype=object),
            "imu_location": np.array(imu_sensor if imu_output_format == "7d" else "multi_sensor_legacy", dtype=object),
            "imu_ids": imu_ids,
            "gt_person_ids": gt_person_ids,
            "gt_bboxes": gt_bboxes,
            "gt_visibility": gt_visibility,
            "gt_skeleton": gt_skeleton,
            "gt_skeleton_meters": skel17_meters[:, np.newaxis, :, :].astype(np.float32),
        }
        write_sequence_npz(sequences_dir / f"{sequence_id}.npz", payload)
        write_sequence_meta(sequences_dir / f"{sequence_id}.json", {"dataset": "totalcapture", "sequence_id": sequence_id, "video_path": str(video_path) if video_path is not None else ""})

    write_video_manifest(manifest_path, video_paths)
    print(f"Wrote {len(sequences)} totalcapture sequence NPZ files to {sequences_dir}")
    print(f"Manifest: {manifest_path}")
    return resolved_output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the TotalCapture dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--manifest_csv", default=None)
    args = parser.parse_args(argv)
    run_preprocess(args.config, output_dir=args.output_dir, manifest_csv=args.manifest_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
