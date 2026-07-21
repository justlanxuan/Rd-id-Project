"""TotalCapture-specific preprocess entrypoint."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.common.slice import (
    convert_imu_to_48,
    map_totalcapture21_to_h36m17,
    normalize_skeleton,
    parse_sensor_order,
    parse_vicon_pos,
    parse_xsens_sensors,
)
from src.preprocess.common.video import find_video_for_sequence, get_video_resolution, write_video_manifest
from src.utils.config import load_config


def load_preprocess_cfg(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    data = load_config(str(config_path))
    preprocess = data.get("preprocess", {})
    if preprocess is None:
        return {}
    if not isinstance(preprocess, dict):
        raise ValueError(f"Invalid preprocess section in config: {config_path}")
    return preprocess


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_synthetic_root(root: Path, side: str | None) -> Path:
    if not side:
        return root
    token = str(side).strip().lower()
    candidates = [root / token, root / token.upper(), root / token.capitalize()]
    for cand in candidates:
        if cand.exists():
            return cand
    return root


def _collect_synthetic_npzs(root: Path, side: str | None) -> list[Path]:
    resolved = _resolve_synthetic_root(root, side)
    direct = sorted(resolved.glob("*.npz"))
    if direct:
        return direct
    if side:
        token = str(side).strip().lower()
        return sorted(p for p in resolved.rglob("*.npz") if token in p.name.lower())
    return sorted(resolved.rglob("*.npz"))


def run_preprocess(
    config_path: str | Path | None,
    output_dir: str | Path | None = None,
    manifest_csv: str | Path | None = None,
) -> Path:
    preprocess_cfg = load_preprocess_cfg(str(config_path) if config_path is not None else None)

    raw_root = Path(preprocess_cfg.get("raw_root", "/data/fzliang/totalcapture")).expanduser().resolve()
    camera = str(preprocess_cfg.get("camera", "cam1"))

    if output_dir is not None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
    else:
        default_manifest = Path(
            preprocess_cfg.get(
                "output",
                "/data/fzliang/reid-project/totalcapture/preprocessed/default/video_manifest.csv",
            )
        ).expanduser().resolve()
        resolved_output_dir = default_manifest.parent

    resolved_manifest_csv = Path(
        manifest_csv if manifest_csv is not None else preprocess_cfg.get("output", str(resolved_output_dir / "video_manifest.csv"))
    ).expanduser().resolve()

    seq_dir = resolved_output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    imu_source = str(preprocess_cfg.get("imu_source", "xsens") or "xsens").strip().lower()
    if imu_source == "synthetic":
        synthetic_root = preprocess_cfg.get("synthetic_imu_root")
        if not synthetic_root:
            raise ValueError("preprocess.synthetic_imu_root is required when imu_source=synthetic")
        synthetic_side = preprocess_cfg.get("synthetic_imu_side")
        synthetic_root = Path(synthetic_root).expanduser().resolve()
        npz_paths = _collect_synthetic_npzs(synthetic_root, synthetic_side)
        if not npz_paths:
            raise FileNotFoundError(f"No synthetic npz files found under {synthetic_root}")

        manifest_rows = []
        seen_videos = set()
        for npz_path in npz_paths:
            shutil.copy2(npz_path, seq_dir / npz_path.name)
            try:
                data = dict(np.load(npz_path, allow_pickle=True))
            except Exception:
                continue
            video_path = ""
            if "video_path" in data:
                try:
                    video_path = str(data["video_path"].item())
                except Exception:
                    video_path = ""
            if video_path and video_path not in seen_videos:
                seen_videos.add(video_path)
                manifest_rows.append({"video_path": video_path})

        write_video_manifest(resolved_manifest_csv, [row["video_path"] for row in manifest_rows])
        print(f"Preprocessed {len(npz_paths)} sequences -> {resolved_output_dir}")
        print(f"Manifest: {resolved_manifest_csv} ({len(manifest_rows)} videos)")
        return resolved_output_dir

    sensor_order = parse_sensor_order(preprocess_cfg.get("sensor_order"))
    skeleton_normalize = _parse_bool(preprocess_cfg.get("skeleton_normalize"), default=True)
    sequences = []
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

    manifest_rows: list[dict[str, str]] = []
    for subject, session, vicon_path, imu_path in sequences:
        quat4, acc3 = parse_xsens_sensors(imu_path, sensor_order)
        joint_names, xyz21 = parse_vicon_pos(vicon_path)
        skel17 = map_totalcapture21_to_h36m17(joint_names, xyz21)

        tlen = min(skel17.shape[0], quat4.shape[0])
        if tlen == 0:
            print(f"Warning: empty sequence {subject}_{session}, skipping")
            continue

        skel17 = skel17[:tlen]
        quat4 = quat4[:tlen]
        acc3 = acc3[:tlen]
        imu48 = convert_imu_to_48(quat4, acc3)
        skel17_meters = skel17.copy().astype(np.float32)
        skel17 = normalize_skeleton(skel17) if skeleton_normalize else skel17.astype(np.float32)

        video_path = find_video_for_sequence(raw_root, subject, session, camera)
        if video_path is not None and video_path.exists():
            w, h = get_video_resolution(video_path)
            gt_bboxes = np.tile(np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32), (tlen, 1, 1))
            manifest_rows.append({"video_path": str(video_path)})
        else:
            gt_bboxes = np.zeros((tlen, 1, 4), dtype=np.float32)
            print(f"Warning: video not found for {subject}_{session} camera={camera}")

        sequence_id = f"totalcapture_{subject}_{session}_{camera}"
        frame_ids = np.arange(tlen, dtype=np.int64)
        imu = imu48[:, np.newaxis, :].astype(np.float32)
        imu_ids = np.array([0], dtype=np.int64)
        gt_person_ids = np.array([0], dtype=np.int64)
        gt_skeleton = skel17[:, np.newaxis, :, :].astype(np.float32)
        gt_visibility = np.ones((tlen, 1), dtype=bool)

        np.savez_compressed(
            seq_dir / f"{sequence_id}.npz",
            video_path=np.array(str(video_path) if video_path else "", dtype=object),
            dataset=np.array("totalcapture", dtype=object),
            sequence_id=np.array(sequence_id, dtype=object),
            frame_ids=frame_ids,
            imu=imu,
            imu_ids=imu_ids,
            gt_person_ids=gt_person_ids,
            gt_bboxes=gt_bboxes,
            gt_visibility=gt_visibility,
            gt_skeleton=gt_skeleton,
            gt_skeleton_meters=skel17_meters[:, np.newaxis, :, :].astype(np.float32),
        )

    write_video_manifest(resolved_manifest_csv, [row["video_path"] for row in manifest_rows])

    print(f"Preprocessed {len(sequences)} sequences -> {resolved_output_dir}")
    print(f"Manifest: {resolved_manifest_csv} ({len(manifest_rows)} videos)")
    return resolved_output_dir


def main() -> None:
    run_preprocess(None)


if __name__ == "__main__":
    main()

