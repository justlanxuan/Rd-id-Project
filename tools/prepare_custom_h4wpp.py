"""Build a session-aligned prepared custom dataset using Hand4Whole++."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from preprocess.common.imu import convert_single_imu_to_7d, parse_imu_csv, resample_imu_to_target
from preprocess.common.sequence import write_sequence_meta, write_sequence_npz
from preprocess.common.slice import run_slice_from_npz


def _read_annotations(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = np.asarray([float(row["timestamp_ms"]) for row in rows], dtype=np.float64)
    boxes = np.zeros((len(rows), 2, 4), dtype=np.float32)
    visibility = np.zeros((len(rows), 2), dtype=bool)
    for t, row in enumerate(rows):
        for p in range(2):
            prefix = f"p{p + 1}_"
            x = float(row[prefix + "bbox_x"])
            y = float(row[prefix + "bbox_y"])
            w = float(row[prefix + "bbox_w"])
            h = float(row[prefix + "bbox_h"])
            boxes[t, p] = [x, y, x + w, y + h]
            visibility[t, p] = str(row[prefix + "is_absent"]).strip() not in {"1", "true", "True"}
    return timestamps, boxes, visibility


def _ordered_imu_paths(session_dir: Path, raw_root: Path, session: str) -> list[Path]:
    """Return IMU files in annotation person order (person1, person2).

    The recording's mapping file says person1=f8... and person2=da....
    Lexicographic filename order is the reverse, so relying on ``sorted``
    silently swaps every IMU/video identity pair.
    """
    paths = sorted(session_dir.glob(f"{session}_*.csv"))
    mapping_path = raw_root / "annotations" / "imu_person_mapping.json"
    if not mapping_path.is_file():
        return paths[:2]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mac_by_person = [str(mapping.get("person1", "")), str(mapping.get("person2", ""))]
    by_mac = {path.name.rsplit("_", 1)[-1].removesuffix(".csv"): path for path in paths}
    ordered = [by_mac[mac] for mac in mac_by_person if mac in by_mac]
    return ordered if len(ordered) == 2 else paths[:2]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _load_h4w(path: Path, frame_count: int):
    entries = json.loads(path.read_text(encoding="utf-8"))
    frames: dict[int, list[dict]] = {}
    for entry in entries:
        try:
            frame = int(Path(str(entry.get("image_id", "0.jpg"))).stem)
        except ValueError:
            continue
        keypoints = np.asarray(entry.get("keypoints", []), dtype=np.float32)
        if keypoints.size != 17 * 3:
            continue
        box = np.asarray(entry.get("box", [0, 0, 0, 0]), dtype=np.float32)
        if box.size < 4:
            continue
        x, y, w, h = box[:4]
        frames.setdefault(frame, []).append(
            {
                "track_id": int(entry.get("idx", 0)),
                "box": np.asarray([x, y, x + w, y + h], dtype=np.float32),
                "keypoints": keypoints.reshape(17, 3),
            }
        )
    return frames, frame_count


def _extract(args, session: str, video: Path, tracks: Path, output: Path) -> Path:
    skeleton = output / "extracts" / session / "skeleton.json"
    if skeleton.exists() and args.skip_existing:
        return skeleton
    skeleton.parent.mkdir(parents=True, exist_ok=True)
    env = dict(**__import__("os").environ)
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["HEADLESS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python_paths = [
        Path(args.h4w_root) / "main",
        Path(args.h4w_root) / "common",
        Path(args.h4w_root) / "common" / "nets" / "WiLoR",
        Path(args.h4w_root) / "common" / "nets" / "mmpose",
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in python_paths), existing_pythonpath]
        if existing_pythonpath
        else [str(path) for path in python_paths]
    )
    cmd = [
        args.h4w_python,
        str(Path(__file__).with_name("h4wpp_extract_custom.py")),
        "--h4w-root", args.h4w_root,
        "--checkpoint", args.checkpoint,
        "--video", str(video),
        "--tracks", str(tracks),
        "--output", str(skeleton),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--frame-stride", str(args.frame_stride),
    ]
    subprocess.run(cmd, check=True, cwd=str(Path(args.h4w_root) / "demo"), env=env)
    return skeleton


def prepare(args) -> Path:
    raw = Path(args.raw_root).resolve()
    tracks_root = Path(args.tracks_root).resolve()
    out = Path(args.output).resolve()
    extract_out = Path(args.extract_root).resolve() if args.extract_root else out
    seq_dir = out / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p.name for p in raw.iterdir() if (p / "video").is_dir() and (p / "imu").is_dir())
    for session in sessions:
        video = raw / session / "video" / f"{session}.mp4"
        anno = raw / "annotations" / f"{session}.anno.csv"
        imu_paths = _ordered_imu_paths(raw / session / "imu", raw, session)
        tracks = tracks_root / session / "skeleton.json"
        if not (video.exists() and anno.exists() and len(imu_paths) >= 2 and tracks.exists()):
            print(f"[WARN] skipping incomplete session {session}")
            continue
        timestamps, gt_boxes, gt_vis = _read_annotations(anno)
        imu_list = []
        for path in imu_paths[:2]:
            src_ts, quat, acc = parse_imu_csv(path)
            values = convert_single_imu_to_7d(quat, acc)
            # Camera and IMU clocks differ by at most one sample at the tail
            # in these recordings; edge-clamp interpolation keeps the complete
            # annotated video timeline while avoiding artificial zero padding.
            aligned = resample_imu_to_target(src_ts, values, np.clip(timestamps, src_ts[0], src_ts[-1]))
            if not np.isfinite(aligned).all():
                raise ValueError(f"IMU interpolation failed: {path}")
            imu_list.append(aligned)
        imu = np.stack(imu_list, axis=1).astype(np.float32)
        h4w_path = _extract(args, session, video, tracks, extract_out)
        detections, _ = _load_h4w(h4w_path, len(timestamps))
        inference_stride = max(1, int(args.frame_stride))
        if inference_stride > 1:
            detections = {
                frame: frame_detections
                for frame, frame_detections in detections.items()
                if frame % inference_stride == 0
            }
        skeleton = np.zeros((len(timestamps), 2, 17, 3), dtype=np.float32)
        extract_boxes = np.zeros((len(timestamps), 2, 4), dtype=np.float32)
        extract_vis = np.zeros((len(timestamps), 2), dtype=bool)
        for t, dets in detections.items():
            if t >= len(timestamps):
                continue
            # Match each H4W crop to the annotated person by box IoU.
            used: set[int] = set()
            for det in sorted(dets, key=lambda item: item["track_id"]):
                candidates = [p for p in range(2) if gt_vis[t, p] and p not in used]
                if not candidates:
                    candidates = [p for p in range(2) if p not in used]
                p = max(candidates, key=lambda i: _iou(det["box"], gt_boxes[t, i]))
                used.add(p)
                skeleton[t, p] = det["keypoints"]
                extract_boxes[t, p] = det["box"]
                extract_vis[t, p] = True
        # H4W++ was evaluated at a lower frame rate than the camera. Carry the
        # latest valid estimate through intermediate frames (and backfill an
        # initial gap) so every IMU/video window remains temporally dense.
        for p in range(2):
            valid = np.flatnonzero(extract_vis[:, p])
            if len(valid) == 0:
                continue
            for t in range(1, len(timestamps)):
                if not extract_vis[t, p]:
                    extract_boxes[t, p] = extract_boxes[t - 1, p]
                    skeleton[t, p] = skeleton[t - 1, p]
                    extract_vis[t, p] = True
            for t in range(valid[0] - 1, -1, -1):
                extract_boxes[t, p] = extract_boxes[valid[0], p]
                skeleton[t, p] = skeleton[valid[0], p]
                extract_vis[t, p] = True
        # Keep the canonical gt field populated with H4W output: this makes the
        # standard vicon/default training path consume the new 3-D skeleton.
        payload = {
            "schema_version": np.array("1.0", dtype=object),
            "video_path": np.array(str(video), dtype=object),
            "dataset": np.array("custom", dtype=object),
            "sequence_id": np.array(f"custom_{session}", dtype=object),
            "frame_ids": np.arange(len(timestamps), dtype=np.int64),
            "imu": imu,
            "imu_ids": np.arange(2, dtype=np.int64),
            "gt_person_ids": np.arange(2, dtype=np.int64),
            "gt_bboxes": gt_boxes,
            "gt_visibility": gt_vis,
            "gt_skeleton": skeleton,
            "gt_skeleton_meters": skeleton.copy(),
            "extract_person_ids": np.arange(2, dtype=np.int64),
            "extract_bboxes": extract_boxes,
            "extract_visibility": extract_vis,
            "extract_skeleton": skeleton.copy(),
            "gt_to_extract_map": np.tile(np.arange(2, dtype=np.int64), (len(timestamps), 1)),
            "imu_person_map": np.array(json.dumps({"0": 0, "1": 1}), dtype=object),
            "frame_timestamps_ms": timestamps,
        }
        path = seq_dir / f"custom_{session}.npz"
        write_sequence_npz(path, payload)
        write_sequence_meta(path.with_suffix(".json"), {"dataset": "custom", "sequence_id": f"custom_{session}", "n_frames": len(timestamps), "imu_dim": 7, "h4wpp_skeleton": str(h4w_path), "h4wpp_inference_frame_stride": inference_stride})
        print(f"[prepared] {session}: frames={len(timestamps)}, visible={int(extract_vis.sum())}/{extract_vis.size}")
    slice_cfg = {
        "window_len": args.window_len,
        "stride": args.stride,
        "train_sessions": args.train_sessions,
        "val_sessions": args.val_sessions,
        "test_sessions": args.test_sessions,
        "multi_person": True,
        "skeleton_source": "vicon",
        "skeleton_normalize": args.skeleton_normalize,
    }
    run_slice_from_npz(out, out, slice_cfg)
    (out / "prepared.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "h4wpp": True,
                "h4wpp_inference_frame_stride": max(1, int(args.frame_stride)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main():
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_h4w_root = os.environ.get(
        "REID_H4WPP_ROOT", str(repo_root / "third-party" / "Hand4Whole-plus-plus_RELEASE")
    )
    default_checkpoint = os.environ.get(
        "REID_H4WPP_CHECKPOINT", str(repo_root / "models" / "hand4whole_plus_plus" / "snapshot_6.pth")
    )
    p.add_argument("--raw-root", required=True)
    p.add_argument("--tracks-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--extract-root",
        help="Optional shared root for H4W++ skeleton JSON outputs; keeps extraction separate from fold-specific slices.",
    )
    p.add_argument("--h4w-root", default=default_h4w_root)
    p.add_argument("--checkpoint", default=default_checkpoint)
    p.add_argument("--h4w-python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--window-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--train-sessions", default="20260211_171423,20260211_171724,20260211_172522")
    p.add_argument("--val-sessions", default="20260211_172257")
    p.add_argument("--test-sessions", default="20260211_172257")
    p.add_argument("--skeleton-normalize", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
