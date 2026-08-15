"""Shared utilities for video skeleton extractors."""

from __future__ import annotations

import json
from pathlib import Path


def convert_bytetrack_txt_to_detfile(track_txt: Path, frame_dir: Path, out_json: Path) -> None:
    rows = []
    with track_txt.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])
            rows.append((frame_id, track_id, x, y, w, h, score))

    if not rows:
        raise RuntimeError(f"No valid rows parsed from {track_txt}")

    min_frame = min(r[0] for r in rows)
    dets = []
    for frame_id, track_id, x, y, w, h, score in rows:
        frame_idx = frame_id - min_frame
        image_path = frame_dir / f"{frame_idx}.jpg"
        if not image_path.exists():
            continue
        dets.append({"image_id": str(image_path), "bbox": [x, y, w, h], "score": score, "idx": track_id})

    if not dets:
        raise RuntimeError("Converted detfile is empty. Check ByteTrack output and extracted frames.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(dets, f)


def extract_video_frames(video_path: Path, frame_dir: Path) -> None:
    import cv2

    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frame_dir / f"{idx}.jpg"), frame)
        idx += 1
    cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted from input video.")
