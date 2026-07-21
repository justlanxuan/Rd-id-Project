"""Shared video-side preprocessing helpers.

This module keeps dataset-specific preprocess entrypoints thin by providing
common logic for video file discovery, manifest writing, and skeleton JSON
materialization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1920, 1080
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 30.0


def find_video_for_sequence(raw_root: Path, subject: str, session: str, camera: str, ext: str = ".mp4") -> Path | None:
    """Find a matching video under a dataset root.

    The primary layout is ``raw_root/<session>/TC_<subject>_<session>_<camera>.mp4``.
    A recursive fallback is used for legacy layouts.
    """
    primary = raw_root / session / f"TC_{subject}_{session}_{camera}{ext}"
    if primary.exists():
        return primary

    candidates = [
        path for path in raw_root.rglob(f"*{ext}") if subject in path.name and session in path.name and camera in path.name
    ]
    if candidates:
        return sorted(candidates)[0]
    return None


def write_video_manifest(manifest_csv: Path, video_paths: Iterable[str]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for video_path in video_paths:
        text = str(video_path).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append({"video_path": text})

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        writer.writerows(rows)


def save_skeleton_json(entries: Sequence[dict], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(list(entries)))


def pose2d_to_bbox(pose2d: np.ndarray, margin: float = 0.05) -> np.ndarray:
    t_len = pose2d.shape[0]
    bboxes = np.zeros((t_len, 4), dtype=np.float32)
    for t in range(t_len):
        pts = pose2d[t]
        mask = np.logical_and(pts[:, 0] > 0, pts[:, 1] > 0)
        if mask.sum() == 0:
            continue
        xs = pts[mask, 0]
        ys = pts[mask, 1]
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1
        bboxes[t] = np.array([x1 - w * margin, y1 - h * margin, x2 + w * margin, y2 + h * margin], dtype=np.float32)
    return bboxes