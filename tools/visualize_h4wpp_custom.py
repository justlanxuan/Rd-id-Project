"""Render H4W++ custom skeletons and torso orientation overlays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# H36M-17 order produced by h4wpp_extract_custom.py.
EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16), (8, 10), (10, 9),
]


def _load(path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        try:
            frame = int(Path(str(item.get("image_id", "0.jpg"))).stem)
        except ValueError:
            continue
        kpt = np.asarray(item.get("keypoints", []), dtype=np.float32)
        box = np.asarray(item.get("box", [0, 0, 0, 0]), dtype=np.float32)
        if kpt.size != 17 * 3 or box.size < 4:
            continue
        x, y, w, h = box[:4]
        out.setdefault(frame, []).append({
            "id": int(item.get("idx", 0)),
            "kpt": kpt.reshape(17, 3),
            "box": np.asarray([x, y, w, h], dtype=np.float32),
        })
    return out


def _orientation(k: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a camera-ground forward vector and yaw in degrees."""
    # H36M indices: pelvis=0, neck/nose=9, left/right shoulders=11/14.
    left_shoulder, right_shoulder = k[11], k[14]
    up = k[9] - k[0]
    across = right_shoulder - left_shoulder
    forward = np.cross(across, up)
    nose_dir = k[9] - ((left_shoulder + right_shoulder) * 0.5)
    if np.dot(forward, nose_dir) < 0:
        forward = -forward
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32), float("nan")
    forward = forward / norm
    # Camera X/Z top-down yaw; z sign follows the model camera convention.
    yaw = math.degrees(math.atan2(float(forward[2]), float(forward[0])))
    return forward, yaw


def _project_relative_2d(k: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Draw a diagnostic 2-D relative projection inside the tracked box.

    H4W++ keypoints are camera-coordinate 3-D points.  This overlay is only a
    visual aid because the prepared JSON intentionally does not retain the
    crop-to-camera intrinsics needed for a metric reprojection.  In the H4W++
    camera convention positive y is up, so no y flip is applied here.
    """
    x, y, w, h = [float(v) for v in box]
    xy = k[:, :2].copy()
    lo, hi = np.nanpercentile(xy, 2, axis=0), np.nanpercentile(xy, 98, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    p = (xy - lo) / span
    return np.stack([x + p[:, 0] * w, y + p[:, 1] * h], axis=1)


def _project_3d(k: np.ndarray, origin: tuple[int, int], scale: float) -> np.ndarray:
    """Project true 3-D camera coordinates into an oblique diagnostic view."""
    # x is horizontal, y is up, and z is depth.  A small z contribution makes
    # depth visible while keeping the anatomically meaningful y-up direction.
    x = k[:, 0] + 0.45 * k[:, 2]
    y = k[:, 1] - 0.20 * k[:, 2]
    ox, oy = origin
    return np.stack([ox + scale * x, oy - scale * y], axis=1)


def _draw_3d_panel(panel: np.ndarray, detections: list[dict], colors) -> None:
    """Render a fixed-scale 3-D skeleton panel, preserving depth and axes."""
    height, width = panel.shape[:2]
    panel[:] = (24, 28, 34)
    cv2.putText(panel, "H4W++ true 3D (camera x/y/z)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cx, cy = width // 2, int(height * 0.58)
    scale = min(width, height) * 0.34
    # Reference axes: x right, y up, z depth (oblique screen direction).
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    origin = (cx, cy)
    for label, vec, color in (("x", axes["x"], (100, 170, 255)),
                              ("y", axes["y"], (120, 255, 140)),
                              ("z", axes["z"], (255, 170, 100))):
        end = _project_3d(vec[None, :], origin, scale * 0.65)[0].astype(int)
        cv2.arrowedLine(panel, origin, tuple(end), color, 2, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(panel, label, tuple(end + np.array([5, 5])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    cv2.circle(panel, origin, 3, (220, 220, 220), -1, cv2.LINE_AA)
    for det in sorted(detections, key=lambda item: item["id"]):
        color = colors[det["id"] % len(colors)]
        pts = _project_3d(det["kpt"], origin, scale)
        for a, b in EDGES:
            pa, pb = tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int))
            cv2.line(panel, pa, pb, color, 3, cv2.LINE_AA)
        for px, py in np.round(pts).astype(int):
            cv2.circle(panel, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
        pelvis = tuple(np.round(pts[0]).astype(int))
        cv2.putText(panel, f"ID {det['id']}", (pelvis[0] + 8, pelvis[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _draw_topdown(frame: np.ndarray, origin: tuple[int, int], forward: np.ndarray, color):
    ox, oy = origin
    cv2.rectangle(frame, (ox - 34, oy - 34), (ox + 34, oy + 34), (30, 30, 30), -1)
    cv2.rectangle(frame, (ox - 34, oy - 34), (ox + 34, oy + 34), color, 1)
    cv2.line(frame, (ox, oy - 27), (ox, oy + 27), (100, 100, 100), 1)
    cv2.line(frame, (ox - 27, oy), (ox + 27, oy), (100, 100, 100), 1)
    if np.linalg.norm(forward) > 1e-6:
        # top-down panel: camera X horizontal, camera Z vertical
        dx, dz = float(forward[0]), float(forward[2])
        end = (int(ox + 28 * dx), int(oy - 28 * dz))
        cv2.arrowedLine(frame, (ox, oy), end, color, 3, tipLength=0.25)


def render(video: Path, skeleton: Path, output: Path, sample_every: int = 1) -> None:
    detections = _load(skeleton)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_width = max(360, width // 2)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + panel_width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output {output}")
    last: dict[int, dict] = {}
    frame_idx = 0
    colors = [(0, 220, 255), (255, 150, 0), (120, 255, 80), (255, 100, 220)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in detections:
            for det in detections[frame_idx]:
                last[det["id"]] = det
        if sample_every <= 1 or frame_idx % sample_every == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 36), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
            cv2.putText(frame, f"Hand4Whole++ | frame {frame_idx} | 3D skeleton + torso orientation", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            for id_, det in sorted(last.items()):
                box = det["box"]
                x, y, w, h = [int(round(v)) for v in box]
                color = colors[id_ % len(colors)]
                pts = _project_relative_2d(det["kpt"], box)
                for a, b in EDGES:
                    pa, pb = tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int))
                    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
                for px, py in np.round(pts).astype(int):
                    cv2.circle(frame, (int(px), int(py)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(px), int(py)), 2, color, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                forward, yaw = _orientation(det["kpt"])
                # Put the top-down orientation inset beside the person box so
                # it does not cover the torso joints being inspected.
                if x + w + 76 < width:
                    panel_x = x + w + 40
                else:
                    panel_x = x - 40
                panel_y = y + min(max(h // 2, 42), max(height - 42, 42))
                _draw_topdown(frame, (min(max(panel_x, 38), width - 38), min(max(panel_y, 38), height - 38)), forward, color)
                text = f"ID {id_}  yaw {yaw:+.1f} deg" if np.isfinite(yaw) else f"ID {id_}  yaw n/a"
                cv2.putText(frame, text, (max(4, x), max(52, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "left: relative 2D overlay (not metric reprojection)", (8, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        _draw_3d_panel(panel, list(last.values()), colors)
        frame = np.concatenate([frame, panel], axis=1)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"Rendered {frame_idx} frames -> {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-every", type=int, default=1)
    a = p.parse_args()
    render(Path(a.video), Path(a.skeleton), Path(a.output), a.sample_every)


if __name__ == "__main__":
    main()
