"""Full-session Custom evaluation data with opaque, unmerged tracklet IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.common.extract import coco_to_h36m17
from preprocess.common.slice import normalize_skeleton
from preprocess.datasets.custom import load_custom_rawcsv_7d_sequence, load_custom_rawcsv_feature_sequence
from src.features.imu import IMUFeatureSpec, select_imu_features


@dataclass(frozen=True)
class CustomTrackletSession:
    sequence_id: str
    frame_ids: np.ndarray
    imu: np.ndarray
    imu_channels: tuple[str, ...]
    imu_ids: np.ndarray
    gt_person_ids: np.ndarray
    gt_bboxes: np.ndarray
    gt_visibility: np.ndarray
    tracklet_labels: tuple[str, ...]
    extract_bboxes: np.ndarray
    extract_visibility: np.ndarray
    extract_skeleton: np.ndarray
    gt_to_extract_map: np.ndarray


def canonical_tracklet_label(raw_id: Any) -> str:
    """Serialize a raw tracker ID without expanding composite/list IDs."""
    return json.dumps(raw_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _load_atomic_tracklets(path: Path) -> tuple[dict[int, dict[str, dict[str, Any]]], tuple[str, ...]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Tracklet JSON must contain a list: {path}")
    frames: dict[int, dict[str, dict[str, Any]]] = {}
    first_seen: dict[str, int] = {}
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        keypoints = entry.get("keypoints", [])
        if not isinstance(keypoints, list) or len(keypoints) < 17 * 3:
            continue
        try:
            frame_idx = int(Path(str(entry.get("image_id", "0"))).stem)
        except ValueError:
            continue
        label = canonical_tracklet_label(entry.get("idx", 0))
        first_seen.setdefault(label, order)
        coco = np.asarray(keypoints[: 17 * 3], dtype=np.float32).reshape(1, 17, 3)
        if not np.isfinite(coco).all():
            continue
        box = entry.get("box", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(box, list) or len(box) < 4:
            box = [0.0, 0.0, 0.0, 0.0]
        x, y, width, height = (float(value) for value in box[:4])
        detection = {
            "bbox": np.asarray([x, y, x + width, y + height], dtype=np.float32),
            "keypoints": coco_to_h36m17(coco)[0],
            "score": float(entry.get("score", 0.0)),
        }
        # A tracker should emit at most one detection per ID/frame.  If its
        # artifact violates that rule, retain the highest-confidence row.
        previous = frames.setdefault(frame_idx, {}).get(label)
        if previous is None or detection["score"] > previous["score"]:
            frames[frame_idx][label] = detection
    labels = tuple(sorted(first_seen, key=lambda label: first_seen[label]))
    return frames, labels


def load_custom_tracklet_session(
    aligned_npz: str | Path,
    tracklet_json: str | Path,
    *,
    custom_imu_root: str | Path | None = None,
    raw_swap: bool = False,
    normalize_extract_skeleton: bool = True,
    imu_feature_spec: IMUFeatureSpec | None = None,
    legacy_sensor: str = "L_LowArm",
) -> CustomTrackletSession:
    """Load one complete session; no segment slicing or tracklet linking occurs."""
    npz_path = Path(aligned_npz).expanduser().resolve()
    json_path = Path(tracklet_json).expanduser().resolve()
    with np.load(npz_path, allow_pickle=True) as source:
        data = {key: source[key].copy() for key in source.files}
    sequence_id = str(data["sequence_id"].item())
    session = sequence_id.removeprefix("custom_")
    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
    t_len = int(len(frame_ids))
    gt_person_ids = np.asarray(data["gt_person_ids"], dtype=np.int64)
    gt_bboxes = np.asarray(data["gt_bboxes"], dtype=np.float32)[:t_len]
    gt_visibility = np.asarray(data["gt_visibility"], dtype=bool)[:t_len]
    if len(gt_bboxes) < t_len or len(gt_visibility) < t_len:
        raise ValueError(f"Ground-truth timeline is shorter than frame_ids in {npz_path}")

    frames, labels = _load_atomic_tracklets(json_path)
    label_to_index = {label: index for index, label in enumerate(labels)}
    n_tracks = len(labels)
    extract_bboxes = np.zeros((t_len, n_tracks, 4), dtype=np.float32)
    extract_skeleton = np.zeros((t_len, n_tracks, 17, 3), dtype=np.float32)
    extract_visibility = np.zeros((t_len, n_tracks), dtype=bool)
    for time_index, frame_id in enumerate(frame_ids):
        for label, detection in frames.get(int(frame_id), {}).items():
            track_index = label_to_index[label]
            extract_bboxes[time_index, track_index] = detection["bbox"]
            extract_skeleton[time_index, track_index] = detection["keypoints"]
            extract_visibility[time_index, track_index] = True
    if normalize_extract_skeleton:
        for track_index in range(n_tracks):
            visible = extract_visibility[:, track_index]
            if visible.any():
                extract_skeleton[visible, track_index] = normalize_skeleton(
                    extract_skeleton[visible, track_index]
                )

    gt_to_extract_map = np.full((t_len, len(gt_person_ids)), -1, dtype=np.int64)
    for time_index in range(t_len):
        for gt_index in range(len(gt_person_ids)):
            if not gt_visibility[time_index, gt_index]:
                continue
            active = np.flatnonzero(extract_visibility[time_index])
            if active.size == 0:
                continue
            ious = np.asarray(
                [
                    _iou(gt_bboxes[time_index, gt_index], extract_bboxes[time_index, track_index])
                    for track_index in active
                ]
            )
            best = int(np.argmax(ious))
            if ious[best] > 0.0:
                gt_to_extract_map[time_index, gt_index] = int(active[best])

    if custom_imu_root is None:
        values = np.asarray(data["imu"], dtype=np.float32)[:t_len]
        if values.ndim == 2:
            values = values[:, np.newaxis, :]
        if values.ndim != 3:
            raise ValueError(f"Expected aligned IMU [T,C] or [T,P,C], got {values.shape}")
        if imu_feature_spec is None:
            if values.shape[-1] >= 48:
                from preprocess.datasets.custom import legacy_imu48_sensor_to_7d

                imu = legacy_imu48_sensor_to_7d(values, legacy_sensor)
            elif values.shape[-1] >= 7:
                imu = values[..., :7]
            else:
                raise ValueError(f"Aligned IMU has no compatible 7D view: {values.shape}")
            imu_channels = (
                "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"
            )
        else:
            imu = np.stack(
                [
                    select_imu_features(
                        values[:, person],
                        data.get("imu_channels"),
                        imu_feature_spec,
                        legacy_sensor=legacy_sensor,
                    )
                    for person in range(values.shape[1])
                ],
                axis=1,
            )
            imu_channels = imu_feature_spec.channels
    else:
        imu_person_map = str(data["imu_person_map"].item()) if "imu_person_map" in data else None
        if imu_feature_spec is None:
            imu = load_custom_rawcsv_7d_sequence(
                Path(custom_imu_root),
                session,
                frame_ids,
                imu_person_map=imu_person_map,
                n_persons=len(gt_person_ids),
            )
            imu_channels = (
                "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"
            )
        else:
            imu = load_custom_rawcsv_feature_sequence(
                Path(custom_imu_root),
                session,
                frame_ids,
                imu_feature_spec,
                imu_person_map=imu_person_map,
                n_persons=len(gt_person_ids),
                legacy_sensor=legacy_sensor,
            )
            imu_channels = imu_feature_spec.channels
    if raw_swap:
        imu = imu[:, ::-1].copy()

    return CustomTrackletSession(
        sequence_id=sequence_id,
        frame_ids=frame_ids,
        imu=imu,
        imu_channels=imu_channels,
        imu_ids=np.asarray(data.get("imu_ids", gt_person_ids), dtype=np.int64),
        gt_person_ids=gt_person_ids,
        gt_bboxes=gt_bboxes,
        gt_visibility=gt_visibility,
        tracklet_labels=labels,
        extract_bboxes=extract_bboxes,
        extract_visibility=extract_visibility,
        extract_skeleton=extract_skeleton,
        gt_to_extract_map=gt_to_extract_map,
    )


__all__ = [
    "CustomTrackletSession",
    "canonical_tracklet_label",
    "load_custom_tracklet_session",
]
