# Experiment Note: D8-custom-detector-id-audit
"""Audit Custom AlphaPose detector IDs against GT boxes without relinking tracks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ALIGNED_ROOT = Path(
    "/data/fzliang/reid-project/custom/preprocessed/"
    "custom_hybrid_finetune_from_egohumans/aligned_sequences"
)
TRACKLET_ROOT = Path("/data/fzliang/reid-project/custom/skeleton/alphapose")
SESSIONS = ("20260211_171423", "20260211_171724", "20260211_172257", "20260211_172522")


def iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def label(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_session(session: str, aligned_root: Path, tracklet_root: Path) -> dict[str, Any]:
    with np.load(aligned_root / f"custom_{session}.npz", allow_pickle=True) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        gt_boxes = np.asarray(archive["gt_bboxes"], dtype=np.float32)
        gt_visibility = np.asarray(archive["gt_visibility"], dtype=bool)
    rows = json.loads((tracklet_root / session / "skeleton_unmerged.json").read_text())
    detections: dict[int, list[dict[str, Any]]] = defaultdict(list)
    labels_seen: set[str] = set()
    duplicate_id_frame = 0
    seen_pairs: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            frame = int(Path(str(row.get("image_id", "0"))).stem)
        except ValueError:
            continue
        box = row.get("box", [0, 0, 0, 0])
        if not isinstance(box, list) or len(box) < 4:
            continue
        track_id = label(row.get("idx", 0))
        pair = (frame, track_id)
        if pair in seen_pairs:
            duplicate_id_frame += 1
        seen_pairs.add(pair)
        labels_seen.add(track_id)
        x, y, width, height = (float(v) for v in box[:4])
        detections[frame].append({"id": track_id, "box": np.asarray([x, y, x + width, y + height], dtype=np.float32)})

    id_to_gt: dict[str, list[tuple[int, int]]] = defaultdict(list)
    gt_to_id: dict[int, list[tuple[int, str]]] = defaultdict(list)
    matched = 0
    candidate_frames = 0
    frame_to_index = {int(frame): index for index, frame in enumerate(frame_ids)}
    for frame, frame_detections in detections.items():
        time_index = frame_to_index.get(frame)
        if time_index is None:
            continue
        active_gt = np.flatnonzero(gt_visibility[time_index])
        if active_gt.size == 0 or not frame_detections:
            continue
        candidate_frames += 1
        matrix = np.asarray(
            [[iou(detection["box"], gt_boxes[time_index, gt]) for gt in active_gt] for detection in frame_detections]
        )
        rows_i, columns_i = linear_sum_assignment(-matrix)
        for detection_index, gt_index in zip(rows_i, columns_i, strict=True):
            if float(matrix[detection_index, gt_index]) < 0.1:
                continue
            track_id = frame_detections[int(detection_index)]["id"]
            gt_id = int(active_gt[int(gt_index)])
            id_to_gt[track_id].append((frame, gt_id))
            gt_to_id[gt_id].append((frame, track_id))
            matched += 1

    def transition_count(values: dict[Any, list[tuple[int, Any]]]) -> int:
        total = 0
        for sequence in values.values():
            sequence = sorted(sequence)
            previous = None
            for _, current in sequence:
                if previous is not None and current != previous:
                    total += 1
                previous = current
        return total

    return {
        "session": session,
        "frames": int(frame_ids.size),
        "detector_rows": int(len(rows)),
        "detector_labels": int(len(labels_seen)),
        "frames_with_detections": int(len(detections)),
        "candidate_frames_with_gt": int(candidate_frames),
        "matched_detection_rows_iou_ge_0.1": int(matched),
        "duplicate_id_frame_rows": int(duplicate_id_frame),
        "raw_id_to_gt_transition_count": transition_count(id_to_gt),
        "gt_to_raw_id_transition_count": transition_count(gt_to_id),
        "identity_provenance": "raw AlphaPose idx joined to GT boxes by per-frame Hungarian IoU; thresholds and transitions are diagnostic, not relinking",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path, default=ALIGNED_ROOT)
    parser.add_argument("--tracklet-root", type=Path, default=TRACKLET_ROOT)
    parser.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    parser.add_argument("--output", type=Path, default=Path("/data/fzliang/reid-project/g9/e2_multimodal/custom_detector_id_audit.json"))
    args = parser.parse_args()
    sessions = list(args.sessions)
    report = {
        "schema_version": "g9-e2-custom-detector-id-audit-1",
        "protocol": {"iou_threshold": 0.1, "association": "per-frame Hungarian IoU", "sessions": sessions},
        "sessions": {session: audit_session(session, args.aligned_root, args.tracklet_root) for session in sessions},
        "s06_limitation": "S06 algorithm NPZs contain inherited person order and no independent detector track IDs; ID switches for S06 remain unobservable.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "sessions": len(sessions), "raw_id_to_gt_transitions": sum(v["raw_id_to_gt_transition_count"] for v in report["sessions"].values()), "gt_to_raw_id_transitions": sum(v["gt_to_raw_id_transition_count"] for v in report["sessions"].values())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
