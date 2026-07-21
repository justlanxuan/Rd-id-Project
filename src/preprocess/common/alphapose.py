"""AlphaPose adapter helpers used by preprocessing pipelines.

This module centralizes the conversion and loading logic for AlphaPose
skeleton JSON files so preprocessing code can depend on a single shared
implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np


# COCO joint order (17 joints)
COCO_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# H36M 17 joint names (order used in the training code)
H36M_JOINTS = [
    'Hip',           # 0 - root/pelvis (virtual center)
    'RightHip',      # 1
    'RightKnee',     # 2
    'RightAnkle',    # 3
    'LeftHip',       # 4
    'LeftKnee',      # 5
    'LeftAnkle',     # 6
    'Spine',         # 7 - spine (virtual)
    'Thorax',        # 8 - thorax (virtual)
    'Neck/Nose',     # 9 - neck/nose
    'Head',          # 10 - head (virtual)
    'LeftShoulder',  # 11
    'LeftElbow',     # 12
    'LeftWrist',     # 13
    'RightShoulder', # 14
    'RightElbow',    # 15
    'RightWrist',    # 16
]


def coco_to_h36m17(coco_keypoints: np.ndarray) -> np.ndarray:
    """Convert COCO 17 joints to H36M 17 joints.

    Args:
        coco_keypoints: (N, 17, 3) array of COCO joints [x, y, confidence]

    Returns:
        h36m_joints: (N, 17, 3) array of H36M joints [x, y, z]
    """
    N = coco_keypoints.shape[0]
    h36m = np.zeros((N, 17, 3), dtype=np.float32)

    # Build COCO name to index map
    coco_map = {name: i for i, name in enumerate(COCO_JOINTS)}

    # Direct mapping for available joints
    # H36M -> COCO mappings
    h36m_to_coco = {
        0: None,   # Hip - compute as mid of hips
        1: coco_map['right_hip'],
        2: coco_map['right_knee'],
        3: coco_map['right_ankle'],
        4: coco_map['left_hip'],
        5: coco_map['left_knee'],
        6: coco_map['left_ankle'],
        7: None,   # Spine - compute
        8: None,   # Thorax - compute
        9: coco_map['nose'],  # Neck/Nose
        10: None,  # Head - compute
        11: coco_map['left_shoulder'],
        12: coco_map['left_elbow'],
        13: coco_map['left_wrist'],
        14: coco_map['right_shoulder'],
        15: coco_map['right_elbow'],
        16: coco_map['right_wrist'],
    }

    for h36m_idx, coco_idx in h36m_to_coco.items():
        if coco_idx is not None:
            h36m[:, h36m_idx, :2] = coco_keypoints[:, coco_idx, :2]  # x, y
            h36m[:, h36m_idx, 2] = 0.0  # z = 0 for 2D skeleton

    # Compute virtual joints
    # Hip (root) = mid of left and right hip
    h36m[:, 0, :2] = (coco_keypoints[:, coco_map['left_hip'], :2] +
                      coco_keypoints[:, coco_map['right_hip'], :2]) / 2

    # Spine = mid of hips and shoulders
    hips = h36m[:, 0, :2]
    shoulders = (coco_keypoints[:, coco_map['left_shoulder'], :2] +
                 coco_keypoints[:, coco_map['right_shoulder'], :2]) / 2
    h36m[:, 7, :2] = (hips + shoulders) / 2

    # Thorax = mid of shoulders and slightly above
    h36m[:, 8, :2] = shoulders

    # Head = above nose (estimate)
    nose = coco_keypoints[:, coco_map['nose'], :2]
    # Head position: extend from nose upward
    neck = (shoulders + nose) / 2
    head_offset = nose - neck
    h36m[:, 10, :2] = nose + head_offset * 0.5

    return h36m


def load_alphapose_skeleton(skeleton_json: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load AlphaPose skeleton from skeleton.json.

    Args:
        skeleton_json: Path to skeleton.json file

    Returns:
        keypoints_3d: (N, 17, 3) array of 3D keypoints
        scores: (N,) array of detection scores
    """
    with open(skeleton_json) as f:
        data = json.load(f)

    # Sort by image_id (frame number)
    # image_id format: "0.jpg", "1.jpg", etc.
    def get_frame_num(item):
        try:
            return int(item['image_id'].split('.')[0])
        except (ValueError, IndexError):
            return 0

    data = sorted(data, key=get_frame_num)

    N = len(data)
    coco_kpts = np.zeros((N, 17, 3), dtype=np.float32)
    scores = np.zeros(N, dtype=np.float32)

    for i, frame in enumerate(data):
        keypoints = frame['keypoints']
        # keypoints is flat list: [x1, y1, c1, x2, y2, c2, ...]
        for j in range(17):
            coco_kpts[i, j, 0] = keypoints[j * 3]      # x
            coco_kpts[i, j, 1] = keypoints[j * 3 + 1]  # y
            coco_kpts[i, j, 2] = keypoints[j * 3 + 2]  # confidence
        scores[i] = frame.get('score', 0.0)

    # Convert to H36M format
    h36m_kpts = coco_to_h36m17(coco_kpts)

    return h36m_kpts, scores


def find_skeleton_for_sequence(
    subject: str,
    session: str,
    skeleton_root: Path
) -> Path | None:
    """Find skeleton.json for a given sequence.

    Args:
        subject: Subject ID (e.g., "S1")
        session: Session name (e.g., "acting1")
        skeleton_root: Root directory containing skeleton results

    Returns:
        Path to skeleton.json or None if not found
    """
    # Accept both TC_S1_acting1_cam1 and S1_acting1_cam1 naming
    patterns = [f"TC_{subject}_{session}_cam1", f"{subject}_{session}_cam1"]

    for subdir in skeleton_root.iterdir():
        if not subdir.is_dir():
            continue
        if any(pat in subdir.name for pat in patterns):
            skeleton_file = subdir / "skeleton.json"
            if skeleton_file.exists():
                return skeleton_file

    return None


def _frame_num_from_image_id(image_id: str) -> int:
    """Extract frame number from AlphaPose image_id like '0.jpg' or 'path/to/0.jpg'."""
    stem = Path(image_id).stem
    try:
        return int(stem)
    except ValueError:
        return 0


def load_alphapose_multiperson(skeleton_json: Path) -> tuple[dict[int, list[dict]], list[int]]:
    """Load AlphaPose skeleton JSON supporting multiple tracks per frame.

    Each detection is converted to H36M 17-joint format. Bbox is converted
    from [x, y, w, h] (COCO) to [x1, y1, x2, y2].

    Args:
        skeleton_json: Path to AlphaPose skeleton.json

    Returns:
        frames: dict mapping frame_idx -> list of detection dicts.
                Each dict has keys: 'track_id', 'bbox' [x1,y1,x2,y2],
                'keypoints' [17,3], 'score'.
        track_ids: sorted list of unique track ids across the whole sequence.
    """
    with open(skeleton_json) as f:
        data = json.load(f)

    frames: dict[int, list[dict]] = {}
    track_ids_set: set[int] = set()

    for entry in data:
        image_id = entry.get("image_id", "")
        frame_idx = _frame_num_from_image_id(image_id)

        keypoints_flat = entry.get("keypoints", [])
        if len(keypoints_flat) < 17 * 3:
            continue

        coco_kpts = np.zeros((1, 17, 3), dtype=np.float32)
        for j in range(17):
            coco_kpts[0, j, 0] = keypoints_flat[j * 3]
            coco_kpts[0, j, 1] = keypoints_flat[j * 3 + 1]
            coco_kpts[0, j, 2] = keypoints_flat[j * 3 + 2]

        h36m_kpts = coco_to_h36m17(coco_kpts)[0]  # [17, 3]

        # bbox from AlphaPose is usually [x, y, w, h]
        box = entry.get("box", [0.0, 0.0, 0.0, 0.0])
        if len(box) < 4:
            box = [0.0, 0.0, 0.0, 0.0]
        x, y, w, h = box
        bbox = np.array([float(x), float(y), float(x + w), float(y + h)], dtype=np.float32)

        raw_idx = entry.get("idx", 0)
        if isinstance(raw_idx, list):
            track_id_list = [int(x) for x in raw_idx]
        else:
            track_id_list = [int(raw_idx)]

        for track_id in track_id_list:
            track_ids_set.add(track_id)
            det = {
                "track_id": track_id,
                "bbox": bbox,
                "keypoints": h36m_kpts,
                "score": float(entry.get("score", 0.0)),
            }
            frames.setdefault(frame_idx, []).append(det)

    return frames, sorted(track_ids_set)
