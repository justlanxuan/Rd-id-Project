#!/usr/bin/env python
"""
Convert MoBInd EgoHumans extracted_data/*.npy into Autism-project unified NPZ format.

Inputs (from MoBInd):
    /data/lyxie/ReID/Data/egohumans/extracted_data/
    └── 01_001_aria01.npy   # keys: imu (T,5,7), pose2d (T,17,2), pose3d (T,24,3)

Outputs:
    data/interim/egohumans/preprocess/sequences/
    └── egohumans_01_001.npz   # unified multi-person NPZ
        keys: video_path, dataset, sequence_id, frame_ids,
              imu (T, N_imu, 48), imu_ids (N_imu,),
              gt_person_ids (N_gt,), gt_bboxes (T, N_gt, 4),
              gt_visibility (T, N_gt,), gt_skeleton (T, N_gt, 17, 3)
"""
import argparse
import re
from pathlib import Path

import numpy as np


# SMPL 24-joint index -> H36M 17-joint index
SMPL24_TO_H36M17 = [
    0,   # 0  Pelvis     -> Hip
    2,   # 1  R_Hip      -> RightHip
    5,   # 2  R_Knee     -> RightKnee
    8,   # 3  R_Ankle    -> RightAnkle
    1,   # 4  L_Hip      -> LeftHip
    4,   # 5  L_Knee     -> LeftKnee
    7,   # 6  L_Ankle    -> LeftAnkle
    6,   # 7  Spine2     -> Spine
    9,   # 8  Spine3     -> Thorax
    12,  # 9  Neck       -> Neck/Nose
    15,  # 10 Head       -> Head
    16,  # 11 L_Shoulder -> LeftShoulder
    18,  # 12 L_Elbow    -> LeftElbow
    20,  # 13 L_Wrist    -> LeftWrist
    17,  # 14 R_Shoulder -> RightShoulder
    19,  # 15 R_Elbow    -> RightElbow
    21,  # 16 R_Wrist    -> RightWrist
]


def smpl24_to_h36m17(smpl_joints: np.ndarray) -> np.ndarray:
    """Map SMPL 24 joints (T, 24, 3) to H36M 17 joints (T, 17, 3)."""
    return smpl_joints[:, SMPL24_TO_H36M17, :].astype(np.float32)


# MoBInd synthetic IMU sensor order and channel layout
# imu shape: (T, 5, 7)
# sensors: [LeftWrist, RightWrist, LeftKnee, RightKnee, Head]
# channels per sensor: [acc_x, acc_y, acc_z, q_w, q_x, q_y, z]
MOBIND_SENSOR_ORDER = ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]

# Autism-project / SIE expects (T, 48) as 4 sensors x (9D rotation matrix + 3D accel)
# sensor order: [L_LowLeg, R_LowLeg, L_LowArm, R_LowArm]
AUTISM_SENSOR_ORDER = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]

# Map MoBInd limb -> Autism limb
MOBIND_TO_AUTISM = {
    "LeftKnee": "L_LowLeg",
    "RightKnee": "R_LowLeg",
    "LeftWrist": "L_LowArm",
    "RightWrist": "R_LowArm",
}


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (..., 4) in order [w, x, y, z]; returns (..., 3, 3) rotation matrix."""
    q = q.astype(np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.zeros(q.shape[:-1] + (3, 3), dtype=np.float32)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def mobind_imu_to_48(imu: np.ndarray) -> np.ndarray:
    """Convert MoBInd (T, 5, 7) IMU to Autism-project (T, 48) format.

    The 5 MoBInd sensors are mapped to the 4 Autism sensor slots (Head is dropped).
    Channels are converted from [acc(3), quat_wxyz(4)] to [rotmat_flat(9), acc(3)].
    """
    T = imu.shape[0]
    out = np.zeros((T, 48), dtype=np.float32)

    autism_to_mobind = {
        autism_name: MOBIND_SENSOR_ORDER.index(mobind_name)
        for mobind_name, autism_name in MOBIND_TO_AUTISM.items()
    }

    for i, autism_name in enumerate(AUTISM_SENSOR_ORDER):
        mobind_idx = autism_to_mobind[autism_name]
        acc = imu[:, mobind_idx, 0:3].astype(np.float32)      # (T, 3)
        quat = imu[:, mobind_idx, 3:7].astype(np.float32)     # (T, 4), wxyz
        rot = quat_to_rotmat(quat).reshape(T, 9)              # (T, 9)
        out[:, i * 9:(i + 1) * 9] = rot
        out[:, 36 + i * 3:36 + (i + 1) * 3] = acc

    return out


def pose2d_to_bbox(pose2d: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Derive a single bbox from COCO 17 2D keypoints (T, 17, 2).
    Returns (T, 4) in [x1, y1, x2, y2]."""
    T = pose2d.shape[0]
    valid = pose2d > 0
    bboxes = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        pts = pose2d[t]
        mask = np.logical_and(pts[:, 0] > 0, pts[:, 1] > 0)
        if mask.sum() == 0:
            continue
        xs = pts[mask, 0]
        ys = pts[mask, 1]
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1
        dx = w * margin
        dy = h * margin
        bboxes[t] = np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy], dtype=np.float32)
    return bboxes


def convert_sequence(mobind_files: list[Path], output_npz: Path, action_id: str, seq_id: str) -> None:
    """Convert all person .npy files for one sequence into one unified NPZ."""
    persons = []
    for p in sorted(mobind_files):
        data = np.load(p, allow_pickle=True).item()
        persons.append({
            'imu': mobind_imu_to_48(data['imu']),
            'pose2d': data['pose2d'].astype(np.float32),
            'pose3d': smpl24_to_h36m17(data['pose3d']),
        })

    # Align time length across persons
    tlen = min(p['imu'].shape[0] for p in persons)
    for p in persons:
        p['imu'] = p['imu'][:tlen]
        p['pose2d'] = p['pose2d'][:tlen]
        p['pose3d'] = p['pose3d'][:tlen]

    n_persons = len(persons)
    imu = np.stack([p['imu'] for p in persons], axis=1).astype(np.float32)        # (T, N, 48)
    gt_skeleton = np.stack([p['pose3d'] for p in persons], axis=1).astype(np.float32)  # (T, N, 17, 3)
    gt_bboxes = np.stack([pose2d_to_bbox(p['pose2d']) for p in persons], axis=1).astype(np.float32)  # (T, N, 4)
    gt_visibility = np.ones((tlen, n_persons), dtype=bool)

    imu_ids = np.arange(n_persons, dtype=np.int64)
    gt_person_ids = np.arange(n_persons, dtype=np.int64)
    frame_ids = np.arange(tlen, dtype=np.int64)

    # Use 'custom_' prefix so the existing TotalCaptureAdapter parses the session
    # correctly for session-based train/val/test splits.
    sequence_id = f"custom_{action_id}_{seq_id}"
    output_npz = output_npz.parent / f"{sequence_id}.npz"
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        video_path=np.array("", dtype=object),
        dataset=np.array("egohumans", dtype=object),
        sequence_id=np.array(sequence_id, dtype=object),
        frame_ids=frame_ids,
        imu=imu,
        imu_ids=imu_ids,
        gt_person_ids=gt_person_ids,
        gt_bboxes=gt_bboxes,
        gt_visibility=gt_visibility,
        gt_skeleton=gt_skeleton,
        gt_skeleton_meters=gt_skeleton.copy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/data/lyxie/ReID/Data/egohumans/extracted_data')
    ap.add_argument('--output', default='./data/interim/egohumans/preprocess/sequences')
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by sequence: MoBInd filename is {action_id}_{seq_id}_{person_id}.npy
    pattern = re.compile(r'^(\d+)_(\d+)_(.+?)\.npy$')
    groups = {}
    for npy_path in sorted(input_dir.glob('*.npy')):
        m = pattern.match(npy_path.name)
        if not m:
            print(f"Skipping unexpected file: {npy_path.name}")
            continue
        action_id, seq_id, person_id = m.groups()
        seq_key = (action_id, seq_id)
        groups.setdefault(seq_key, []).append(npy_path)

    print(f"Found {len(groups)} sequences with {sum(len(v) for v in groups.values())} person files.")

    for (action_id, seq_id), files in sorted(groups.items()):
        out_npz = output_dir / f"egohumans_{action_id}_{seq_id}.npz"
        if out_npz.exists():
            print(f"Skipping existing {out_npz.name}")
            continue
        convert_sequence(files, out_npz, action_id, seq_id)
        print(f"Saved {out_npz.name} ({len(files)} persons)")

    print(f"Done. Output: {output_dir}")


if __name__ == '__main__':
    main()
