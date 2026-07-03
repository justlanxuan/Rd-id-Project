#!/usr/bin/env python
"""
Convert realistic EgoHumans synthetic IMU (acc + quat + gyro + mag) into
Autism-project unified NPZ format.

Inputs:
    /data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data/
    └── 01_001_aria01.npy   # keys: imu (T,5,13), acc (T,5,3), quat (T,5,4), ...

Outputs:
    data/interim/egohumans_realistic/sequences/
    └── custom_{action_id}_{seq_id}.npz
        keys: video_path, dataset, sequence_id, frame_ids,
              imu (T, N_imu, 48), imu_ids (N_imu,),
              gt_person_ids (N_gt,), gt_bboxes (T, N_gt, 4),
              gt_visibility (T, N_gt,), gt_skeleton (T, N_gt, 17, 3)
"""
import argparse
import re
from pathlib import Path

import numpy as np

from convert_mobind_to_pipeline import (
    smpl24_to_h36m17,
    AUTISM_SENSOR_ORDER,
    MOBIND_TO_AUTISM,
    quat_to_rotmat,
    pose2d_to_bbox,
)


def realistic_imu_to_48(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert realistic (T, 5, 3) acc + (T, 5, 4) quat to Autism (T, 48)."""
    # Reorder MoBInd sensors to Autism sensor order, drop Head
    MOBIND_SENSOR_ORDER = ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]
    autism_to_mobind = {
        autism_name: MOBIND_SENSOR_ORDER.index(mobind_name)
        for mobind_name, autism_name in MOBIND_TO_AUTISM.items()
    }

    T = acc.shape[0]
    out = np.zeros((T, 48), dtype=np.float32)

    for i, autism_name in enumerate(AUTISM_SENSOR_ORDER):
        mobind_idx = autism_to_mobind[autism_name]
        a = acc[:, mobind_idx, :].astype(np.float32)
        q = quat[:, mobind_idx, :].astype(np.float32)
        rot = quat_to_rotmat(q).reshape(T, 9)
        out[:, i * 9:(i + 1) * 9] = rot
        out[:, 36 + i * 3:36 + (i + 1) * 3] = a

    return out


def convert_sequence(mobind_files: list[Path], output_npz: Path, action_id: str, seq_id: str) -> None:
    persons = []
    for p in sorted(mobind_files):
        data = np.load(p, allow_pickle=True).item()
        persons.append({
            'imu': realistic_imu_to_48(data['acc'], data['quat']),
            'pose2d': data['pose2d'].astype(np.float32),
            'pose3d': smpl24_to_h36m17(data['pose3d']),
        })

    tlen = min(p['imu'].shape[0] for p in persons)
    for p in persons:
        p['imu'] = p['imu'][:tlen]
        p['pose2d'] = p['pose2d'][:tlen]
        p['pose3d'] = p['pose3d'][:tlen]

    n_persons = len(persons)
    imu = np.stack([p['imu'] for p in persons], axis=1).astype(np.float32)
    gt_skeleton = np.stack([p['pose3d'] for p in persons], axis=1).astype(np.float32)
    gt_bboxes = np.stack([pose2d_to_bbox(p['pose2d']) for p in persons], axis=1).astype(np.float32)
    gt_visibility = np.ones((tlen, n_persons), dtype=bool)

    imu_ids = np.arange(n_persons, dtype=np.int64)
    gt_person_ids = np.arange(n_persons, dtype=np.int64)
    frame_ids = np.arange(tlen, dtype=np.int64)

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
    ap.add_argument('--input', default='/data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data')
    ap.add_argument('--output', default='./data/interim/egohumans_realistic/sequences')
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r'^(\d+)_(\d+)_(.+?)\.npy$')
    groups = {}
    for npy_path in sorted(input_dir.glob('*.npy')):
        # Skip sequences without pose2d for the current supervised pipeline
        if any(npy_path.name.startswith(f'03_0{i}_') for i in (11, 12, 13, 14)):
            continue
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
