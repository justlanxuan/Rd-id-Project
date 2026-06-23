#!/usr/bin/env python
"""
Convert EgoHumans processed_data/poses2d/cam03/*.npy detections into
Autism-project video-pipeline skeleton.json format.

EgoHumans pose2d .npy contains a 1-D array of dicts, one per person:
    {
        'bbox': [x1, y1, x2, y2, score],
        'human_name': 'aria01',
        'human_id': 0,
        'color': [255, 0, 0],
        'keypoints': (K, 3) array,  # K >= 17; first 17 are COCO body joints
    }

Output skeleton.json is a flat list of AlphaPose-style entries:
    {
        'image_id': f'{frame_idx:05d}.jpg',
        'category_id': 1,
        'keypoints': [x, y, conf, ...] * 17,
        'score': person_score,
        'box': [x, y, w, h],
        'idx': track_id,
    }

The frame index is 0-based to match the NPZ frame_ids.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def convert_sequence(pose2d_dir: Path, out_json: Path) -> None:
    """Convert one sequence's cam03 pose2d .npy files to skeleton.json."""
    npy_files = sorted(pose2d_dir.glob("*.npy"))
    if not npy_files:
        print(f"[WARN] No pose2d files in {pose2d_dir}")
        return

    entries = []
    # Build a stable track_id map from human_name.
    name_to_tid = {}
    next_tid = 0

    for npy_file in npy_files:
        # EgoHumans files are 1-based: 00001.npy -> frame 0 in NPZ.
        frame_idx = int(npy_file.stem) - 1
        if frame_idx < 0:
            continue

        data = np.load(npy_file, allow_pickle=True)
        for det in data:
            name = det.get("human_name", "")
            if name not in name_to_tid:
                name_to_tid[name] = next_tid
                next_tid += 1
            track_id = name_to_tid[name]

            bbox = det["bbox"]  # [x1, y1, x2, y2, score]
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            w, h = x2 - x1, y2 - y1

            kpts = det["keypoints"]  # (K, 3)
            # Take first 17 COCO body joints.
            coco17 = kpts[:17].astype(float)
            keypoints_flat = []
            for j in range(17):
                keypoints_flat.extend([
                    float(coco17[j, 0]),
                    float(coco17[j, 1]),
                    float(coco17[j, 2]),
                ])

            score = float(bbox[4]) if len(bbox) > 4 else float(np.mean(coco17[:, 2]))

            entries.append({
                "image_id": f"{frame_idx:05d}.jpg",
                "category_id": 1,
                "keypoints": keypoints_flat,
                "score": score,
                "box": [x1, y1, w, h],
                "idx": track_id,
            })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(entries, f)
    print(f"Saved {out_json} ({len(entries)} detections, {next_tid} tracks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/lyxie/ReID/Data/egohumans/data")
    ap.add_argument("--output_root", default="/home/fzliang/Autism-project/data/interim/egohumans_extract_poses2d")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    action_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.endswith(
        ("_tagging", "_lego", "_fencing", "_basketball", "_volleyball", "_badminton", "_tennis"))])

    for action_dir in action_dirs:
        seq_dirs = sorted([d for d in action_dir.iterdir() if d.is_dir() and not d.name.endswith(".tar.gz")])
        for seq_dir in seq_dirs:
            pose2d_dir = seq_dir / "processed_data" / "poses2d" / "cam03" / "rgb"
            if not pose2d_dir.exists():
                continue
            action_id = action_dir.name.split("_")[0]
            seq_id = seq_dir.name.split("_")[0]
            seq_key = f"custom_{action_id}_{seq_id}"
            out_json = output_root / seq_key / "skeleton.json"
            if out_json.exists():
                print(f"Skipping existing {out_json}")
                continue
            convert_sequence(pose2d_dir, out_json)

    print(f"Done. Output: {output_root}")


if __name__ == "__main__":
    main()
