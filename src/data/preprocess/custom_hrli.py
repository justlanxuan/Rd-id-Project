"""Preprocess for HRli annotated dataset (batch_20260505_v02).

Reads:
  /home/hrli/data_annotation/annotation/batch_20260505_v02/auto_clean_after_stage_1/
  - <segment_id>/
    - annotation/<mac>.csv       (per-person frame annotations)
    - imu/<seg>_<mac>.csv        (per-person IMU data)
    - video/<seg>_retimed.mp4    (video file)
    - metadata.json              (person mapping)

Outputs (compatible with existing custom pipeline):
  data/interim/custom_hrli/
  - sequences/custom_hrli_<segment_id>.npz
  - sequences/custom_hrli_<segment_id>.json
  - video_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.preprocess.custom import (
    convert_single_imu_to_48,
    lowpass_filter_fft,
    parse_imu_csv,
    resample_imu_to_target,
)
from src.datasets.totalcapture import quat_to_rotmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HRli custom dataset preprocess")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/home/hrli/data_annotation/annotation/batch_20260505_v02/auto_clean_after_stage_1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/interim/custom_hrli",
    )
    parser.add_argument(
        "--min_persons",
        type=int,
        default=4,
        help="Minimum number of persons to include a segment",
    )
    parser.add_argument(
        "--max_persons",
        type=int,
        default=7,
        help="Maximum number of persons to include a segment",
    )
    parser.add_argument(
        "--imu_lowpass_cutoff_hz",
        type=float,
        default=None,
        help="IMU low-pass filter cutoff (None to skip)",
    )
    parser.add_argument(
        "--imu_lowpass_fs_hz",
        type=float,
        default=100.0,
        help="IMU sampling frequency for filter",
    )
    return parser.parse_args()


def parse_hrli_annotation(anno_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse single-person HRli annotation CSV.

    Returns:
        frame_indices: [T] int64
        timestamps_ms: [T] float64
        bboxes: [T, 4] float32 in [x, y, w, h]
    """
    with anno_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty annotation file: {anno_path}")

    T = len(rows)
    frame_indices = np.zeros(T, dtype=np.int64)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    bboxes = np.zeros((T, 4), dtype=np.float32)

    for t, row in enumerate(rows):
        frame_indices[t] = int(row["frame_index"])
        timestamps_ms[t] = float(row["timestamp_ms"])
        bboxes[t] = np.array(
            [
                float(row["bbox_x"]),
                float(row["bbox_y"]),
                float(row["bbox_w"]),
                float(row["bbox_h"]),
            ],
            dtype=np.float32,
        )

    return frame_indices, timestamps_ms, bboxes


def merge_annotations(
    anno_paths: List[Path],
    person_names: List[str],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge per-person annotations into unified frame-level annotations.

    Returns:
        n_persons: number of persons
        frame_indices: [T] int64
        timestamps_ms: [T] float64
        bboxes: [T, N, 4] float32 in [x1, y1, x2, y2]
        visibility: [T, N] bool
    """
    n_persons = len(anno_paths)

    # Collect all unique frame indices and timestamps
    all_frames: Dict[int, float] = {}
    person_bboxes: List[Dict[int, np.ndarray]] = [{} for _ in range(n_persons)]

    for p, anno_path in enumerate(anno_paths):
        frame_indices, timestamps_ms, bboxes = parse_hrli_annotation(anno_path)
        for t in range(len(frame_indices)):
            fid = int(frame_indices[t])
            all_frames[fid] = timestamps_ms[t]
            person_bboxes[p][fid] = np.array(
                [
                    bboxes[t, 0],
                    bboxes[t, 1],
                    bboxes[t, 0] + bboxes[t, 2],
                    bboxes[t, 1] + bboxes[t, 3],
                ],
                dtype=np.float32,
            )

    # Sort by frame index
    sorted_frames = sorted(all_frames.items())
    T = len(sorted_frames)

    frame_indices = np.zeros(T, dtype=np.int64)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    bboxes = np.zeros((T, n_persons, 4), dtype=np.float32)
    visibility = np.zeros((T, n_persons), dtype=bool)

    for t, (fid, ts) in enumerate(sorted_frames):
        frame_indices[t] = fid
        timestamps_ms[t] = ts
        for p in range(n_persons):
            if fid in person_bboxes[p]:
                bboxes[t, p] = person_bboxes[p][fid]
                visibility[t, p] = True

    return n_persons, frame_indices, timestamps_ms, bboxes, visibility


def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    processed_count = 0
    skipped_count = 0

    for segment_dir in sorted(raw_root.iterdir()):
        if not segment_dir.is_dir():
            continue

        segment_id = segment_dir.name
        sequence_id = f"custom_hrli_{segment_id}"

        # Check required files
        metadata_path = segment_dir / "metadata.json"
        video_dir = segment_dir / "video"
        imu_dir = segment_dir / "imu"
        anno_dir = segment_dir / "annotation"

        if not metadata_path.exists():
            print(f"[SKIP] {sequence_id}: no metadata.json")
            skipped_count += 1
            continue

        # Load metadata
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        exported = metadata.get("exported_imu_annotations", [])
        if not exported:
            print(f"[SKIP] {sequence_id}: no exported annotations")
            skipped_count += 1
            continue

        n_persons = len(exported)
        if n_persons < args.min_persons or n_persons > args.max_persons:
            print(
                f"[SKIP] {sequence_id}: {n_persons} persons (not in [{args.min_persons}, {args.max_persons}])"
            )
            skipped_count += 1
            continue

        # Find video file
        video_files = list(video_dir.glob("*_retimed.mp4"))
        if not video_files:
            print(f"[SKIP] {sequence_id}: no video file")
            skipped_count += 1
            continue
        video_path = video_files[0]

        # Build ordered lists of annotation and IMU files
        anno_paths: List[Path] = []
        imu_paths: List[Path] = []
        person_names: List[str] = []

        for person_info in exported:
            imu_id = person_info["imu_id"]
            person_name = person_info.get("person_name", "unknown")

            # Find annotation file by MAC address
            anno_file = anno_dir / f"{imu_id}.csv"
            if not anno_file.exists():
                print(f"[WARN] {sequence_id}: annotation not found for {imu_id}")
                continue

            # Find IMU file by MAC address (contains imu_id in filename)
            imu_files = [f for f in imu_dir.glob("*.csv") if imu_id in f.name]
            if not imu_files:
                print(f"[WARN] {sequence_id}: IMU not found for {imu_id}")
                continue

            anno_paths.append(anno_file)
            imu_paths.append(imu_files[0])
            person_names.append(person_name)

        n_valid = len(anno_paths)
        if n_valid < args.min_persons:
            print(f"[SKIP] {sequence_id}: only {n_valid} valid persons")
            skipped_count += 1
            continue

        # Merge annotations
        try:
            n_persons, frame_indices, anno_ts, anno_bboxes, anno_visibility = merge_annotations(
                anno_paths, person_names
            )
        except Exception as e:
            print(f"[ERROR] {sequence_id}: failed to merge annotations: {e}")
            skipped_count += 1
            continue

        # Load and resample IMU
        imu_data_list = []
        for imu_path in imu_paths:
            try:
                imu_ts, quat4, acc3 = parse_imu_csv(imu_path)
                imu48 = convert_single_imu_to_48(quat4, acc3)
                if args.imu_lowpass_cutoff_hz is not None:
                    imu48 = lowpass_filter_fft(
                        imu48, args.imu_lowpass_cutoff_hz, args.imu_lowpass_fs_hz
                    )
                imu_data_list.append((imu_ts, imu48))
            except Exception as e:
                print(f"[WARN] {sequence_id}: failed to parse IMU {imu_path.name}: {e}")
                continue

        if len(imu_data_list) < args.min_persons:
            print(f"[SKIP] {sequence_id}: only {len(imu_data_list)} valid IMUs")
            skipped_count += 1
            continue

        # Find overlapping time range
        valid_start_ms = max(data[0][0] for data in imu_data_list)
        valid_end_ms = min(data[0][-1] for data in imu_data_list)

        crop_mask = (anno_ts >= valid_start_ms) & (anno_ts <= valid_end_ms)
        if not crop_mask.any():
            print(f"[SKIP] {sequence_id}: no overlap between IMU and annotation")
            skipped_count += 1
            continue

        crop_indices = np.where(crop_mask)[0]
        first_idx = int(crop_indices[0])
        last_idx = int(crop_indices[-1])

        target_ts = anno_ts[first_idx : last_idx + 1]
        T = len(target_ts)
        out_frame_ids = np.arange(T, dtype=np.int64)
        bboxes = anno_bboxes[first_idx : last_idx + 1]
        visibility = anno_visibility[first_idx : last_idx + 1]

        n_imu = len(imu_data_list)
        imu_out = np.zeros((T, n_imu, 48), dtype=np.float32)
        for i, (imu_ts, imu48) in enumerate(imu_data_list):
            resampled = resample_imu_to_target(imu_ts, imu48, target_ts)
            nan_mask = np.isnan(resampled).any(axis=1)
            resampled = np.nan_to_num(resampled, nan=0.0)
            imu_out[:, i] = resampled
            if nan_mask.any():
                print(f"[WARN] {sequence_id} IMU {i} has {nan_mask.sum()} out-of-range frames")

        person_ids = np.arange(n_persons, dtype=np.int64)
        imu_ids = np.arange(n_imu, dtype=np.int64)

        # Build MAC-to-person map
        imu_person_map: Dict[str, int] = {}
        for i, fpath in enumerate(imu_paths):
            mac = fpath.stem.split("_")[-1] if "_" in fpath.stem else fpath.stem
            imu_person_map[mac] = int(person_ids[i])

        # Save NPZ
        npz_path = seq_dir / f"{sequence_id}.npz"
        np.savez_compressed(
            npz_path,
            video_path=np.array(str(video_path), dtype=object),
            dataset=np.array("custom_hrli", dtype=object),
            sequence_id=np.array(sequence_id, dtype=object),
            frame_ids=out_frame_ids,
            imu=imu_out,
            imu_ids=imu_ids,
            gt_person_ids=person_ids,
            gt_bboxes=bboxes,
            gt_visibility=visibility,
            imu_person_map=np.array(json.dumps(imu_person_map), dtype=object),
        )

        # Save metadata JSON
        meta = {
            "video_path": str(video_path),
            "dataset": "custom_hrli",
            "sequence_id": sequence_id,
            "n_frames": int(T),
            "n_imu": int(n_imu),
            "n_gt": int(n_persons),
            "has_gt_skeleton": False,
            "imu_ids": imu_ids.tolist(),
            "gt_person_ids": person_ids.tolist(),
            "extract_person_ids": [],
            "imu_person_map": imu_person_map,
        }
        (seq_dir / f"{sequence_id}.json").write_text(json.dumps(meta, indent=2))

        manifest_rows.append({"video_path": str(video_path)})
        processed_count += 1
        print(f"[OK] {sequence_id}: {T} frames, {n_imu} IMUs, {n_persons} persons")

    # Save manifest
    manifest_csv = output_dir / "video_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"\nDone: {processed_count} processed, {skipped_count} skipped")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
