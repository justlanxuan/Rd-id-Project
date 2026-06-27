"""Convert HRli annotated data to existing custom dataset format.

Input: /home/hrli/data_annotation/annotation/batch_20260505_v02/auto_clean_after_stage_1/
Output: /data/fzliang/custom/batch_20260505/
"""

import csv
import json
import shutil
from pathlib import Path


def parse_hrli_annotation(anno_path: Path) -> dict:
    """Parse single-person HRli annotation CSV."""
    with anno_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    frames = {}
    for row in rows:
        fid = int(row["frame_index"])
        frames[fid] = {
            "timestamp_ms": float(row["timestamp_ms"]),
            "bbox_x": float(row["bbox_x"]),
            "bbox_y": float(row["bbox_y"]),
            "bbox_w": float(row["bbox_w"]),
            "bbox_h": float(row["bbox_h"]),
        }
    return frames


def main() -> None:
    src_root = Path("/home/hrli/data_annotation/annotation/batch_20260505_v02/auto_clean_after_stage_1")
    dst_root = Path("/data/fzliang/custom/batch_20260505")
    dst_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for segment_dir in sorted(src_root.iterdir()):
        if not segment_dir.is_dir():
            continue

        segment_id = segment_dir.name
        metadata_path = segment_dir / "metadata.json"

        if not metadata_path.exists():
            print(f"[SKIP] {segment_id}: no metadata.json")
            skipped += 1
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        exported = metadata.get("exported_imu_annotations", [])
        if not exported:
            print(f"[SKIP] {segment_id}: no exported annotations")
            skipped += 1
            continue

        n_persons = len(exported)
        print(f"[INFO] {segment_id}: {n_persons} persons")

        # Organize by person count
        person_count_dir = dst_root / f"{n_persons}person"
        person_count_dir.mkdir(exist_ok=True)
        anno_dir = person_count_dir / "annotations"
        anno_dir.mkdir(exist_ok=True)

        # Create session directory structure
        session_dir = person_count_dir / segment_id
        session_dir.mkdir(exist_ok=True)
        (session_dir / "imu").mkdir(exist_ok=True)
        (session_dir / "video").mkdir(exist_ok=True)

        # Copy/rename video file
        video_files = list((segment_dir / "video").glob("*_retimed.mp4"))
        if video_files:
            src_video = video_files[0]
            dst_video = session_dir / "video" / f"{segment_id}.mp4"
            if not dst_video.exists():
                shutil.copy2(str(src_video), str(dst_video))
                print(f"  [COPY] video -> {dst_video}")

        # Copy IMU files with renamed format
        imu_mapping = {}
        for i, person_info in enumerate(exported):
            imu_id = person_info["imu_id"]
            person_key = f"person{i + 1}"
            imu_mapping[person_key] = imu_id

            # Find and copy IMU file
            imu_files = list((segment_dir / "imu").glob(f"*{imu_id}*.csv"))
            if imu_files:
                src_imu = imu_files[0]
                dst_imu = session_dir / "imu" / f"{segment_id}_{imu_id}.csv"
                if not dst_imu.exists():
                    shutil.copy2(str(src_imu), str(dst_imu))

        # Generate per-session imu_person_mapping.json
        mapping_path = anno_dir / "imu_person_mapping.json"
        existing_mapping = {}
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                existing_mapping = json.load(f)
        existing_mapping.update(imu_mapping)
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(existing_mapping, f, indent=2, ensure_ascii=False)

        # Merge annotations into .anno.csv format
        all_person_frames = []
        for i, person_info in enumerate(exported):
            imu_id = person_info["imu_id"]
            anno_file = segment_dir / "annotation" / f"{imu_id}.csv"
            if anno_file.exists():
                frames = parse_hrli_annotation(anno_file)
                all_person_frames.append(frames)

        # Find all unique frame indices
        all_frame_ids = set()
        for frames in all_person_frames:
            all_frame_ids.update(frames.keys())
        sorted_frames = sorted(all_frame_ids)

        if not sorted_frames:
            print(f"[SKIP] {segment_id}: no frames")
            skipped += 1
            continue

        # Write .anno.csv
        anno_csv_path = anno_dir / f"{segment_id}.anno.csv"
        with anno_csv_path.open("w", newline="", encoding="utf-8") as f:
            # Build header
            fields = ["video_stem", "frame_index", "timestamp_ms"]
            for p in range(1, len(all_person_frames) + 1):
                fields.extend([
                    f"p{p}_bbox_x", f"p{p}_bbox_y", f"p{p}_bbox_w", f"p{p}_bbox_h",
                    f"p{p}_is_absent",
                ])
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for fid in sorted_frames:
                # Get timestamp from first available person
                ts = 0.0
                for frames in all_person_frames:
                    if fid in frames:
                        ts = frames[fid]["timestamp_ms"]
                        break

                row = {
                    "video_stem": segment_id,
                    "frame_index": fid,
                    "timestamp_ms": ts,
                }

                for p, frames in enumerate(all_person_frames, start=1):
                    if fid in frames:
                        info = frames[fid]
                        row[f"p{p}_bbox_x"] = info["bbox_x"]
                        row[f"p{p}_bbox_y"] = info["bbox_y"]
                        row[f"p{p}_bbox_w"] = info["bbox_w"]
                        row[f"p{p}_bbox_h"] = info["bbox_h"]
                        row[f"p{p}_is_absent"] = 0
                    else:
                        row[f"p{p}_bbox_x"] = 0
                        row[f"p{p}_bbox_y"] = 0
                        row[f"p{p}_bbox_w"] = 0
                        row[f"p{p}_bbox_h"] = 0
                        row[f"p{p}_is_absent"] = 1

                writer.writerow(row)

        processed += 1
        print(f"  [OK] anno.csv -> {anno_csv_path} ({len(sorted_frames)} frames)")

    print(f"\nDone: {processed} processed, {skipped} skipped")
    print(f"Output: {dst_root}")


if __name__ == "__main__":
    main()
