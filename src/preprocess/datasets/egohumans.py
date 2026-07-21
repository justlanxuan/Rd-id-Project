"""EgoHumans preprocessing pipeline.

The dataset entrypoint intentionally exposes the dataset-owned production
stages:

1. extract: run true video skeleton extraction through the shared dispatcher.
2. preprocess: convert raw IMU arrays and optional stored pose files.
3. pack: group processed people into unified per-sequence NPZ files.
4. slice: create train/val/test window CSV files from unified NPZ files.

Converting EgoHumans' bundled pose2d files is preprocessing, not extraction.
Set ``preprocess.skeleton_source: pose2d`` to materialize those files as
``skeleton.json`` and skip the extract stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.preprocess.common.extract import run_video_skeleton_extraction
from src.preprocess.common.slice import load_slice_cfg
from src.preprocess.common.video import pose2d_to_bbox, write_video_manifest
from src.utils.config import load_config


SMPL24_TO_H36M17 = [
    0, 2, 5, 8, 1, 4, 7, 6, 9, 12, 15, 16, 18, 20, 17, 19, 21,
]
MOBIND_SENSOR_ORDER = ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]
LEFT_WRIST_INDEX = MOBIND_SENSOR_ORDER.index("LeftWrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EgoHumans preprocess")
    parser.add_argument("--task", choices=["extract", "preprocess", "pack", "slice"], default="preprocess")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override preprocess output directory")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Override video manifest path")
    parser.add_argument("--max_sequences", type=int, default=0, help="Optional debug limit")
    return parser.parse_args()


def load_preprocess_cfg(config_path: str) -> dict[str, Any]:
    data = load_config(config_path)
    preprocess = data.get("preprocess", {})
    if preprocess is None:
        return {}
    if not isinstance(preprocess, dict):
        raise ValueError(f"Invalid preprocess section in config: {config_path}")
    return preprocess


def output_paths(args: argparse.Namespace, preprocess_cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    manifest = Path(args.manifest_csv or preprocess_cfg.get(
        "output",
        "/data/fzliang/reid-project/egohumans/preprocessed/default/video_manifest.csv",
    )).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else manifest.parent
    processed_dir = Path(preprocess_cfg.get("processed_root", output_root / "processed")).expanduser().resolve()
    sequence_dir = Path(preprocess_cfg.get("sequence_root", output_root / "sequences")).expanduser().resolve()
    return output_root, processed_dir, sequence_dir, manifest


def raw_root_from_cfg(preprocess_cfg: dict[str, Any]) -> Path:
    return Path(preprocess_cfg.get("raw_root", "/data/lyxie/ReID/Data/egohumans")).expanduser().resolve()


def extracted_root_from_cfg(preprocess_cfg: dict[str, Any]) -> Path:
    imu_source = str(preprocess_cfg.get("imu_source") or "realistic").strip().lower()
    default = (
        "/data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data"
        if imu_source == "realistic"
        else "/data/lyxie/ReID/Data/egohumans/extracted_data"
    )
    return Path(preprocess_cfg.get("extracted_root", default)).expanduser().resolve()


def skeleton_root_from_cfg(preprocess_cfg: dict[str, Any]) -> Path:
    return Path(preprocess_cfg.get(
        "pose2d_output_root",
        "/data/fzliang/reid-project/egohumans/skeleton/alphapose",
    )).expanduser().resolve()


def raw_workflow_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).expanduser().open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    return data


def egohumans_camera(config_path: str, preprocess_cfg: dict[str, Any]) -> str:
    raw = raw_workflow_config(config_path)
    raw_preprocess = raw.get("preprocess", {})
    if isinstance(raw_preprocess, dict) and raw_preprocess.get("camera"):
        return str(raw_preprocess["camera"])
    value = str(preprocess_cfg.get("camera") or "").strip()
    return value if value and value != "cam1" else "cam03"


def smpl24_to_h36m17(smpl_joints: np.ndarray) -> np.ndarray:
    return smpl_joints[:, SMPL24_TO_H36M17, :].astype(np.float32)


def mobind_leftwrist_imu(imu: np.ndarray) -> np.ndarray:
    return imu[:, LEFT_WRIST_INDEX, 0:7].astype(np.float32)


def realistic_leftwrist_imu(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [acc[:, LEFT_WRIST_INDEX, :].astype(np.float32), quat[:, LEFT_WRIST_INDEX, :].astype(np.float32)],
        axis=-1,
    )


def convert_pose2d_sequence(pose2d_dir: Path, out_json: Path) -> bool:
    npy_files = sorted(pose2d_dir.glob("*.npy"))
    if not npy_files:
        return False

    entries: list[dict[str, Any]] = []
    name_to_track_id: dict[str, int] = {}
    for npy_file in npy_files:
        frame_idx = int(npy_file.stem) - 1
        if frame_idx < 0:
            continue
        frame_data = np.load(npy_file, allow_pickle=True)
        for det in frame_data:
            name = str(det.get("human_name", ""))
            if name not in name_to_track_id:
                name_to_track_id[name] = len(name_to_track_id)
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(float, bbox[:4])
            coco17 = det["keypoints"][:17].astype(float)
            keypoints = []
            for joint_idx in range(17):
                keypoints.extend([
                    float(coco17[joint_idx, 0]),
                    float(coco17[joint_idx, 1]),
                    float(coco17[joint_idx, 2]),
                ])
            entries.append({
                "image_id": f"{frame_idx:05d}.jpg",
                "category_id": 1,
                "keypoints": keypoints,
                "score": float(bbox[4]) if len(bbox) > 4 else float(np.mean(coco17[:, 2])),
                "box": [x1, y1, x2 - x1, y2 - y1],
                "idx": name_to_track_id[name],
            })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(entries))
    print(f"[preprocess:pose2d] {out_json} ({len(entries)} detections)")
    return True


def materialize_pose2d_skeletons(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    data_root = raw_root_from_cfg(cfg) / "data"
    output_root = skeleton_root_from_cfg(cfg)
    camera = egohumans_camera(args.config, cfg)
    suffixes = ("_tagging", "_lego", "_fencing", "_basketball", "_volleyball", "_badminton", "_tennis")
    action_dirs = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.endswith(suffixes))

    converted = 0
    for action_dir in action_dirs:
        for seq_dir in sorted(d for d in action_dir.iterdir() if d.is_dir()):
            if args.max_sequences and converted >= args.max_sequences:
                print(f"[preprocess:pose2d] done: {converted} sequences -> {output_root}")
                return converted
            pose2d_dir = seq_dir / "processed_data" / "poses2d" / camera / "rgb"
            if not pose2d_dir.exists():
                continue
            action_id = action_dir.name.split("_")[0]
            seq_id = seq_dir.name.split("_")[0]
            out_json = output_root / f"egohumans_{action_id}_{seq_id}" / "skeleton.json"
            if out_json.exists() and bool(cfg.get("skip_existing", True)):
                print(f"[preprocess:pose2d] skip existing {out_json}")
                converted += 1
                continue
            if convert_pose2d_sequence(pose2d_dir, out_json):
                converted += 1
    print(f"[preprocess:pose2d] done: {converted} sequences -> {output_root}")
    return converted


def should_materialize_pose2d_skeletons(config_path: str, preprocess_cfg: dict[str, Any]) -> bool:
    if bool(preprocess_cfg.get("pose2d_to_skeleton", False)):
        return True
    source = str(preprocess_cfg.get("skeleton_source", "")).strip().lower()
    if source in {"pose2d", "stored_pose", "egohumans_pose"}:
        return True
    raw = raw_workflow_config(config_path)
    slice_cfg = raw.get("slice", {})
    has_explicit_extract = isinstance(raw.get("extract"), dict)
    return (
        isinstance(slice_cfg, dict)
        and str(slice_cfg.get("skeleton_source", "")).strip().lower() == "alphapose"
        and not has_explicit_extract
    )


def write_egohumans_extract_manifest(manifest_csv: Path, rows: list[tuple[Path, str]]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "sequence_id", "result_name"])
        writer.writeheader()
        for video_path, sequence_id in rows:
            writer.writerow({
                "video_path": str(video_path),
                "sequence_id": sequence_id,
                "result_name": sequence_id,
            })


def iter_egohumans_videos(config_path: str, preprocess_cfg: dict[str, Any]) -> list[tuple[Path, str]]:
    data_root = raw_root_from_cfg(preprocess_cfg) / "data"
    camera = egohumans_camera(config_path, preprocess_cfg)
    suffixes = ("_tagging", "_lego", "_fencing", "_basketball", "_volleyball", "_badminton", "_tennis")
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos: list[tuple[Path, str]] = []
    for action_dir in sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.endswith(suffixes)):
        for seq_dir in sorted(d for d in action_dir.iterdir() if d.is_dir()):
            action_id = action_dir.name.split("_")[0]
            seq_id = seq_dir.name.split("_")[0]
            sequence_id = f"egohumans_{action_id}_{seq_id}"
            cam_dir = seq_dir / "exo" / camera
            if not cam_dir.exists():
                continue
            direct = sorted(path for path in cam_dir.iterdir() if path.is_file() and path.suffix.lower() in video_exts)
            if direct:
                videos.extend((path, sequence_id) for path in direct)
                continue
            recursive = sorted(path for path in cam_dir.rglob("*") if path.is_file() and path.suffix.lower() in video_exts)
            videos.extend((path, sequence_id) for path in recursive)
    return videos


def run_extract(args: argparse.Namespace) -> None:
    raw = raw_workflow_config(args.config)
    if not isinstance(raw.get("extract"), dict):
        print("[extract] skipped: no explicit extract section in config.")
        return

    cfg = load_preprocess_cfg(args.config)
    _, _, _, manifest_csv = output_paths(args, cfg)
    videos = iter_egohumans_videos(args.config, cfg)
    if args.max_sequences:
        videos = videos[: args.max_sequences]
    if not videos:
        raise FileNotFoundError(
            f"No EgoHumans video files found under {raw_root_from_cfg(cfg) / 'data'} "
            f"for camera={egohumans_camera(args.config, cfg)}. "
            "If you are using bundled processed_data/poses2d, skip extract and run preprocess."
        )
    write_egohumans_extract_manifest(manifest_csv, videos)
    print(f"[extract] manifest: {manifest_csv} ({len(videos)} videos)")
    run_video_skeleton_extraction(args.config)


def processed_person_path(processed_dir: Path, action_id: str, seq_id: str, person_name: str) -> Path:
    return processed_dir / f"egohumans_{action_id}_{seq_id}" / f"{person_name}.npz"


def iter_raw_person_arrays(extracted_root: Path) -> dict[tuple[str, str], list[Path]]:
    pattern = re.compile(r"^(\d+)_(\d+)_(.+?)\.npy$")
    groups: dict[tuple[str, str], list[Path]] = {}
    for npy_path in sorted(extracted_root.glob("*.npy")):
        match = pattern.match(npy_path.name)
        if not match:
            print(f"[preprocess] skip unexpected file: {npy_path.name}")
            continue
        action_id, seq_id, _ = match.groups()
        groups.setdefault((action_id, seq_id), []).append(npy_path)
    return groups


def run_preprocess(args: argparse.Namespace) -> None:
    cfg = load_preprocess_cfg(args.config)
    _, processed_dir, _, _ = output_paths(args, cfg)
    extracted_root = extracted_root_from_cfg(cfg)
    imu_source = str(cfg.get("imu_source") or "realistic").strip().lower()
    if not extracted_root.exists():
        raise FileNotFoundError(f"EgoHumans extracted_root not found: {extracted_root}")
    if imu_source not in {"realistic", "mobind"}:
        raise ValueError(f"Unsupported EgoHumans imu_source={imu_source!r}; expected 'realistic' or 'mobind'.")

    groups = iter_raw_person_arrays(extracted_root)
    processed = 0
    for (action_id, seq_id), files in sorted(groups.items()):
        if args.max_sequences and processed >= args.max_sequences:
            break
        if imu_source == "realistic" and action_id == "03" and seq_id in {"011", "012", "013", "014"}:
            continue
        for path in sorted(files):
            out_path = processed_person_path(processed_dir, action_id, seq_id, path.stem)
            if out_path.exists() and bool(cfg.get("skip_existing", True)):
                continue
            raw = np.load(path, allow_pickle=True).item()
            imu = realistic_leftwrist_imu(raw["acc"], raw["quat"]) if imu_source == "realistic" else mobind_leftwrist_imu(raw["imu"])
            pose2d = raw["pose2d"].astype(np.float32)
            pose3d = smpl24_to_h36m17(raw["pose3d"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, imu=imu, pose2d=pose2d, pose3d=pose3d)
        processed += 1

    if should_materialize_pose2d_skeletons(args.config, cfg):
        materialize_pose2d_skeletons(args, cfg)

    print(f"[preprocess] done: {processed} sequences -> {processed_dir}")


def save_sequence_npz(persons: list[dict[str, np.ndarray]], output_dir: Path, sequence_id: str) -> Path:
    t_len = min(person["imu"].shape[0] for person in persons)
    imu = np.stack([person["imu"][:t_len] for person in persons], axis=1).astype(np.float32)
    pose2d = np.stack([person["pose2d"][:t_len] for person in persons], axis=1).astype(np.float32)
    pose3d = np.stack([person["pose3d"][:t_len] for person in persons], axis=1).astype(np.float32)
    n_persons = len(persons)

    out_npz = output_dir / f"{sequence_id}.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        video_path=np.array("", dtype=object),
        dataset=np.array("egohumans", dtype=object),
        sequence_id=np.array(sequence_id, dtype=object),
        frame_ids=np.arange(t_len, dtype=np.int64),
        imu=imu,
        imu_ids=np.arange(n_persons, dtype=np.int64),
        gt_person_ids=np.arange(n_persons, dtype=np.int64),
        gt_bboxes=np.stack([pose2d_to_bbox(pose2d[:, person_idx]) for person_idx in range(n_persons)], axis=1),
        gt_visibility=np.ones((t_len, n_persons), dtype=bool),
        gt_skeleton=pose3d,
        gt_skeleton_meters=pose3d,
    )
    return out_npz


def run_pack(args: argparse.Namespace) -> None:
    cfg = load_preprocess_cfg(args.config)
    _, processed_dir, sequence_dir, manifest_csv = output_paths(args, cfg)
    if not processed_dir.exists():
        raise FileNotFoundError(f"EgoHumans processed directory not found: {processed_dir}")

    packed = 0
    for seq_dir in sorted(d for d in processed_dir.iterdir() if d.is_dir()):
        if args.max_sequences and packed >= args.max_sequences:
            break
        sequence_id = seq_dir.name
        out_npz = sequence_dir / f"{sequence_id}.npz"
        if out_npz.exists() and bool(cfg.get("skip_existing", True)):
            packed += 1
            continue
        persons = [dict(np.load(path, allow_pickle=True)) for path in sorted(seq_dir.glob("*.npz"))]
        if not persons:
            continue
        save_sequence_npz(persons, sequence_dir, sequence_id)
        packed += 1

    write_video_manifest(manifest_csv, [])
    print(f"[pack] done: {packed} unified sequences -> {sequence_dir}")
    print(f"[pack] manifest: {manifest_csv}")


def run_slice(args: argparse.Namespace) -> None:
    from src.preprocess.common.slice import run_slice_from_npz

    cfg = dict(load_slice_cfg(args.config))
    cfg.setdefault("multi_person", True)
    cfg.setdefault("skeleton_source", "vicon")
    root = Path(cfg.get("root", "/data/fzliang/totalcapture"))
    out_dir = Path(cfg.get("out_dir", "/data/fzliang/reid-project/totalcapture/preprocessed/default"))
    run_slice_from_npz(root, out_dir, cfg)


def main() -> None:
    args = parse_args()
    if args.task == "extract":
        run_extract(args)
    elif args.task == "preprocess":
        run_preprocess(args)
    elif args.task == "pack":
        run_pack(args)
    elif args.task == "slice":
        run_slice(args)
    else:
        raise ValueError(f"Unsupported EgoHumans task: {args.task}")


if __name__ == "__main__":
    main()
