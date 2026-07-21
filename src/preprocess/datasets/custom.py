"""Preprocess for Custom dataset: generate standardized NPZ + video manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.preprocess.common import (
    convert_single_imu_to_7d,
    lowpass_filter_fft,
    parse_imu_csv,
    resample_imu_to_target,
    rotmat_to_quat_wxyz,
)
from src.preprocess.common.alphapose import load_alphapose_multiperson
from src.preprocess.common.slice import _align_extract_to_npz, load_slice_cfg, write_csv
from src.utils.config import load_config


def legacy_imu48_sensor_to_7d(imu48: np.ndarray, sensor_name: str = "L_LowArm") -> np.ndarray:
    """Extract acc3 + quat4 from legacy custom 48D IMU.

    Legacy custom 48D stores four sensors in order:
    L_LowLeg, R_LowLeg, L_LowArm, R_LowArm. For each sensor, the first block
    contains a 3x3 rotation matrix and the final block contains acceleration.
    """
    order = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
    if sensor_name not in order:
        raise ValueError(f"Unsupported legacy sensor {sensor_name!r}; expected one of {order}")
    x = np.asarray(imu48, dtype=np.float32)
    if x.shape[-1] < 48:
        raise ValueError(f"Expected legacy 48D IMU, got shape {x.shape}")
    idx = order.index(sensor_name)
    rot = x[..., idx * 9 : (idx + 1) * 9].reshape(*x.shape[:-1], 3, 3)
    acc = x[..., 36 + idx * 3 : 36 + (idx + 1) * 3].astype(np.float32)
    return np.concatenate([acc, rotmat_to_quat_wxyz(rot)], axis=-1).astype(np.float32)


def load_custom_split_7d_sequence(
    root: Path,
    session: str,
    seg_idx: int,
    person: int,
    target_len: int | None = None,
) -> np.ndarray:
    """Load a full custom 7D sequence from train/val/test split files.

    The historical custom extraction stores one continuous segment as three
    chronological files: train, val, and test. Evaluation must reconstruct the
    full sequence instead of using only the test fragment.
    """
    parts = []
    for split in ("train", "val", "test"):
        path = root / f"{session}_seg{seg_idx}_person{person}_{split}.npy"
        if not path.exists():
            parts = []
            break
        arr = np.load(path, allow_pickle=True).item()["imu"].astype(np.float32)
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        parts.append(arr[:, :7])
    if not parts:
        path = root / f"{session}_seg{seg_idx}_person{person}_test.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, allow_pickle=True).item()["imu"].astype(np.float32)
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        parts = [arr[:, :7]]

    full = np.concatenate(parts, axis=0).astype(np.float32)
    if target_len is None:
        return full
    if len(full) >= target_len:
        return full[:target_len]
    pad = np.zeros((target_len - len(full), full.shape[1]), dtype=np.float32)
    return np.concatenate([full, pad], axis=0)


def load_custom_rawcsv_7d_sequence(
    root: Path,
    session: str,
    frame_ids: np.ndarray,
    imu_person_map: str | dict | None = None,
    n_persons: int = 2,
) -> np.ndarray:
    """Load raw custom session IMU CSV and align it to video frame ids.

    The raw custom layout is expected to be:

    ``root/<session>/video/<session>_frame_timestamps_retimed.csv``
    ``root/<session>/imu/<session>_<mac>.csv``

    Output is ordered by person id, using ``imu_person_map`` when available.
    ``frame_ids`` are the zero-based frame ids stored in segment NPZ files.
    """
    session_dir = Path(root) / session
    ts_path = session_dir / "video" / f"{session}_frame_timestamps_retimed.csv"
    if not ts_path.exists():
        ts_path = session_dir / "video" / f"{session}_frame_timestamps.csv"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)

    with ts_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        ts_rows = list(reader)
    if not ts_rows:
        raise ValueError(f"Empty timestamp file: {ts_path}")

    frame_to_ts: dict[int, float] = {}
    for row in ts_rows:
        # Raw timestamp CSV is 1-based; segment NPZ frame_ids are 0-based.
        frame_to_ts[int(row["frame_index"]) - 1] = float(row["timestamp_ms"])
    target_ts = np.array([frame_to_ts[int(fid)] for fid in frame_ids], dtype=np.float64)

    mapping: dict[str, int] = {}
    if isinstance(imu_person_map, str) and imu_person_map.strip():
        mapping = {str(k): int(v) for k, v in json.loads(imu_person_map).items()}
    elif isinstance(imu_person_map, dict):
        mapping = {str(k): int(v) for k, v in imu_person_map.items()}

    imu_dir = session_dir / "imu"
    imu_files = sorted(imu_dir.glob(f"{session}_*.csv"))
    if not imu_files:
        raise FileNotFoundError(f"No IMU CSV files found under {imu_dir}")

    out = np.zeros((len(frame_ids), n_persons, 7), dtype=np.float32)
    used = np.zeros((n_persons,), dtype=bool)
    fallback_person = 0
    for imu_path in imu_files:
        mac = imu_path.stem.replace(f"{session}_", "")
        if mac in mapping:
            person = int(mapping[mac])
        else:
            while fallback_person < n_persons and used[fallback_person]:
                fallback_person += 1
            if fallback_person >= n_persons:
                continue
            person = fallback_person
        if not (0 <= person < n_persons):
            continue
        imu_ts, quat4, acc3 = parse_imu_csv(imu_path)
        imu7 = convert_single_imu_to_7d(quat4, acc3)
        aligned = resample_imu_to_target(imu_ts, imu7, target_ts)
        out[:, person] = np.nan_to_num(aligned, nan=0.0)
        used[person] = True

    if not used.all():
        missing = np.where(~used)[0].tolist()
        raise ValueError(f"Missing raw CSV IMU for session={session}, person indices={missing}")
    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom dataset preprocess")
    parser.add_argument("--task", choices=["preprocess", "pack_segments", "slice"], default="preprocess")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--raw_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--segment_root", default=None)
    parser.add_argument("--skeleton_root", default=None)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--legacy_sensor", default=None)
    parser.add_argument("--custom_imu_root", default=None)
    parser.add_argument("--raw_imu_root", default=None)
    parser.add_argument("--raw_swap_sessions", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


DEFAULT_SEGMENT_ROOT = Path(
    "/home/fzliang/Autism-project/experiments/G1:egohumans/E10:joint_pretraining/data/custom_segments/sequences"
)
DEFAULT_OUTPUT_ROOT = Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_segmentimu")
DEFAULT_SESSIONS = ("20260211_171423", "20260211_171724", "20260211_172257", "20260211_172522")
DEFAULT_VAL_BY_TEST = {
    "20260211_171423": "20260211_171724",
    "20260211_171724": "20260211_172257",
    "20260211_172257": "20260211_172522",
    "20260211_172522": "20260211_171423",
}


def load_segments_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = load_config(config_path)
    segments = data.get("segments", {})
    if segments is None:
        return {}
    if not isinstance(segments, dict):
        raise ValueError(f"Invalid segments section in config: {config_path}")
    return segments


def _has_custom_segments(segment_root: Path) -> bool:
    return segment_root.exists() and any(segment_root.glob("custom_*.npz"))


def _find_skeleton_json(sequence_id: str, video_path: str, skeleton_root: Path) -> Path | None:
    video_stem = Path(video_path).stem if video_path else ""
    core = sequence_id[len("custom_") :] if sequence_id.startswith("custom_") else sequence_id
    session = core.split("_seg", 1)[0]
    candidates = []
    for name in (video_stem, core, session, sequence_id):
        if name:
            candidates.extend([
                skeleton_root / name / "skeleton.json",
                skeleton_root / name / "skeleton_merged.json",
            ])
    for path in candidates:
        if path.exists():
            return path
    for subdir in skeleton_root.iterdir() if skeleton_root.exists() else []:
        if not subdir.is_dir():
            continue
        if any(token and token in subdir.name for token in (video_stem, core, session)):
            skeleton_json = subdir / "skeleton.json"
            if skeleton_json.exists():
                return skeleton_json
    return None


def run_pack_segments(args: argparse.Namespace) -> None:
    """Pack preprocessed custom sequences and extracted skeletons into segment NPZs."""
    cfg = load_segments_cfg(args.config)
    preprocess_cfg = load_preprocess_cfg(args.config)
    extract_cfg = load_config(args.config).get("extract", {}) if args.config else {}

    input_root = Path(
        cfg.get("input_root")
        or cfg.get("preprocess_root")
        or Path(preprocess_cfg.get("output", "/data/fzliang/reid-project/custom/preprocessed/default/video_manifest.csv")).parent
    ).expanduser().resolve()
    skeleton_root = Path(
        args.skeleton_root
        or cfg.get("skeleton_root")
        or extract_cfg.get("results_root")
        or "/data/fzliang/reid-project/custom/skeleton/alphapose"
    ).expanduser().resolve()
    output_root = Path(
        args.segment_root
        or cfg.get("output_root")
        or "/data/fzliang/reid_project/interim/custom_segments"
    ).expanduser().resolve()
    normalize_extract_skeleton = bool(cfg.get("normalize_extract_skeleton", True))
    skip_existing = bool(cfg.get("skip_existing", True))
    force = bool(args.force or cfg.get("force", False))

    seq_in = input_root / "sequences"
    seq_out = output_root / "sequences"
    if skip_existing and _has_custom_segments(seq_out):
        summary = {
            "input_root": str(input_root),
            "skeleton_root": str(skeleton_root),
            "output_root": str(output_root),
            "packed": 0,
            "skipped": len(list(seq_out.glob("custom_*.npz"))),
            "missing_skeleton": [],
        }
        (output_root / "pack_segments_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    if not seq_in.exists():
        raise FileNotFoundError(f"Preprocessed custom sequences not found: {seq_in}")
    if force and seq_out.exists():
        import shutil

        shutil.rmtree(seq_out)
    seq_out.mkdir(parents=True, exist_ok=True)

    packed = 0
    skipped = 0
    missing: list[str] = []
    for src_npz in sorted(seq_in.glob("custom_*.npz")):
        out_npz = seq_out / src_npz.name
        if skip_existing and out_npz.exists():
            skipped += 1
            continue
        data = dict(np.load(src_npz, allow_pickle=True))
        sequence_id = str(data["sequence_id"].item())
        video_path = str(data.get("video_path", ""))
        if hasattr(data.get("video_path"), "item"):
            video_path = str(data["video_path"].item())
        skeleton_json = _find_skeleton_json(sequence_id, video_path, skeleton_root)
        if skeleton_json is None:
            missing.append(sequence_id)
            continue
        alphapose_frames, track_ids = load_alphapose_multiperson(skeleton_json)
        packed_data = _align_extract_to_npz(
            data,
            alphapose_frames,
            track_ids,
            normalize_extract_skeleton=normalize_extract_skeleton,
        )
        packed_data["extract_source"] = str(skeleton_json)
        np.savez_compressed(out_npz, **packed_data)
        packed += 1

    summary = {
        "input_root": str(input_root),
        "skeleton_root": str(skeleton_root),
        "output_root": str(output_root),
        "packed": packed,
        "skipped": skipped,
        "missing_skeleton": missing,
    }
    (output_root / "pack_segments_summary.json").write_text(json.dumps(summary, indent=2))
    if missing:
        raise FileNotFoundError("Missing skeleton.json for sequences:\n" + "\n".join(missing))
    print(json.dumps(summary, indent=2))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "subject",
        "session",
        "split",
        "npz_path",
        "window_start",
        "window_end",
        "window_len",
        "skeleton_source",
        "person_idx",
        "imu_idx",
        "source_sequence",
        "source_person",
        "source_window_start",
    ]
    write_csv(path, rows, fieldnames)


def _session_from_sequence(sequence_id: str) -> str:
    return sequence_id.split("_seg", 1)[0].split("custom_", 1)[1]


def _split_for_session(session: str, test_session: str, val_session: str) -> str:
    if session == test_session:
        return "test"
    if session == val_session:
        return "val"
    return "train"


def _target_fold_specs(slice_cfg: dict) -> list[dict[str, str]]:
    policy = str(slice_cfg.get("split_policy") or "").strip().lower()
    folds = slice_cfg.get("folds") or []
    if policy:
        if folds:
            raise ValueError("Use either slice.split_policy or slice.folds, not both.")
        if policy != "leave_one_session_out_20260211":
            raise ValueError(f"Unknown custom slice.split_policy: {policy}")
        return [
            {
                "name": f"fold{idx}_{test_session}",
                "test_session": test_session,
                "val_session": DEFAULT_VAL_BY_TEST[test_session],
            }
            for idx, test_session in enumerate(DEFAULT_SESSIONS, start=1)
        ]
    if folds:
        specs: list[dict[str, str]] = []
        for idx, fold in enumerate(folds, start=1):
            if not isinstance(fold, dict):
                raise ValueError(f"slice.folds[{idx}] must be a mapping, got {type(fold).__name__}")
            test_session = str(fold.get("test_session") or fold.get("session") or "").strip()
            val_session = str(fold.get("val_session") or "").strip()
            name = str(fold.get("name") or f"fold{idx}_{test_session}").strip()
            if not test_session or not val_session:
                raise ValueError(f"slice.folds[{idx}] requires test_session and val_session")
            specs.append({"name": name, "test_session": test_session, "val_session": val_session})
        return specs
    return [
        {
            "name": f"fold{idx}_{test_session}",
            "test_session": test_session,
            "val_session": DEFAULT_VAL_BY_TEST[test_session],
        }
        for idx, test_session in enumerate(DEFAULT_SESSIONS, start=1)
    ]


def run_slice(args: argparse.Namespace) -> None:
    slice_cfg = load_slice_cfg(args.config)
    segment_root = Path(args.segment_root or slice_cfg.get("segment_root", DEFAULT_SEGMENT_ROOT)).expanduser().resolve()
    output_root = Path(args.output_root or slice_cfg.get("output_root", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    window_len = int(args.window_len or slice_cfg.get("window_len", 24))
    stride = int(args.stride or slice_cfg.get("stride", 8))
    legacy_sensor = str(args.legacy_sensor or slice_cfg.get("legacy_sensor", "L_LowArm"))
    custom_imu_root = Path(args.custom_imu_root or slice_cfg.get("custom_imu_root", "")).expanduser().resolve() if (args.custom_imu_root or slice_cfg.get("custom_imu_root")) else None
    raw_imu_root = Path(args.raw_imu_root or slice_cfg.get("raw_imu_root", "")).expanduser().resolve() if (args.raw_imu_root or slice_cfg.get("raw_imu_root")) else None
    raw_swap_sessions = {
        s.strip() for s in str(args.raw_swap_sessions or slice_cfg.get("raw_swap_sessions", "")).split(",") if s.strip()
    }
    force = bool(args.force or slice_cfg.get("force", False))

    if custom_imu_root is not None and raw_imu_root is not None:
        raise ValueError("Use either --custom_imu_root or --raw_imu_root, not both.")

    fold_specs = _target_fold_specs(slice_cfg)

    def build_fold(fold: dict[str, str]) -> dict[str, int]:
        test_session = fold["test_session"]
        val_session = fold["val_session"]
        fold_dir = output_root / fold["name"]
        seq_dir = fold_dir / "sequences"
        if force and fold_dir.exists():
            import shutil

            shutil.rmtree(fold_dir)
        seq_dir.mkdir(parents=True, exist_ok=True)

        rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
        counts = {"train": 0, "val": 0, "test": 0}
        for path in sorted(segment_root.glob("custom_*.npz")):
            data = np.load(path, allow_pickle=True)
            sequence_id = str(data["sequence_id"].item())
            session = _session_from_sequence(sequence_id)
            split = _split_for_session(session, test_session, val_session)
            pose = data["extract_skeleton"][:, :, :, :2].astype(np.float32)
            visibility = data["extract_visibility"].astype(bool)
            t_len, n_person = pose.shape[:2]
            if raw_imu_root is not None:
                imu_person_map = None
                if "imu_person_map" in data.files:
                    imu_person_map = str(data["imu_person_map"].item())
                imu7 = load_custom_rawcsv_7d_sequence(
                    raw_imu_root,
                    session,
                    data["frame_ids"].astype(np.int64),
                    imu_person_map=imu_person_map,
                    n_persons=n_person,
                )
                if raw_swap_sessions and session in raw_swap_sessions:
                    imu7 = imu7[:, ::-1].copy()
            elif custom_imu_root is None:
                imu7 = legacy_imu48_sensor_to_7d(data["imu"].astype(np.float32), legacy_sensor)
            else:
                seg_idx = int(sequence_id.rsplit("_seg", 1)[1])
                imu7 = np.stack(
                    [
                        load_custom_split_7d_sequence(custom_imu_root, session, seg_idx, person, target_len=t_len)
                        for person in range(n_person)
                    ],
                    axis=1,
                )

            for person in range(n_person):
                for start in range(0, t_len - window_len + 1, stride):
                    end = start + window_len
                    if not visibility[start:end, person].any():
                        continue
                    rel = f"sequences/{sequence_id}_p{person}_{start}_{end}.npz"
                    np.savez_compressed(
                        fold_dir / rel,
                        skeleton=pose[start:end, person],
                        imu=imu7[start:end, person],
                    )
                    rows_by_split[split].append(
                        {
                            "subject": f"P{person}",
                            "session": session,
                            "split": split,
                            "npz_path": rel,
                            "window_start": "0",
                            "window_end": str(window_len),
                            "window_len": str(window_len),
                            "skeleton_source": "gt",
                            "person_idx": "0",
                            "imu_idx": "0",
                            "source_sequence": sequence_id,
                            "source_person": str(person),
                            "source_window_start": str(start),
                        }
                    )
                    counts[split] += 1

        for split, rows in rows_by_split.items():
            _write_csv(fold_dir / f"windows_{split}.csv", rows)
        (fold_dir / "slice_summary.json").write_text(
            json.dumps(
                {
                    "test_session": test_session,
                    "val_session": val_session,
                    "window_len": window_len,
                    "stride": stride,
                    "legacy_sensor": legacy_sensor,
                    "custom_imu_root": "" if custom_imu_root is None else str(custom_imu_root),
                    "raw_imu_root": "" if raw_imu_root is None else str(raw_imu_root),
                    "raw_swap_sessions": sorted(raw_swap_sessions or []),
                    "counts": counts,
                },
                indent=2,
            )
        )
        return counts

    summary = {fold["name"]: build_fold(fold) for fold in fold_specs}
    print(json.dumps(summary, indent=2))


def load_imu_person_mapping(annotations_dir: Path) -> dict[str, str] | None:
    """Load IMU-to-person mapping from annotations/imu_person_mapping.json.

    Expected format:
        {"person1": "f8:a2:fd:ea:fb:80", "person2": "da:19:a9:ac:6d:fe"}

    Returns None if the mapping file does not exist.
    """
    mapping_path = annotations_dir / "imu_person_mapping.json"
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid IMU person mapping format in {mapping_path}")
    return mapping


def load_preprocess_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = load_config(config_path)
    preprocess = data.get("preprocess", {})
    if preprocess is None:
        return {}
    if not isinstance(preprocess, dict):
        raise ValueError(f"Invalid preprocess section in config: {config_path}")
    return preprocess


def parse_annotations(anno_path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse annotation CSV.

    Returns:
        n_persons: number of persons
        frame_indices: [T] int64
        timestamps_ms: [T] float64
        bboxes: [T, N, 4] float32 in [x1, y1, x2, y2]
        visibility: [T, N] bool
    """
    with anno_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty annotation file: {anno_path}")

    cols = set(rows[0].keys())
    n_persons = 0
    while f"p{n_persons + 1}_bbox_x" in cols:
        n_persons += 1

    if n_persons == 0:
        raise ValueError(f"No person bbox columns found in {anno_path}")

    T = len(rows)
    frame_indices = np.zeros(T, dtype=np.int64)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    bboxes = np.zeros((T, n_persons, 4), dtype=np.float32)
    visibility = np.zeros((T, n_persons), dtype=bool)

    for t, row in enumerate(rows):
        frame_indices[t] = int(row["frame_index"])
        timestamps_ms[t] = float(row["timestamp_ms"])
        for p in range(n_persons):
            prefix = f"p{p + 1}_"
            x = float(row[f"{prefix}bbox_x"])
            y = float(row[f"{prefix}bbox_y"])
            w = float(row[f"{prefix}bbox_w"])
            h = float(row[f"{prefix}bbox_h"])
            bboxes[t, p] = np.array([x, y, x + w, y + h], dtype=np.float32)
            visibility[t, p] = int(row[f"{prefix}is_absent"]) == 0

    return n_persons, frame_indices, timestamps_ms, bboxes, visibility


def _find_col(candidates: List[str], row: dict) -> str:
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"Could not find any of {candidates}. Available: {list(row.keys())}")


def _interpolate_sparse_bboxes(
    frame_indices: np.ndarray,
    bboxes: np.ndarray,
    visibility: np.ndarray,
    n_video_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate bboxes for frames between first and last valid annotation.

    For each person, finds the first and last frames where visibility=True.
    Within this range, missing frames get linearly interpolated bboxes.
    Frames outside any person's valid range are marked invisible and excluded.

    Args:
        frame_indices: [T_anno] annotated frame indices
        bboxes: [T_anno, N, 4]
        visibility: [T_anno, N] bool
        n_video_frames: total video frames

    Returns:
        out_bboxes: [T_out, N, 4]
        out_visibility: [T_out, N] bool
        out_frame_ids: [T_out] frame indices (0-based) in output range
    """
    n_persons = bboxes.shape[1]
    per_person: list[dict] = []

    for p in range(n_persons):
        valid_mask = visibility[:, p]
        valid_frames = frame_indices[valid_mask]
        if len(valid_frames) == 0:
            raise ValueError(f"Person {p} has no valid annotations")
        per_person.append({
            "first": int(valid_frames[0]),
            "last": int(valid_frames[-1]),
            "valid_frames": valid_frames,
            "valid_bboxes": bboxes[valid_mask, p],
        })

    # Global output range: from min first_valid to max last_valid
    output_start = max(0, min(v["first"] for v in per_person))
    output_end = min(n_video_frames - 1, max(v["last"] for v in per_person))
    if output_start > output_end:
        raise ValueError("Invalid output range: start > end")

    T_out = output_end - output_start + 1
    out_frame_ids = np.arange(output_start, output_end + 1, dtype=np.int64)
    out_bboxes = np.zeros((T_out, n_persons, 4), dtype=np.float32)
    out_visibility = np.zeros((T_out, n_persons), dtype=bool)

    for p in range(n_persons):
        pv = per_person[p]
        first_v = pv["first"]
        last_v = pv["last"]
        valid_frames = pv["valid_frames"]
        valid_bboxes = pv["valid_bboxes"]

        # Build frame->bbox lookup for original annotations
        frame_to_bbox: dict[int, np.ndarray] = {}
        for i, f in enumerate(frame_indices):
            if visibility[i, p]:
                frame_to_bbox[int(f)] = bboxes[i, p].copy()

        for idx, f in enumerate(out_frame_ids):
            f = int(f)
            if f in frame_to_bbox:
                out_bboxes[idx, p] = frame_to_bbox[f]
                out_visibility[idx, p] = True
            elif first_v <= f <= last_v:
                # Linearly interpolate between nearest valid frames
                pos = int(np.searchsorted(valid_frames, f))
                if pos == 0:
                    out_bboxes[idx, p] = valid_bboxes[0]
                elif pos >= len(valid_frames):
                    out_bboxes[idx, p] = valid_bboxes[-1]
                else:
                    f_high = int(valid_frames[pos])
                    f_low = int(valid_frames[pos - 1])
                    alpha = (f - f_low) / (f_high - f_low) if f_high != f_low else 0.0
                    out_bboxes[idx, p] = valid_bboxes[pos - 1] + alpha * (
                        valid_bboxes[pos] - valid_bboxes[pos - 1]
                    )
                out_visibility[idx, p] = True
            # else: remains zero bbox, visibility=False

    return out_bboxes, out_visibility, out_frame_ids


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 30.0


def main() -> None:
    args = parse_args()
    if args.task == "pack_segments":
        run_pack_segments(args)
        return
    if args.task == "slice":
        run_slice(args)
        return
    preprocess_cfg = load_preprocess_cfg(args.config)
    
    # Get IMU filter parameters from config (applied before downsampling)
    imu_cfg = preprocess_cfg.get("imu", {})

    def _parse_float_or_none(x):
        if x is None:
            return None
        if isinstance(x, str):
            xs = x.strip()
            if xs.lower() in ("none", "null", ""):
                return None
            try:
                return float(xs)
            except ValueError:
                raise ValueError(f"Invalid numeric value for IMU config: {x}")
        return float(x)

    imu_lowpass_cutoff_hz = _parse_float_or_none(imu_cfg.get("lowpass_cutoff_hz", None))
    imu_lowpass_fs_hz = _parse_float_or_none(imu_cfg.get("lowpass_fs_hz", 100.0))  # Default: 100Hz raw IMU

    raw_root = Path(args.raw_root if args.raw_root else preprocess_cfg.get("raw_root", "/data/fzliang/custom")).expanduser().resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        default_manifest = Path(preprocess_cfg.get(
            "output",
            "/data/fzliang/reid-project/custom/preprocessed/default/video_manifest.csv",
        )).expanduser().resolve()
        output_dir = default_manifest.parent

    manifest_csv = Path(
        args.manifest_csv if args.manifest_csv else preprocess_cfg.get("output", str(output_dir / "video_manifest.csv"))
    ).expanduser().resolve()

    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []

    def collect_sessions(base_dir: Path, annotations_dir: Path, prefix: str = "custom"):
        """Collect all session directories under base_dir."""
        for session_dir in sorted(base_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if not (session_dir / "video").exists():
                continue
            session_stem = session_dir.name
            sequence_id = f"{prefix}_{session_stem}"
            video_path = session_dir / "video" / f"{session_stem}.mp4"
            anno_path = annotations_dir / f"{session_stem}.anno.csv"
            imu_dir = session_dir / "imu"
            yield session_stem, sequence_id, video_path, anno_path, imu_dir, annotations_dir

    # Determine directory layout
    # Layout A: raw_root/<person_count>/<session>/  (old)
    # Layout B: raw_root/<session>/  (new, e.g. batch_20260505)
    has_person_count_subdirs = any(
        d.is_dir() and (d / "annotations").exists()
        for d in raw_root.iterdir()
    )

    if has_person_count_subdirs:
        # Old layout: iterate over person_count subdirs
        sessions_iter = []
        for person_count_dir in sorted(raw_root.iterdir()):
            if not person_count_dir.is_dir():
                continue
            if not (person_count_dir / "annotations").exists():
                continue
            sessions_iter.extend(
                collect_sessions(person_count_dir, person_count_dir / "annotations", prefix="custom")
            )
    else:
        # New layout: sessions directly under raw_root
        sessions_iter = list(collect_sessions(raw_root, raw_root / "annotations", prefix="custom"))

    for session_stem, sequence_id, video_path, anno_path, imu_dir, annotations_dir in sessions_iter:

            if not video_path.exists():
                print(f"Warning: video not found for {sequence_id}, skipping")
                continue
            if not anno_path.exists():
                print(f"Warning: annotation not found for {sequence_id}, skipping")
                continue

            imu_files = sorted(imu_dir.glob("*.csv"))
            if not imu_files:
                print(f"Warning: no IMU files for {sequence_id}, skipping")
                continue

            n_persons, frame_indices, anno_ts, anno_bboxes, anno_visibility = parse_annotations(anno_path)
            if len(imu_files) != n_persons:
                print(f"Warning: IMU count ({len(imu_files)}) != person count ({n_persons}) for {sequence_id}")

            # Load IMU-to-person mapping if available
            mapping = load_imu_person_mapping(annotations_dir)
            if mapping is not None:
                ordered_files: list[Path | None] = [None] * n_persons
                used_indices: set[int] = set()
                for person_key, mac_addr in mapping.items():
                    if not person_key.startswith("person"):
                        continue
                    try:
                        person_idx = int(person_key.replace("person", "")) - 1
                    except ValueError:
                        continue
                    if not (0 <= person_idx < n_persons):
                        continue
                    # Find IMU file whose stem contains the MAC address
                    for fpath in imu_files:
                        if mac_addr in fpath.name and fpath not in [ordered_files[i] for i in used_indices if ordered_files[i] is not None]:
                            ordered_files[person_idx] = fpath
                            used_indices.add(person_idx)
                            break
                # Fill remaining slots with unused files (alphabetical order)
                unused_files = [f for f in imu_files if f not in ordered_files]
                for i in range(n_persons):
                    if ordered_files[i] is None and unused_files:
                        ordered_files[i] = unused_files.pop(0)
                imu_files = [f for f in ordered_files if f is not None]
                if len(imu_files) != n_persons:
                    print(f"Warning: could not resolve full IMU mapping for {sequence_id}, using partial")

            imu_data_list = []
            for imu_path in imu_files:
                imu_ts, quat4, acc3 = parse_imu_csv(imu_path)
                imu7 = convert_single_imu_to_7d(quat4, acc3)
                # Apply FFT low-pass filter before resampling (at 100Hz raw rate)
                if imu_lowpass_cutoff_hz is not None:
                    imu7 = lowpass_filter_fft(imu7, imu_lowpass_cutoff_hz, imu_lowpass_fs_hz)
                imu_data_list.append((imu_ts, imu7))

            valid_start_ms = max(data[0][0] for data in imu_data_list)
            valid_end_ms = min(data[0][-1] for data in imu_data_list)

            # Determine output frame coverage
            # For sparse annotations (fewer rows than video frames), expand to all video frames
            cap = cv2.VideoCapture(str(video_path))
            n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
            cap.release()

            sparse_annotation = len(anno_ts) < n_video_frames * 0.8
            if sparse_annotation and n_video_frames > 0:
                # Interpolate bboxes between first and last valid annotations per person
                # Frames outside any person's valid range are excluded
                try:
                    anno_bboxes, anno_visibility, frame_ids = _interpolate_sparse_bboxes(
                        frame_indices, anno_bboxes, anno_visibility, n_video_frames
                    )
                except ValueError as e:
                    print(f"Warning: failed to interpolate annotations for {sequence_id}: {e}, skipping")
                    continue

                T = len(frame_ids)
                if T == 0:
                    print(f"Warning: no valid frames after interpolation for {sequence_id}, skipping")
                    continue

                # Compute video start time from annotation frame_index vs timestamp_ms
                if len(frame_indices) >= 2:
                    frame_interval_ms = 1000.0 / fps
                    offsets = anno_ts - frame_indices * frame_interval_ms
                    video_start_ms = float(np.median(offsets))
                elif len(frame_indices) == 1:
                    video_start_ms = float(anno_ts[0] - frame_indices[0] * (1000.0 / fps))
                else:
                    video_start_ms = float(anno_ts[0])

                target_ts = video_start_ms + frame_ids.astype(np.float64) * (1000.0 / fps)
            else:
                # Dense annotation mode: use annotation timestamps directly (legacy behavior)
                crop_mask = (anno_ts >= valid_start_ms) & (anno_ts <= valid_end_ms)
                if not crop_mask.any():
                    print(f"Warning: no overlap between IMU and annotation for {sequence_id}, skipping")
                    continue

                crop_indices = np.where(crop_mask)[0]
                first_idx = int(crop_indices[0])
                last_idx = int(crop_indices[-1])

                target_ts = anno_ts[first_idx : last_idx + 1]
                T = len(target_ts)
                frame_ids = np.arange(T, dtype=np.int64)
                bboxes = anno_bboxes[first_idx : last_idx + 1]
                visibility = anno_visibility[first_idx : last_idx + 1]

            n_imu = len(imu_data_list)
            imu_out = np.zeros((T, n_imu, 7), dtype=np.float32)
            for i, (imu_ts, imu7) in enumerate(imu_data_list):
                resampled = resample_imu_to_target(imu_ts, imu7, target_ts)
                nan_mask = np.isnan(resampled).any(axis=1)
                resampled = np.nan_to_num(resampled, nan=0.0)
                imu_out[:, i] = resampled
                if nan_mask.any():
                    print(f"Warning: {sequence_id} IMU {i} has {nan_mask.sum()} out-of-range frames")

            person_ids = np.arange(n_persons, dtype=np.int64)
            imu_ids = np.arange(n_imu, dtype=np.int64)

            # Build MAC-to-person map for storage (derived from file ordering)
            imu_person_map: dict[str, int] = {}
            for i, fpath in enumerate(imu_files):
                # Extract MAC from filename: {stem}_{mac}.csv
                mac = fpath.stem.split("_")[-1] if "_" in fpath.stem else fpath.stem
                imu_person_map[mac] = int(person_ids[i])

            npz_path = seq_dir / f"{sequence_id}.npz"
            np.savez_compressed(
                npz_path,
                video_path=np.array(str(video_path), dtype=object),
                dataset=np.array("custom", dtype=object),
                sequence_id=np.array(sequence_id, dtype=object),
                frame_ids=frame_ids,
                imu=imu_out,
                imu_ids=imu_ids,
                gt_person_ids=person_ids,
                gt_bboxes=anno_bboxes,
                gt_visibility=anno_visibility,
                imu_person_map=np.array(json.dumps(imu_person_map), dtype=object),
            )

            meta = {
                "video_path": str(video_path),
                "dataset": "custom",
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
            print(f"Processed {sequence_id}: {T} frames, {n_imu} IMUs, {n_persons} persons")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Preprocessed {len(manifest_rows)} sequences -> {output_dir}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
