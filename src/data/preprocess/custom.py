"""Preprocess for Custom dataset: generate standardized NPZ + video manifest."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.datasets.totalcapture import quat_to_rotmat
from src.utils.config import load_config




def lowpass_filter_fft(signal: np.ndarray, cutoff_hz: float | None, fs_hz: float) -> np.ndarray:
    """Apply FFT-domain low-pass filter along time axis [axis 0].
    
    Args:
        signal: [T, ...] array with time on axis 0
        cutoff_hz: Cutoff frequency in Hz (None to skip filtering)
        fs_hz: Sampling frequency in Hz
    
    Returns:
        Filtered signal with same shape as input
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal.astype(np.float32, copy=False)
    
    time_len = signal.shape[0]
    nyquist = fs_hz / 2.0
    effective_cutoff = float(cutoff_hz)
    
    # Safety clip to Nyquist
    if effective_cutoff >= nyquist:
        warnings.warn(
            f"Requested IMU low-pass cutoff {effective_cutoff:.3f} Hz exceeds Nyquist {nyquist:.3f} Hz; "
            f"clipping to {nyquist * 0.95:.3f} Hz.",
            RuntimeWarning
        )
        effective_cutoff = max(nyquist * 0.95, 1e-6)
    
    # FFT-domain filtering
    freq = np.fft.rfftfreq(time_len, d=1.0 / fs_hz)
    spectrum = np.fft.rfft(signal, axis=0)
    spectrum[freq > effective_cutoff, ...] = 0.0
    return np.fft.irfft(spectrum, n=time_len, axis=0).astype(np.float32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom dataset preprocess")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--raw_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    return parser.parse_args()


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


def parse_imu_csv(imu_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse custom IMU CSV.

    Returns:
        timestamps_ms: [T] float64
        quat4: [T, 4] float32
        acc3: [T, 3] float32
    """
    with imu_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty IMU file: {imu_path}")

    ts_col = _find_col(["epoch_ms"], rows[0])
    q0_col = _find_col(["四元数0()"], rows[0])
    q1_col = _find_col(["四元数1()"], rows[0])
    q2_col = _find_col(["四元数2()"], rows[0])
    q3_col = _find_col(["四元数3()"], rows[0])
    ax_col = _find_col(["加速度X(g)"], rows[0])
    ay_col = _find_col(["加速度Y(g)"], rows[0])
    az_col = _find_col(["加速度Z(g)"], rows[0])

    T = len(rows)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    quat4 = np.zeros((T, 4), dtype=np.float32)
    acc3 = np.zeros((T, 3), dtype=np.float32)

    for t, row in enumerate(rows):
        timestamps_ms[t] = float(row[ts_col])
        quat4[t] = np.array([float(row[q0_col]), float(row[q1_col]), float(row[q2_col]), float(row[q3_col])], dtype=np.float32)
        # Convert acceleration from g to m/s² to match TotalCapture units
        acc3[t] = np.array([float(row[ax_col]), float(row[ay_col]), float(row[az_col])], dtype=np.float32) * 9.80665

    return timestamps_ms, quat4, acc3


def convert_single_imu_to_48(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    """Convert single-sensor IMU to 48D by repeating 12D four times."""
    T = quat4.shape[0]
    rot = quat_to_rotmat(quat4).reshape(T, 9)
    acc = acc3

    out = np.zeros((T, 48), dtype=np.float32)
    for i in range(4):
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def resample_imu_to_target(src_ts: np.ndarray, src_values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Linearly interpolate IMU values to target timestamps."""
    valid_start = src_ts[0]
    valid_end = src_ts[-1]

    out = np.zeros((len(target_ts), src_values.shape[1]), dtype=np.float32)
    for d in range(src_values.shape[1]):
        out[:, d] = np.interp(target_ts, src_ts, src_values[:, d])

    out[target_ts < valid_start] = np.nan
    out[target_ts > valid_end] = np.nan
    return out


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 30.0


def main() -> None:
    args = parse_args()
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
        default_manifest = Path(preprocess_cfg.get("output", "./data/interim/video_manifest.csv")).expanduser().resolve()
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
                imu48 = convert_single_imu_to_48(quat4, acc3)
                # Apply FFT low-pass filter before resampling (at 100Hz raw rate)
                if imu_lowpass_cutoff_hz is not None:
                    imu48 = lowpass_filter_fft(imu48, imu_lowpass_cutoff_hz, imu_lowpass_fs_hz)
                imu_data_list.append((imu_ts, imu48))

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
            imu_out = np.zeros((T, n_imu, 48), dtype=np.float32)
            for i, (imu_ts, imu48) in enumerate(imu_data_list):
                resampled = resample_imu_to_target(imu_ts, imu48, target_ts)
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
