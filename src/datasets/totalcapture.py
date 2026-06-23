"""TotalCapture dataset adapter for preprocessing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.adapters.alphapose import load_alphapose_skeleton, find_skeleton_for_sequence, load_alphapose_multiperson


SENSOR_ORDER = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]


@dataclass
class SequenceMeta:
    subject: str
    session: str
    split: str
    npz_path: str
    num_frames: int


def parse_subjects(spec: str | int | None) -> List[str]:
    if spec is None:
        return []
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def parse_sensor_order(spec: Sequence[str] | str | None) -> List[str]:
    if spec is None:
        return list(SENSOR_ORDER)
    if isinstance(spec, str):
        return [x.strip() for x in spec.split(",") if x.strip()]
    return [str(x).strip() for x in spec if str(x).strip()]


def parse_bool(spec, default: bool = True) -> bool:
    if spec is None:
        return default
    if isinstance(spec, bool):
        return spec
    s = str(spec).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    # q: [N, 4], in order [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def parse_vicon_pos(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid Vicon file: {path}")

    joints = [x for x in lines[0].split("\t") if x]
    rows: List[np.ndarray] = []
    for ln in lines[1:]:
        parts = [x for x in ln.split("\t") if x]
        if len(parts) < len(joints):
            continue
        coords = []
        ok = True
        for token in parts[: len(joints)]:
            vals = token.split()
            if len(vals) != 3:
                ok = False
                break
            coords.append([float(vals[0]), float(vals[1]), float(vals[2])])
        if ok:
            rows.append(np.asarray(coords, dtype=np.float32))

    if not rows:
        raise ValueError(f"No valid rows in Vicon file: {path}")
    return joints, np.stack(rows, axis=0)


def parse_xsens_sensors(path: Path, selected: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty sensors file: {path}")

    first = lines[0].split()
    if len(first) < 2:
        raise ValueError(f"Invalid sensors header: {path}")
    n_sensors = int(first[0])

    quats = []
    accs = []

    i = 1
    while i < len(lines):
        # frame index line
        _ = lines[i]
        i += 1
        if i + n_sensors > len(lines):
            break

        sensor_map: Dict[str, np.ndarray] = {}
        for _k in range(n_sensors):
            toks = lines[i].split()
            i += 1
            if len(toks) < 8:
                continue
            name = toks[0]
            vals = np.array([float(x) for x in toks[1:8]], dtype=np.float32)
            sensor_map[name] = vals

        if not all(name in sensor_map for name in selected):
            continue

        q_frame = []
        a_frame = []
        for name in selected:
            vals = sensor_map[name]
            q_frame.append(vals[:4])
            a_frame.append(vals[4:7])
        quats.append(np.stack(q_frame, axis=0))
        accs.append(np.stack(a_frame, axis=0))

    if not quats:
        raise ValueError(f"No valid IMU frames found in {path}")

    return np.stack(quats, axis=0), np.stack(accs, axis=0)


def convert_imu_to_48(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    # quat4: [T, 4 sensors, 4], acc3: [T, 4 sensors, 3]
    tlen = quat4.shape[0]
    out = np.zeros((tlen, 48), dtype=np.float32)
    for i in range(4):
        rot = quat_to_rotmat(quat4[:, i, :]).reshape(tlen, 9)
        acc = acc3[:, i, :]
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def map_totalcapture21_to_h36m17(joint_names: Sequence[str], xyz21: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(joint_names)}

    def j(name: str) -> np.ndarray:
        return xyz21[:, idx[name], :]

    y = np.zeros((xyz21.shape[0], 17, 3), dtype=np.float32)
    y[:, 0, :] = j("Hips")
    y[:, 1, :] = j("RightUpLeg")
    y[:, 2, :] = j("RightLeg")
    y[:, 3, :] = j("RightFoot")
    y[:, 4, :] = j("LeftUpLeg")
    y[:, 5, :] = j("LeftLeg")
    y[:, 6, :] = j("LeftFoot")
    y[:, 7, :] = j("Spine2")
    y[:, 8, :] = j("Spine3")
    y[:, 9, :] = j("Neck")
    y[:, 10, :] = j("Head")
    y[:, 11, :] = j("LeftShoulder")
    y[:, 12, :] = j("LeftArm")
    y[:, 13, :] = j("LeftForeArm")
    y[:, 14, :] = j("RightShoulder")
    y[:, 15, :] = j("RightArm")
    y[:, 16, :] = j("RightForeArm")
    return y


def normalize_skeleton(skel: np.ndarray) -> np.ndarray:
    # Root-relative and scale-normalized to improve training stability.
    root = skel[:, 0:1, :]
    skel = skel - root
    scale = np.linalg.norm(skel[:, 8, :] - skel[:, 0, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    skel = skel / scale[:, None, :]
    return skel.astype(np.float32)


def find_sequences(root: Path) -> List[Tuple[str, str, Path, Path]]:
    seqs: List[Tuple[str, str, Path, Path]] = []
    for subject_dir in sorted(root.glob("S[1-5]")):
        subject = subject_dir.name
        imu_subject = subject.lower()
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name
            vicon_pos = session_dir / "gt_skel_gbl_pos.txt"
            imu_file = root / imu_subject / f"{imu_subject}_{session}_Xsens.sensors"
            if vicon_pos.exists() and imu_file.exists():
                seqs.append((subject, session, vicon_pos, imu_file))
    return seqs


def subject_to_split(subject: str, train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> str:
    if subject in train:
        return "train"
    if subject in val:
        return "val"
    if subject in test:
        return "test"
    raise ValueError(f"Subject {subject} not assigned to split")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _align_extract_to_npz(
    data: dict[str, np.ndarray],
    alphapose_frames: dict[int, list[dict]],
    track_ids: list[int],
    normalize_extract_skeleton: bool = True,
) -> dict[str, np.ndarray]:
    """Align AlphaPose multi-person output into NPZ.

    Args:
        data: NPZ dict (must contain gt_bboxes, gt_visibility, gt_person_ids, frame_ids)
        alphapose_frames: mapping frame_idx -> list of detection dicts
        track_ids: sorted unique track ids from AlphaPose output

    Returns:
        Updated data dict with extract_* and gt_to_extract_map fields.
    """
    T = int(data["frame_ids"].shape[0])
    N_gt = int(data["gt_person_ids"].shape[0])
    N_pred = len(track_ids)

    extract_bboxes = np.zeros((T, N_pred, 4), dtype=np.float32)
    extract_skeleton = np.zeros((T, N_pred, 17, 3), dtype=np.float32)
    extract_visibility = np.zeros((T, N_pred), dtype=bool)

    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    for t in range(T):
        frame_idx = int(data["frame_ids"][t])
        if frame_idx in alphapose_frames:
            for det in alphapose_frames[frame_idx]:
                p_idx = track_id_to_idx.get(det["track_id"])
                if p_idx is None:
                    continue
                extract_bboxes[t, p_idx] = det["bbox"]
                extract_skeleton[t, p_idx] = det["keypoints"]
                extract_visibility[t, p_idx] = True

    # Normalize each extracted track independently (same as GT skeleton preprocessing).
    if normalize_extract_skeleton:
        for p in range(N_pred):
            if extract_visibility[:, p].any():
                extract_skeleton[:, p] = normalize_skeleton(extract_skeleton[:, p])

    gt_to_extract_map = np.full((T, N_gt), -1, dtype=np.int64)
    gt_bboxes = data["gt_bboxes"]
    gt_visibility = data["gt_visibility"]

    for t in range(T):
        for g in range(N_gt):
            if not gt_visibility[t, g]:
                continue
            gt_box = gt_bboxes[t, g]
            best_iou = -1.0
            best_p = -1
            for p in range(N_pred):
                if not extract_visibility[t, p]:
                    continue
                iou = _compute_iou(gt_box, extract_bboxes[t, p])
                if iou > best_iou:
                    best_iou = iou
                    best_p = p
            if best_iou > 0.0:
                gt_to_extract_map[t, g] = best_p

    data = dict(data)
    data["extract_person_ids"] = np.array(track_ids, dtype=np.int64)
    data["extract_bboxes"] = extract_bboxes
    data["extract_visibility"] = extract_visibility
    data["extract_skeleton"] = extract_skeleton
    data["gt_to_extract_map"] = gt_to_extract_map
    return data


class TotalCaptureAdapter:
    """Adapter that preprocesses TotalCapture raw data into NPZ + CSV."""

    def __init__(self, slice_cfg: dict):
        self.cfg = slice_cfg
        self.root = Path(slice_cfg.get("root", "/data/fzliang/totalcapture"))
        self.out_dir = Path(slice_cfg.get("out_dir", "data/processed/totalcapture"))
        self.window_len = int(slice_cfg.get("window_len", 24))
        self.stride = int(slice_cfg.get("stride", 16))
        self.sensor_order = parse_sensor_order(slice_cfg.get("sensor_order", SENSOR_ORDER))
        self.train_subj = parse_subjects(slice_cfg.get("train_subjects", "S1,S2,S3"))
        self.val_subj = parse_subjects(slice_cfg.get("val_subjects", "S4"))
        self.test_subj = parse_subjects(slice_cfg.get("test_subjects", "S5"))
        self.train_sessions = parse_subjects(slice_cfg.get("train_sessions", ""))
        self.val_sessions = parse_subjects(slice_cfg.get("val_sessions", ""))
        self.test_sessions = parse_subjects(slice_cfg.get("test_sessions", ""))
        self.max_sequences = int(slice_cfg.get("max_sequences", 0))
        self.skeleton_normalize = parse_bool(slice_cfg.get("skeleton_normalize", True), default=True)
        self.skeleton_source = slice_cfg.get("skeleton_source", "vicon")
        self.skeleton_root = None
        if self.skeleton_source == "alphapose":
            self.skeleton_root = Path(slice_cfg.get("skeleton_root", "/home/fzliang/MotionBERT/results_totalcapture_video"))
        self.multi_person = parse_bool(slice_cfg.get("multi_person"), default=False)

    def run(self) -> Path:
        return self._run_slice()

    def _run_slice(self) -> Path:
        """Slice from NPZs produced by the preprocess stage."""
        import shutil

        seq_dir = self.out_dir / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)

        npz_paths = sorted((self.root / "sequences").glob("*.npz"))
        if self.max_sequences > 0:
            npz_paths = npz_paths[: self.max_sequences]

        sequence_rows: List[Dict[str, object]] = []
        window_rows: List[Dict[str, object]] = []

        for npz_path in npz_paths:
            data = dict(np.load(npz_path, allow_pickle=True))
            sequence_id = str(data["sequence_id"].item())
            # Parse subject/session from sequence_id
            if sequence_id.startswith("totalcapture_"):
                parts = sequence_id.split("_")
                subject = parts[1]
                session = "_".join(parts[2:-1])
            elif sequence_id.startswith("custom_"):
                session = sequence_id[len("custom_"):]
                subject = "all"
            else:
                subject = "unknown"
                session = sequence_id

            splits = []
            use_session_split = bool(self.train_sessions or self.val_sessions or self.test_sessions)
            if use_session_split:
                if session in self.train_sessions:
                    splits.append("train")
                if session in self.val_sessions:
                    splits.append("val")
                if session in self.test_sessions:
                    splits.append("test")
            else:
                if subject in self.train_subj:
                    splits.append("train")
                if subject in self.val_subj:
                    splits.append("val")
                if subject in self.test_subj:
                    splits.append("test")
            if not splits:
                print(f"Warning: {sequence_id} (subject={subject}, session={session}) not assigned to any split, skipping...")
                continue

            tlen = int(data["frame_ids"].shape[0])

            # Align extracted skeleton if requested
            if self.skeleton_source == "alphapose" and self.skeleton_root is not None:
                extract_dir = self._find_extract_dir(sequence_id)
                if extract_dir is not None:
                    skeleton_json = extract_dir / "skeleton.json"
                    if skeleton_json.exists():
                        alphapose_frames, track_ids = load_alphapose_multiperson(skeleton_json)
                        data = _align_extract_to_npz(
                            data,
                            alphapose_frames,
                            track_ids,
                            normalize_extract_skeleton=self.skeleton_normalize,
                        )
                        data["extract_source"] = str(skeleton_json)
                    else:
                        print(f"Warning: skeleton.json not found in {extract_dir} for {sequence_id}")
                else:
                    print(f"Warning: No extract result found for {sequence_id}")

            rel_npz = Path("sequences") / f"{sequence_id}.npz"
            out_npz = self.out_dir / rel_npz
            np.savez_compressed(out_npz, **data)

            sequence_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "split": ",".join(splits),
                    "npz_path": str(rel_npz),
                    "num_frames": int(tlen),
                }
            )

            # Choose skeleton source for training windows
            train_skeleton_source = self.skeleton_source
            has_gt_skeleton = "gt_skeleton" in data
            n_imu = int(data["imu_ids"].shape[0])
            n_gt = int(data["gt_person_ids"].shape[0]) if "gt_person_ids" in data else 0
            has_extract = "extract_skeleton" in data and data["extract_skeleton"].shape[1] > 0

            if tlen >= self.window_len:
                for st in range(0, tlen - self.window_len + 1, self.stride):
                    ed = st + self.window_len
                    for split in splits:
                        if train_skeleton_source == "vicon" and has_gt_skeleton:
                            skeleton_source = "gt"
                        elif has_extract:
                            skeleton_source = "extract"
                        else:
                            skeleton_source = train_skeleton_source

                        if self.multi_person:
                            # EgoHumans-style: imu axis 1 corresponds to persons.
                            # Emit only matched IMU-person pairs.
                            for person_idx in range(max(n_gt, 1)):
                                imu_idx = person_idx
                                if imu_idx >= n_imu:
                                    continue
                                window_rows.append(
                                    {
                                        "subject": f"{subject}_P{person_idx}" if subject != "all" else f"P{person_idx}",
                                        "session": session,
                                        "split": split,
                                        "npz_path": str(rel_npz),
                                        "window_start": int(st),
                                        "window_end": int(ed),
                                        "window_len": int(self.window_len),
                                        "skeleton_source": skeleton_source,
                                        "person_idx": person_idx,
                                        "imu_idx": imu_idx,
                                    }
                                )
                        else:
                            for person_idx in range(max(n_gt, 1)):
                                for imu_idx in range(n_imu):
                                    window_rows.append(
                                        {
                                            "subject": subject,
                                            "session": session,
                                            "split": split,
                                            "npz_path": str(rel_npz),
                                            "window_start": int(st),
                                            "window_end": int(ed),
                                            "window_len": int(self.window_len),
                                            "skeleton_source": skeleton_source,
                                            "person_idx": person_idx,
                                            "imu_idx": imu_idx,
                                        }
                                    )

        write_csv(
            self.out_dir / "sequences.csv",
            sequence_rows,
            ["subject", "session", "split", "npz_path", "num_frames"],
        )

        write_csv(
            self.out_dir / "windows_all.csv",
            window_rows,
            ["subject", "session", "split", "npz_path", "window_start", "window_end", "window_len", "skeleton_source", "person_idx", "imu_idx"],
        )

        for split in ["train", "val", "test"]:
            split_rows = [r for r in window_rows if r["split"] == split]
            write_csv(
                self.out_dir / f"windows_{split}.csv",
                split_rows,
                ["subject", "session", "split", "npz_path", "window_start", "window_end", "window_len", "skeleton_source", "person_idx", "imu_idx"],
            )

        summary = {
            "num_sequences": len(sequence_rows),
            "num_windows": len(window_rows),
            "window_len": self.window_len,
            "stride": self.stride,
            "sensor_order": self.sensor_order,
            "train_subjects": self.train_subj,
            "val_subjects": self.val_subj,
            "test_subjects": self.test_subj,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("Slice done")
        print(f"Output dir: {self.out_dir}")
        print(f"Sequences: {len(sequence_rows)}, windows: {len(window_rows)}")
        return self.out_dir

    def _find_extract_dir(self, sequence_id: str) -> Path | None:
        """Find extract result directory matching sequence_id under skeleton_root."""
        if self.skeleton_root is None:
            return None
        if sequence_id.startswith("totalcapture_"):
            core = sequence_id[len("totalcapture_"):]
            patterns = [core, f"TC_{core}"]
        elif sequence_id.startswith("custom_"):
            core = sequence_id[len("custom_"):]
            patterns = [core]
        else:
            patterns = [sequence_id]
        for subdir in self.skeleton_root.iterdir():
            if not subdir.is_dir():
                continue
            if any(pat in subdir.name for pat in patterns):
                return subdir
        return None
