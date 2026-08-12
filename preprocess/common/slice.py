"""Shared helpers for dataset slicing entrypoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .extract import load_alphapose_multiperson

SENSOR_ORDER = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]

EGOHUMANS_MOBIND_E21_SOURCE_SPLIT = {
    "train_sessions": [
        "01_003", "01_004", "01_005", "01_006", "01_007", "01_008", "01_009", "01_010", "01_011", "01_012", "01_013", "01_014",
        "02_001", "02_002", "02_003", "02_004", "02_005", "02_006",
        "03_005", "03_006", "03_007", "03_008", "03_009", "03_010",
        "04_002", "04_003", "04_004", "04_005", "04_006", "04_007", "04_008",
        "05_001", "05_002", "05_003", "05_004", "05_005", "05_006", "05_007", "05_008",
        "06_001", "06_002", "06_003", "06_004", "06_005", "06_006", "06_007", "06_008", "06_009", "06_010", "06_011", "06_012", "06_013", "06_014", "06_015", "06_016", "06_017", "06_018", "06_019", "06_020", "06_021", "06_022", "06_023", "06_024", "06_025", "06_026", "06_027", "06_028", "06_029", "06_030", "06_031", "06_032", "06_033",
        "07_001", "07_002", "07_003", "07_004", "07_005", "07_006", "07_007", "07_008", "07_009", "07_010", "07_011", "07_012", "07_013", "07_014", "07_015", "07_016", "07_017", "07_018", "07_019", "07_020", "07_021", "07_022",
    ],
    "val_sessions": ["01_002", "03_004", "04_001"],
    "test_sessions": ["01_001", "03_001", "03_002", "03_003"],
}


def load_slice_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = load_config(config_path)
    slice_cfg = data.get("slice", {})
    if slice_cfg is None:
        return {}
    if not isinstance(slice_cfg, dict):
        raise ValueError(f"Invalid slice section in config: {config_path}")
    return slice_cfg


def resolve_output_paths(
    output_dir: str | None,
    manifest_csv: str | None,
    default_manifest: str,
) -> tuple[Path, Path]:
    default_manifest_path = Path(default_manifest).expanduser().resolve()
    if output_dir:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
    else:
        resolved_output_dir = default_manifest_path.parent
    if manifest_csv:
        resolved_manifest_csv = Path(manifest_csv).expanduser().resolve()
    else:
        resolved_manifest_csv = default_manifest_path
    return resolved_output_dir, resolved_manifest_csv


def parse_subjects(spec: Any) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip() for x in spec if str(x).strip()]
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def parse_sensor_order(spec: Any) -> list[str]:
    if spec is None:
        return list(SENSOR_ORDER)
    if isinstance(spec, str):
        return [x.strip() for x in spec.split(",") if x.strip()]
    return [str(x).strip() for x in spec if str(x).strip()]


def parse_bool(spec: Any, default: bool = True) -> bool:
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


def apply_split_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    policy = str(cfg.get("split_policy") or "").strip().lower()
    if not policy:
        return cfg
    if policy == "mobind_e21_source":
        for key, value in EGOHUMANS_MOBIND_E21_SOURCE_SPLIT.items():
            cfg.setdefault(key, value)
        return cfg
    raise ValueError(f"Unknown slice.split_policy: {policy}")


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
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
    rows: list[np.ndarray] = []
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


def parse_xsens_sensors(path: Path, selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        _ = lines[i]
        i += 1
        if i + n_sensors > len(lines):
            break
        sensor_map: dict[str, np.ndarray] = {}
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
    tlen = quat4.shape[0]
    out = np.zeros((tlen, 48), dtype=np.float32)
    for i in range(4):
        rot = quat_to_rotmat(quat4[:, i, :]).reshape(tlen, 9)
        acc = acc3[:, i, :]
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def map_totalcapture21_to_h36m17(joint_names: list[str], xyz21: np.ndarray) -> np.ndarray:
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
    root = skel[:, 0:1, :]
    skel = skel - root
    scale = np.linalg.norm(skel[:, 8, :] - skel[:, 0, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    skel = skel / scale[:, None, :]
    return skel.astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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


def assign_frame_acc_candidate_groups(
    window_rows: list[dict[str, Any]],
    cross_sequence_group_size: int = 0,
) -> dict[str, int]:
    """Assign deterministic candidate groups without inventing singleton accuracy.

    Multi-person windows retain their natural synchronized group.  Remaining
    single-person rows can be grouped across sequences at the same relative
    window position, which provides a reproducible source-domain retrieval
    protocol for datasets such as TotalCapture.
    """
    natural_groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in window_rows:
        row["candidate_group_id"] = ""
        row["candidate_index"] = ""
        key = (
            str(row["split"]),
            str(row["npz_path"]),
            int(row["window_start"]),
            int(row["window_end"]),
        )
        natural_groups.setdefault(key, []).append(row)

    assigned_groups = 0
    assigned_rows = 0
    unresolved: list[dict[str, Any]] = []
    for key in sorted(natural_groups):
        rows = natural_groups[key]
        if len(rows) < 2:
            unresolved.extend(rows)
            continue
        group_id = f"native:{key[0]}:{key[1]}:{key[2]}:{key[3]}"
        for candidate_index, row in enumerate(rows):
            row["candidate_group_id"] = group_id
            row["candidate_index"] = candidate_index
        assigned_groups += 1
        assigned_rows += len(rows)

    if cross_sequence_group_size >= 2:
        by_position: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for row in unresolved:
            key = (str(row["split"]), int(row["window_start"]), int(row["window_end"]))
            by_position.setdefault(key, []).append(row)
        for key in sorted(by_position):
            rows = sorted(
                by_position[key],
                key=lambda row: (str(row["npz_path"]), int(row["person_idx"]), int(row["imu_idx"])),
            )
            for chunk_index, start in enumerate(range(0, len(rows), cross_sequence_group_size)):
                chunk = rows[start : start + cross_sequence_group_size]
                if len(chunk) < 2:
                    continue
                group_id = f"cross:{key[0]}:{key[1]}:{key[2]}:{chunk_index}"
                for candidate_index, row in enumerate(chunk):
                    row["candidate_group_id"] = group_id
                    row["candidate_index"] = candidate_index
                assigned_groups += 1
                assigned_rows += len(chunk)

    return {
        "candidate_groups": assigned_groups,
        "candidate_rows": assigned_rows,
        "singleton_rows": len(window_rows) - assigned_rows,
    }


def run_slice_from_npz(
    root: Path,
    out_dir: Path,
    slice_cfg: dict[str, Any],
) -> Path:
    cfg = dict(slice_cfg)
    cfg = apply_split_policy(cfg)
    window_len = int(cfg.get("window_len", 24))
    stride = int(cfg.get("stride", 16))
    sensor_order = parse_sensor_order(cfg.get("sensor_order", SENSOR_ORDER))
    train_subj = parse_subjects(cfg.get("train_subjects", "S1,S2,S3"))
    val_subj = parse_subjects(cfg.get("val_subjects", "S4"))
    test_subj = parse_subjects(cfg.get("test_subjects", "S5"))
    train_sessions = parse_subjects(cfg.get("train_sessions", ""))
    val_sessions = parse_subjects(cfg.get("val_sessions", ""))
    test_sessions = parse_subjects(cfg.get("test_sessions", ""))
    max_sequences = int(cfg.get("max_sequences", 0))
    skeleton_normalize = parse_bool(cfg.get("skeleton_normalize", True), default=True)
    skeleton_source = cfg.get("skeleton_source", "vicon")
    skeleton_root = None
    if skeleton_source == "alphapose":
        skeleton_root = Path(cfg.get("skeleton_root", "/data/fzliang/reid-project/totalcapture/skeleton/alphapose"))
    multi_person = parse_bool(cfg.get("multi_person"), default=False)
    candidate_group_size = int(cfg.get("frame_acc_candidate_group_size", 0))
    require_multi_candidate_test = parse_bool(cfg.get("require_multi_candidate_test"), default=False)

    npz_paths = sorted((root / "sequences").glob("*.npz"))
    if max_sequences > 0:
        npz_paths = npz_paths[: max_sequences]

    sequence_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for npz_path in npz_paths:
        data = dict(np.load(npz_path, allow_pickle=True))
        sequence_id = str(data["sequence_id"].item())
        if sequence_id.startswith("totalcapture_"):
            parts = sequence_id.split("_")
            subject = parts[1]
            session = "_".join(parts[2:-1])
        elif sequence_id.startswith("custom_plus_"):
            session = sequence_id[len("custom_plus_"):]
            subject = "all"
        elif sequence_id.startswith("custom_"):
            session = sequence_id[len("custom_"):]
            subject = "all"
        elif sequence_id.startswith("egohumans_"):
            session = sequence_id[len("egohumans_"):]
            subject = "all"
        else:
            subject = "unknown"
            session = sequence_id

        splits = []
        use_session_split = bool(train_sessions or val_sessions or test_sessions)
        if use_session_split:
            if session in train_sessions:
                splits.append("train")
            if session in val_sessions:
                splits.append("val")
            if session in test_sessions:
                splits.append("test")
        else:
            if subject in train_subj:
                splits.append("train")
            if subject in val_subj:
                splits.append("val")
            if subject in test_subj:
                splits.append("test")
        if not splits:
            print(f"Warning: {sequence_id} (subject={subject}, session={session}) not assigned to any split, skipping...")
            continue

        if skeleton_source == "alphapose" and skeleton_root is not None:
            extract_dir = None
            if sequence_id.startswith("totalcapture_"):
                core = sequence_id[len("totalcapture_"):]
                patterns = [core, f"TC_{core}"]
            elif sequence_id.startswith("custom_"):
                core = sequence_id[len("custom_"):]
                patterns = [core]
            elif sequence_id.startswith("egohumans_"):
                core = sequence_id[len("egohumans_"):]
                patterns = [core]
            else:
                patterns = [sequence_id]
            for subdir in skeleton_root.iterdir():
                if not subdir.is_dir():
                    continue
                if any(pat in subdir.name for pat in patterns):
                    extract_dir = subdir
                    break
            if extract_dir is not None:
                skeleton_json = extract_dir / "skeleton.json"
                if skeleton_json.exists():
                    alphapose_frames, track_ids = load_alphapose_multiperson(skeleton_json)
                    data = _align_extract_to_npz(
                        data,
                        alphapose_frames,
                        track_ids,
                        normalize_extract_skeleton=skeleton_normalize,
                    )
                    data["extract_source"] = str(skeleton_json)
                else:
                    print(f"Warning: skeleton.json not found in {extract_dir} for {sequence_id}")
            else:
                print(f"Warning: No extract result found for {sequence_id}")

        rel_npz = Path("sequences") / npz_path.name
        if skeleton_source == "alphapose" and "extract_skeleton" in data:
            aligned_dir = out_dir / "aligned_sequences"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            rel_npz = Path("aligned_sequences") / f"{sequence_id}.npz"
            np.savez_compressed(out_dir / rel_npz, **data)

        sequence_rows.append(
            {
                "subject": subject,
                "session": session,
                "split": ",".join(splits),
                "npz_path": str(rel_npz),
                "num_frames": int(data["frame_ids"].shape[0]),
            }
        )

        train_skeleton_source = skeleton_source
        has_gt_skeleton = "gt_skeleton" in data
        n_imu = int(data["imu_ids"].shape[0])
        n_gt = int(data["gt_person_ids"].shape[0]) if "gt_person_ids" in data else 0
        has_extract = "extract_skeleton" in data and data["extract_skeleton"].shape[1] > 0

        if data["frame_ids"].shape[0] >= window_len:
            for st in range(0, data["frame_ids"].shape[0] - window_len + 1, stride):
                ed = st + window_len
                for split in splits:
                    if train_skeleton_source == "vicon" and has_gt_skeleton:
                        skeleton_source_for_window = "gt"
                    elif has_extract:
                        skeleton_source_for_window = "extract"
                    else:
                        skeleton_source_for_window = train_skeleton_source

                    if multi_person:
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
                                    "window_len": int(window_len),
                                    "skeleton_source": skeleton_source_for_window,
                                    "person_idx": person_idx,
                                    "imu_idx": imu_idx,
                                    "source_sequence": sequence_id,
                                    "source_person": person_idx,
                                    "source_window_start": int(st),
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
                                        "window_len": int(window_len),
                                        "skeleton_source": skeleton_source_for_window,
                                        "person_idx": person_idx,
                                        "imu_idx": imu_idx,
                                        "source_sequence": sequence_id,
                                        "source_person": person_idx,
                                        "source_window_start": int(st),
                                    }
                                )

    candidate_stats = assign_frame_acc_candidate_groups(window_rows, candidate_group_size)
    window_fieldnames = [
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
        "candidate_group_id",
        "candidate_index",
    ]
    write_csv(out_dir / "sequences.csv", sequence_rows, ["subject", "session", "split", "npz_path", "num_frames"])
    write_csv(out_dir / "windows_all.csv", window_rows, window_fieldnames)
    for split in ["train", "val", "test"]:
        split_rows = [r for r in window_rows if r["split"] == split]
        if split == "test" and require_multi_candidate_test:
            split_rows = [r for r in split_rows if r["candidate_group_id"]]
        write_csv(
            out_dir / f"windows_{split}.csv",
            split_rows,
            window_fieldnames,
        )
    summary = {
        "num_sequences": len(sequence_rows),
        "num_windows": len(window_rows),
        "window_len": window_len,
        "stride": stride,
        "sensor_order": sensor_order,
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
        "frame_acc_candidate_group_size": candidate_group_size,
        "require_multi_candidate_test": require_multi_candidate_test,
        **candidate_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir
