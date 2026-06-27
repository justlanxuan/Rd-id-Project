"""Physics-based matching evaluation entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.modules.matchers.physics_matchers.frequency import (
    FrequencyPhysicsMatcher,
    build_sequence_windows,
)
from src.utils.config import resolve_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-based matching evaluation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _unique_test_sequences(rows: List[Dict[str, str]], root_dir: Path) -> List[Tuple[str, Path, List[Dict[str, str]]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    order: List[str] = []
    for row in rows:
        if row.get("split", "") != "test":
            continue
        npz_rel = row["npz_path"]
        npz_path = (root_dir / npz_rel).resolve()
        key = str(npz_path)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    sequences: List[Tuple[str, Path, List[Dict[str, str]]]] = []
    for key in order:
        npz_path = Path(key)
        data = np.load(npz_path, allow_pickle=True)
        sequence_id = str(data["sequence_id"].item())
        sequences.append((sequence_id, npz_path, grouped[key]))
    return sequences


def _resolve_test_output_dir(cfg: Dict[str, object]) -> Path:
    test_cfg = cfg.get("test", {}) if isinstance(cfg, dict) else {}
    out = test_cfg.get("output", {}) if isinstance(test_cfg, dict) else {}
    work_dir = Path(cfg.get("work_dir", ".")).expanduser().resolve()
    return (
        work_dir
        / out.get("output_root", "test")
        / out.get("run_name", "")
    ).resolve()


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _build_sliding_windows(t_len: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """Build sliding windows [start, end) over a sequence length.

    If t_len < window_size, returns one fallback window [0, t_len).
    """
    if t_len <= 0:
        return []

    w = max(int(window_size), 1)
    s = max(int(stride), 1)

    if t_len <= w:
        return [(0, t_len)]

    windows: List[Tuple[int, int]] = []
    for st in range(0, t_len - w + 1, s):
        windows.append((st, st + w))
    if windows and windows[-1][1] < t_len:
        windows.append((t_len - w, t_len))
    return windows


def _sequence_person_arrays(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a single-person IMU/skeleton/visibility triplet from one sequence."""
    imu = np.asarray(data["imu"], dtype=np.float32)
    if imu.ndim == 3:
        imu = imu[:, 0]

    if "extract_skeleton" in data and "extract_visibility" in data:
        skeleton = np.asarray(data["extract_skeleton"], dtype=np.float32)
        visibility = np.asarray(data["extract_visibility"], dtype=bool)
    else:
        skeleton = np.asarray(data["gt_skeleton"], dtype=np.float32)
        visibility = np.asarray(data["gt_visibility"], dtype=bool)

    if skeleton.ndim == 4:
        skeleton = skeleton[:, 0]
    if visibility.ndim == 2:
        visibility = visibility[:, 0]

    return imu, skeleton, visibility


def _build_multi_sequence_data(
    selected: List[Dict[str, object]],
    starts: List[int],
    length: int,
    imu_labels: np.ndarray | None = None,
    person_labels: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Stack several single-person sequences into one synthetic multi-person sample."""
    imu_list: List[np.ndarray] = []
    skel_list: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []

    for item, start in zip(selected, starts):
        data = item["data"]
        imu, skeleton, visibility = _sequence_person_arrays(data)
        end = start + length
        imu_list.append(imu[start:end])
        skel_list.append(skeleton[start:end])
        vis_list.append(visibility[start:end])

    group_size = len(selected)
    imu_stack = np.stack(imu_list, axis=1)  # [T, G, 48]
    skeleton_stack = np.stack(skel_list, axis=1)  # [T, G, 17, 3]
    visibility_stack = np.stack(vis_list, axis=1)  # [T, G]

    if imu_labels is None:
        imu_labels = np.arange(group_size, dtype=np.int64)
    else:
        imu_labels = np.asarray(imu_labels, dtype=np.int64)
    if person_labels is None:
        person_labels = np.arange(group_size, dtype=np.int64)
    else:
        person_labels = np.asarray(person_labels, dtype=np.int64)

    imu_label_to_pos = np.empty(group_size, dtype=np.int64)
    person_label_to_pos = np.empty(group_size, dtype=np.int64)
    for pos, label in enumerate(imu_labels.tolist()):
        imu_label_to_pos[int(label)] = int(pos)
    for pos, label in enumerate(person_labels.tolist()):
        person_label_to_pos[int(label)] = int(pos)

    return {
        "frame_ids": np.arange(length, dtype=np.int64),
        "imu": imu_stack,
        "imu_ids": imu_labels,
        "imu_label_to_pos": imu_label_to_pos,
        "gt_person_ids": person_labels,
        "person_label_to_pos": person_label_to_pos,
        "gt_visibility": visibility_stack,
        "gt_skeleton": skeleton_stack,
        "gt_to_extract_map": np.tile(np.arange(group_size, dtype=np.int64)[np.newaxis, :], (length, 1)),
        "extract_person_ids": person_labels,
        "extract_visibility": visibility_stack,
        "extract_skeleton": skeleton_stack,
    }


def _make_shuffled_labels(group_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Create independent shuffled IMU/person labels for a synthetic group."""
    imu_labels = rng.permutation(np.arange(group_size, dtype=np.int64))
    person_labels = rng.permutation(np.arange(group_size, dtype=np.int64))
    return imu_labels, person_labels


def _assignment_accuracy(
    assignments: List[Tuple[int, int]],
    imu_label_to_pos: np.ndarray,
    person_label_to_pos: np.ndarray,
) -> float:
    correct = 0
    total = 0
    for imu_label, person_label in assignments:
        imu_pos = int(imu_label_to_pos[int(imu_label)])
        person_pos = int(person_label_to_pos[int(person_label)])
        total += 1
        if imu_pos == person_pos:
            correct += 1
    return float(correct) / float(total) if total > 0 else 0.0


def _select_group_members(
    sequences: List[Dict[str, object]],
    group_size: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    if len(sequences) < group_size:
        return []
    indices = rng.choice(len(sequences), size=group_size, replace=False)
    return [sequences[int(i)] for i in indices]


def _evaluate_physics_grouped(
    sequences: List[Dict[str, object]],
    matcher: FrequencyPhysicsMatcher,
    group_size: int,
    num_trials: int,
    chunk_windows: int,
    min_chunk_windows: int,
    seed: int,
) -> Dict[str, object]:
    if len(sequences) < group_size:
        return {
            "group_size": group_size,
            "num_sequences": len(sequences),
            "num_trials": 0,
            "mean_acc": None,
            "std_acc": None,
            "note": f"insufficient sequences ({len(sequences)} < {group_size})",
        }

    rng = np.random.default_rng(seed)
    trial_acc: List[float] = []
    trial_scores: List[float] = []

    for _ in range(num_trials):
        selected = _select_group_members(sequences, group_size, rng)
        if not selected:
            continue

        lengths = [int(item["length"]) for item in selected]
        length = min(min(lengths), int(chunk_windows))
        if length < int(min_chunk_windows):
            continue

        starts: List[int] = []
        for item in selected:
            seq_len = int(item["length"])
            max_start = max(seq_len - length, 0)
            starts.append(int(rng.integers(0, max_start + 1)) if max_start > 0 else 0)

        imu_labels, person_labels = _make_shuffled_labels(group_size, rng)
        data = _build_multi_sequence_data(selected, starts, length, imu_labels=imu_labels, person_labels=person_labels)
        result = matcher.match_sequence(data, [(0, length)])
        assignments = result.get("assignments", [])

        trial_acc.append(_assignment_accuracy(assignments, data["imu_label_to_pos"], data["person_label_to_pos"]))
        if result.get("scores"):
            trial_scores.append(float(np.mean(result["scores"])))

    return {
        "group_size": group_size,
        "num_sequences": len(sequences),
        "num_trials": int(len(trial_acc)),
        "requested_trials": int(num_trials),
        "chunk_windows": int(chunk_windows),
        "min_chunk_windows": int(min_chunk_windows),
        "mean_acc": float(np.mean(trial_acc)) if trial_acc else None,
        "std_acc": float(np.std(trial_acc)) if trial_acc else None,
        "mean_score": float(np.mean(trial_scores)) if trial_scores else None,
    }


def _evaluate_physics_synchronous(
    sequences: List[Dict[str, object]],
    matcher: FrequencyPhysicsMatcher,
    group_size: int,
    window_size: int,
    stride: int,
    num_trials: int,
    seed: int,
) -> Dict[str, object]:
    if len(sequences) < group_size:
        return {
            "group_size": group_size,
            "num_sequences": len(sequences),
            "num_trials": 0,
            "mean_acc": None,
            "std_acc": None,
            "note": f"insufficient sequences ({len(sequences)} < {group_size})",
        }

    rng = np.random.default_rng(seed)
    trial_acc: List[float] = []
    trial_num_windows: List[int] = []

    for _ in range(num_trials):
        selected = _select_group_members(sequences, group_size, rng)
        if not selected:
            continue

        lengths = [int(item["length"]) for item in selected]
        length = min(lengths)
        if length <= 0:
            continue

        starts: List[int] = []
        for item in selected:
            seq_len = int(item["length"])
            max_start = max(seq_len - length, 0)
            starts.append(int(rng.integers(0, max_start + 1)) if max_start > 0 else 0)

        windows = _build_sliding_windows(length, window_size, stride)
        if not windows:
            continue

        imu_labels, person_labels = _make_shuffled_labels(group_size, rng)
        data = _build_multi_sequence_data(selected, starts, length, imu_labels=imu_labels, person_labels=person_labels)
        result = matcher.match_sequence(data, windows)
        assignments = result.get("assignments", [])

        trial_acc.append(_assignment_accuracy(assignments, data["imu_label_to_pos"], data["person_label_to_pos"]))
        trial_num_windows.append(len(windows))

    return {
        "group_size": group_size,
        "num_sequences": len(sequences),
        "num_trials": int(len(trial_acc)),
        "requested_trials": int(num_trials),
        "window_size": int(window_size),
        "stride": int(stride),
        "mean_num_windows": float(np.mean(trial_num_windows)) if trial_num_windows else None,
        "mean_acc": float(np.mean(trial_acc)) if trial_acc else None,
        "std_acc": float(np.std(trial_acc)) if trial_acc else None,
    }


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    test_cfg = cfg.get("test", {})
    matcher_cfg = test_cfg.get("matcher", {})
    physics_cfg = matcher_cfg.get("physics_based_matcher", {})
    frequency_cfg = physics_cfg.get("frequency", {})

    if not physics_cfg.get("enabled", False):
        raise ValueError("Physics matcher is disabled. Enable test.matcher.physics_based_matcher.enabled in config.")
    if not frequency_cfg.get("enabled", True):
        raise ValueError("Frequency matcher is disabled. Enable test.matcher.physics_based_matcher.frequency.enabled in config.")

    paths = cfg.get("paths", {})
    test_csv = Path(paths.get("test_csv", "")).expanduser().resolve()
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    data_root = Path(paths.get("data_root", test_csv.parent)).expanduser().resolve()
    rows = _read_csv_rows(test_csv)
    sequences = _unique_test_sequences(rows, data_root)
    if not sequences:
        print("No test sequences found.")
        return

    sequence_records: List[Dict[str, object]] = []
    for sequence_id, npz_path, seq_rows in sequences:
        data = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
        imu = np.asarray(data["imu"])
        if imu.ndim == 3:
            length = int(imu.shape[0])
        else:
            length = int(imu.shape[0])
        sequence_records.append(
            {
                "sequence_id": sequence_id,
                "npz_path": npz_path,
                "rows": seq_rows,
                "data": data,
                "length": length,
            }
        )

    matcher = FrequencyPhysicsMatcher(frequency_cfg)
    eval_profile = str(physics_cfg.get("evaluation_profile", "single_sequence")).strip() or "single_sequence"
    results = []

    for item in sequence_records:
        sequence_id = str(item["sequence_id"])
        npz_path = Path(item["npz_path"])
        seq_rows = item["rows"]
        data = item["data"]

        gt_person_ids = np.asarray(data.get("gt_person_ids"), dtype=np.int64)
        gt_imu_ids = np.asarray(data.get("imu_ids"), dtype=np.int64)

        has_extract = all(
            key in data for key in ["extract_skeleton", "extract_visibility", "extract_person_ids", "extract_bboxes"]
        )

        if has_extract:
            gt_bboxes = np.asarray(data.get("gt_bboxes"), dtype=np.float32)
            gt_visibility = np.asarray(data.get("gt_visibility"), dtype=bool)
            pred_bboxes = np.asarray(data.get("extract_bboxes"), dtype=np.float32)
            pred_visibility = np.asarray(data.get("extract_visibility"), dtype=bool)

            if gt_bboxes.ndim != 3 or pred_bboxes.ndim != 3:
                raise ValueError(f"Missing bbox tensors in {npz_path}")

            T = int(gt_bboxes.shape[0])
            N_gt = int(gt_bboxes.shape[1])
            N_pred = int(pred_bboxes.shape[1])
            data_for_match = data
            eval_mode = "bbox_iou"
        else:
            # ID-only fallback (e.g., TotalCapture Vicon): no extracted bboxes/tracks.
            if "gt_skeleton" not in data or "gt_visibility" not in data or "gt_person_ids" not in data:
                raise ValueError(f"Missing required GT tensors for ID-only evaluation in {npz_path}")
            gt_skeleton = np.asarray(data.get("gt_skeleton"), dtype=np.float32)
            gt_visibility = np.asarray(data.get("gt_visibility"), dtype=bool)
            if gt_skeleton.ndim != 4 or gt_visibility.ndim != 2:
                raise ValueError(f"Invalid GT tensor shapes for ID-only evaluation in {npz_path}")

            T = int(gt_skeleton.shape[0])
            N_gt = int(gt_skeleton.shape[1])
            N_pred = N_gt

            # Repackage GT as pseudo-extract tracks for physics matching.
            data_for_match = dict(data)
            data_for_match["extract_skeleton"] = gt_skeleton
            data_for_match["extract_visibility"] = gt_visibility
            data_for_match["extract_person_ids"] = gt_person_ids
            eval_mode = "id_only"

        if eval_profile == "multi_video":
            continue

        iou_thresh = float(physics_cfg.get("iou_threshold", 0.5))

        # Reuse synchronous sliding window settings when enabled in test config.
        synchronous_cfg = test_cfg.get("synchronous_test", {}) if isinstance(test_cfg, dict) else {}
        use_sync_windows = bool(synchronous_cfg.get("enabled", False))
        if use_sync_windows:
            window_size = int(synchronous_cfg.get("window_size", 24))
            stride = int(synchronous_cfg.get("stride", 1))
            windows = _build_sliding_windows(T, window_size, stride)
        else:
            windows = build_sequence_windows(seq_rows)
        if not windows:
            continue

        # Per-window independent matching + per-frame multi-window voting.
        # frame_votes[t][track_id][imu_id] accumulates weighted votes from windows covering frame t.
        frame_votes: List[Dict[int, Dict[int, float]]] = [dict() for _ in range(T)]
        window_results: List[Dict[str, object]] = []

        for st, ed in windows:
            local_result = matcher.match_sequence(data_for_match, [(st, ed)])
            assignments = local_result.get("assignments", [])
            confidences = local_result.get("confidences", [])

            for k, pair in enumerate(assignments):
                imu_id = int(pair[0])
                track_id = int(pair[1])
                weight = float(confidences[k]) if k < len(confidences) else 1.0
                for t in range(st, min(ed, T)):
                    if track_id not in frame_votes[t]:
                        frame_votes[t][track_id] = {}
                    frame_votes[t][track_id][imu_id] = frame_votes[t][track_id].get(imu_id, 0.0) + weight

            window_results.append(
                {
                    "window_start": int(st),
                    "window_end": int(ed),
                    "assignments": assignments,
                    "scores": local_result.get("scores", []),
                    "confidences": confidences,
                }
            )

        # Build final assignment for each frame from multi-window votes.
        frame_track_to_imu: List[Dict[int, int]] = []
        for t in range(T):
            mapping_t: Dict[int, int] = {}
            for track_id, imu_votes in frame_votes[t].items():
                if not imu_votes:
                    continue
                best_imu = max(imu_votes.items(), key=lambda item: item[1])[0]
                mapping_t[int(track_id)] = int(best_imu)
            frame_track_to_imu.append(mapping_t)

        # Evaluate each frame once using the voted per-frame assignment.
        total_correct = 0
        total_count = 0
        frame_accs: List[float] = []

        for t in range(T):
            gt_indices = [g for g in range(N_gt) if gt_visibility[t, g]]

            if not gt_indices:
                continue

            frame_total = len(gt_indices)
            frame_correct = 0

            if eval_mode == "bbox_iou":
                pred_indices = [p for p in range(N_pred) if pred_visibility[t, p]]
                if pred_indices:
                    iou_mat = np.zeros((len(gt_indices), len(pred_indices)), dtype=np.float32)
                    for gi, g in enumerate(gt_indices):
                        for pi, p in enumerate(pred_indices):
                            iou_mat[gi, pi] = _iou(gt_bboxes[t, g], pred_bboxes[t, p])

                    row_ind, col_ind = linear_sum_assignment(-iou_mat)
                    for r, c in zip(row_ind, col_ind):
                        if iou_mat[r, c] < iou_thresh:
                            continue
                        g = gt_indices[r]
                        p = pred_indices[c]
                        pred_track_id = int(data["extract_person_ids"][p])
                        pred_imu = frame_track_to_imu[t].get(pred_track_id, None)
                        gt_imu = int(gt_imu_ids[g]) if g < len(gt_imu_ids) else int(gt_person_ids[g])
                        if pred_imu is not None and pred_imu == gt_imu:
                            frame_correct += 1
            else:
                # ID-only mode: each GT person has an intrinsic id per segment.
                for g in gt_indices:
                    gt_pid = int(gt_person_ids[g])
                    pred_imu_idx = frame_track_to_imu[t].get(gt_pid, None)
                    if pred_imu_idx is None:
                        continue
                    if 0 <= int(pred_imu_idx) < len(gt_imu_ids):
                        if int(gt_imu_ids[int(pred_imu_idx)]) == gt_pid:
                            frame_correct += 1

            frame_accs.append(frame_correct / frame_total)
            total_correct += frame_correct
            total_count += frame_total

        acc = float(total_correct / total_count) if total_count > 0 else 0.0
        seq_result = {
            "sequence_id": sequence_id,
            "num_windows": len(windows),
            "window_source": "synchronous_test" if use_sync_windows else "slice_csv",
            "eval_mode": eval_mode,
            "num_imus": int(np.asarray(data["imu"]).shape[1] if np.asarray(data["imu"]).ndim == 3 else 1),
            "num_persons": int(N_gt),
            "accuracy": acc,
            "mean_frame_accuracy": float(np.mean(frame_accs)) if frame_accs else 0.0,
            "num_frames_with_votes": int(sum(1 for m in frame_track_to_imu if m)),
            "window_results": window_results,
        }
        results.append(seq_result)
        print(
            json.dumps(
                {
                    "sequence_id": sequence_id,
                    "accuracy": acc,
                    "mean_frame_accuracy": seq_result["mean_frame_accuracy"],
                    "num_frames_with_votes": seq_result["num_frames_with_votes"],
                    "first_window_assignments": window_results[0]["assignments"] if window_results else [],
                },
                indent=2,
            )
        )

    grouped_cfg = test_cfg.get("grouped_test", {}) if isinstance(test_cfg, dict) else {}
    synchronous_cfg = test_cfg.get("synchronous_test", {}) if isinstance(test_cfg, dict) else {}
    grouped_summary = None
    synchronous_summary = None
    if eval_profile == "multi_video":
        group_sizes_raw = str(grouped_cfg.get("group_sizes", "2,4,6,8,16"))
        group_sizes = [int(x.strip()) for x in group_sizes_raw.split(",") if x.strip()]
        grouped_results: List[Dict[str, object]] = []
        for group_size in group_sizes:
            grouped_results.append(
                _evaluate_physics_grouped(
                    sequence_records,
                    matcher,
                    group_size=group_size,
                    num_trials=int(grouped_cfg.get("num_trials", 50)),
                    chunk_windows=int(grouped_cfg.get("chunk_windows", 30)),
                    min_chunk_windows=int(grouped_cfg.get("min_chunk_windows", 15)),
                    seed=int(grouped_cfg.get("seed", 42)),
                )
            )
        grouped_summary = {
            "matcher": "physics_frequency",
            "mode": "grouped_multi_video",
            "num_sequences": len(sequence_records),
            "results": grouped_results,
        }
        print(json.dumps(grouped_summary, indent=2))

        if synchronous_cfg.get("enabled", False):
            sync_group_size = int(synchronous_cfg.get("group_size", 2))
            synchronous_summary = {
                "matcher": "physics_frequency",
                "mode": "synchronous_multi_video",
                "group_size": sync_group_size,
                "num_sequences": len(sequence_records),
                "result": _evaluate_physics_synchronous(
                    sequence_records,
                    matcher,
                    group_size=sync_group_size,
                    window_size=int(synchronous_cfg.get("window_size", 24)),
                    stride=int(synchronous_cfg.get("stride", 1)),
                    num_trials=int(synchronous_cfg.get("num_trials", grouped_cfg.get("num_trials", 50))),
                    seed=int(synchronous_cfg.get("seed", 42)),
                ),
            }
            print(json.dumps(synchronous_summary, indent=2))

    summary = {
        "matcher": "physics_frequency",
        "evaluation_profile": eval_profile,
        "num_sequences": len(sequence_records),
        "sequences": results,
    }
    if results:
        summary["mean_accuracy"] = float(np.mean([r["accuracy"] for r in results]))
        summary["mean_frame_accuracy"] = float(np.mean([r["mean_frame_accuracy"] for r in results]))
        summary["mean_num_frames_with_votes"] = float(np.mean([r["num_frames_with_votes"] for r in results]))
    if grouped_summary is not None:
        summary["grouped_test"] = grouped_summary
    if synchronous_summary is not None:
        summary["synchronous_test"] = synchronous_summary

    print(json.dumps(summary, indent=2))

    save_json = args.save_json.strip()
    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    else:
        out_dir = _resolve_test_output_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "physics_results.json").write_text(json.dumps(summary, indent=2))
        if grouped_summary is not None:
            (out_dir / "physics_grouped_results.json").write_text(json.dumps(grouped_summary, indent=2))
        if synchronous_summary is not None:
            (out_dir / "physics_synchronous_results.json").write_text(json.dumps(synchronous_summary, indent=2))


if __name__ == "__main__":
    main()
