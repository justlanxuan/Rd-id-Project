"""Synchronous multi-person evaluation with HOTA/AssA metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from trackeval.metrics import HOTA

from src.engine.common import build_alignment_model
from src.datasets.alignment_dataset import lowpass_filter_fft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous multi-person evaluation")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=24, help="Chunk size for Hungarian matching")
    parser.add_argument("--stride", type=int, default=1, help="Stride between chunks (default 1 = evaluate every frame)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--save_mot_pred", type=str, default="")
    parser.add_argument("--save_mot_gt", type=str, default="")
    parser.add_argument("--imu_stats_json", type=str, default="")
    parser.add_argument("--per_session_stats_dir", type=str, default="", help="Directory containing per-session imu_stats.json files (e.g., <session>_imu_stats.json)")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    parser.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None, help="FFT low-pass cutoff for IMU windows in Hz; set <= 0 to disable.")
    parser.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0, help="Sampling rate used by the IMU low-pass filter.")
    parser.add_argument("--smooth_window", type=int, default=0, help="Temporal smoothing window size for frame assignments (0 = disabled)")
    parser.add_argument("--group_windows", type=int, default=1, help="Aggregate this many consecutive windows into one matching decision (1 = disabled)")
    # Global motion options (must match training config)
    parser.add_argument("--use_global_motion", action="store_true")
    parser.add_argument("--global_motion_input_dim", type=int, default=2)
    parser.add_argument("--global_motion_hidden_dim", type=int, default=64)
    parser.add_argument("--global_motion_num_layers", type=int, default=2)
    parser.add_argument("--global_motion_dropout", type=float, default=0.1)
    parser.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    parser.add_argument("--global_motion_fusion_type", type=str, default="concat")
    parser.add_argument("--global_motion_fusion_proj", action="store_true")
    parser.add_argument("--global_motion_root_source", type=str, default="auto")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_unique_sequences(rows: List[Dict[str, str]], root_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (sequence_id, npz_path) for test split sequences."""
    seen = set()
    seqs = []
    for row in rows:
        if row.get("split", "") != "test":
            continue
        npz_rel = row["npz_path"]
        npz_path = (root_dir / npz_rel).resolve()
        key = str(npz_path)
        if key in seen:
            continue
        seen.add(key)
        data = np.load(npz_path, allow_pickle=True)
        sequence_id = str(data["sequence_id"].item())
        seqs.append((sequence_id, npz_path))
    return seqs


def _batch_infer_embeddings(encoder, windows: List[np.ndarray], device: torch.device, batch_size: int = 64, root_windows: List[np.ndarray] | None = None) -> np.ndarray:
    """Run encoder on a list of window tensors and return embeddings."""
    if not windows:
        return np.zeros((0,), dtype=np.float32)
    all_emb: List[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(np.stack(windows[i : i + batch_size], axis=0)).to(device)
        with torch.no_grad():
            if root_windows is not None:
                root_batch = torch.from_numpy(np.stack(root_windows[i : i + batch_size], axis=0)).to(device)
                emb = encoder(batch, root_batch).cpu().numpy()
            else:
                emb = encoder(batch).cpu().numpy()
        all_emb.append(emb)
    return np.concatenate(all_emb, axis=0)


def _build_root_traj(
    data: Dict[str, np.ndarray],
    start: int,
    end: int,
    person_idx: int,
    root_source: str,
) -> np.ndarray:
    """Build global trajectory features for one person/window."""
    src = (root_source or "auto").lower()

    if src in ("root_3d", "auto") and "gt_skeleton_meters" in data:
        skel = data["gt_skeleton_meters"][start:end, person_idx]
        if skel.ndim == 3 and skel.shape[-1] >= 3:
            return skel[:, 0, :3].astype(np.float32)

    if "extract_bboxes" not in data:
        raise ValueError("extract_bboxes not found in data for global trajectory extraction")

    bboxes = data["extract_bboxes"][start:end, person_idx]
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

    if src in ("bbox_full", "full"):
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        return np.stack([cx, cy, w, h], axis=-1).astype(np.float32)

    if src in ("bbox_vel", "vel"):
        vx = np.zeros_like(cx)
        vy = np.zeros_like(cy)
        vx[1:] = cx[1:] - cx[:-1]
        vy[1:] = cy[1:] - cy[:-1]
        return np.stack([cx, cy, vx, vy], axis=-1).astype(np.float32)

    # default: bbox_center / auto fallback
    return np.stack([cx, cy], axis=-1).astype(np.float32)


def evaluate_sequence(
    model,
    data: Dict[str, np.ndarray],
    window_size: int,
    stride: int,
    device: torch.device,
    batch_size: int = 64,
    root_source: str = "auto",
    imu_lowpass_cutoff_hz: float | None = None,
    imu_lowpass_fs_hz: float = 30.0,
    group_windows: int = 1,
) -> np.ndarray:
    """Run synchronous Hungarian matching on a full sequence.

    For each chunk [start, end), we compute embeddings for all visible
    tracks and IMUs, run Hungarian matching, and assign the result to the
    center frame of the chunk. With stride=1, every frame receives exactly
    one assignment (from the chunk whose center is closest to it).

    Embeddings are computed in batches for efficiency.

    Returns:
        frame_assignments: [T, N_imu] array mapping each IMU to an extract track index per frame.
    """
    T = int(data["frame_ids"].shape[0])
    N_imu = int(data["imu_ids"].shape[0])
    N_pred = int(data["extract_person_ids"].shape[0])

    # ------------------------------------------------------------------
    # 1. Collect all windows that need inference
    # ------------------------------------------------------------------
    window_meta: List[Tuple[int, int, np.ndarray, List[int] | None, List[int] | None]] = []
    imu_windows: List[np.ndarray] = []
    skel_windows: List[np.ndarray] = []
    imu_index_map: Dict[Tuple[int, int], int] = {}
    skel_index_map: Dict[Tuple[int, int], int] = {}

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        vis_pred = data["extract_visibility"][start:end].any(axis=0)
        active_pred = np.where(vis_pred)[0]

        if len(active_pred) == 0 or N_imu == 0:
            window_meta.append((start, end, active_pred, None, None))
            continue

        w_idx = len(window_meta)
        imu_ids_for_win: List[int] = []
        skel_ids_for_win: List[int] = []

        for i in range(N_imu):
            imu_index_map[(w_idx, i)] = len(imu_windows)
            imu_window = data["imu"][start:end, i].astype(np.float32)
            if imu_lowpass_cutoff_hz is not None and imu_lowpass_cutoff_hz > 0:
                imu_window = lowpass_filter_fft(imu_window, imu_lowpass_cutoff_hz, imu_lowpass_fs_hz)
            imu_windows.append(imu_window)
            imu_ids_for_win.append(i)

        for p in active_pred:
            skel_index_map[(w_idx, int(p))] = len(skel_windows)
            skel_windows.append(data["extract_skeleton"][start:end, p].astype(np.float32))
            skel_ids_for_win.append(int(p))

        window_meta.append((start, end, active_pred, imu_ids_for_win, skel_ids_for_win))

    # ------------------------------------------------------------------
    # 2. Batch inference for all collected windows
    # ------------------------------------------------------------------
    model.eval()
    z_imu_all = _batch_infer_embeddings(model.imu_encoder, imu_windows, device, batch_size)
    
    # Check if video encoder supports global motion (has local_encoder attr)
    use_global = hasattr(model.video_encoder, 'local_encoder')
    if use_global:
        # Extract root trajectory features for each skeleton window
        skel_root_windows: List[np.ndarray] = []
        root_index_map: Dict[Tuple[int, int], int] = {}
        for w_idx, (start, end, active_pred, imu_ids_for_win, skel_ids_for_win) in enumerate(window_meta):
            if skel_ids_for_win is None:
                continue
            for p in skel_ids_for_win:
                root_index_map[(w_idx, int(p))] = len(skel_root_windows)
                root_traj = _build_root_traj(data, start, end, int(p), root_source)
                skel_root_windows.append(root_traj)
        z_vid_all = _batch_infer_embeddings(model.video_encoder, skel_windows, device, batch_size, skel_root_windows)
    else:
        z_vid_all = _batch_infer_embeddings(model.video_encoder, skel_windows, device, batch_size)

    # ------------------------------------------------------------------
    # 3. Hungarian matching (per window or grouped windows)
    # ------------------------------------------------------------------
    group_windows = max(1, group_windows)
    centers: List[int] = []
    center_assignments: List[np.ndarray] = []

    if group_windows == 1:
        for w_idx, (start, end, active_pred, imu_ids_for_win, skel_ids_for_win) in enumerate(window_meta):
            if imu_ids_for_win is None or skel_ids_for_win is None:
                continue

            z_imu = np.stack([z_imu_all[imu_index_map[(w_idx, i)]] for i in imu_ids_for_win], axis=0)
            z_vid = np.stack([z_vid_all[skel_index_map[(w_idx, p)]] for p in skel_ids_for_win], axis=0)

            sim = np.dot(z_imu, z_vid.T)  # [N_active_imu, N_active_pred]
            row_ind, col_ind = linear_sum_assignment(-sim)

            center = (start + end) // 2
            assignment = np.full(N_imu, -1, dtype=np.int64)
            for r, c in zip(row_ind, col_ind):
                assignment[imu_ids_for_win[r]] = skel_ids_for_win[c]
            centers.append(center)
            center_assignments.append(assignment)
    else:
        num_w = len(window_meta)
        for w in range(num_w - group_windows + 1):
            # Union of active tracks across the group
            active_group: set[int] = set()
            for gw in range(w, w + group_windows):
                _, _, active_pred, _, _ = window_meta[gw]
                if active_pred is not None:
                    active_group.update(active_pred.tolist())
            if not active_group:
                continue
            active_group = sorted(active_group)

            # Average IMU embeddings across the group
            z_imu_list = []
            for i in range(N_imu):
                embs = [z_imu_all[imu_index_map[(gw, i)]]
                        for gw in range(w, w + group_windows)
                        if (gw, i) in imu_index_map]
                z_imu_list.append(np.mean(embs, axis=0) if embs else np.zeros_like(z_imu_all[0]))
            z_imu = np.stack(z_imu_list, axis=0)

            # Average video embeddings for each active track (only over windows where it is active)
            z_vid_list = []
            for p in active_group:
                embs = [z_vid_all[skel_index_map[(gw, p)]]
                        for gw in range(w, w + group_windows)
                        if (gw, p) in skel_index_map]
                if embs:
                    z_vid_list.append(np.mean(embs, axis=0))
            if not z_vid_list:
                continue
            z_vid = np.stack(z_vid_list, axis=0)

            sim = np.dot(z_imu, z_vid.T)
            row_ind, col_ind = linear_sum_assignment(-sim)

            start0 = window_meta[w][0]
            center = start0 + (group_windows - 1) * stride // 2 + window_size // 2
            assignment = np.full(N_imu, -1, dtype=np.int64)
            for r, c in zip(row_ind, col_ind):
                assignment[r] = active_group[c]
            centers.append(center)
            center_assignments.append(assignment)

    # ------------------------------------------------------------------
    # 4. Assign each frame to the nearest chunk center
    # ------------------------------------------------------------------
    frame_assignments = np.full((T, N_imu), -1, dtype=np.int64)
    for t in range(T):
        for i in range(N_imu):
            best_dist = float("inf")
            best_pred = -1
            for center, assign in zip(centers, center_assignments):
                dist = abs(t - center)
                if dist < best_dist and assign[i] != -1:
                    best_dist = dist
                    best_pred = assign[i]
            frame_assignments[t, i] = best_pred

    return frame_assignments


def smooth_frame_assignments(frame_assignments: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply sliding-window majority vote smoothing per IMU.
    
    Args:
        frame_assignments: [T, N_imu] array
        window: smoothing window size (must be odd)
    
    Returns:
        smoothed [T, N_imu] array
    """
    if window <= 1:
        return frame_assignments
    T, N_imu = frame_assignments.shape
    smoothed = frame_assignments.copy()
    half = window // 2
    
    for i in range(N_imu):
        for t in range(T):
            start = max(0, t - half)
            end = min(T, t + half + 1)
            vals = frame_assignments[start:end, i]
            valid = vals[vals != -1]
            if len(valid) > 0:
                # Majority vote
                from collections import Counter
                c = Counter(valid)
                smoothed[t, i] = c.most_common(1)[0][0]
            else:
                smoothed[t, i] = -1
    return smoothed


def build_hota_data(
    data: Dict[str, np.ndarray],
    frame_assignments: np.ndarray,
) -> Dict[str, object]:
    """Build trackeval HOTA input from frame assignments.

    GT ids are mapped to contiguous indices 0..N_gt-1.
    Tracker ids are IMU indices 0..N_imu-1.
    """
    T = int(data["frame_ids"].shape[0])
    N_gt = int(data["gt_person_ids"].shape[0])
    N_imu = int(data["imu_ids"].shape[0])

    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []

    for t in range(T):
        gt_present = []
        for g in range(N_gt):
            if data["gt_visibility"][t, g]:
                gt_present.append(g)

        tracker_present = []
        for i in range(N_imu):
            if frame_assignments[t, i] != -1:
                tracker_present.append(i)

        gt_ids_list.append(np.array(gt_present, dtype=np.int64))
        tracker_ids_list.append(np.array(tracker_present, dtype=np.int64))

        sim = np.zeros((len(gt_present), len(tracker_present)), dtype=np.float32)
        for gi, g in enumerate(gt_present):
            p_gt = data["gt_to_extract_map"][t, g]
            for ti, i in enumerate(tracker_present):
                p_pred = frame_assignments[t, i]
                if p_gt != -1 and p_pred != -1 and p_gt == p_pred:
                    sim[gi, ti] = 1.0
        similarity_scores_list.append(sim)

    num_gt_dets = sum(len(x) for x in gt_ids_list)
    num_tracker_dets = sum(len(x) for x in tracker_ids_list)

    return {
        "num_tracker_dets": num_tracker_dets,
        "num_gt_dets": num_gt_dets,
        "num_gt_ids": N_gt,
        "num_tracker_ids": N_imu,
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_scores_list,
    }


def compute_frame_matching_accuracy(
    data: Dict[str, np.ndarray],
    frame_assignments: np.ndarray,
) -> float:
    """Compute per-frame IMU-to-person matching accuracy against GT.

    For each visible GT person, we find the extract track mapped to them via
    gt_to_extract_map. If the Hungarian assignment sends the corresponding
    IMU (matched by imu_id == gt_person_id) to that same extract track, the
    frame is counted as correct.
    """
    T = int(data["frame_ids"].shape[0])
    N_gt = int(data["gt_person_ids"].shape[0])
    gt_to_extract_map = data["gt_to_extract_map"]
    gt_visibility = data["gt_visibility"]
    gt_person_ids = data["gt_person_ids"]
    imu_ids = data["imu_ids"]

    correct = 0
    total = 0
    for t in range(T):
        for g in range(N_gt):
            if not gt_visibility[t, g]:
                continue
            p_gt = gt_to_extract_map[t, g]
            if p_gt == -1:
                continue
            total += 1
            matched_imus = np.where(frame_assignments[t] == p_gt)[0]
            if len(matched_imus) == 1:
                i = int(matched_imus[0])
                if imu_ids[i] == gt_person_ids[g]:
                    correct += 1
    return float(correct / total) if total > 0 else 0.0


def write_mot_format(path: Path, data: Dict[str, np.ndarray], frame_assignments: np.ndarray | None) -> None:
    """Write MOT challenge format txt file.

    If frame_assignments is None, write GT; otherwise write predictions.
    """
    T = int(data["frame_ids"].shape[0])
    rows = []
    if frame_assignments is None:
        N_gt = int(data["gt_person_ids"].shape[0])
        for t in range(T):
            for g in range(N_gt):
                if not data["gt_visibility"][t, g]:
                    continue
                bbox = data["gt_bboxes"][t, g]
                tid = int(data["gt_person_ids"][g])
                x, y, x2, y2 = bbox
                w, h = max(0.0, x2 - x), max(0.0, y2 - y)
                rows.append(f"{t+1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    else:
        N_imu = int(data["imu_ids"].shape[0])
        for t in range(T):
            for i in range(N_imu):
                pred_track = frame_assignments[t, i]
                if pred_track == -1:
                    continue
                # Find GT person mapped to this pred_track
                for g in range(data["gt_person_ids"].shape[0]):
                    if not data["gt_visibility"][t, g]:
                        continue
                    if data["gt_to_extract_map"][t, g] == pred_track:
                        bbox = data["gt_bboxes"][t, g]
                        tid = int(data["imu_ids"][i])
                        x, y, x2, y2 = bbox
                        w, h = max(0.0, x2 - x), max(0.0, y2 - y)
                        rows.append(f"{t+1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
                        break

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, _ = build_alignment_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    root_dir = Path(args.data_root) if args.data_root else Path(args.test_csv).parent
    rows = _read_csv_rows(Path(args.test_csv))
    sequences = _extract_unique_sequences(rows, root_dir)

    if not sequences:
        print("No test sequences found.")
        return

    metric = HOTA()
    all_results = []

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)

    per_session_stats = {}
    if args.per_session_stats_dir:
        psd = Path(args.per_session_stats_dir)
        for p in psd.glob("*_imu_stats.json"):
            session_id = p.stem.replace("_imu_stats", "")
            stats = json.loads(p.read_text())
            per_session_stats[session_id] = (
                np.asarray(stats["imu_mean"], dtype=np.float32),
                np.asarray(stats["imu_std"], dtype=np.float32),
            )
        print(f"[INFO] Loaded per-session stats for {len(per_session_stats)} sessions: {sorted(per_session_stats.keys())}")

    for seq_id, npz_path in sequences:
        data = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
        print(f"Evaluating {seq_id} ...")

        if "extract_person_ids" not in data or data["extract_person_ids"].shape[0] == 0:
            print(f"  Skipping: no extracted skeletons.")
            continue

        # Apply IMU normalization: per-session takes precedence over global
        session_key = seq_id.replace("custom_", "")
        if session_key in per_session_stats:
            sess_mean, sess_std = per_session_stats[session_key]
            data["imu"] = (data["imu"] - sess_mean) / np.maximum(sess_std, 1e-6)
            print(f"  [Per-session norm] Using stats for session {session_key}")
        elif imu_mean is not None and imu_std is not None:
            data["imu"] = (data["imu"] - imu_mean) / np.maximum(imu_std, 1e-6)

        frame_assignments = evaluate_sequence(
            model, data, args.window_size, args.stride, device, args.batch_size, args.global_motion_root_source,
            args.imu_lowpass_cutoff_hz, args.imu_lowpass_fs_hz, args.group_windows
        )
        
        if args.smooth_window > 0:
            frame_assignments = smooth_frame_assignments(frame_assignments, window=args.smooth_window)

        hota_data = build_hota_data(data, frame_assignments)
        res = metric.eval_sequence(hota_data)
        frame_acc = compute_frame_matching_accuracy(data, frame_assignments)

        alpha_idx = len(metric.array_labels) // 2
        seq_result = {
            "sequence_id": seq_id,
            "HOTA": float(res["HOTA"][alpha_idx]),
            "AssA": float(res["AssA"][alpha_idx]),
            "AssRe": float(res["AssRe"][alpha_idx]),
            "AssPr": float(res["AssPr"][alpha_idx]),
            "DetA": float(res["DetA"][alpha_idx]),
            "DetRe": float(res["DetRe"][alpha_idx]),
            "DetPr": float(res["DetPr"][alpha_idx]),
            "LocA": float(res["LocA"][alpha_idx]),
            "frame_matching_accuracy": frame_acc,
        }
        all_results.append(seq_result)
        print(
            f"  HOTA={seq_result['HOTA']:.3f} AssA={seq_result['AssA']:.3f} "
            f"DetA={seq_result['DetA']:.3f} FrameAcc={seq_result['frame_matching_accuracy']:.3f}"
        )

        if args.save_mot_pred:
            pred_path = Path(args.save_mot_pred) / f"{seq_id}_pred.txt"
            write_mot_format(pred_path, data, frame_assignments)
        if args.save_mot_gt:
            gt_path = Path(args.save_mot_gt) / f"{seq_id}_gt.txt"
            write_mot_format(gt_path, data, None)

    summary = {
        "num_sequences": len(all_results),
        "sequences": all_results,
    }
    if all_results:
        for key in ["HOTA", "AssA", "AssRe", "AssPr", "DetA", "DetRe", "DetPr", "LocA", "frame_matching_accuracy"]:
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in all_results]))

    print(json.dumps(summary, indent=2))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
