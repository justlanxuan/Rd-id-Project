#!/usr/bin/env python3
"""
Experiment Note: A1-eval-mobind-frameacc
Evaluate the official MoBInd stage2 checkpoint on our synchronous-style
per-frame IMU-to-person identification task (FrameAcc).

This script uses the same extracted skeleton tracks and gt_to_extract_map as
our pipeline, but replaces the embedding model with MoBInd.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Add MoBInd repo to path so we can import its modules
MOBIND_ROOT = Path("/home/fzliang/MoBind")
sys.path.insert(0, str(MOBIND_ROOT))

from builder import build_model  # noqa: E402
from configs.config import DATASET_CONFIG  # noqa: E402
from datasets.utils import extract_limb_motion, normalize_pose  # noqa: E402

def compute_frame_matching_accuracy(data, frame_assignments):
    """Local copy of the FrameAcc helper from src/engine/eval_synchronous.py."""
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


# MoBInd EgoHumans official test/val sequences (from configs/config.py)
MOBIND_TEST_SEQS = {
    ("01", "001"), ("01", "002"), ("01", "003"), ("01", "004"),
    ("03", "001"), ("03", "002"), ("03", "003"), ("03", "004"),
    ("04", "001"), ("04", "002"), ("04", "003"), ("04", "004"),
    ("05", "001"), ("05", "002"), ("05", "003"), ("05", "004"),
    ("06", "001"), ("06", "002"), ("06", "003"), ("06", "004"),
    ("07", "001"), ("07", "002"), ("07", "003"), ("07", "004"),
}
MOBIND_VAL_SEQS = {
    ("01", "005"), ("03", "005"), ("04", "005"),
    ("05", "005"), ("06", "005"), ("07", "005"),
}

# Our pipeline's test sequences that are NOT in MoBInd official test/val
SELECTED_SEQS = {
    "01_011", "02_001", "03_009", "04_011", "05_007",
    "06_024", "06_040", "06_041", "06_054", "06_019",
    "06_036", "06_006", "06_025", "06_060", "07_011", "07_007",
}


def parse_npz_sequence_id(seq_id: str):
    """custom_01_002 -> ('01', '002')."""
    parts = seq_id.split("_")
    return parts[1], parts[2]


def load_skeleton_json(path: Path, tlen: int):
    """Load AlphaPose-style skeleton.json and build (T, N_tracks, 17, 2) array."""
    with open(path, "r") as f:
        entries = json.load(f)

    # Discover track ids
    track_ids = sorted(set(e["idx"] for e in entries))
    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    n_tracks = len(track_ids)

    pose2d = np.zeros((tlen, n_tracks, 17, 2), dtype=np.float32)
    visibility = np.zeros((tlen, n_tracks), dtype=bool)

    for e in entries:
        frame = int(e["image_id"].split(".")[0])
        if frame < 0 or frame >= tlen:
            continue
        tid = e["idx"]
        kpts = np.array(e["keypoints"], dtype=np.float32).reshape(17, 3)
        pose2d[frame, tid_to_idx[tid], :, :2] = kpts[:, :2]
        visibility[frame, tid_to_idx[tid]] = (kpts[:, 2] > 0).any()

    return pose2d, visibility, track_ids


def build_mobind_motion_batch(pose2d_window: np.ndarray, limb_list, image_size):
    """
    Args:
        pose2d_window: (T, N_tracks, 17, 2) pixel coords, last dim = (x,y)
    Returns:
        motion_input: (N_active, S, 6, T)
    """
    T, P, J, C = pose2d_window.shape
    # Convert to (T, P, 2, 17) so normalize_pose and extract_limb_motion work
    pose2d_perm = pose2d_window.transpose(0, 1, 3, 2)
    pose2d_norm = normalize_pose(pose2d_perm, "EgoHumans")

    # Extract limb motion per limb -> (T, P, 2, J_limb)
    limb_tensors = []
    for limb in limb_list:
        lm = extract_limb_motion(pose2d_norm, "EgoHumans", limb)
        limb_tensors.append(lm)
    # Stack -> (S, T, P, 2, J_limb)
    motion = np.stack(limb_tensors, axis=0)
    S = motion.shape[0]
    # Permute to (P, S, 2, J_limb, T) then reshape to (P, S, 6, T)
    motion = motion.transpose(2, 0, 3, 4, 1)
    motion = motion.reshape(P, S, -1, T)
    return torch.from_numpy(motion).float()


def build_mobind_imu_batch(imu_window: np.ndarray):
    """
    Args:
        imu_window: (T, N_imu, 5, 7)
    Returns:
        imu_input: (N_imu, 5, 7, T)
    """
    # (T, P, N, C) -> (P, N, C, T)
    imu = imu_window.transpose(1, 2, 3, 0)
    return torch.from_numpy(imu).float()


def evaluate_sequence_mobind(model, npz_data, pose2d, imu_raw,
                             limb_list, image_size, window_size, stride, device):
    """Return frame_assignments (T, N_imu) for one sequence."""
    T = int(npz_data["frame_ids"].shape[0])
    N_imu = int(npz_data["imu_ids"].shape[0])
    extract_visibility = npz_data["extract_visibility"]

    # Make sure all inputs are aligned to T
    pose2d = pose2d[:T]
    imu_raw = imu_raw[:T]

    centers = []
    center_assignments = []

    num_windows = 0
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        num_windows += 1
        active_pred = np.where(extract_visibility[start:end].any(axis=0))[0]
        if active_pred.size == 0:
            continue

        imu_input = build_mobind_imu_batch(imu_window=imu_raw[start:end]).to(device)
        motion_input = build_mobind_motion_batch(
            pose2d_window=pose2d[start:end, active_pred],
            limb_list=limb_list,
            image_size=image_size,
        ).to(device)

        with torch.no_grad():
            out = model(
                {"imu": imu_input, "motion": motion_input},
                global_weight=1.0,
                local_weight=0.0,
            )
            z_imu = out["cls_i"]  # (N_imu, D)
            z_vid = out["cls_m"]  # (N_active, D)

        sim = (z_imu @ z_vid.t()).cpu().numpy()  # (N_imu, N_active)
        row_ind, col_ind = linear_sum_assignment(-sim)

        # assignment[i] = track id or -1
        assignment = np.full(N_imu, -1, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            assignment[r] = int(active_pred[c])

        center = (start + end) // 2
        centers.append(center)
        center_assignments.append(assignment)

    if not centers:
        return np.full((T, N_imu), -1, dtype=np.int64)

    centers = np.array(centers)
    # Build per-frame assignments by nearest center
    frame_assignments = np.full((T, N_imu), -1, dtype=np.int64)
    for t in range(T):
        # nearest center with a valid assignment for each IMU
        for i in range(N_imu):
            valid = [idx for idx, a in enumerate(center_assignments) if a[i] != -1]
            if not valid:
                continue
            best_idx = min(valid, key=lambda idx: abs(centers[idx] - t))
            frame_assignments[t, i] = center_assignments[best_idx][i]

    return frame_assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", default="/home/fzliang/Autism-project/data/interim/egohumans_full_extract/slice/sequences")
    parser.add_argument("--mobind_extracted_root", default="/data/lyxie/ReID/Data/egohumans/extracted_data")
    parser.add_argument("--mobind_exp_dir", default="/home/fzliang/MoBind/checkpoints/EgoHumans/stage2_repro")
    parser.add_argument("--window_size", type=int, default=100, help="Frames per MoBInd window (5s at 20fps)")
    parser.add_argument("--stride", type=int, default=16, help="Stride between windows in frames")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default="/home/fzliang/Autism-project/experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/results/mobind_frameacc.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MoBInd model
    import omegaconf
    config_path = Path(args.mobind_exp_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(str(config_path))
    # Make stage1 path absolute so build_model can find it regardless of cwd
    config.model.stage1_exp = "/home/fzliang/MoBind/checkpoints/EgoHumans/stage1"
    model = build_model(config)
    ckpt_path = Path(args.mobind_exp_dir) / "checkpoints" / "best.pt"
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(device)
    model.eval()
    print(f"Loaded MoBInd model from {ckpt_path}")

    limb_list = DATASET_CONFIG["EgoHumans"]["limb_list"]
    image_size = DATASET_CONFIG["EgoHumans"]["image_size"]

    npz_root = Path(args.npz_root)
    mobind_extracted_root = Path(args.mobind_extracted_root)

    # Enumerate all NPZ sequences and filter to MoBInd-train-only
    all_npz = sorted(npz_root.glob("custom_*.npz"))
    selected = []
    for p in all_npz:
        seq_id = p.stem
        action, seq = parse_npz_sequence_id(seq_id)
        session = f"{action}_{seq}"
        if session not in SELECTED_SEQS:
            continue
        if (action, seq) in MOBIND_TEST_SEQS or (action, seq) in MOBIND_VAL_SEQS:
            continue
        selected.append((seq_id, action, seq, p))

    print(f"Selected {len(selected)} sequences for evaluation (MoBInd train-only)")

    results = []
    for seq_id, action, seq, npz_path in tqdm(selected, desc="MoBInd FrameAcc"):
        npz_data = np.load(npz_path, allow_pickle=True)

        # Load skeleton.json from extract_source path stored in NPZ
        skeleton_json = Path(npz_data["extract_source"].item())
        if not skeleton_json.exists():
            print(f"[WARN] skeleton.json not found for {seq_id}: {skeleton_json}")
            continue

        T_npz = int(npz_data["frame_ids"].shape[0])
        pose2d, skel_vis, track_ids = load_skeleton_json(skeleton_json, T_npz)

        # Sanity: track ids should match extract_person_ids sorted order
        extract_pids = npz_data["extract_person_ids"]
        if not np.array_equal(np.array(track_ids), extract_pids):
            print(f"[WARN] {seq_id} track_ids {track_ids} != extract_person_ids {extract_pids}")

        # Load raw MoBInd IMU files for this sequence, sorted by aria ID
        imu_files = sorted(mobind_extracted_root.glob(f"{action}_{seq}_aria*.npy"))
        if len(imu_files) != npz_data["imu_ids"].shape[0]:
            print(f"[WARN] {seq_id} IMU file count {len(imu_files)} != imu_ids {npz_data['imu_ids'].shape[0]}")

        raw_imus = [np.load(f, allow_pickle=True).item()["imu"].astype(np.float32) for f in imu_files]
        # Align time lengths
        tlen = min(T_npz, min(arr.shape[0] for arr in raw_imus), pose2d.shape[0])
        raw_imus = [arr[:tlen] for arr in raw_imus]
        imu_raw = np.stack(raw_imus, axis=1)  # (T, N_imu, 5, 7)
        pose2d = pose2d[:tlen]

        # Truncate NPZ metadata to tlen for FrameAcc computation
        npz_trunc = {}
        for k in npz_data.files:
            arr = npz_data[k]
            if (hasattr(arr, "shape") and arr.ndim > 0 and arr.shape[0] == T_npz and
                    k not in ("sequence_id", "dataset", "video_path", "extract_source")):
                npz_trunc[k] = arr[:tlen]
            else:
                npz_trunc[k] = arr

        frame_assignments = evaluate_sequence_mobind(
            model=model,
            npz_data=npz_trunc,
            pose2d=pose2d,
            imu_raw=imu_raw,
            limb_list=limb_list,
            image_size=image_size,
            window_size=args.window_size,
            stride=args.stride,
            device=device,
        )

        frame_acc = compute_frame_matching_accuracy(npz_trunc, frame_assignments)
        results.append({
            "sequence_id": seq_id,
            "frame_acc": float(frame_acc),
            "num_frames": int(tlen),
        })
        print(f"{seq_id}: FrameAcc = {frame_acc:.4f}")

    mean_acc = float(np.mean([r["frame_acc"] for r in results])) if results else 0.0
    summary = {
        "model": "MoBInd_stage2_official",
        "num_sequences": len(results),
        "sequences": results,
        "mean_frame_acc": mean_acc,
        "window_size": args.window_size,
        "stride": args.stride,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMoBInd mean FrameAcc over {len(results)} sequences: {mean_acc:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
