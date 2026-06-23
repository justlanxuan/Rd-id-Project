#!/usr/bin/env python3
"""
Experiment Note: A5-eval-mobind-aligned-test
Evaluate the official MoBInd stage2 checkpoint on its own official test split
(24 sequences), using the same synchronous-style FrameAcc metric.
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

MOBIND_ROOT = Path("/home/fzliang/MoBind")
sys.path.insert(0, str(MOBIND_ROOT))

from builder import build_model  # noqa: E402
from configs.config import DATASET_CONFIG  # noqa: E402
from datasets.utils import extract_limb_motion, normalize_pose  # noqa: E402

# MoBInd official test sequences
SELECTED_SEQS = {
    "01_001", "01_002", "01_003", "01_004",
    "03_001", "03_002", "03_003", "03_004",
    "04_001", "04_002", "04_003", "04_004",
    "05_001", "05_002", "05_003", "05_004",
    "06_001", "06_002", "06_003", "06_004",
    "07_001", "07_002", "07_003", "07_004",
}


def compute_frame_matching_accuracy(data, frame_assignments):
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


def parse_npz_sequence_id(seq_id: str):
    parts = seq_id.split("_")
    return parts[1], parts[2]


def load_skeleton_json(path: Path, tlen: int):
    with open(path, "r") as f:
        entries = json.load(f)
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
    T, P, J, C = pose2d_window.shape
    pose2d_perm = pose2d_window.transpose(0, 1, 3, 2)
    pose2d_norm = normalize_pose(pose2d_perm, "EgoHumans")
    limb_tensors = []
    for limb in limb_list:
        lm = extract_limb_motion(pose2d_norm, "EgoHumans", limb)
        limb_tensors.append(lm)
    motion = np.stack(limb_tensors, axis=0)
    motion = motion.transpose(2, 0, 3, 4, 1)
    motion = motion.reshape(P, motion.shape[1], -1, T)
    return torch.from_numpy(motion).float()


def build_mobind_imu_batch(imu_window: np.ndarray):
    imu = imu_window.transpose(1, 2, 3, 0)
    return torch.from_numpy(imu).float()


def evaluate_sequence_mobind(model, npz_data, pose2d, imu_raw,
                             limb_list, image_size, window_size, stride, device):
    T = int(npz_data["frame_ids"].shape[0])
    N_imu = int(npz_data["imu_ids"].shape[0])
    extract_visibility = npz_data["extract_visibility"]
    pose2d = pose2d[:T]
    imu_raw = imu_raw[:T]
    centers = []
    center_assignments = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        active_pred = np.where(extract_visibility[start:end].any(axis=0))[0]
        if active_pred.size == 0:
            continue
        imu_input = build_mobind_imu_batch(imu_window=imu_raw[start:end]).to(device)
        # Use all tracks for motion so batch size matches IMU; restrict embeddings to active tracks afterwards.
        motion_input = build_mobind_motion_batch(
            pose2d_window=pose2d[start:end],
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
            z_vid_all = out["cls_m"]  # (N_tracks, D)
            z_vid = z_vid_all[active_pred]  # (N_active, D)
        sim = (z_imu @ z_vid.t()).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-sim)
        assignment = np.full(N_imu, -1, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            assignment[r] = int(active_pred[c])
        centers.append((start + end) // 2)
        center_assignments.append(assignment)
    if not centers:
        return np.full((T, N_imu), -1, dtype=np.int64)
    centers = np.array(centers)
    frame_assignments = np.full((T, N_imu), -1, dtype=np.int64)
    for t in range(T):
        for i in range(N_imu):
            valid = [idx for idx, a in enumerate(center_assignments) if a[i] != -1]
            if not valid:
                continue
            best_idx = min(valid, key=lambda idx: abs(centers[idx] - t))
            frame_assignments[t, i] = center_assignments[best_idx][i]
    return frame_assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", default="/home/fzliang/Autism-project/data/interim/egohumans_mobind_aligned/slice/sequences")
    parser.add_argument("--mobind_extracted_root", default="/data/lyxie/ReID/Data/egohumans/extracted_data")
    parser.add_argument("--mobind_exp_dir", default="/home/fzliang/MoBind/checkpoints/EgoHumans/stage2_repro")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", default="/home/fzliang/Autism-project/experiments/G_egohumans/E5:mobind_aligned_splits/results/mobind_frameacc_aligned_test.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    import omegaconf
    config_path = Path(args.mobind_exp_dir) / "config.yaml"
    config = omegaconf.OmegaConf.load(str(config_path))
    config.model.stage1_exp = "/home/fzliang/MoBind/checkpoints/EgoHumans/stage1"
    model = build_model(config)
    ckpt_path = Path(args.mobind_exp_dir) / "checkpoints" / "best.pt"
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(device)
    model.eval()

    limb_list = DATASET_CONFIG["EgoHumans"]["limb_list"]
    image_size = DATASET_CONFIG["EgoHumans"]["image_size"]
    npz_root = Path(args.npz_root)
    mobind_extracted_root = Path(args.mobind_extracted_root)

    all_npz = sorted(npz_root.glob("custom_*.npz"))
    selected = []
    for p in all_npz:
        seq_id = p.stem
        action, seq = parse_npz_sequence_id(seq_id)
        session = f"{action}_{seq}"
        if session in SELECTED_SEQS:
            selected.append((seq_id, action, seq, p))

    print(f"Selected {len(selected)} MoBInd test sequences")
    results = []
    for seq_id, action, seq, npz_path in tqdm(selected, desc="MoBInd FrameAcc"):
        npz_data = np.load(npz_path, allow_pickle=True)
        skeleton_json = Path(npz_data["extract_source"].item())
        if not skeleton_json.exists():
            print(f"[WARN] skeleton.json not found for {seq_id}: {skeleton_json}")
            continue
        T_npz = int(npz_data["frame_ids"].shape[0])
        pose2d, skel_vis, track_ids = load_skeleton_json(skeleton_json, T_npz)
        imu_files = sorted(mobind_extracted_root.glob(f"{action}_{seq}_aria*.npy"))
        if len(imu_files) != npz_data["imu_ids"].shape[0]:
            print(f"[WARN] {seq_id} IMU file count {len(imu_files)} != imu_ids {npz_data['imu_ids'].shape[0]}")
        raw_imus = [np.load(f, allow_pickle=True).item()["imu"].astype(np.float32) for f in imu_files]
        tlen = min(T_npz, min(arr.shape[0] for arr in raw_imus), pose2d.shape[0])
        raw_imus = [arr[:tlen] for arr in raw_imus]
        imu_raw = np.stack(raw_imus, axis=1)
        pose2d = pose2d[:tlen]
        npz_trunc = {}
        for k in npz_data.files:
            arr = npz_data[k]
            if (hasattr(arr, "shape") and arr.ndim > 0 and arr.shape[0] == T_npz and
                    k not in ("sequence_id", "dataset", "video_path", "extract_source")):
                npz_trunc[k] = arr[:tlen]
            else:
                npz_trunc[k] = arr
        frame_assignments = evaluate_sequence_mobind(
            model=model, npz_data=npz_trunc, pose2d=pose2d, imu_raw=imu_raw,
            limb_list=limb_list, image_size=image_size,
            window_size=args.window_size, stride=args.stride, device=device,
        )
        frame_acc = compute_frame_matching_accuracy(npz_trunc, frame_assignments)
        results.append({"sequence_id": seq_id, "frame_acc": float(frame_acc), "num_frames": int(tlen)})
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
