#!/usr/bin/env python3
"""Intra-sequence windowed IMU-video matching evaluation (v3).

Provides multiple metrics that are more generous than exact Hungarian accuracy
while preserving relative model rankings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.engine.common import build_alignment_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Windowed intra evaluation v3 with generous metrics")
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    p.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    p.add_argument("--motionbert_ckpt", type=str, default="")
    p.add_argument("--skip_motionbert_ckpt", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--imu_stats_json", type=str, default="")
    p.add_argument("--imu_sensor", type=str, default="R_LowArm")
    p.add_argument("--repeat_single_sensor", type=int, default=4)
    p.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None)
    p.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0)
    p.add_argument("--num_trials_per_seq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(sim_matrix: np.ndarray, gt_mapping: Dict[int, int]) -> Dict[str, float]:
    """Compute multiple matching metrics from similarity matrix.
    
    gt_mapping: imu_idx -> person_idx (ground truth)
    """
    n = sim_matrix.shape[0]
    imu_indices = sorted(gt_mapping.keys())
    person_indices = sorted(set(gt_mapping.values()))
    
    # Create index mappings
    imu_to_row = {idx: i for i, idx in enumerate(imu_indices)}
    person_to_col = {idx: j for j, idx in enumerate(person_indices)}
    
    # 1. Hungarian Exact Match
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    hungarian_correct = 0
    for r, c in zip(row_ind, col_ind):
        imu_idx = imu_indices[r]
        person_idx = person_indices[c]
        if gt_mapping.get(imu_idx) == person_idx:
            hungarian_correct += 1
    hungarian_acc = hungarian_correct / n
    
    # 2. Top-K Accuracy (per IMU)
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    mrr_sum = 0.0
    
    for imu_idx in imu_indices:
        r = imu_to_row[imu_idx]
        gt_person = gt_mapping[imu_idx]
        gt_col = person_to_col[gt_person]
        
        # Rank persons by similarity for this IMU
        ranked = np.argsort(-sim_matrix[r])
        rank_of_gt = np.where(ranked == gt_col)[0][0] + 1  # 1-based rank
        
        if rank_of_gt == 1:
            top1_correct += 1
        if rank_of_gt <= 2:
            top2_correct += 1
        if rank_of_gt <= 3:
            top3_correct += 1
        
        mrr_sum += 1.0 / rank_of_gt
    
    top1_acc = top1_correct / n
    top2_acc = top2_correct / n
    top3_acc = top3_correct / n
    mrr = mrr_sum / n
    
    # 3. Normalized Accuracy (relative to random baseline)
    # Random baseline for N-person Hungarian: 1/N!
    random_baseline = 1.0 / math.factorial(n)
    normalized_acc = max(0.0, (hungarian_acc - random_baseline) / (1.0 - random_baseline))
    
    # 4. Pairwise AUC-like metric
    # For each correct pair (imu_i, person_i), compare to all incorrect pairs
    correct_pairs = []
    incorrect_pairs = []
    for imu_idx in imu_indices:
        r = imu_to_row[imu_idx]
        gt_person = gt_mapping[imu_idx]
        gt_col = person_to_col[gt_person]
        correct_sim = sim_matrix[r, gt_col]
        correct_pairs.append(correct_sim)
        
        for person_idx in person_indices:
            c = person_to_col[person_idx]
            if person_idx != gt_person:
                incorrect_pairs.append(sim_matrix[r, c])
    
    if correct_pairs and incorrect_pairs:
        correct_pairs = np.array(correct_pairs)
        incorrect_pairs = np.array(incorrect_pairs)
        # Fraction of correct pairs that have higher sim than incorrect pairs
        pairwise_correct = 0
        total_pairs = 0
        for cs in correct_pairs:
            for ic in incorrect_pairs:
                total_pairs += 1
                if cs > ic:
                    pairwise_correct += 1
                elif cs == ic:
                    pairwise_correct += 0.5
        pairwise_auc = pairwise_correct / total_pairs if total_pairs > 0 else 0.0
    else:
        pairwise_auc = 0.0
    
    return {
        "hungarian": hungarian_acc,
        "top1": top1_acc,
        "top2": top2_acc,
        "top3": top3_acc,
        "mrr": mrr,
        "normalized": normalized_acc,
        "pairwise_auc": pairwise_auc,
        "random_baseline": random_baseline,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)

    return_root = getattr(args, "use_global_motion", False)
    root_source = getattr(args, "global_motion_root_source", "auto")
    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=args.imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
        return_root_trajectory=return_root,
        root_source=root_source,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    with open(args.test_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    model, _ = build_alignment_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print(f"Computing embeddings for {len(ds)} windows...")
    imu_all = []
    vid_all = []
    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(device)
            skel = batch["skeleton"].to(device)
            forward_kwargs = {"imu": imu, "skeleton": skel}
            if "root_trajectory" in batch:
                forward_kwargs["root_trajectory"] = batch["root_trajectory"].to(device)
            out = model(**forward_kwargs)
            imu_all.append(out["imu"].detach().cpu().numpy())
            vid_all.append(out["video"].detach().cpu().numpy())

    imu_all = np.concatenate(imu_all, axis=0)
    vid_all = np.concatenate(vid_all, axis=0)

    # Group by (session, window_start)
    window_map: Dict[Tuple[str, str], Dict] = {}
    for i, row in enumerate(rows):
        key = (row["session"], row["window_start"])
        if key not in window_map:
            window_map[key] = {
                "session": row["session"],
                "person_embs": {},
            }
        person_idx = int(row["person_idx"])
        imu_idx = int(row["imu_idx"])
        if person_idx not in window_map[key]["person_embs"]:
            window_map[key]["person_embs"][person_idx] = []
        window_map[key]["person_embs"][person_idx].append({
            "imu_idx": imu_idx,
            "imu_emb": imu_all[i],
            "vid_emb": vid_all[i],
        })

    seq_windows: Dict[str, List[Dict]] = {}
    for key, win_data in window_map.items():
        session = key[0]
        if session not in seq_windows:
            seq_windows[session] = []
        seq_windows[session].append(win_data)

    rng = np.random.default_rng(args.seed)

    all_results = []
    for session, windows in sorted(seq_windows.items()):
        n_windows = len(windows)
        if n_windows == 0:
            continue

        first_window = windows[0]
        n_persons = len(first_window["person_embs"])
        
        # Ground truth mapping: imu_idx == person_idx
        gt_mapping = {i: i for i in range(n_persons)}

        trial_metrics = {k: [] for k in ["hungarian", "top1", "top2", "top3", "mrr", "normalized", "pairwise_auc"]}

        for trial in range(args.num_trials_per_seq):
            win_idx = rng.integers(0, n_windows)
            win = windows[win_idx]

            # Extract embeddings
            vid_embs_per_person = {}
            imu_embs_per_imu = {}
            for person_idx, entries in win["person_embs"].items():
                vid_embs_per_person[person_idx] = entries[0]["vid_emb"]
                for entry in entries:
                    imu_idx = entry["imu_idx"]
                    if imu_idx not in imu_embs_per_imu:
                        imu_embs_per_imu[imu_idx] = entry["imu_emb"]

            sorted_imu_keys = sorted(imu_embs_per_imu.keys())
            sorted_person_keys = sorted(vid_embs_per_person.keys())
            n = len(sorted_imu_keys)

            sim_matrix = np.zeros((n, n), dtype=np.float32)
            for i, imu_idx in enumerate(sorted_imu_keys):
                for j, person_idx in enumerate(sorted_person_keys):
                    imu_emb = imu_embs_per_imu[imu_idx]
                    vid_emb = vid_embs_per_person[person_idx]
                    sim = np.dot(imu_emb, vid_emb) / (np.linalg.norm(imu_emb) * np.linalg.norm(vid_emb) + 1e-8)
                    sim_matrix[i, j] = sim

            metrics = compute_metrics(sim_matrix, gt_mapping)
            for k, v in metrics.items():
                if k != "random_baseline":
                    trial_metrics[k].append(v)

        result = {
            "sequence_id": session,
            "n_persons": n_persons,
            "n_windows": n_windows,
        }
        for k, vals in trial_metrics.items():
            result[k] = round(float(np.mean(vals)), 4)
        
        all_results.append(result)

    # Aggregate
    agg = {}
    for k in trial_metrics.keys():
        agg[k] = round(float(np.mean([r[k] for r in all_results])), 4)

    print(f"\n{'='*90}")
    print(f"{'Metric':<25} {'Value':<10} {'vs Hungarian':<15} {'Description'}")
    print(f"{'-'*90}")
    
    descriptions = {
        "hungarian": "Exact Hungarian match",
        "top1": "Each IMU's top-1 choice is correct",
        "top2": "Correct person in IMU's top-2 choices",
        "top3": "Correct person in IMU's top-3 choices",
        "mrr": "Mean Reciprocal Rank (1/rank)",
        "normalized": "(acc - random) / (1 - random)",
        "pairwise_auc": "Correct pair sim > incorrect pair sim",
    }
    
    for k in ["hungarian", "top1", "top2", "top3", "mrr", "normalized", "pairwise_auc"]:
        v = agg[k]
        vs = "baseline" if k == "hungarian" else f"{(v/agg['hungarian'] - 1)*100:+.1f}%"
        print(f"{descriptions[k]:<25} {v:<10.4f} {vs:<15} ")
    
    print(f"{'='*90}")
    
    print(f"\nPer-sequence breakdown:")
    print(f"{'Sequence':<50} {'Persons':<8} " + " ".join(f"{k:<10}" for k in ["hungarian", "top2", "mrr", "pairwise_auc"]))
    print(f"{'-'*100}")
    for r in all_results:
        seq_short = r['sequence_id'][:45]
        vals = " ".join(f"{r[k]:<10.2%}" for k in ["hungarian", "top2", "mrr", "pairwise_auc"])
        print(f"{seq_short:<50} {r['n_persons']:<8} {vals}")


if __name__ == "__main__":
    main()
