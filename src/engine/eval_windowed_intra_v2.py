#!/usr/bin/env python3
"""Intra-sequence windowed IMU-video matching evaluation (v2).

Compares Hungarian matching vs greedy (independent) matching.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.engine.common import build_alignment_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intra-sequence windowed matching evaluation v2")
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
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def greedy_matching(sim_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy matching: each IMU picks best available person in order.
    
    Strategy: process IMUs in order 0, 1, 2, ...
    Each IMU picks the highest-similarity person not yet taken.
    """
    n = sim_matrix.shape[0]
    used_persons = set()
    matches = []
    for i in range(n):
        prefs = np.argsort(-sim_matrix[i])
        for j in prefs:
            if j not in used_persons:
                matches.append((i, j))
                used_persons.add(j)
                break
    return matches


def greedy_by_confidence(sim_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy matching: process IMUs by confidence gap (best - second_best).
    
    Most confident IMU picks first.
    """
    n = sim_matrix.shape[0]
    # Calculate confidence for each IMU
    confidences = []
    for i in range(n):
        sorted_sims = np.sort(sim_matrix[i])[::-1]
        gap = sorted_sims[0] - sorted_sims[1] if n > 1 else sorted_sims[0]
        confidences.append((i, gap, sorted_sims[0]))
    
    # Sort by confidence (highest gap first), then by best sim
    confidences.sort(key=lambda x: (-x[1], -x[2]))
    
    used_persons = set()
    matches = []
    for i, _, _ in confidences:
        prefs = np.argsort(-sim_matrix[i])
        for j in prefs:
            if j not in used_persons:
                matches.append((i, j))
                used_persons.add(j)
                break
    return matches


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
                "window_start": int(row["window_start"]),
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

        trial_hungarian = []
        trial_greedy = []
        trial_greedy_conf = []

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

            # Hungarian
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            hungarian_correct = sum(
                sorted_imu_keys[r] == sorted_person_keys[c]
                for r, c in zip(row_ind, col_ind)
            )
            trial_hungarian.append(hungarian_correct / n_persons)

            # Greedy (order)
            greedy_pairs = greedy_matching(sim_matrix)
            greedy_correct = sum(
                sorted_imu_keys[r] == sorted_person_keys[c]
                for r, c in greedy_pairs
            )
            trial_greedy.append(greedy_correct / n_persons)

            # Greedy (confidence)
            greedy_conf_pairs = greedy_by_confidence(sim_matrix)
            greedy_conf_correct = sum(
                sorted_imu_keys[r] == sorted_person_keys[c]
                for r, c in greedy_conf_pairs
            )
            trial_greedy_conf.append(greedy_conf_correct / n_persons)

        result = {
            "sequence_id": session,
            "n_persons": n_persons,
            "n_windows": n_windows,
            "hungarian": round(float(np.mean(trial_hungarian)), 4),
            "greedy": round(float(np.mean(trial_greedy)), 4),
            "greedy_conf": round(float(np.mean(trial_greedy_conf)), 4),
        }
        all_results.append(result)
        print(f"  {session}: {n_persons}p, {n_windows}w | Hungarian={result['hungarian']:.2%} | Greedy={result['greedy']:.2%} | GreedyConf={result['greedy_conf']:.2%}")

    print(f"\n{'='*80}")
    print(f"{'Method':<20} {'Mean Acc':<12} {'vs Hungarian':<15}")
    print(f"{'-'*80}")
    
    hungarian_mean = float(np.mean([r["hungarian"] for r in all_results]))
    greedy_mean = float(np.mean([r["greedy"] for r in all_results]))
    greedy_conf_mean = float(np.mean([r["greedy_conf"] for r in all_results]))
    
    print(f"{'Hungarian':<20} {hungarian_mean:.4f}     baseline")
    print(f"{'Greedy (order)':<20} {greedy_mean:.4f}     {(greedy_mean/hungarian_mean - 1)*100:+.1f}%")
    print(f"{'Greedy (confidence)':<20} {greedy_conf_mean:.4f}     {(greedy_conf_mean/hungarian_mean - 1)*100:+.1f}%")
    print(f"{'='*80}")

    summary = {
        "num_sequences": len(all_results),
        "num_trials_per_seq": args.num_trials_per_seq,
        "hungarian_mean": round(hungarian_mean, 4),
        "greedy_mean": round(greedy_mean, 4),
        "greedy_conf_mean": round(greedy_conf_mean, 4),
        "sequences": all_results,
    }

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
