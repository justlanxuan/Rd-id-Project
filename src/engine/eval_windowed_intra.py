#!/usr/bin/env python3
"""Intra-sequence windowed IMU-video matching evaluation.

For each test sequence, randomly sample N window positions.
Within each window, perform Hungarian matching between all IMUs and all video tracks.
The group size is determined by the number of people in the video (not configurable).
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
    p = argparse.ArgumentParser(description="Intra-sequence windowed matching evaluation")
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
    p.add_argument("--num_trials_per_seq", type=int, default=10, help="Number of random window samples per sequence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


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

    imu_all = np.concatenate(imu_all, axis=0)  # [N_total, embed_dim]
    vid_all = np.concatenate(vid_all, axis=0)  # [N_total, embed_dim]

    # Group by (session, window_start, person_idx, imu_idx)
    # Each window has N_person × N_imu rows
    window_map: Dict[Tuple[str, str], Dict] = {}
    for i, row in enumerate(rows):
        key = (row["session"], row["window_start"])
        if key not in window_map:
            window_map[key] = {
                "session": row["session"],
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "person_embs": {},  # person_idx -> list of (imu_idx, imu_emb, vid_emb)
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

    # Group windows by session
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

        # Determine group size = number of people in this sequence
        # All windows in a sequence should have the same number of people
        first_window = windows[0]
        n_persons = len(first_window["person_embs"])
        n_imus = len(first_window["person_embs"][0]) if n_persons > 0 else 0

        trial_accs = []
        for trial in range(args.num_trials_per_seq):
            # Randomly sample a window (with replacement if n_windows < num_trials)
            win_idx = rng.integers(0, n_windows)
            win = windows[win_idx]

            # Build similarity matrix: [n_imus, n_persons]
            # Each person_idx has one video embedding (we use the one matched to itself)
            # and multiple IMU embeddings (one per imu_idx)
            sim_matrix = np.zeros((n_imus, n_persons), dtype=np.float32)

            for person_idx, entries in win["person_embs"].items():
                for entry in entries:
                    imu_idx = entry["imu_idx"]
                    # Use the video embedding from the (person_idx, imu_idx=person_idx) pair
                    # Actually, for video embedding, it should be the same regardless of imu_idx
                    # because video embedding only depends on skeleton. So we can use any entry's vid_emb.
                    # But to be precise, let's use the entry where imu_idx == person_idx
                    pass

            # Re-organize: get video embedding for each person
            # Video embedding should be the same for all imu_idx of the same person_idx
            # Let's verify and pick the first one
            vid_embs_per_person = {}
            imu_embs_per_imu = {}

            for person_idx, entries in win["person_embs"].items():
                # All entries for the same person should have the same vid_emb
                vid_embs_per_person[person_idx] = entries[0]["vid_emb"]
                for entry in entries:
                    imu_idx = entry["imu_idx"]
                    if imu_idx not in imu_embs_per_imu:
                        imu_embs_per_imu[imu_idx] = entry["imu_emb"]

            # Build similarity matrix
            for i, imu_idx in enumerate(sorted(imu_embs_per_imu.keys())):
                for j, person_idx in enumerate(sorted(vid_embs_per_person.keys())):
                    imu_emb = imu_embs_per_imu[imu_idx]
                    vid_emb = vid_embs_per_person[person_idx]
                    # Cosine similarity
                    sim = np.dot(imu_emb, vid_emb) / (np.linalg.norm(imu_emb) * np.linalg.norm(vid_emb) + 1e-8)
                    sim_matrix[i, j] = sim

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)

            # Check correctness: imu_idx should match person_idx
            correct = 0
            for r, c in zip(row_ind, col_ind):
                imu_idx = sorted(imu_embs_per_imu.keys())[r]
                person_idx = sorted(vid_embs_per_person.keys())[c]
                if imu_idx == person_idx:
                    correct += 1

            acc = correct / n_persons if n_persons > 0 else 0.0
            trial_accs.append(acc)

        mean_acc = float(np.mean(trial_accs))
        std_acc = float(np.std(trial_accs))

        result = {
            "sequence_id": session,
            "n_persons": n_persons,
            "n_windows": n_windows,
            "num_trials": args.num_trials_per_seq,
            "mean_acc": round(mean_acc, 4),
            "std_acc": round(std_acc, 4),
            "trial_accs": [round(a, 4) for a in trial_accs],
        }
        all_results.append(result)
        print(f"  {session}: {n_persons} persons, {n_windows} windows, mean_acc={mean_acc:.4f} ± {std_acc:.4f}")

    overall_mean = float(np.mean([r["mean_acc"] for r in all_results]))
    overall_std = float(np.mean([r["std_acc"] for r in all_results]))

    summary = {
        "num_sequences": len(all_results),
        "num_trials_per_seq": args.num_trials_per_seq,
        "overall_mean_acc": round(overall_mean, 4),
        "overall_std_acc": round(overall_std, 4),
        "sequences": all_results,
    }

    print(f"\n{'='*60}")
    print(f"Overall mean accuracy: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"{'='*60}")

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
