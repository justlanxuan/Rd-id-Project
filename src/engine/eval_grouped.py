"""Grouped IMU-video matching evaluation (adapts from MotionBERT)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset, lowpass_filter_fft
from src.engine.common import build_alignment_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped IMU-video matching evaluation")
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
    p.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None, help="FFT low-pass cutoff for IMU windows in Hz; set <= 0 to disable.")
    p.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0, help="Sampling rate used by the IMU low-pass filter.")
    p.add_argument("--chunk_windows", type=int, default=30)
    p.add_argument("--min_chunk_windows", type=int, default=15)
    p.add_argument("--group_sizes", type=str, default="2,4,6,8,16")
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_csv", type=str, default="")
    p.add_argument("--per_subject_split", action="store_true",
                   help="If set, each group is constructed from different subjects, forcing cross-subject matching.")
    p.add_argument("--shuffle_match", action="store_true", default=True,
                   help="If set (default), randomly shuffle IMU-side units before matching to prevent Hungarian algorithm from defaulting to identity permutation on degenerate matrices.")
    p.add_argument("--no_shuffle_match", action="store_true",
                   help="If set, disable IMU-side shuffling (legacy behavior).")

    # Global motion options (must match training config)
    p.add_argument("--use_global_motion", action="store_true")
    p.add_argument("--global_motion_input_dim", type=int, default=2)
    p.add_argument("--global_motion_hidden_dim", type=int, default=64)
    p.add_argument("--global_motion_num_layers", type=int, default=2)
    p.add_argument("--global_motion_dropout", type=float, default=0.1)
    p.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    p.add_argument("--global_motion_fusion_type", type=str, default="concat")
    p.add_argument("--global_motion_fusion_proj", action="store_true")
    p.add_argument("--global_motion_root_source", type=str, default="auto")
    p.add_argument("--global_motion_train_only", action="store_true")
    p.add_argument("--global_motion_aux_weight", type=float, default=0.0)
    return p.parse_args()


def parse_group_sizes(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def pair_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean cosine similarity between two sequences of embeddings."""
    n = min(len(a), len(b))
    if n <= 0:
        return -1.0
    sims = [1.0 - cosine(a[t], b[t]) for t in range(n)]
    return float(np.mean(sims))


def build_chunk_units(seq_embeddings: List[Dict], chunk_windows: int = 30, min_chunk_windows: int = 15) -> List[Dict]:
    """Build chunk units from sequences for grouped evaluation."""
    units = []
    for seq in seq_embeddings:
        imu_emb = seq["imu_emb"]
        vid_emb = seq["vid_emb"]
        seq_name = seq["seq_name"]

        n = min(len(imu_emb), len(vid_emb))
        if n < min_chunk_windows:
            continue

        start = 0
        cid = 0
        while start < n:
            end = min(start + chunk_windows, n)
            if end - start >= min_chunk_windows:
                units.append(
                    {
                        "unit_id": f"{seq_name}_c{cid:03d}",
                        "seq_name": seq_name,
                        "imu_emb": imu_emb[start:end],
                        "vid_emb": vid_emb[start:end],
                    }
                )
                cid += 1
            start += chunk_windows
    return units


def evaluate_grouped(units: List[Dict], group_size: int, num_trials: int = 50, seed: int = 42, shuffle_match: bool = True) -> Dict:
    """Evaluate grouped matching accuracy for a given group size.

    Args:
        shuffle_match: If True (default), randomly shuffle the IMU-side units
            before constructing the similarity matrix. This prevents the
            Hungarian algorithm from exploiting degenerate matrices (e.g.
            constant-row matrices when video embeddings collapse) by always
            defaulting to the identity permutation.
    """
    if len(units) < group_size:
        return {
            "group_size": group_size,
            "num_units": len(units),
            "num_trials": 0,
            "mean_acc": None,
            "std_acc": None,
            "mean_diag_sim": None,
            "mean_offdiag_sim": None,
            "note": f"insufficient units ({len(units)} < {group_size})",
        }

    rng = np.random.default_rng(seed)
    trial_acc = []
    trial_diag = []
    trial_offdiag = []

    for _ in range(num_trials):
        idx = rng.choice(len(units), size=group_size, replace=False)
        sel = [units[i] for i in idx]

        # Randomly shuffle IMU-side order so true match is not on diagonal
        if shuffle_match and group_size > 1:
            perm = rng.permutation(group_size)
            imu_sel = [sel[perm[i]] for i in range(group_size)]
        else:
            perm = np.arange(group_size)
            imu_sel = sel

        sim = np.zeros((group_size, group_size), dtype=np.float32)
        for i in range(group_size):
            for j in range(group_size):
                sim[i, j] = pair_similarity(imu_sel[i]["imu_emb"], sel[j]["vid_emb"])

        row_ind, col_ind = linear_sum_assignment(-sim)
        # With shuffled IMU side, correct matching is: perm[row_ind[k]] == col_ind[k]
        if shuffle_match and group_size > 1:
            correct = np.sum(perm[row_ind] == col_ind)
        else:
            correct = np.sum(row_ind == col_ind)
        trial_acc.append(float(correct) / float(group_size))

        trial_diag.append(float(np.mean(np.diag(sim))))
        if group_size > 1:
            mask = ~np.eye(group_size, dtype=bool)
            trial_offdiag.append(float(np.mean(sim[mask])))
        else:
            trial_offdiag.append(float("nan"))

    return {
        "group_size": group_size,
        "num_units": len(units),
        "num_trials": num_trials,
        "mean_acc": float(np.mean(trial_acc)),
        "std_acc": float(np.std(trial_acc)),
        "mean_diag_sim": float(np.mean(trial_diag)),
        "mean_offdiag_sim": float(np.mean(trial_offdiag)),
    }


def evaluate_grouped_per_subject(units_by_subject: Dict[str, List[Dict]], group_size: int, num_trials: int = 50, seed: int = 42, shuffle_match: bool = True) -> Dict:
    """Evaluate grouped matching with per-subject splitting.

    Each group is constructed by selecting exactly one unit from each of
    `group_size` different subjects. This forces cross-subject matching.
    """
    subjects = list(units_by_subject.keys())
    if len(subjects) < group_size:
        return {
            "group_size": group_size,
            "num_subjects": len(subjects),
            "num_trials": 0,
            "mean_acc": None,
            "std_acc": None,
            "mean_diag_sim": None,
            "mean_offdiag_sim": None,
            "note": f"insufficient subjects ({len(subjects)} < {group_size})",
        }

    rng = np.random.default_rng(seed)
    trial_acc = []
    trial_diag = []
    trial_offdiag = []

    for _ in range(num_trials):
        # Select group_size different subjects
        selected_subjects = rng.choice(subjects, size=group_size, replace=False)
        # Select one random unit from each subject
        sel = [rng.choice(units_by_subject[s]) for s in selected_subjects]

        # Randomly shuffle IMU-side order so true match is not on diagonal
        if shuffle_match and group_size > 1:
            perm = rng.permutation(group_size)
            imu_sel = [sel[perm[i]] for i in range(group_size)]
        else:
            perm = np.arange(group_size)
            imu_sel = sel

        sim = np.zeros((group_size, group_size), dtype=np.float32)
        for i in range(group_size):
            for j in range(group_size):
                sim[i, j] = pair_similarity(imu_sel[i]["imu_emb"], sel[j]["vid_emb"])

        row_ind, col_ind = linear_sum_assignment(-sim)
        if shuffle_match and group_size > 1:
            correct = np.sum(perm[row_ind] == col_ind)
        else:
            correct = np.sum(row_ind == col_ind)
        trial_acc.append(float(correct) / float(group_size))

        trial_diag.append(float(np.mean(np.diag(sim))))
        if group_size > 1:
            mask = ~np.eye(group_size, dtype=bool)
            trial_offdiag.append(float(np.mean(sim[mask])))
        else:
            trial_offdiag.append(float("nan"))

    return {
        "group_size": group_size,
        "num_subjects": len(subjects),
        "num_trials": num_trials,
        "mean_acc": float(np.mean(trial_acc)) if trial_acc else None,
        "std_acc": float(np.std(trial_acc)) if trial_acc else None,
        "mean_diag_sim": float(np.mean(trial_diag)) if trial_diag else None,
        "mean_offdiag_sim": float(np.mean(trial_offdiag)) if trial_offdiag else None,
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

    # Group embeddings by sequence (subject_session)
    seq_map = {}
    for i, row in enumerate(rows):
        seq_name = f"{row['subject']}_{row['session']}"
        if seq_name not in seq_map:
            seq_map[seq_name] = {"imu": [], "vid": []}
        seq_map[seq_name]["imu"].append(imu_all[i])
        seq_map[seq_name]["vid"].append(vid_all[i])

    seq_embeddings = []
    for seq_name, seq_data in seq_map.items():
        seq_embeddings.append(
            {
                "seq_name": seq_name,
                "imu_emb": np.stack(seq_data["imu"], axis=0),
                "vid_emb": np.stack(seq_data["vid"], axis=0),
            }
        )

    units = build_chunk_units(seq_embeddings, chunk_windows=args.chunk_windows, min_chunk_windows=args.min_chunk_windows)
    print(f"Built {len(units)} chunk units from {len(seq_embeddings)} sequences")

    group_sizes = parse_group_sizes(args.group_sizes)
    results = []

    if args.per_subject_split:
        # Group units by subject (e.g., "S4" from seq_name "S4_acting1")
        units_by_subject: Dict[str, List[Dict]] = {}
        for unit in units:
            subject = unit["seq_name"].split("_")[0]
            units_by_subject.setdefault(subject, []).append(unit)
        print(f"Per-subject split: {len(units_by_subject)} subjects")
        for subj, us in units_by_subject.items():
            print(f"  {subj}: {len(us)} units")
        for gs in group_sizes:
            print(f"Evaluating group_size={gs} (per-subject)...")
            res = evaluate_grouped_per_subject(units_by_subject, gs, num_trials=args.num_trials, seed=args.seed, shuffle_match=not args.no_shuffle_match)
            results.append(res)
            print(json.dumps(res, indent=2))
        summary = {
            "num_sequences": len(seq_embeddings),
            "num_units": len(units),
            "per_subject_split": True,
            "num_subjects": len(units_by_subject),
            "chunk_windows": args.chunk_windows,
            "min_chunk_windows": args.min_chunk_windows,
            "results": results,
        }
    else:
        for gs in group_sizes:
            print(f"Evaluating group_size={gs}...")
            res = evaluate_grouped(units, gs, num_trials=args.num_trials, seed=args.seed, shuffle_match=not args.no_shuffle_match)
            results.append(res)
            print(json.dumps(res, indent=2))
        summary = {
            "num_sequences": len(seq_embeddings),
            "num_units": len(units),
            "per_subject_split": False,
            "chunk_windows": args.chunk_windows,
            "min_chunk_windows": args.min_chunk_windows,
            "results": results,
        }

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved JSON: {out_json}")

    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            fieldnames = list(dict.fromkeys(k for r in results for k in r.keys()))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
