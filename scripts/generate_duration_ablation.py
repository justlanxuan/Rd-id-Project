"""Generate duration ablation subsets from an existing sliced dataset.

This script creates downsampled versions of windows_train.csv by either:
1. temporal_truncation: limit each sequence to max_frames (from the start)
2. subject_subsetting: keep only specific subjects
3. sequence_subsetting: keep only N sequences per subject

Usage:
    python scripts/generate_duration_ablation.py \
        --root data/interim/totalcapture_video/slice \
        --mode temporal_truncation \
        --max_frames 2000 \
        --out_dir data/interim/totalcapture_video_ablation/trunc_2000
"""

import argparse
import csv
import json
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ablation_temporal_truncation(root: Path, max_frames: int, out_dir: Path):
    """Keep only windows where window_end <= max_frames for each train sequence."""
    seqs = load_csv(root / "sequences.csv")
    windows_all = load_csv(root / "windows_all.csv")

    # Build sequence frame lookup
    seq_frames = {r["npz_path"]: int(r["num_frames"]) for r in seqs}

    filtered = []
    for w in windows_all:
        nf = seq_frames.get(w["npz_path"], 0)
        limit = min(nf, max_frames)
        if int(w["window_end"]) <= limit:
            filtered.append(w)

    # Write per-split CSVs
    for split in ["train", "val", "test"]:
        rows = [r for r in filtered if r["split"] == split]
        save_csv(out_dir / f"windows_{split}.csv", rows, filtered[0].keys() if filtered else [])

    # Copy sequences.csv as-is (metadata)
    save_csv(out_dir / "sequences.csv", seqs, seqs[0].keys())

    # Stats
    train_rows = [r for r in filtered if r["split"] == "train"]
    train_frames = sum(min(seq_frames.get(r["npz_path"], 0), max_frames) for r in seqs if "train" in r["split"])
    print(f"[temporal_truncation max_frames={max_frames}] -> {out_dir}")
    print(f"  Train windows: {len(train_rows)}")
    print(f"  Approx train frames (capped): {train_frames} ({train_frames/30/60:.2f} min)")


def ablation_subject_subset(root: Path, subjects: list[str], out_dir: Path):
    """Keep only windows from specified subjects."""
    seqs = load_csv(root / "sequences.csv")
    windows_all = load_csv(root / "windows_all.csv")

    subject_set = set(subjects)
    filtered = [w for w in windows_all if w["subject"] in subject_set]

    # Also filter sequences.csv for stats
    filtered_seqs = [r for r in seqs if r["subject"] in subject_set]

    for split in ["train", "val", "test"]:
        rows = [r for r in filtered if r["split"] == split]
        save_csv(out_dir / f"windows_{split}.csv", rows, filtered[0].keys() if filtered else [])

    save_csv(out_dir / "sequences.csv", filtered_seqs, filtered_seqs[0].keys())

    train_rows = [r for r in filtered if r["split"] == "train"]
    train_frames = sum(int(r["num_frames"]) for r in filtered_seqs if "train" in r["split"])
    print(f"[subject_subset subjects={subjects}] -> {out_dir}")
    print(f"  Train windows: {len(train_rows)}")
    print(f"  Train frames: {train_frames} ({train_frames/30/60:.2f} min)")


def ablation_sequence_subset(root: Path, n_per_subject: int, out_dir: Path):
    """Keep only first N sequences per subject."""
    seqs = load_csv(root / "sequences.csv")
    windows_all = load_csv(root / "windows_all.csv")

    from collections import defaultdict
    subject_seqs = defaultdict(list)
    for r in seqs:
        subject_seqs[r["subject"]].append(r["npz_path"])

    allowed = set()
    for subj, paths in subject_seqs.items():
        allowed.update(paths[:n_per_subject])

    filtered = [w for w in windows_all if w["npz_path"] in allowed]
    filtered_seqs = [r for r in seqs if r["npz_path"] in allowed]

    for split in ["train", "val", "test"]:
        rows = [r for r in filtered if r["split"] == split]
        save_csv(out_dir / f"windows_{split}.csv", rows, filtered[0].keys() if filtered else [])

    save_csv(out_dir / "sequences.csv", filtered_seqs, filtered_seqs[0].keys())

    train_rows = [r for r in filtered if r["split"] == "train"]
    train_frames = sum(int(r["num_frames"]) for r in filtered_seqs if "train" in r["split"])
    print(f"[sequence_subset n_per_subject={n_per_subject}] -> {out_dir}")
    print(f"  Train windows: {len(train_rows)}")
    print(f"  Train frames: {train_frames} ({train_frames/30/60:.2f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Path to existing slice dir (contains sequences.csv, windows_all.csv)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for ablation subset")
    parser.add_argument("--mode", choices=["temporal_truncation", "subject_subset", "sequence_subset"], required=True)
    parser.add_argument("--max_frames", type=int, default=None, help="For temporal_truncation")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects for subject_subset")
    parser.add_argument("--n_per_subject", type=int, default=None, help="For sequence_subset")
    args = parser.parse_args()

    if args.mode == "temporal_truncation":
        if args.max_frames is None:
            raise ValueError("--max_frames required for temporal_truncation")
        ablation_temporal_truncation(args.root, args.max_frames, args.out_dir)
    elif args.mode == "subject_subset":
        if args.subjects is None:
            raise ValueError("--subjects required for subject_subset")
        ablation_subject_subset(args.root, args.subjects.split(","), args.out_dir)
    elif args.mode == "sequence_subset":
        if args.n_per_subject is None:
            raise ValueError("--n_per_subject required for sequence_subset")
        ablation_sequence_subset(args.root, args.n_per_subject, args.out_dir)


if __name__ == "__main__":
    main()
