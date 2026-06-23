#!/usr/bin/env python3
"""
Experiment Note: A3-compare-results
Load MoBInd and our-pipeline FrameAcc results (single-window, 4-window mean,
and 4-window vote) on the same 16 sequences, print side-by-side tables,
and generate charts.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_seq_acc(path: Path, key: str = "frame_acc"):
    with open(path) as f:
        data = json.load(f)
    mean_key = "mean_frame_acc" if key == "frame_acc" else "mean_frame_matching_accuracy"
    return {r["sequence_id"]: r[key] for r in data["sequences"]}, data[mean_key]


def main():
    mobind_path = RESULTS_DIR / "mobind_frameacc.json"
    ours_path = RESULTS_DIR / "ours_frameacc.json"
    ours4_path = RESULTS_DIR / "ours_frameacc_4window.json"
    ours4v_path = RESULTS_DIR / "ours_frameacc_4window_vote.json"

    mobind_seq, mobind_mean = load_seq_acc(mobind_path, key="frame_acc")
    ours_seq, ours_mean = load_seq_acc(ours_path, key="frame_matching_accuracy")
    ours4_seq, ours4_mean = (None, None)
    ours4v_seq, ours4v_mean = (None, None)
    if ours4_path.exists():
        ours4_seq, ours4_mean = load_seq_acc(ours4_path, key="frame_matching_accuracy")
    if ours4v_path.exists():
        ours4v_seq, ours4v_mean = load_seq_acc(ours4v_path, key="frame_matching_accuracy")

    seq_ids = sorted(mobind_seq.keys())
    mobind_vals = [mobind_seq[s] for s in seq_ids]
    ours_vals = [ours_seq[s] for s in seq_ids]

    print("=" * 90)
    header = f"{'Sequence':<20} {'MoBInd':>12} {'Ours(1w)':>12}"
    if ours4_seq is not None:
        header += f" {'Ours(4w-mean)':>14}"
    if ours4v_seq is not None:
        header += f" {'Ours(4w-vote)':>14}"
    print(header)
    print("-" * 90)
    for s in seq_ids:
        line = f"{s:<20} {mobind_seq[s]:>12.4f} {ours_seq[s]:>12.4f}"
        if ours4_seq is not None:
            line += f" {ours4_seq[s]:>14.4f}"
        if ours4v_seq is not None:
            line += f" {ours4v_seq[s]:>14.4f}"
        print(line)
    print("-" * 90)
    mean_line = f"{'Mean':<20} {mobind_mean:>12.4f} {ours_mean:>12.4f}"
    if ours4_mean is not None:
        mean_line += f" {ours4_mean:>14.4f}"
    if ours4v_mean is not None:
        mean_line += f" {ours4v_mean:>14.4f}"
    print(mean_line)
    print("=" * 90)

    # Save table as JSON
    summary = {
        "num_sequences": len(seq_ids),
        "mobind_mean_frame_acc": mobind_mean,
        "ours_1window_mean_frame_acc": ours_mean,
        "per_sequence": [
            {
                "sequence_id": s,
                "mobind": mobind_seq[s],
                "ours_1window": ours_seq[s],
            }
            for s in seq_ids
        ],
    }
    if ours4_mean is not None:
        summary["ours_4window_mean_frame_acc"] = ours4_mean
        for entry, s in zip(summary["per_sequence"], seq_ids):
            entry["ours_4window_mean"] = ours4_seq[s]
    if ours4v_mean is not None:
        summary["ours_4window_vote_mean_frame_acc"] = ours4v_mean
        for entry, s in zip(summary["per_sequence"], seq_ids):
            entry["ours_4window_vote"] = ours4v_seq[s]

    with open(RESULTS_DIR / "frameacc_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Bar chart
    x = np.arange(len(seq_ids))
    width = 0.18
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - 1.5 * width, mobind_vals, width, label="MoBInd")
    ax.bar(x - 0.5 * width, ours_vals, width, label="Ours (1 window)")
    if ours4_seq is not None:
        ours4_vals = [ours4_seq[s] for s in seq_ids]
        ax.bar(x + 0.5 * width, ours4_vals, width, label="Ours (4 windows mean)")
    if ours4v_seq is not None:
        ours4v_vals = [ours4v_seq[s] for s in seq_ids]
        ax.bar(x + 1.5 * width, ours4v_vals, width, label="Ours (4 windows vote)")
    ax.set_ylabel("FrameAcc")
    ax.set_title("FrameAcc comparison on 16 MoBInd-train-only EgoHumans sequences")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, rotation=45, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.axhline(mobind_mean, color="C0", linestyle="--", linewidth=1)
    ax.axhline(ours_mean, color="C1", linestyle="--", linewidth=1)
    if ours4_mean is not None:
        ax.axhline(ours4_mean, color="C2", linestyle="--", linewidth=1)
    if ours4v_mean is not None:
        ax.axhline(ours4v_mean, color="C3", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frameacc_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved chart to {FIGURES_DIR / 'frameacc_comparison.png'}")


if __name__ == "__main__":
    main()
