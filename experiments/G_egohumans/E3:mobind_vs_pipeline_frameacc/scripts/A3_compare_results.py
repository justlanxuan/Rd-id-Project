#!/usr/bin/env python3
"""
Experiment Note: A3-compare-results
Load MoBInd and our-pipeline FrameAcc results on the same 16 sequences,
print a side-by-side table, and generate a comparison bar chart.
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

    mobind_seq, mobind_mean = load_seq_acc(mobind_path, key="frame_acc")
    ours_seq, ours_mean = load_seq_acc(ours_path, key="frame_matching_accuracy")

    seq_ids = sorted(mobind_seq.keys())
    mobind_vals = [mobind_seq[s] for s in seq_ids]
    ours_vals = [ours_seq[s] for s in seq_ids]

    print("=" * 70)
    print(f"{'Sequence':<20} {'MoBInd':>12} {'Ours':>12} {'Δ':>12}")
    print("-" * 70)
    for s in seq_ids:
        d = mobind_seq[s] - ours_seq[s]
        print(f"{s:<20} {mobind_seq[s]:>12.4f} {ours_seq[s]:>12.4f} {d:>+12.4f}")
    print("-" * 70)
    print(f"{'Mean':<20} {mobind_mean:>12.4f} {ours_mean:>12.4f} {mobind_mean - ours_mean:>+12.4f}")
    print("=" * 70)

    # Save table as JSON
    summary = {
        "num_sequences": len(seq_ids),
        "mobind_mean_frame_acc": mobind_mean,
        "ours_mean_frame_acc": ours_mean,
        "difference": mobind_mean - ours_mean,
        "per_sequence": [
            {
                "sequence_id": s,
                "mobind": mobind_seq[s],
                "ours": ours_seq[s],
                "delta": mobind_seq[s] - ours_seq[s],
            }
            for s in seq_ids
        ],
    }
    with open(RESULTS_DIR / "frameacc_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Bar chart
    x = np.arange(len(seq_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, mobind_vals, width, label="MoBInd")
    ax.bar(x + width / 2, ours_vals, width, label="Our pipeline")
    ax.set_ylabel("FrameAcc")
    ax.set_title("FrameAcc comparison on 16 MoBInd-train-only EgoHumans sequences")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_ids, rotation=45, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.axhline(mobind_mean, color="C0", linestyle="--", linewidth=1)
    ax.axhline(ours_mean, color="C1", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frameacc_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved chart to {FIGURES_DIR / 'frameacc_comparison.png'}")


if __name__ == "__main__":
    main()
