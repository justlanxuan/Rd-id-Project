#!/usr/bin/env python3
"""
Experiment Note: A5-compare-4imu-vs-1imu
Aggregate 4-IMU (E3) and 1-IMU (E4) results into a comparison table and bar chart.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/fzliang/Autism-project")
E3_RESULTS = ROOT / "experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/results"
E4_RESULTS = ROOT / "experiments/G_egohumans/E4:single_imu_right_wrist/results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    e3_full = load_json(ROOT / "data/interim/egohumans_full_extract/test/egohumans_full_extract/synchronous_results.json")
    e3_subset = load_json(E3_RESULTS / "frameacc_comparison.json")

    e4_full = load_json(E4_RESULTS / "full_test_1imu.json")
    e4_1w = load_json(E4_RESULTS / "ours_1imu_1window.json")
    e4_vote = load_json(E4_RESULTS / "ours_1imu_4window_vote.json")

    # Overall comparison table
    overall = {
        "Setting": ["Full test (20 seq)", "Train-only subset 1-window (16 seq)", "Train-only subset 4-window vote (16 seq)"],
        "4-IMU": [
            e3_full["mean_frame_matching_accuracy"],
            e3_subset["ours_1window_mean_frame_acc"],
            e3_subset["ours_4window_vote_mean_frame_acc"],
        ],
        "1-IMU (R_LowArm)": [
            e4_full["mean_frame_matching_accuracy"],
            e4_1w["mean_frame_matching_accuracy"],
            e4_vote["mean_frame_matching_accuracy"],
        ],
    }
    overall["Δ (pp)"] = [round((a - b) * 100, 2) for a, b in zip(overall["4-IMU"], overall["1-IMU (R_LowArm)"])]

    print("=" * 60)
    print("4-IMU vs 1-IMU FrameAcc comparison")
    print("=" * 60)
    print(f"{'Setting':<45} {'4-IMU':>8} {'1-IMU':>8} {'Δ(pp)':>8}")
    for i, setting in enumerate(overall["Setting"]):
        print(f"{setting:<45} {overall['4-IMU'][i]:>8.4f} {overall['1-IMU (R_LowArm)'][i]:>8.4f} {overall['Δ (pp)'][i]:>8.2f}")

    # Per-sequence comparison (16 train-only sequences)
    seq_4w = {s["sequence_id"]: s for s in e3_subset["per_sequence"]}
    seq_1w_4imu = {s["sequence_id"]: s["frame_matching_accuracy"] for s in load_json(ROOT / "data/interim/egohumans_full_extract/test/egohumans_full_extract/synchronous_results.json")["sequences"]}

    seq_ids = [s["sequence_id"] for s in e3_subset["per_sequence"]]
    x4_1w = [seq_4w[s]["ours_1window"] for s in seq_ids]
    x4_vote = [seq_4w[s]["ours_4window_vote"] for s in seq_ids]
    x1_1w = [e4_1w["sequences"][i]["frame_matching_accuracy"] for i, s in enumerate(seq_ids)]
    x1_vote = [e4_vote["sequences"][i]["frame_matching_accuracy"] for i, s in enumerate(seq_ids)]

    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(seq_ids)
    ind = np.arange(n)
    width = 0.2

    ax.bar(ind - 1.5 * width, x4_1w, width, label="4-IMU 1-window")
    ax.bar(ind - 0.5 * width, x4_vote, width, label="4-IMU 4-window vote")
    ax.bar(ind + 0.5 * width, x1_1w, width, label="1-IMU 1-window")
    ax.bar(ind + 1.5 * width, x1_vote, width, label="1-IMU 4-window vote")

    ax.set_ylabel("FrameAcc")
    ax.set_title("Per-sequence FrameAcc: 4-IMU vs 1-IMU (R_LowArm)")
    ax.set_xticks(ind)
    ax.set_xticklabels([s.replace("custom_", "") for s in seq_ids], rotation=45, ha="right")
    ax.set_ylim(0.85, 1.02)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig_path = E4_RESULTS / "figures" / "frameacc_4imu_vs_1imu.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    # Write results.md
    results_md = E4_RESULTS / "results.md"
    with open(results_md, "w") as f:
        f.write("# E4 Results: 4-IMU vs 1-IMU (Right Wrist)\n\n")
        f.write("## Overall FrameAcc Comparison\n\n")
        f.write("| Setting | 4-IMU | 1-IMU (R_LowArm) | Δ (pp) |\n")
        f.write("|---|---|---|---|\n")
        for i, setting in enumerate(overall["Setting"]):
            f.write(f"| {setting} | {overall['4-IMU'][i]:.4f} | {overall['1-IMU (R_LowArm)'][i]:.4f} | {overall['Δ (pp)'][i]:.2f} |\n")
        f.write("\n")
        f.write("## Per-sequence FrameAcc (16 MoBInd-train-only sequences)\n\n")
        f.write("| Sequence | 4-IMU 1w | 4-IMU 4w vote | 1-IMU 1w | 1-IMU 4w vote |\n")
        f.write("|---|---|---|---|---|\n")
        for i, sid in enumerate(seq_ids):
            f.write(f"| {sid} | {x4_1w[i]:.4f} | {x4_vote[i]:.4f} | {x1_1w[i]:.4f} | {x1_vote[i]:.4f} |\n")
        f.write("\n")
        f.write("## AI Reflection\n\n")
        f.write("- The 1-IMU model is a strict ablation: only the right-wrist sensor is kept; all other hyperparameters and data splits are identical.\n")
        f.write("- We expect a measurable but hopefully small drop in FrameAcc, because wrist motion is usually highly correlated with full-body motion in EgoHumans activities.\n")
        f.write("- Human review: Does this drop justify the hardware simplification, or should we explore fusing wrist + one other sensor as a middle ground?\n")

    print(f"Saved results to {results_md}")

    summary_json = E4_RESULTS / "comparison_summary.json"
    with open(summary_json, "w") as f:
        json.dump({
            "overall": overall,
            "per_sequence": [
                {
                    "sequence_id": sid,
                    "four_imu_1window": x4_1w[i],
                    "four_imu_4window_vote": x4_vote[i],
                    "one_imu_1window": x1_1w[i],
                    "one_imu_4window_vote": x1_vote[i],
                }
                for i, sid in enumerate(seq_ids)
            ],
        }, f, indent=2)
    print(f"Saved summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
