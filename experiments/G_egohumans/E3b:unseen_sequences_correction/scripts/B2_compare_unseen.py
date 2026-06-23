#!/usr/bin/env python3
"""
Experiment Note: B2-compare-unseen
Generate the corrected E3 comparison table and figure on the 4 sequences that
are unseen by both MoBInd and our 4-IMU pipeline.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/fzliang/Autism-project")
E3B_RESULTS = ROOT / "experiments/G_egohumans/E3b:unseen_sequences_correction/results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    mobind = load_json(E3B_RESULTS / "mobind_frameacc_unseen.json")
    ours_1w = load_json(E3B_RESULTS / "ours_4imu_1window_unseen.json")
    ours_vote = load_json(E3B_RESULTS / "ours_4imu_4window_vote_unseen.json")

    seq_ids = [s["sequence_id"] for s in mobind["sequences"]]
    m_vals = [s["frame_acc"] for s in mobind["sequences"]]
    o1_vals = [s["frame_matching_accuracy"] for s in ours_1w["sequences"]]
    ov_vals = [s["frame_matching_accuracy"] for s in ours_vote["sequences"]]

    means = {
        "MoBInd (5s window)": mobind["mean_frame_acc"],
        "Our pipeline (1 window)": ours_1w["mean_frame_matching_accuracy"],
        "Our pipeline (4-window vote)": ours_vote["mean_frame_matching_accuracy"],
    }

    print("=" * 60)
    print("Corrected E3: FrameAcc on sequences unseen by both models")
    print("=" * 60)
    print(f"{'Sequence':<20} {'MoBInd':>10} {'Ours 1w':>10} {'Ours 4w-vote':>14}")
    for sid, m, o1, ov in zip(seq_ids, m_vals, o1_vals, ov_vals):
        print(f"{sid:<20} {m:>10.4f} {o1:>10.4f} {ov:>14.4f}")
    print("-" * 60)
    for name, val in means.items():
        print(f"{name:<30} {val:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(seq_ids))
    width = 0.25
    ax.bar(x - width, m_vals, width, label="MoBInd (5s window)")
    ax.bar(x, o1_vals, width, label="Our pipeline (1 window)")
    ax.bar(x + width, ov_vals, width, label="Our pipeline (4-window vote)")
    ax.set_ylabel("FrameAcc")
    ax.set_title("FrameAcc on 4 sequences unseen by both models")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("custom_", "") for s in seq_ids])
    ax.set_ylim(0.9, 1.01)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig_path = E3B_RESULTS / "figures" / "frameacc_unseen_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    # results.md
    results_md = E3B_RESULTS / "results.md"
    with open(results_md, "w") as f:
        f.write("# E3b Corrected Results: Unseen-by-Both Sequences\n\n")
        f.write("## 说明\n\n")
        f.write("E3 之前使用的 16 个序列属于 MoBInd 官方 train split，导致 MoBInd checkpoint 在训练时见过它们，")
        f.write("不是严格的泛化测试。本实验改用 **双方都没见过的 4 个序列**（我们的 test_sessions 与 MoBInd test/val 的交集）：\n\n")
        f.write("```\n01_002, 03_001, 04_005, 05_002\n```\n\n")
        f.write("## FrameAcc 对比\n\n")
        f.write("| Sequence | MoBInd (5s window) | Our pipeline (1 window) | Our pipeline (4-window vote) |\n")
        f.write("|---|---|---|---|\n")
        for sid, m, o1, ov in zip(seq_ids, m_vals, o1_vals, ov_vals):
            f.write(f"| {sid} | {m:.4f} | {o1:.4f} | {ov:.4f} |\n")
        f.write("| **Mean** | **{:.4f}** | **{:.4f}** | **{:.4f}** |\n".format(
            means["MoBInd (5s window)"], means["Our pipeline (1 window)"], means["Our pipeline (4-window vote)"]
        ))
        f.write("\n## 结论\n\n")
        f.write("- 在严格 unseen 的 4 个序列上，MoBInd 的 mean FrameAcc 为 **{:.4f}**。\n".format(means["MoBInd (5s window)"]))
        f.write("- 我们的 pipeline 1-window 为 **{:.4f}**（落后 {:.2f} pp）。\n".format(
            means["Our pipeline (1 window)"], (means["MoBInd (5s window)"] - means["Our pipeline (1 window)"]) * 100
        ))
        f.write("- 我们的 pipeline 4-window vote 为 **{:.4f}**，与 MoBInd 基本持平（差距 {:.2f} pp）。\n".format(
            means["Our pipeline (4-window vote)"],
            (means["MoBInd (5s window)"] - means["Our pipeline (4-window vote)"]) * 100,
        ))
        f.write("\n## AI Reflection\n\n")
        f.write("- 之前 E3 的 16-sequence 'train-only' subset 对 MoBInd 不够公平；本 4-sequence unseen subset 才是严格对齐的对比。\n")
        f.write("- 4-window vote 在这个小集合上表现出色，与 MoBInd 5s 窗口相当，但样本量仅 4 个序列，结论需谨慎推广。\n")
        f.write("- 建议后续在更大的 unseen 集合上验证，或直接使用完整 20-sequence test set 作为 secondary 参考。\n")

    print(f"Saved results to {results_md}")


if __name__ == "__main__":
    main()
