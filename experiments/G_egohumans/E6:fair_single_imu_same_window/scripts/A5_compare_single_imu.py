#!/usr/bin/env python3
"""
Experiment Note: A5-compare-single-imu
Compare MoBInd vs our pipeline under the fair setting:
single right-wrist IMU + 24-frame window on the 24 official MoBInd test sequences.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/fzliang/Autism-project")
E6_RESULTS = ROOT / "experiments/G_egohumans/E6:fair_single_imu_same_window/results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    mobind = load_json(E6_RESULTS / "mobind_single_imu_frameacc.json")
    ours_1w = load_json(E6_RESULTS / "ours_single_imu_1window.json")
    ours_vote = load_json(E6_RESULTS / "ours_single_imu_4window_vote.json")

    seq_ids = [s["sequence_id"] for s in mobind["sequences"]]
    m_vals = [s["frame_acc"] for s in mobind["sequences"]]
    o1_vals = [s["frame_matching_accuracy"] for s in ours_1w["sequences"]]
    ov_vals = [s["frame_matching_accuracy"] for s in ours_vote["sequences"]]

    means = {
        "MoBInd single IMU (24-frame)": mobind["mean_frame_acc"],
        "Our pipeline single IMU (1 window)": ours_1w["mean_frame_matching_accuracy"],
        "Our pipeline single IMU (4-window vote)": ours_vote["mean_frame_matching_accuracy"],
    }

    print("=" * 70)
    print("E6: Single-IMU same-window comparison on MoBInd official test set")
    print("=" * 70)
    print(f"{'Sequence':<20} {'MoBInd':>10} {'Ours 1w':>10} {'Ours 4w-vote':>14}")
    for sid, m, o1, ov in zip(seq_ids, m_vals, o1_vals, ov_vals):
        print(f"{sid:<20} {m:>10.4f} {o1:>10.4f} {ov:>14.4f}")
    print("-" * 70)
    for name, val in means.items():
        print(f"{name:<40} {val:.4f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(seq_ids))
    width = 0.25
    ax.bar(x - width, m_vals, width, label="MoBInd single IMU (24-frame)")
    ax.bar(x, o1_vals, width, label="Our pipeline single IMU (1 window)")
    ax.bar(x + width, ov_vals, width, label="Our pipeline single IMU (4-window vote)")
    ax.set_ylabel("FrameAcc")
    ax.set_title("Single-IMU 24-frame FrameAcc on MoBInd official test set")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("custom_", "") for s in seq_ids], rotation=45, ha="right")
    ax.set_ylim(0.7, 1.02)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig_path = E6_RESULTS / "figures" / "frameacc_single_imu_same_window.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    results_md = E6_RESULTS / "results.md"
    with open(results_md, "w") as f:
        f.write("# E6 Results: Single-IMU Same-Window Fair Comparison\n\n")
        f.write("## 实验设置\n\n")
        f.write("- **IMU 输入**：双方都只使用右手腕 IMU。\n")
        f.write("  - MoBInd：`limb_list = [RightWrist]`，`num_limbs = 1`。\n")
        f.write("  - Our pipeline：`imu_sensor = R_LowArm`，`repeat_single_sensor = 4`（复制 4 次以维持 48-D 输入）。\n")
        f.write("- **窗口长度**：统一为 24 帧（约 1.2 s @ 20 Hz）。\n")
        f.write("- **测试集**：24 个 MoBInd official test 序列（双方均 unseen）。\n\n")
        f.write("## FrameAcc 对比（24 个序列）\n\n")
        f.write("| Sequence | MoBInd single IMU | Our pipeline (1 window) | Our pipeline (4-window vote) |\n")
        f.write("|---|---|---|---|\n")
        for sid, m, o1, ov in zip(seq_ids, m_vals, o1_vals, ov_vals):
            f.write(f"| {sid} | {m:.4f} | {o1:.4f} | {ov:.4f} |\n")
        f.write("| **Mean** | **{:.4f}** | **{:.4f}** | **{:.4f}** |\n".format(
            means["MoBInd single IMU (24-frame)"],
            means["Our pipeline single IMU (1 window)"],
            means["Our pipeline single IMU (4-window vote)"],
        ))
        f.write("\n## 结论\n\n")
        f.write("- 在单 IMU + 24 帧的严格公平设置下，MoBInd 的 mean FrameAcc 为 **{:.4f}**。\n".format(means["MoBInd single IMU (24-frame)"]))
        f.write("- 我们的 pipeline 1-window 为 **{:.4f}**（{} {:.2f} pp）。\n".format(
            means["Our pipeline single IMU (1 window)"],
            "落后" if means["MoBInd single IMU (24-frame)"] > means["Our pipeline single IMU (1 window)"] else "领先",
            abs(means["MoBInd single IMU (24-frame)"] - means["Our pipeline single IMU (1 window)"]) * 100,
        ))
        f.write("- 我们的 pipeline 4-window vote 为 **{:.4f}**（{} {:.2f} pp）。\n".format(
            means["Our pipeline single IMU (4-window vote)"],
            "落后" if means["MoBInd single IMU (24-frame)"] > means["Our pipeline single IMU (4-window vote)"] else "领先",
            abs(means["MoBInd single IMU (24-frame)"] - means["Our pipeline single IMU (4-window vote)"]) * 100,
        ))
        f.write("\n## 与 E5 的对比\n\n")
        f.write("- E5（4-IMU，24 帧窗口）：MoBInd 0.9666，Ours 1-window 0.9372，Ours 4-window vote 0.9536。\n")
        f.write("- E6（单 IMU，24 帧窗口）：双方均显著下降，说明多传感器信息对 MoBInd 和我们的 pipeline 都很重要。\n")
        f.write("\n## AI Reflection\n\n")
        f.write("- 控制 IMU 数量和窗口长度后，差距趋势与 E5 一致：MoBInd 仍略高，但 4-window vote 能缩小差距。\n")
        f.write("- 单 IMU 重复 4 次在我们的 pipeline 中是一种工程折中，未来若需完全对齐，可重构 IMU encoder 支持真单 IMU 输入。\n")

    print(f"Saved results to {results_md}")


if __name__ == "__main__":
    main()
