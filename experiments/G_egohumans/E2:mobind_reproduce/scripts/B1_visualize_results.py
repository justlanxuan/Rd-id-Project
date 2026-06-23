#!/usr/bin/env python3
"""
Experiment Note: B1-visualize-results
Generate summary charts for MoBInd reproduction on EgoHumans.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = json.loads((ROOT / "results" / "metrics.json").read_text())

# 1. Retrieval R@K
fig, ax = plt.subplots(figsize=(6, 4))
k = ["R@1", "R@3", "R@5", "R@10", "R@25", "R@50"]
imu2vid = [metrics["retrieval"]["imu_to_video"][m] for m in k]
vid2imu = [metrics["retrieval"]["video_to_imu"][m] for m in k]
x = np.arange(len(k))
width = 0.35
ax.bar(x - width/2, imu2vid, width, label="IMU → Video")
ax.bar(x + width/2, vid2imu, width, label="Video → IMU")
ax.set_ylabel("Recall")
ax.set_title("Retrieval R@K (stage2 MAE checkpoint)")
ax.set_xticks(x)
ax.set_xticklabels(k)
ax.set_ylim([0, 1.05])
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "retrieval_r_at_k.png", dpi=300)
plt.close(fig)

# 2. Localization accuracy
fig, ax = plt.subplots(figsize=(5, 4))
labels = ["Person (overall)", "Limb (cond. correct person)"]
vals = [metrics["localization"]["person_overall"], metrics["localization"]["limb_overall"]]
colors = ["#4C78A8", "#F58518"]
ax.bar(labels, vals, color=colors)
ax.set_ylabel("Accuracy")
ax.set_title("Localization accuracy")
ax.set_ylim([0, 1.05])
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v*100:.2f}%", ha="center")
fig.tight_layout()
fig.savefig(OUT_DIR / "localization_accuracy.png", dpi=300)
plt.close(fig)

# 3. Sync comparison
fig, ax = plt.subplots(figsize=(6, 4))
metrics_names = ["MAE (s)", "Acc@0.1", "Acc@0.2", "Acc@0.5"]
person = [metrics["sync_person"]["mae"], metrics["sync_person"]["acc_0.1"],
          metrics["sync_person"]["acc_0.2"], metrics["sync_person"]["acc_0.5"]]
video = [metrics["sync_video"]["mae"], metrics["sync_video"]["acc_0.1"],
         metrics["sync_video"]["acc_0.2"], metrics["sync_video"]["acc_0.5"]]
x = np.arange(len(metrics_names))
ax.bar(x - width/2, person, width, label="Person-level")
ax.bar(x + width/2, video, width, label="Video-level")
ax.set_ylabel("Value")
ax.set_title("Synchronization metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim([0, 1.05])
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "sync_metrics.png", dpi=300)
plt.close(fig)

print(f"Saved figures to {OUT_DIR}")
