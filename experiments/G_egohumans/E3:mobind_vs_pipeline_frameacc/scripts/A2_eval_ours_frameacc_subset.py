#!/usr/bin/env python3
"""
Experiment Note: A2-eval-ours-frameacc-subset
Run our own pipeline's synchronous eval on the same 16 MoBInd-train-only
sequences used in A1, so the FrameAcc numbers are directly comparable.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# Sequences to keep (MoBInd train-only)
SELECTED_SEQS = {
    "01_011", "02_001", "03_009", "04_011", "05_007",
    "06_024", "06_040", "06_041", "06_054", "06_019",
    "06_036", "06_006", "06_025", "06_060", "07_011", "07_007",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="/home/fzliang/Autism-project/data/interim/egohumans_full_extract/slice/windows_test.csv")
    parser.add_argument("--output_json", default="/home/fzliang/Autism-project/experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/results/ours_frameacc.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_csv = out_dir / "windows_test_mobind_train_only.csv"

    # Filter CSV to selected sessions
    with open(test_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r["session"] in SELECTED_SEQS]

    if not rows:
        print("No rows matched selected sequences.")
        sys.exit(1)

    with open(filtered_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Filtered CSV: {len(rows)} rows, {len(SELECTED_SEQS)} sessions -> {filtered_csv}")

    # Run our synchronous eval
    cmd = [
        sys.executable,
        "-m",
        "src.engine.eval_synchronous",
        "--test_csv", str(filtered_csv),
        "--data_root", str(test_csv.parent),
        "--motionbert_root", "/home/fzliang/origin/MotionBERT",
        "--motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml",
        "--motionbert_ckpt", "checkpoint/pretrain/MB_lite_models.bin",
        "--checkpoint", "/home/fzliang/Autism-project/data/interim/egohumans_full_extract/train/egohumans_full_extract/best.pt",
        "--imu_stats_json", "/home/fzliang/Autism-project/data/interim/egohumans_full_extract/train/egohumans_full_extract/imu_stats.json",
        "--window_size", "24",
        "--stride", "16",
        "--batch_size", "64",
        "--imu_sensor", "",
        "--repeat_single_sensor", "1",
        "--device", args.device,
        "--save_json", str(args.output_json),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/home/fzliang/Autism-project", check=True)
    print(f"Saved our FrameAcc results to {args.output_json}")


if __name__ == "__main__":
    main()
