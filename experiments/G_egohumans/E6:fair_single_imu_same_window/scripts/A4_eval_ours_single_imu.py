#!/usr/bin/env python3
"""
Experiment Note: A4-eval-ours-single-imu
Evaluate our single-IMU (R_LowArm repeated 4x) 24-frame model on the
MoBInd-aligned test set using synchronous eval.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="/home/fzliang/Autism-project/data/interim/egohumans_mobind_aligned_single_imu_24/slice/windows_test.csv")
    parser.add_argument("--checkpoint", default="/home/fzliang/Autism-project/data/interim/egohumans_mobind_aligned_single_imu_24/train/egohumans_mobind_aligned_single_imu_24/best.pt")
    parser.add_argument("--imu_stats_json", default="/home/fzliang/Autism-project/data/interim/egohumans_mobind_aligned_single_imu_24/train/egohumans_mobind_aligned_single_imu_24/imu_stats.json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--group_windows", type=int, default=1)
    parser.add_argument("--group_vote", action="store_true")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.engine.eval_synchronous",
        "--test_csv", str(args.test_csv),
        "--data_root", str(Path(args.test_csv).parent),
        "--motionbert_root", "/home/fzliang/origin/MotionBERT",
        "--motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml",
        "--motionbert_ckpt", "checkpoint/pretrain/MB_lite_models.bin",
        "--checkpoint", str(args.checkpoint),
        "--imu_stats_json", str(args.imu_stats_json),
        "--window_size", "24",
        "--stride", "16",
        "--group_windows", str(args.group_windows),
        "--batch_size", "64",
        "--imu_sensor", "R_LowArm",
        "--repeat_single_sensor", "4",
        "--device", args.device,
        "--save_json", str(args.output_json),
    ]
    if args.group_vote:
        cmd.append("--group_vote")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/home/fzliang/Autism-project", check=True)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
