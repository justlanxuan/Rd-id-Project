#!/usr/bin/env python3
"""
Experiment Note: A3-eval-1imu-full-test
Run the newly trained 1-IMU (right wrist) model on the full EgoHumans test set
using the standard synchronous evaluator.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="/home/fzliang/Autism-project/data/interim/egohumans_full_extract/slice/windows_test.csv")
    parser.add_argument("--checkpoint", default="/home/fzliang/Autism-project/data/interim/egohumans_full_extract/train/egohumans_right_wrist/best.pt")
    parser.add_argument("--imu_stats_json", default="/home/fzliang/Autism-project/data/interim/egohumans_full_extract/train/egohumans_right_wrist/imu_stats.json")
    parser.add_argument("--output_json", default="/home/fzliang/Autism-project/experiments/G_egohumans/E4:single_imu_right_wrist/results/full_test_1imu.json")
    parser.add_argument("--device", default="cuda:0")
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
        "--batch_size", "64",
        "--imu_sensor", "R_LowArm",
        "--repeat_single_sensor", "4",
        "--device", args.device,
        "--save_json", str(args.output_json),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/home/fzliang/Autism-project", check=True)
    print(f"Saved full-test 1-IMU results to {args.output_json}")


if __name__ == "__main__":
    main()
