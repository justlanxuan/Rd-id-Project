"""Alignment entrypoint for TotalCapture IMU-video data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config, resolve_config
from src.datasets.totalcapture import TotalCaptureAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align TotalCapture IMU and skeleton data into NPZ + CSV")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with an slice section")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--sensor_order", type=str, default=None, help="Comma-separated IMU sensor order")
    parser.add_argument("--train_subjects", type=str, default=None)
    parser.add_argument("--val_subjects", type=str, default=None)
    parser.add_argument("--test_subjects", type=str, default=None)
    parser.add_argument("--max_sequences", type=int, default=None, help="0 means all")
    parser.add_argument("--skeleton_source", type=str, default=None, choices=["vicon", "alphapose"], help="Skeleton data source")
    parser.add_argument("--skeleton_root", type=str, default=None, help="Root directory for AlphaPose skeleton data (if skeleton_source=alphapose)")
    parser.add_argument("--multi_person", action="store_true", help="Treat IMU axis 1 as persons and only emit matched person_idx==imu_idx pairs.")
    return parser.parse_args()


def load_slice_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = resolve_config(config_path)
    slice = data.get("slice", {})
    if slice is None:
        return {}
    if not isinstance(slice, dict):
        raise ValueError(f"Invalid slice section in config: {config_path}")
    return slice


def main() -> None:
    args = parse_args()
    slice_cfg = load_slice_cfg(args.config)

    # Override config with CLI args
    if args.root is not None:
        slice_cfg["root"] = args.root
    if args.out_dir is not None:
        slice_cfg["out_dir"] = args.out_dir
    if args.window_len is not None:
        slice_cfg["window_len"] = args.window_len
    if args.stride is not None:
        slice_cfg["stride"] = args.stride
    if args.sensor_order is not None:
        slice_cfg["sensor_order"] = args.sensor_order
    if args.train_subjects is not None:
        slice_cfg["train_subjects"] = args.train_subjects
    if args.val_subjects is not None:
        slice_cfg["val_subjects"] = args.val_subjects
    if args.test_subjects is not None:
        slice_cfg["test_subjects"] = args.test_subjects
    if args.max_sequences is not None:
        slice_cfg["max_sequences"] = args.max_sequences
    if args.skeleton_source is not None:
        slice_cfg["skeleton_source"] = args.skeleton_source
    if args.skeleton_root is not None:
        slice_cfg["skeleton_root"] = args.skeleton_root
    if args.multi_person:
        slice_cfg["multi_person"] = True

    adapter = TotalCaptureAdapter(slice_cfg)
    adapter.run()


if __name__ == "__main__":
    main()
