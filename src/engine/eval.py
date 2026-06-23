"""Evaluate IMU-video alignment model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset, lowpass_filter_fft
from src.engine.common import build_alignment_model, build_loss_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IMU-video alignment model")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--imu_stats_json", type=str, default="")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    parser.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None, help="FFT low-pass cutoff for IMU windows in Hz; set <= 0 to disable.")
    parser.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0, help="Sampling rate used by the IMU low-pass filter.")
    # Physics encoder options
    parser.add_argument("--imu_encoder_type", type=str, default="lstm", choices=["lstm", "physics"])
    parser.add_argument("--physics_d_model", type=int, default=128)
    parser.add_argument("--physics_n_heads", type=int, default=4)
    parser.add_argument("--physics_num_layers", type=int, default=3)
    parser.add_argument("--physics_fs_hz", type=float, default=30.0)
    parser.add_argument("--physics_n_fft", type=int, default=64)
    parser.add_argument("--physics_dropout", type=float, default=0.1)
    # Global motion options (must match training config)
    parser.add_argument("--use_global_motion", action="store_true")
    parser.add_argument("--global_motion_input_dim", type=int, default=2)
    parser.add_argument("--global_motion_hidden_dim", type=int, default=64)
    parser.add_argument("--global_motion_num_layers", type=int, default=2)
    parser.add_argument("--global_motion_dropout", type=float, default=0.1)
    parser.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    parser.add_argument("--global_motion_fusion_type", type=str, default="concat")
    parser.add_argument("--global_motion_fusion_proj", action="store_true")
    parser.add_argument("--global_motion_root_source", type=str, default="auto")
    parser.add_argument("--global_motion_train_only", action="store_true")
    parser.add_argument("--global_motion_aux_weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = torch.tensor(stats["imu_mean"], dtype=torch.float32).cpu().numpy()
        imu_std = torch.tensor(stats["imu_std"], dtype=torch.float32).cpu().numpy()

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    return_root = getattr(args, "use_global_motion", False)
    root_source = getattr(args, "global_motion_root_source", "auto")
    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
        return_root_trajectory=return_root,
        root_source=root_source,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, _ = build_alignment_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    loss_fn = build_loss_fn(temperature=args.temperature, device=device)

    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(device)
            skeleton = batch["skeleton"].to(device)
            forward_kwargs = {"imu": imu, "skeleton": skeleton}
            if "root_trajectory" in batch:
                forward_kwargs["root_trajectory"] = batch["root_trajectory"].to(device)
            out = model(**forward_kwargs)
            total_loss += float(loss_fn(out["imu"], out["video"]).item())
            from src.modules.matchers import retrieval_top1
            total_top1 += retrieval_top1(out["imu"], out["video"])
            n += 1

    metrics = {
        "loss": total_loss / max(n, 1),
        "top1": total_top1 / max(n, 1),
        "num_batches": n,
        "num_windows": len(ds),
    }

    print(json.dumps(metrics, indent=2))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
