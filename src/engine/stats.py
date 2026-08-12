"""Statistics helpers for training."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import WindowAlignmentDataset, lowpass_filter_fft
from src.modules.encoders.hybrid import imu_sequence_features, raw_pose_sequence, skeleton_tokens


def read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_imu_stats_from_train_csv(
    train_csv: str,
    data_root: str | None,
    imu_lowpass_cutoff_hz: float | None = None,
    imu_lowpass_fs_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(train_csv)
    base = Path(data_root) if data_root else Path(train_csv).resolve().parent

    per_source: dict[tuple[str, int], list] = defaultdict(lambda: [None, None, 0])

    for row in rows:
        rel = row["npz_path"]
        imu_idx = int(row.get("imu_idx", 0))
        key = (rel, imu_idx)
        if per_source[key][2] > 0:
            continue

        data = np.load((base / rel).resolve(), allow_pickle=True)
        imu = data["imu"].astype(np.float64)
        if imu.ndim == 3:
            imu = imu[:, imu_idx, :]

        if imu_lowpass_cutoff_hz is not None and imu_lowpass_cutoff_hz > 0:
            imu = lowpass_filter_fft(imu, imu_lowpass_cutoff_hz, imu_lowpass_fs_hz)

        per_source[key][0] = imu.sum(axis=0)
        per_source[key][1] = (imu * imu).sum(axis=0)
        per_source[key][2] = imu.shape[0]

    total_count = sum(v[2] for v in per_source.values())
    if total_count == 0:
        raise ValueError("No IMU frames found while computing stats.")

    sums = np.zeros_like(list(per_source.values())[0][0])
    sq_sums = np.zeros_like(list(per_source.values())[0][1])
    for s, sq, _count in per_source.values():
        sums += s
        sq_sums += sq

    mean = sums / total_count
    var = np.maximum(sq_sums / total_count - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fit_tensor_stats(x: torch.Tensor, dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    mu = x.mean(dim=dims, keepdim=True)
    sd = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return mu, sd


def fit_hybrid_encoder_stats(model: torch.nn.Module, dataset: WindowAlignmentDataset, batch_size: int = 64) -> None:
    """Fit E21-style raw/vector/IMU normalization buffers on the train split."""
    video_encoder = getattr(model, "video_encoder", None)
    imu_encoder = getattr(model, "imu_encoder", None)
    if video_encoder is None or imu_encoder is None:
        return
    if not all(hasattr(video_encoder, k) for k in ("raw_mu", "raw_sd", "vec_mu", "vec_sd")):
        return
    if not all(hasattr(imu_encoder, k) for k in ("imu_mu", "imu_sd")):
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_parts = []
    vec_parts = []
    imu_parts = []
    skel_smooth = int(getattr(video_encoder, "skeleton_smooth_kernel", 9))
    image_height = float(getattr(video_encoder, "image_height", 1080.0))
    image_width = float(getattr(video_encoder, "image_width", 1920.0))
    imu_smooth = int(getattr(imu_encoder, "imu_smooth_kernel", 5))
    imu_feature_mode = str(getattr(imu_encoder, "feature_mode", "raw"))

    with torch.no_grad():
        for batch in loader:
            skel = batch["skeleton"].float()
            imu = batch["imu"].float()
            raw_parts.append(raw_pose_sequence(skel, skel_smooth, image_height, image_width))
            vec_parts.append(skeleton_tokens(skel, skel_smooth, image_height, image_width))
            imu_parts.append(imu_sequence_features(imu, imu_smooth, imu_feature_mode))

    raw = torch.cat(raw_parts, dim=0)
    vec = torch.cat(vec_parts, dim=0)
    imu = torch.cat(imu_parts, dim=0)
    raw_mu, raw_sd = _fit_tensor_stats(raw, (0, 1))
    vec_mu, vec_sd = _fit_tensor_stats(vec, (0, 1, 2))
    imu_mu, imu_sd = _fit_tensor_stats(imu, (0, 1))

    video_encoder.raw_mu.copy_(raw_mu.to(video_encoder.raw_mu.device))
    video_encoder.raw_sd.copy_(raw_sd.to(video_encoder.raw_sd.device))
    video_encoder.vec_mu.copy_(vec_mu.to(video_encoder.vec_mu.device))
    video_encoder.vec_sd.copy_(vec_sd.to(video_encoder.vec_sd.device))
    imu_encoder.imu_mu.copy_(imu_mu.to(imu_encoder.imu_mu.device))
    imu_encoder.imu_sd.copy_(imu_sd.to(imu_encoder.imu_sd.device))
    print("[INFO] Fitted hybrid encoder stats on train split.")
