"""IMU-guided video encoder and IMU statistical feature utilities."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.encoders.base import BaseEncoder


FEATURE_NAMES: List[str] = ["avg", "med", "var", "lq", "uq", "min", "max"]
AXIS_NAMES: List[str] = ["x", "y", "z", "tot"]
IMU_STATS_DIM = len(FEATURE_NAMES) * len(AXIS_NAMES)  # 28


def _build_segments(signal: torch.Tensor, seg_len: int) -> List[torch.Tensor]:
    """Split [T, C] signal into non-overlapping segments for feature estimation."""
    tlen = int(signal.shape[0])
    if tlen <= seg_len:
        return [signal]

    segments: List[torch.Tensor] = []
    start = 0
    while start + seg_len <= tlen:
        segments.append(signal[start : start + seg_len])
        start += seg_len

    remain = tlen - start
    # Keep a reasonably long tail segment instead of dropping it completely.
    if remain >= max(1, seg_len // 2):
        segments.append(signal[start:tlen])

    if not segments:
        segments.append(signal)
    return segments


def _stats_7(seg: torch.Tensor) -> torch.Tensor:
    """Compute 7 statistics over [T, 4] signal and return [28]."""
    avg = seg.mean(dim=0)
    med = seg.median(dim=0).values
    var = seg.var(dim=0, unbiased=False)
    lq = torch.quantile(seg, q=0.25, dim=0)
    uq = torch.quantile(seg, q=0.75, dim=0)
    mn = seg.min(dim=0).values
    mx = seg.max(dim=0).values
    return torch.stack([avg, med, var, lq, uq, mn, mx], dim=0).reshape(-1)


def compute_imu_stats28_from_imu48(
    imu_48d: torch.Tensor,
    fs_hz: float = 30.0,
    segment_seconds: float = 2.0,
) -> torch.Tensor:
    """Compute 28-dim IMU statistical features from 48D IMU stream.

    Args:
        imu_48d: [T, 48] or [B, T, 48]
        fs_hz: Sampling rate in Hz.
        segment_seconds: Segment duration in seconds for feature estimation.

    Returns:
        [28] if input is [T, 48], else [B, 28].
    """
    if imu_48d.ndim not in (2, 3):
        raise ValueError(f"Expected imu_48d shape [T,48] or [B,T,48], got {tuple(imu_48d.shape)}")

    squeeze_back = False
    if imu_48d.ndim == 2:
        imu_48d = imu_48d.unsqueeze(0)
        squeeze_back = True

    if imu_48d.shape[-1] < 39:
        raise ValueError(f"Expected last dim >= 39 for accelerometer channels, got {imu_48d.shape[-1]}")

    # In single-sensor mode, acceleration is repeated to 4 sensor slots. We use the first slot [36:39].
    acc_xyz = imu_48d[..., 36:39]
    acc_tot = torch.linalg.norm(acc_xyz, ord=2, dim=-1, keepdim=True)
    acc_4 = torch.cat([acc_xyz, acc_tot], dim=-1)  # [B, T, 4]

    seg_len = max(1, int(round(float(fs_hz) * float(segment_seconds))))

    feats = []
    for b in range(acc_4.shape[0]):
        sig = acc_4[b]
        segments = _build_segments(sig, seg_len)
        seg_feats = torch.stack([_stats_7(seg) for seg in segments], dim=0)
        feats.append(seg_feats.mean(dim=0))

    out = torch.stack(feats, dim=0)
    if squeeze_back:
        out = out[0]
    return out


class IMUGuidedVideoEncoder(BaseEncoder):
    """Video encoder trained to regress IMU statistical features."""

    def __init__(
        self,
        backbone: nn.Module,
        rep_dim: int = 512,
        temporal_layers: int = 2,
        pred_hidden_dim: int = 256,
        embed_dim: int = 128,
        pred_dim: int = IMU_STATS_DIM,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.joint_compress = nn.Linear(17 * rep_dim, rep_dim)

        self.temporal_lstm = nn.LSTM(
            input_size=rep_dim,
            hidden_size=rep_dim,
            num_layers=temporal_layers,
            batch_first=True,
        )
        self.pred_head = nn.Sequential(
            nn.Linear(rep_dim, pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(pred_hidden_dim, pred_dim),
        )
        self.embed_head = nn.Linear(rep_dim, embed_dim)

    def encode(self, skeleton_xyz: torch.Tensor) -> torch.Tensor:
        return self.forward(skeleton_xyz)["video_embed"]

    def forward(self, skeleton_xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        # skeleton_xyz: [B, T, 17, 3]
        rep = self.backbone(skeleton_xyz, return_rep=True)  # [B, T, 17, 512]
        bsz, tlen, joints, rep_dim = rep.shape

        frame_rep = self.joint_compress(rep.reshape(bsz * tlen, joints * rep_dim)).reshape(bsz, tlen, rep_dim)
        h_0 = torch.zeros(self.temporal_lstm.num_layers, bsz, rep_dim, device=rep.device)
        c_0 = torch.zeros(self.temporal_lstm.num_layers, bsz, rep_dim, device=rep.device)
        out, _ = self.temporal_lstm(frame_rep, (h_0, c_0))
        last = out[:, -1, :]

        pred_stats = self.pred_head(last)
        video_embed = F.normalize(self.embed_head(last), dim=-1)
        return {
            "imu_feature_pred": pred_stats,
            "video_embed": video_embed,
            "video_hidden": last,
        }
