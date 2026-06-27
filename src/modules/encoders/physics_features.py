"""Batch-wise physics feature extraction for IMU windows.

Extracts time-domain statistics and frequency-domain descriptors from
single-sensor IMU data [B, T, 48] (repeated-sensor format).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def compute_channel_stats_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute 8 statistics for a batch of 1D signals.

    Args:
        x: [B, T] signal

    Returns:
        [B, 8] = [mean, std, max, min, energy, zcr, skew, kurt]
    """
    B, T = x.shape
    if T <= 1:
        return torch.zeros(B, 8, device=x.device, dtype=x.dtype)

    mean = x.mean(dim=1, keepdim=True)  # [B, 1]
    std = x.std(dim=1, keepdim=True) + 1e-12  # [B, 1]
    max_v = x.amax(dim=1, keepdim=True)  # [B, 1]
    min_v = x.amin(dim=1, keepdim=True)  # [B, 1]
    energy = (x ** 2).sum(dim=1, keepdim=True)  # [B, 1]

    # Zero-crossing rate (around mean)
    zero_mean = x - mean
    signs = torch.sign(zero_mean)
    zcr = ((signs[:, 1:] * signs[:, :-1]) < 0).float().sum(dim=1, keepdim=True) / max(T - 1, 1)  # [B, 1]

    # Skewness & kurtosis
    centered = (x - mean) / std
    skew = (centered ** 3).mean(dim=1, keepdim=True)  # [B, 1]
    kurt = (centered ** 4).mean(dim=1, keepdim=True)  # [B, 1]

    return torch.cat([mean, std, max_v, min_v, energy, zcr, skew, kurt], dim=1)  # [B, 8]


def compute_freq_features_torch(
    x: torch.Tensor, fs_hz: float = 30.0, n_fft: int = 64
) -> torch.Tensor:
    """Compute 6 frequency-domain features for a batch of 1D signals.

    Args:
        x: [B, T] signal

    Returns:
        [B, 6] = [dom_freq, low_e, mid_e, high_e, entropy, centroid]
    """
    B, T = x.shape
    device = x.device
    dtype = x.dtype

    if T <= 1:
        return torch.zeros(B, 6, device=device, dtype=dtype)

    # Detrend
    x = x - x.mean(dim=1, keepdim=True)  # [B, T]

    # Check for constant signal
    if torch.allclose(x, torch.zeros_like(x), atol=1e-8):
        return torch.zeros(B, 6, device=device, dtype=dtype)

    n_fft = max(n_fft, T)

    # FFT
    spec = torch.abs(torch.fft.rfft(x, n=n_fft, dim=1))  # [B, F]
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs_hz, device=device)  # [F]

    # PSD
    psd = spec ** 2  # [B, F]
    total = psd.sum(dim=1, keepdim=True) + 1e-12  # [B, 1]

    # Band energy ratios
    low_mask = (freqs >= 0.3) & (freqs <= 2.0)   # [F]
    mid_mask = (freqs > 2.0) & (freqs <= 5.0)    # [F]
    high_mask = (freqs > 5.0) & (freqs <= 15.0)  # [F]

    low_e = psd[:, low_mask].sum(dim=1, keepdim=True) / total  # [B, 1]
    mid_e = psd[:, mid_mask].sum(dim=1, keepdim=True) / total  # [B, 1]
    high_e = psd[:, high_mask].sum(dim=1, keepdim=True) / total  # [B, 1]

    # Dominant frequency
    dom_idx = psd.argmax(dim=1)  # [B]
    dom_freq = freqs[dom_idx].unsqueeze(1)  # [B, 1]

    # Spectral entropy
    psd_norm = psd / total  # [B, F]
    entropy = -(psd_norm * torch.log(psd_norm + 1e-12)).sum(dim=1, keepdim=True)  # [B, 1]

    # Spectral centroid
    freqs_exp = freqs.unsqueeze(0).expand(B, -1)  # [B, F]
    centroid = (freqs_exp * psd).sum(dim=1, keepdim=True) / total  # [B, 1]

    return torch.cat([dom_freq, low_e, mid_e, high_e, entropy, centroid], dim=1)  # [B, 6]


class IMUPhysicsFeatureExtractor(nn.Module):
    """Extract physics feature tokens from raw IMU [B, T, 48].

    Outputs a list of 6 token tensors with varying dimensions:
      token 0: acc_x stats      [B, 8]
      token 1: acc_y stats      [B, 8]
      token 2: acc_z stats      [B, 8]
      token 3: acc_norm stats   [B, 8]
      token 4: acc_norm freq    [B, 6]
      token 5: rot stats        [B, 18]
    """

    def __init__(self, fs_hz: float = 30.0, n_fft: int = 64) -> None:
        super().__init__()
        self.fs_hz = fs_hz
        self.n_fft = n_fft

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args: x [B, T, 48]. Returns: list of [B, D] tensors."""
        acc = x[:, :, 36:39]  # [B, T, 3]  first sensor acceleration copy
        rot = x[:, :, 0:9]    # [B, T, 9]  first sensor rotation copy

        feats: List[torch.Tensor] = []

        # Tokens 0-2: per-axis acceleration stats
        for i in range(3):
            feats.append(compute_channel_stats_torch(acc[:, :, i]))

        # Token 3: acceleration norm stats
        acc_norm = torch.norm(acc, dim=-1)  # [B, T]
        feats.append(compute_channel_stats_torch(acc_norm))

        # Token 4: acceleration norm frequency
        feats.append(compute_freq_features_torch(acc_norm, self.fs_hz, self.n_fft))

        # Token 5: rotation mean + std
        rot_mean = rot.mean(dim=1)  # [B, 9]
        rot_std = rot.std(dim=1)    # [B, 9]
        feats.append(torch.cat([rot_mean, rot_std], dim=-1))  # [B, 18]

        return feats
