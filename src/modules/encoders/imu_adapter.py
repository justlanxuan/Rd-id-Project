"""IMU input adapters for distribution alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineIMUAdapter(nn.Module):
    """Element-wise affine transformation: x * scale + shift.

    Parameters: 96 (48 scales + 48 shifts).
    Equivalent to learning per-feature mean/std offset.
    """

    def __init__(self, feat_dim: int = 48) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feat_dim))
        self.shift = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, feat_dim]
        return x * self.scale + self.shift


class PhysicsAwareIMUAdapter(nn.Module):
    """Physics-aware adapter respecting IMU structure.

    Input has 4 sensor slots, each with 9D rotation + 3D acceleration.
    Applies independent transform to rotation and acceleration modalities.

    Rotation: 9x9 mixing matrix + per-element scale/shift.
    Acceleration: 3x3 mixing matrix + per-element scale/shift.
    Parameters: ~138 (9*9 + 9 + 9 + 3*3 + 3 + 3).
    """

    def __init__(
        self,
        num_slots: int = 4,
        rot_dim: int = 9,
        acc_dim: int = 3,
        init_identity: bool = True,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.rot_dim = rot_dim
        self.acc_dim = acc_dim
        self.slot_dim = rot_dim + acc_dim

        # Rotation transform: mixing matrix + affine
        self.rot_mix = nn.Parameter(torch.eye(rot_dim))
        self.rot_scale = nn.Parameter(torch.ones(rot_dim))
        self.rot_shift = nn.Parameter(torch.zeros(rot_dim))

        # Acceleration transform: mixing matrix + affine
        self.acc_mix = nn.Parameter(torch.eye(acc_dim))
        self.acc_scale = nn.Parameter(torch.ones(acc_dim))
        self.acc_shift = nn.Parameter(torch.zeros(acc_dim))

        if not init_identity:
            nn.init.xavier_uniform_(self.rot_mix)
            nn.init.xavier_uniform_(self.acc_mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, num_slots * (rot_dim + acc_dim)]
        B, T, total_dim = x.shape
        x = x.reshape(B, T, self.num_slots, self.slot_dim)

        rot = x[..., : self.rot_dim]  # [B, T, 4, 9]
        acc = x[..., self.rot_dim :]  # [B, T, 4, 3]

        # Apply rotation transform (shared across slots)
        rot = rot @ self.rot_mix  # [B, T, 4, 9]
        rot = rot * self.rot_scale + self.rot_shift

        # Apply acceleration transform (shared across slots)
        acc = acc @ self.acc_mix  # [B, T, 4, 3]
        acc = acc * self.acc_scale + self.acc_shift

        x = torch.cat([rot, acc], dim=-1)  # [B, T, 4, 12]
        return x.reshape(B, T, total_dim)


class TemporalConvIMUAdapter(nn.Module):
    """Temporal convolution adapter with residual connection.

    Captures local temporal noise/drift patterns.
    Parameters: ~14K for feat_dim=48, hidden=48, kernel=3.
    """

    def __init__(
        self,
        feat_dim: int = 48,
        hidden_dim: int = 48,
        kernel_size: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-length padding"
        padding = kernel_size // 2

        layers = []
        in_ch = feat_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else feat_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, feat_dim]
        x = x.transpose(1, 2)  # [B, feat_dim, T]
        out = self.convs(x)  # [B, feat_dim, T]
        out = out + x  # residual
        return out.transpose(1, 2)  # [B, T, feat_dim]
