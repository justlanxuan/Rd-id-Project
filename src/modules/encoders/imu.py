"""IMU encoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.encoders.base import BaseEncoder
from src.modules.encoders.imu_adapter import (
    AffineIMUAdapter,
    PhysicsAwareIMUAdapter,
    TemporalConvIMUAdapter,
)


class IMUEncoder(BaseEncoder):
    """DeSPITE IMU encoder structure (2-layer LSTM, last-step output)."""

    def __init__(
        self,
        input_size: int = 48,
        hidden_size: int = 512,
        num_layers: int = 2,
        device: str = "cuda",
        adapter_type: str | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.adapter = None
        if adapter_type == "affine":
            self.adapter = AffineIMUAdapter(input_size)
        elif adapter_type == "physics":
            self.adapter = PhysicsAwareIMUAdapter()
        elif adapter_type == "temporal_conv":
            self.adapter = TemporalConvIMUAdapter(input_size)
        elif adapter_type is not None and adapter_type != "none":
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            x = self.adapter(x)
        bsz = x.shape[0]
        h_0 = torch.zeros(self.num_layers, bsz, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, bsz, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return out[:, -1, :].reshape(-1, self.hidden_size)
