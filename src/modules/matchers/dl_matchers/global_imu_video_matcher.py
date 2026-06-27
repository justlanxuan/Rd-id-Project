"""Deep-learning matcher with global motion fusion."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.modules.encoders.imu import IMUEncoder
from src.modules.encoders.global_motion import GlobalVideoEncoder


class GlobalIMUVideoMatcher(nn.Module):
    """IMU-Video cross-modal matching model with global motion encoder."""

    def __init__(
        self,
        imu_encoder: IMUEncoder,
        video_encoder: GlobalVideoEncoder,
    ) -> None:
        super().__init__()
        self.imu_encoder = imu_encoder
        self.video_encoder = video_encoder
        self.global_only = False

    def forward(
        self,
        imu: torch.Tensor,
        skeleton: torch.Tensor,
        root_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_imu = self.imu_encoder(imu)
        if self.global_only:
            if root_trajectory is None:
                raise ValueError("global_only mode requires root_trajectory input")
            z_global = self.video_encoder.global_encoder(root_trajectory)
            return {"imu": z_imu, "video": z_global, "video_global": z_global}

        z_vid, z_local, z_global = self.video_encoder(
            skeleton,
            root_trajectory,
            return_components=True,
        )
        out = {"imu": z_imu, "video": z_vid, "video_local": z_local}
        if z_global is not None:
            out["video_global"] = z_global
        return out
