"""Deep-learning matcher: encoder pair wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.modules.domain import DomainClassifier
from src.modules.encoders.imu import IMUEncoder
from src.modules.encoders.video import VideoEncoder


class IMUVideoMatcher(nn.Module):
    """IMU-Video cross-modal matching model with optional domain adversarial head."""

    def __init__(
        self,
        imu_encoder: IMUEncoder,
        video_encoder: VideoEncoder,
        num_domains: int = 0,
        domain_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.imu_encoder = imu_encoder
        self.video_encoder = video_encoder
        self.num_domains = num_domains
        self.domain_hidden_dim = domain_hidden_dim

        self.domain_classifier = None
        if num_domains > 0:
            embed_dim = getattr(imu_encoder, "hidden_size", None)
            if embed_dim is None:
                # Fallback: infer from the last layer's output dimension
                if hasattr(imu_encoder, "lstm"):
                    embed_dim = imu_encoder.lstm.hidden_size
                else:
                    raise ValueError("Cannot infer IMU encoder output dim for domain classifier.")
            self.domain_classifier = DomainClassifier(
                input_dim=embed_dim,
                hidden_dim=domain_hidden_dim,
                num_domains=num_domains,
            )

    def forward(
        self, imu: torch.Tensor, skeleton: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_imu = self.imu_encoder(imu)
        z_vid = self.video_encoder(skeleton)
        out = {"imu": z_imu, "video": z_vid}
        if self.domain_classifier is not None:
            out["domain_logits"] = self.domain_classifier(z_imu)
        return out
