"""Deep-learning matcher: encoder pair wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import ModelCapabilities, ModelOutput
from src.modules.domain import DomainClassifier


class IMUVideoMatcher(nn.Module):
    """IMU-Video cross-modal matching model with optional domain adversarial head."""

    def __init__(
        self,
        imu_encoder: nn.Module,
        video_encoder: nn.Module,
        num_domains: int = 0,
        domain_hidden_dim: int = 256,
        pair_head: bool = False,
        pair_hidden_dim: int = 256,
        cross_pair_head: bool = False,
        cross_pair_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.imu_encoder = imu_encoder
        self.video_encoder = video_encoder
        self.num_domains = num_domains
        self.domain_hidden_dim = domain_hidden_dim
        self.capabilities = ModelCapabilities(
            pair_logits=pair_head,
            cross_pair_logits=cross_pair_head,
            domain_logits=num_domains > 0,
        )
        embed_dim = getattr(imu_encoder, "hidden_size", None)
        if embed_dim is None:
            raise ValueError("Cannot infer IMU encoder output dim.")

        self.domain_classifier = None
        if num_domains > 0:
            self.domain_classifier = DomainClassifier(
                input_dim=embed_dim,
                hidden_dim=domain_hidden_dim,
                num_domains=num_domains,
            )

        self.pair_head = None
        if pair_head:
            self.pair_head = nn.Sequential(
                nn.Linear(embed_dim * 4, pair_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(pair_hidden_dim, pair_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(pair_hidden_dim // 2, 1),
            )

        self.cross_pair_head = None
        if cross_pair_head:
            self.cross_pair_head = nn.Sequential(
                nn.Linear(embed_dim * 4, cross_pair_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cross_pair_hidden_dim, cross_pair_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cross_pair_hidden_dim // 2, 1),
            )

    def forward(self, imu: torch.Tensor, skeleton: torch.Tensor) -> ModelOutput:
        z_imu = self.imu_encoder(imu)
        z_vid = self.video_encoder(skeleton)
        out = {"imu": z_imu, "video": z_vid}
        if self.domain_classifier is not None:
            out["domain_logits"] = self.domain_classifier(z_imu)
        return out

    def pair_logits(self, z_imu: torch.Tensor, z_video: torch.Tensor) -> torch.Tensor:
        if self.pair_head is None:
            raise RuntimeError("pair_logits called but pair_head is disabled.")
        n_imu, n_video = z_imu.shape[0], z_video.shape[0]
        imu = z_imu[:, None, :].expand(n_imu, n_video, -1)
        vid = z_video[None, :, :].expand(n_imu, n_video, -1)
        feat = torch.cat([imu, vid, torch.abs(imu - vid), imu * vid], dim=-1)
        return self.pair_head(feat).squeeze(-1)

    def cross_pair_logits(self, imu: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        if self.cross_pair_head is None:
            raise RuntimeError("cross_pair_logits called but cross_pair_head is disabled.")
        imu_seq = self.imu_encoder.forward_sequence(imu)
        video_seq = self.video_encoder.forward_sequence(skeleton)
        if imu_seq.shape[1] != video_seq.shape[1]:
            raise ValueError(f"Temporal length mismatch: {imu_seq.shape} vs {video_seq.shape}")
        n_imu, n_video = imu_seq.shape[0], video_seq.shape[0]
        imu_pair = imu_seq[:, None, :, :].expand(n_imu, n_video, -1, -1)
        vid_pair = video_seq[None, :, :, :].expand(n_imu, n_video, -1, -1)
        feat = torch.cat([imu_pair, vid_pair, torch.abs(imu_pair - vid_pair), imu_pair * vid_pair], dim=-1)
        frame_logits = self.cross_pair_head(feat).squeeze(-1)
        return frame_logits.mean(dim=-1)
