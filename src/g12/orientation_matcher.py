"""Orientation-aware two-stream matcher for the G12 E4 controlled ablation."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.g10.global_encoder import TemporalEncoder, _safe_normalize


class OrientationAwareMatcher(nn.Module):
    """Keep the original skeleton/IMU streams and add an explicit turning branch."""

    def __init__(
        self,
        skeleton_dim: int,
        imu_dim: int,
        orientation_dim: int = 5,
        hidden: int = 96,
        embedding_dim: int = 64,
        temporal_mode: str = "multiscale",
        multiscale_fusion: str = "hierarchical_attention",
        window_seconds: float | None = None,
        use_orientation: bool = True,
        fusion_mode: str = "gate",
        onset_bins: int = 8,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_orientation = bool(use_orientation)
        if fusion_mode not in {"gate", "concat", "gyro_focus", "cross", "conditional_cross", "residual"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")
        self.fusion_mode = fusion_mode
        self.temporal_mode = str(temporal_mode)
        self.onset_bins = int(onset_bins)
        if self.onset_bins < 2:
            raise ValueError("onset_bins must be >=2")
        self.skeleton_encoder = TemporalEncoder(
            skeleton_dim, hidden, embedding_dim, mode=temporal_mode,
            multiscale_fusion=multiscale_fusion, window_seconds=window_seconds, use_layer_norm=use_layer_norm,
        )
        self.imu_encoder = TemporalEncoder(
            imu_dim, hidden, embedding_dim, mode=temporal_mode,
            multiscale_fusion=multiscale_fusion, window_seconds=window_seconds, use_layer_norm=use_layer_norm,
        )
        if self.use_orientation:
            self.orientation_encoder = TemporalEncoder(
                orientation_dim, hidden, embedding_dim, mode=temporal_mode,
                multiscale_fusion=multiscale_fusion, window_seconds=window_seconds, use_layer_norm=use_layer_norm,
            )
            self.turning_gate = nn.Sequential(
                nn.Linear(embedding_dim, max(8, embedding_dim // 2)),
                nn.GELU(),
                nn.Linear(max(8, embedding_dim // 2), 1),
                nn.Sigmoid(),
            ) if fusion_mode == "gate" else None
            self.skeleton_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.turning_activity_head = nn.Linear(embedding_dim, 1)
            self.orientation_onset_head = nn.Linear(embedding_dim, self.onset_bins)
            if fusion_mode in {"gyro_focus", "cross", "conditional_cross", "residual"}:
                self.gyro_encoder = TemporalEncoder(
                    3, hidden, embedding_dim, mode=temporal_mode,
                    multiscale_fusion=multiscale_fusion, window_seconds=window_seconds, use_layer_norm=use_layer_norm,
                )
                self.imu_fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                self.gyro_turning_gate = nn.Sequential(
                    nn.Linear(embedding_dim, max(8, embedding_dim // 2)),
                    nn.GELU(),
                    nn.Linear(max(8, embedding_dim // 2), 1),
                    nn.Sigmoid(),
                )
                self.gyro_onset_head = nn.Linear(embedding_dim, self.onset_bins)

    def forward(self, skeleton: torch.Tensor, imu: torch.Tensor, orientation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        skeleton_embedding = self.skeleton_encoder(skeleton)
        imu_embedding = self.imu_encoder(imu)
        output = {"imu": imu_embedding}
        if not self.use_orientation:
            output["skeleton"] = skeleton_embedding
            output["video"] = skeleton_embedding
            return output
        if orientation is None:
            raise ValueError("orientation-aware matcher requires orientation features")
        orientation_embedding = self.orientation_encoder(orientation)
        gate = self.turning_gate(orientation_embedding) if self.turning_gate is not None else torch.ones(
            (orientation_embedding.shape[0], 1), device=orientation_embedding.device, dtype=orientation_embedding.dtype
        )
        fused = self.skeleton_fusion(torch.cat([skeleton_embedding, orientation_embedding * gate], dim=-1))
        output["skeleton"] = _safe_normalize(fused)
        if self.fusion_mode == "conditional_cross":
            gyro_embedding = self.gyro_encoder(imu[..., 3:6])
            skeleton_activity_gate = orientation[..., 4].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            imu_activity_gate = self.gyro_turning_gate(gyro_embedding)
            skeleton_delta = self.skeleton_fusion(torch.cat([skeleton_embedding, orientation_embedding], dim=-1))
            imu_delta = self.imu_fusion(torch.cat([imu_embedding, gyro_embedding], dim=-1))
            output["skeleton"] = _safe_normalize(skeleton_embedding + skeleton_activity_gate * skeleton_delta)
            output["imu"] = _safe_normalize(imu_embedding + imu_activity_gate * imu_delta)
            output["imu_turning_gate"] = imu_activity_gate
            output["gyro_onset_logits"] = self.gyro_onset_head(gyro_embedding)
            gate = skeleton_activity_gate
        elif self.fusion_mode in {"gyro_focus", "cross"}:
            gyro_embedding = self.gyro_encoder(imu[..., 3:6])
            output["imu"] = _safe_normalize(self.imu_fusion(torch.cat([imu_embedding, gyro_embedding], dim=-1)))
            output["gyro_onset_logits"] = self.gyro_onset_head(gyro_embedding)
        elif self.fusion_mode == "residual":
            gyro_embedding = self.gyro_encoder(imu[..., 3:6])
            output["skeleton"] = _safe_normalize(skeleton_embedding + 0.3 * orientation_embedding)
            output["imu"] = _safe_normalize(imu_embedding + 0.3 * gyro_embedding)
            output["gyro_onset_logits"] = self.gyro_onset_head(gyro_embedding)
        output["orientation_embedding"] = orientation_embedding
        output["turning_activity_pred"] = self.turning_activity_head(orientation_embedding).squeeze(-1)
        output["orientation_onset_logits"] = self.orientation_onset_head(orientation_embedding)
        output["turning_gate"] = gate
        # ``video`` is the stable official matcher key.  Keep ``skeleton`` as
        # the historical G12 spelling so old reports/checkpoints remain usable.
        output["video"] = output["skeleton"]
        return output
