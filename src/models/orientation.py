"""Official registry adapter for extractor-derived orientation matching."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import ModelCapabilities
from src.models.orientation_matcher import OrientationAwareMatcher


def build_orientation_model(cfg: Any, device: torch.device) -> OrientationAwareMatcher:
    orientation_cfg = cfg.TRAIN.ORIENTATION
    model = OrientationAwareMatcher(
        skeleton_dim=51,
        imu_dim=6,
        orientation_dim=5,
        hidden=int(orientation_cfg.HIDDEN),
        embedding_dim=int(orientation_cfg.EMBEDDING_DIM),
        temporal_mode=str(orientation_cfg.TEMPORAL_MODE),
        multiscale_fusion=str(orientation_cfg.MULTISCALE_FUSION),
        window_seconds=float(orientation_cfg.WINDOW_SECONDS) if orientation_cfg.WINDOW_SECONDS else None,
        use_orientation=bool(orientation_cfg.ENABLED),
        fusion_mode=str(orientation_cfg.FUSION),
        use_layer_norm=False,
    )
    # Orientation datasets perform their own per-window IMU normalization and
    # are intentionally independent from the legacy 7D hybrid statistics.
    model.capabilities = ModelCapabilities(
        external_imu_normalization=False,
        fitted_input_stats=False,
        full_validation_batch=False,
        segment_frame_acc=False,
        preferred_validation_metric="val_loss",
        requires_orientation=True,
    )
    # Keep the model registry contract explicit even though the orientation
    # branch has no hybrid pair/domain heads.
    return model.to(device)


__all__ = ["OrientationAwareMatcher", "build_orientation_model"]
