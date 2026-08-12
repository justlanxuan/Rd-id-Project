"""Builder for the current hybrid IMU/skeleton matcher."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import ModelCapabilities
from src.modules.encoders import HybridIMUEncoder, HybridSkeletonEncoder
from src.modules.matchers import IMUVideoMatcher


def build_hybrid_model(cfg: Any, device: torch.device) -> IMUVideoMatcher:
    model_cfg = cfg.TRAIN.MODEL
    hidden = int(model_cfg.HYBRID_HIDDEN)
    imu_encoder = HybridIMUEncoder(
        hidden_size=hidden,
        imu_smooth_kernel=int(model_cfg.HYBRID_IMU_SMOOTH),
        feature_mode=str(model_cfg.HYBRID_IMU_FEATURE_MODE),
        temporal_layers=int(model_cfg.HYBRID_TEMPORAL_LAYERS),
        temporal_kernel=int(model_cfg.HYBRID_TEMPORAL_KERNEL),
        temporal_mode=str(model_cfg.HYBRID_TEMPORAL_MODE),
        dropout=float(model_cfg.HYBRID_DROPOUT),
    )
    video_encoder = HybridSkeletonEncoder(
        hidden_size=hidden,
        skeleton_smooth_kernel=int(model_cfg.HYBRID_SKELETON_SMOOTH),
        image_height=float(model_cfg.HYBRID_IMAGE_HEIGHT),
        image_width=float(model_cfg.HYBRID_IMAGE_WIDTH),
        token_layers=int(model_cfg.HYBRID_TOKEN_LAYERS),
        token_heads=int(model_cfg.HYBRID_TOKEN_HEADS),
        temporal_layers=int(model_cfg.HYBRID_TEMPORAL_LAYERS),
        temporal_kernel=int(model_cfg.HYBRID_TEMPORAL_KERNEL),
        temporal_mode=str(model_cfg.HYBRID_TEMPORAL_MODE),
        feature_mode=str(model_cfg.HYBRID_SKELETON_FEATURE_MODE),
        dropout=float(model_cfg.HYBRID_DROPOUT),
    )
    model = IMUVideoMatcher(
        imu_encoder=imu_encoder,
        video_encoder=video_encoder,
        num_domains=int(model_cfg.NUM_DOMAINS),
        domain_hidden_dim=int(model_cfg.DOMAIN_HIDDEN_DIM),
        pair_head=bool(model_cfg.PAIR_HEAD),
        pair_hidden_dim=int(model_cfg.PAIR_HIDDEN_DIM),
        cross_pair_head=bool(model_cfg.CROSS_PAIR_HEAD),
        cross_pair_hidden_dim=int(model_cfg.CROSS_PAIR_HIDDEN_DIM),
    )
    model.capabilities = ModelCapabilities(
        pair_logits=bool(model_cfg.PAIR_HEAD),
        cross_pair_logits=bool(model_cfg.CROSS_PAIR_HEAD),
        domain_logits=int(model_cfg.NUM_DOMAINS) > 0,
        fitted_input_stats=True,
        full_validation_batch=True,
        segment_frame_acc=True,
        preferred_validation_metric="val_loss",
    )
    return model.to(device)
