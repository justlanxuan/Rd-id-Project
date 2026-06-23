"""Matchers module for cross-modal association and alignment."""

from src.modules.matchers.base import BaseMatcher
from src.modules.matchers.hungarian import HungarianMatcher
from src.modules.matchers.losses import SymmetricInfoNCE, retrieval_top1

# Deep-learning matchers
from src.modules.matchers.dl_matchers.imu_video_matcher import IMUVideoMatcher
from src.modules.matchers.dl_matchers.global_imu_video_matcher import GlobalIMUVideoMatcher
from src.modules.matchers.dl_matchers.despite_matcher import DeSPITEMatcher
from src.modules.matchers.physics_matchers import FrequencyPhysicsMatcher, FrequencyConfig

# Encoders (backward-compatible re-exports)
from src.modules.encoders.imu import IMUEncoder
from src.modules.encoders.video import VideoEncoder
from src.modules.encoders.utils import (
    build_motionbert_backbone,
    load_motionbert_checkpoint,
    load_despite_imu_weights,
)

__all__ = [
    "BaseMatcher",
    "HungarianMatcher",
    "SymmetricInfoNCE",
    "retrieval_top1",
    "IMUVideoMatcher",
    "GlobalIMUVideoMatcher",
    "DeSPITEMatcher",
    "FrequencyPhysicsMatcher",
    "FrequencyConfig",
    "IMUEncoder",
    "VideoEncoder",
    "build_motionbert_backbone",
    "load_motionbert_checkpoint",
    "load_despite_imu_weights",
]
