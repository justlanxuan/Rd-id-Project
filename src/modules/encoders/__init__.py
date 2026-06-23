"""Encoders module for IMU and video modalities."""

from src.modules.encoders.base import BaseEncoder
from src.modules.encoders.imu import IMUEncoder
from src.modules.encoders.physics_imu import PhysicsIMUEncoder
from src.modules.encoders.video import VideoEncoder
from src.modules.encoders.global_motion import GlobalMotionEncoder, GlobalVideoEncoder
from src.modules.encoders.imu_guided_video import (
    IMUGuidedVideoEncoder,
    compute_imu_stats28_from_imu48,
    IMU_STATS_DIM,
)
from src.modules.encoders.utils import (
    build_motionbert_backbone,
    load_motionbert_checkpoint,
    load_despite_imu_weights,
    resolve_checkpoint_path,
)

__all__ = [
    "BaseEncoder",
    "IMUEncoder",
    "VideoEncoder",
    "GlobalMotionEncoder",
    "GlobalVideoEncoder",
    "IMUGuidedVideoEncoder",
    "compute_imu_stats28_from_imu48",
    "IMU_STATS_DIM",
    "build_motionbert_backbone",
    "load_motionbert_checkpoint",
    "load_despite_imu_weights",
    "resolve_checkpoint_path",
]
