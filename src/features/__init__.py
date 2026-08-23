"""Feature contracts used by the G10 global-motion benchmark."""

from .orientation import (
    OrientationContractError,
    OrientationTrack,
    derive_2d_torso_proxy,
    derive_3d_torso_heading,
    direct_root_orientation,
)

from .global_motion import (
    AnchorBundle,
    FeatureContractError,
    IMUView,
    derive_trajectory_features,
    extract_global_anchors,
    extract_imu_views,
    spectral_summary,
)

__all__ = [
    "AnchorBundle",
    "FeatureContractError",
    "IMUView",
    "derive_trajectory_features",
    "extract_global_anchors",
    "extract_imu_views",
    "spectral_summary",
    "OrientationContractError",
    "OrientationTrack",
    "derive_2d_torso_proxy",
    "derive_3d_torso_heading",
    "direct_root_orientation",
]
