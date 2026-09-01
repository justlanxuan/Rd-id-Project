"""Feature contracts used by the G10 global-motion benchmark."""

from .global_motion import (
    AnchorBundle,
    FeatureContractError,
    IMUView,
    derive_trajectory_features,
    extract_global_anchors,
    extract_imu_views,
    spectral_summary,
)
from .imu import (
    CANONICAL_7D_CHANNELS,
    IMUChannelSpec,
    IMUFeatureSpec,
    channel_specs_from_names,
    feature_spec_from_cfg,
    feature_spec_from_config,
    infer_channel_specs,
    parse_feature_channels,
    select_imu_features,
)
from .orientation import (
    OrientationContractError,
    OrientationTrack,
    derive_2d_torso_proxy,
    derive_3d_torso_heading,
    direct_root_orientation,
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
    "CANONICAL_7D_CHANNELS",
    "IMUChannelSpec",
    "IMUFeatureSpec",
    "channel_specs_from_names",
    "feature_spec_from_cfg",
    "feature_spec_from_config",
    "infer_channel_specs",
    "parse_feature_channels",
    "select_imu_features",
]
