"""Named IMU channel and model-input contracts.

The canonical sequence format stores IMU values as a dense numeric array, but
the last dimension is meaningful only together with its channel catalogue.
This module keeps that catalogue explicit and provides the small amount of
selection logic shared by preprocessing, datasets, models, and checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from preprocess.common.imu import rotmat_to_quat_wxyz

IMU_FEATURE_SCHEMA_VERSION = "imu.feature.v1"
CANONICAL_7D_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)
LEGACY_SENSOR_ORDER = ("L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm")


@dataclass(frozen=True)
class IMUChannelSpec:
    """Semantic metadata for one scalar IMU channel."""

    name: str
    unit: str = "unknown"
    coordinate_frame: str = "unknown"
    source: str = "standardized"
    kind: str = "scalar"
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("IMU channel name must be non-empty")
        if not str(self.unit).strip():
            raise ValueError(f"IMU channel {self.name!r} must declare a unit")
        if not str(self.coordinate_frame).strip():
            raise ValueError(f"IMU channel {self.name!r} must declare a coordinate frame")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["derived_from"] = list(self.derived_from)
        return value


@dataclass(frozen=True)
class IMUFeatureSpec:
    """Ordered model-input contract selected from a canonical IMU catalogue."""

    name: str
    channels: tuple[str, ...]
    normalization: str = "train"
    causal: bool = True
    schema_version: str = IMU_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        channels = tuple(str(channel).strip() for channel in self.channels)
        if not name:
            raise ValueError("IMU feature spec name must be non-empty")
        if not channels or any(not channel for channel in channels):
            raise ValueError(f"IMU feature spec {name!r} must contain non-empty channels")
        if len(set(channels)) != len(channels):
            raise ValueError(f"IMU feature spec {name!r} contains duplicate channels: {channels}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "channels", channels)

    @property
    def input_dim(self) -> int:
        return len(self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "channels": list(self.channels),
            "input_dim": self.input_dim,
            "normalization": self.normalization,
            "causal": self.causal,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def channel_specs_from_names(
    names: Sequence[str],
    *,
    units: Sequence[str] | None = None,
    coordinate_frames: Sequence[str] | None = None,
    source: str = "standardized",
) -> tuple[IMUChannelSpec, ...]:
    """Build channel metadata, using explicit defaults for legacy artifacts."""
    names = tuple(str(name).strip() for name in names)
    if len(set(names)) != len(names):
        raise ValueError(f"IMU channel names must be unique: {names}")
    units = tuple(str(value) for value in units) if units is not None else ("unknown",) * len(names)
    frames = (
        tuple(str(value) for value in coordinate_frames)
        if coordinate_frames is not None
        else ("unknown",) * len(names)
    )
    if len(units) != len(names) or len(frames) != len(names):
        raise ValueError("IMU channel metadata lengths must match channel names")
    return tuple(
        IMUChannelSpec(name, units[index], frames[index], source=source)
        for index, name in enumerate(names)
    )


def infer_channel_specs(names: Sequence[str], *, source: str = "standardized") -> tuple[IMUChannelSpec, ...]:
    """Infer conservative metadata for conventional raw and derived names."""
    specs = []
    for name in names:
        normalized = str(name).strip().lower()
        if normalized.startswith("acc_") or normalized.startswith("acceleration_"):
            unit, frame, kind = "m/s^2", "sensor", "vector_component"
        elif normalized.startswith("gyro_") or normalized.startswith("angular_velocity_"):
            unit, frame, kind = "rad/s", "sensor", "vector_component"
        elif normalized.startswith("quat_"):
            unit, frame, kind = "unitless", "sensor", "quaternion_component"
        elif normalized in {"acc_magnitude", "acc_magnitude_centered"}:
            unit, frame, kind = "m/s^2", "derived", "scalar"
        elif normalized.startswith("acc_d") or normalized == "acc_change_energy":
            unit, frame, kind = "m/s^2", "derived", "scalar"
        elif normalized in {"angular_speed", "ang_speed"}:
            unit, frame, kind = "rad/s", "derived", "scalar"
        else:
            unit, frame, kind = "unknown", "unknown", "scalar"
        specs.append(IMUChannelSpec(str(name), unit, frame, source=source, kind=kind))
    return tuple(specs)


def parse_feature_channels(value: Any) -> tuple[str, ...]:
    """Parse comma-separated or YAML-list feature channel names."""
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise TypeError(f"IMU feature channels must be a string/list, got {type(value).__name__}")


def feature_spec_from_config(
    *,
    view: str = "auto",
    channels: Any = (),
    name: str = "",
    normalization: str = "train",
    causal: bool = True,
) -> IMUFeatureSpec:
    """Resolve explicit channels or one of the stable built-in views."""
    explicit = parse_feature_channels(channels)
    if explicit:
        return IMUFeatureSpec(name or "custom", explicit, normalization, causal)

    normalized_view = str(view or "auto").strip().lower()
    views = {
        "auto": CANONICAL_7D_CHANNELS,
        "canonical_7d": CANONICAL_7D_CHANNELS,
        "legacy_7d": CANONICAL_7D_CHANNELS,
        "acc_quat": CANONICAL_7D_CHANNELS,
        "acc": ("acc_x", "acc_y", "acc_z"),
        "acc_magnitude": ("acc_magnitude", "acc_magnitude_centered"),
        "acc_changes": ("acc_dx", "acc_dy", "acc_dz", "acc_change_energy"),
        "gyro": ("gyro_x", "gyro_y", "gyro_z"),
        "mag": ("mag_x", "mag_y", "mag_z"),
        "acc_gyro": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        "acc_gyro_quat": (
            "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
        ),
        "gyro_quat": ("gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"),
        "legacy_48d": tuple(f"legacy_{index}" for index in range(48)),
    }
    if normalized_view not in views:
        raise ValueError(
            f"Unknown IMU feature view={view!r}; use explicit imu_features or one of {sorted(views)}"
        )
    return IMUFeatureSpec(name or normalized_view, views[normalized_view], normalization, causal)


def _canonical_names_for_array(values: np.ndarray, available_channels: Sequence[str] | None) -> tuple[str, ...]:
    if available_channels is not None:
        channels = tuple(str(value) for value in available_channels)
        if len(channels) != values.shape[-1]:
            raise ValueError(
                f"IMU channel metadata width={len(channels)} does not match values width={values.shape[-1]}"
            )
        if len(set(channels)) != len(channels):
            raise ValueError(f"IMU channel metadata contains duplicates: {channels}")
        return channels
    if values.shape[-1] == len(CANONICAL_7D_CHANNELS):
        return CANONICAL_7D_CHANNELS
    return tuple(f"legacy_{index}" for index in range(values.shape[-1]))


def _legacy_48d_to_7d(values: np.ndarray, sensor_name: str) -> np.ndarray:
    if values.shape[-1] < 48:
        raise ValueError(f"Legacy IMU conversion requires 48 channels, got {values.shape}")
    if sensor_name not in LEGACY_SENSOR_ORDER:
        raise ValueError(
            f"Unsupported legacy IMU sensor={sensor_name!r}; expected one of {LEGACY_SENSOR_ORDER}"
        )
    sensor_index = LEGACY_SENSOR_ORDER.index(sensor_name)
    rotation = values[..., sensor_index * 9 : (sensor_index + 1) * 9].reshape(*values.shape[:-1], 3, 3)
    acceleration = values[..., 36 + sensor_index * 3 : 36 + (sensor_index + 1) * 3]
    quaternion = rotmat_to_quat_wxyz(rotation)
    return np.concatenate([acceleration, quaternion], axis=-1).astype(np.float32)


def select_imu_features(
    values: np.ndarray,
    available_channels: Sequence[str] | None,
    feature_spec: IMUFeatureSpec,
    *,
    legacy_sensor: str = "L_LowArm",
) -> np.ndarray:
    """Select an ordered feature view, with an explicit legacy 48D bridge.

    ``values`` must be ``[T,C]`` at this boundary. Person selection is owned by
    the dataset so this function remains independent of window indexing.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected IMU values [T,C], got {array.shape}")
    channels = _canonical_names_for_array(array, available_channels)
    requested = feature_spec.channels
    if set(requested).issubset(set(channels)):
        indices = [channels.index(channel) for channel in requested]
        return array[:, indices].astype(np.float32, copy=False)

    if array.shape[-1] == 48 and set(requested).issubset(set(CANONICAL_7D_CHANNELS)):
        canonical = _legacy_48d_to_7d(array, legacy_sensor)
        indices = [CANONICAL_7D_CHANNELS.index(channel) for channel in requested]
        return canonical[:, indices]

    missing = [channel for channel in requested if channel not in channels]
    raise ValueError(
        f"IMU feature view {feature_spec.name!r} requires missing channels={missing}; "
        f"available={channels}"
    )


def feature_spec_from_cfg(cfg: Any) -> IMUFeatureSpec:
    """Resolve the training input contract from a YACS or mapping config."""
    if hasattr(cfg, "TRAIN"):
        train = cfg.TRAIN
    elif isinstance(cfg, Mapping):
        train = cfg.get("TRAIN", cfg.get("train", {}))
    else:
        train = {}

    def read(name: str, default: Any) -> Any:
        if isinstance(train, Mapping):
            return train.get(name, train.get(name.lower(), default))
        return getattr(train, name, default)

    view = read("IMU_FEATURE_VIEW", "auto")
    channels = read("IMU_FEATURES", ())
    name = read("IMU_FEATURE_NAME", "")
    normalization = read("IMU_FEATURE_NORMALIZATION", "train")
    causal = read("IMU_FEATURE_CAUSAL", True)
    return feature_spec_from_config(
        view=str(view),
        channels=channels,
        name=str(name),
        normalization=str(normalization),
        causal=bool(causal),
    )


__all__ = [
    "CANONICAL_7D_CHANNELS",
    "IMUChannelSpec",
    "IMUFeatureSpec",
    "IMU_FEATURE_SCHEMA_VERSION",
    "LEGACY_SENSOR_ORDER",
    "channel_specs_from_names",
    "feature_spec_from_cfg",
    "feature_spec_from_config",
    "parse_feature_channels",
    "select_imu_features",
    "infer_channel_specs",
]
