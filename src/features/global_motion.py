"""Explicit global-motion and IMU feature contracts for G10.

The functions in this module deliberately operate on NumPy arrays and do not
know about a dataset directory, a model, or a training split.  A caller must
provide semantic joint names, timestamps, visibility and IMU channel names.
This prevents the common failure mode where a 17-joint array from one source
is silently interpreted using another source's joint order.

The contract distinguishes sensor-proximal anchors (left wrist/elbow/shoulder)
from body-global anchors (pelvis, centroids, bbox and robust transform).  A
wrist trajectory is therefore allowed in the global benchmark, but its
provenance says that it can contain local arm motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "g10.global_motion.v1"
ANCHOR_IDS = (
    "A0_left_wrist",
    "A1_left_elbow",
    "A2_left_shoulder",
    "A3_shoulder_midpoint",
    "A4_pelvis",
    "A5_upper_centroid",
    "A6_full_centroid",
    "A7_bbox_center",
    "A8_robust_similarity_transform",
)

COCO17_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
H36M17_JOINTS = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


class FeatureContractError(ValueError):
    """Raised when a feature input violates the explicit G10 contract."""


@dataclass(frozen=True)
class AnchorBundle:
    """Global anchor trajectories and their validity masks.

    ``trajectories`` maps each anchor ID to ``[T, D]``.  A8 is a transform
    descriptor ``[translation_x, translation_y, log_scale, rotation]`` and is
    consequently always two-dimensional, while the other anchors preserve the
    input coordinate dimension.
    """

    schema_version: str
    coordinate_space: str
    joint_layout: str
    trajectories: Mapping[str, np.ndarray]
    validity: Mapping[str, np.ndarray]
    auxiliary: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class IMUView:
    """One explicit IMU input view with channel provenance."""

    view_id: str
    values: np.ndarray
    channels: tuple[str, ...]
    validity: np.ndarray
    sensor_location: str
    provenance: str


def _as_float_array(value: np.ndarray | Sequence[float], *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise FeatureContractError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise FeatureContractError(f"{name} contains non-finite values")
    return array


def _validate_timestamps(timestamps: np.ndarray | Sequence[float], length: int) -> np.ndarray:
    ts = _as_float_array(timestamps, name="timestamps", ndim=1)
    if len(ts) != length:
        raise FeatureContractError(f"timestamps length={len(ts)} does not match T={length}")
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        raise FeatureContractError("timestamps must be strictly increasing in seconds")
    if len(ts) > 1 and float(np.median(np.diff(ts))) <= 0:
        raise FeatureContractError("timestamps have no positive sampling interval")
    return ts


def _normalise_joint_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _joint_index(joint_names: Sequence[str], candidates: Iterable[str]) -> int | None:
    normalised = {_normalise_joint_name(name): idx for idx, name in enumerate(joint_names)}
    for candidate in candidates:
        idx = normalised.get(_normalise_joint_name(candidate))
        if idx is not None:
            return idx
    return None


def _resolve_layout(joint_names: Sequence[str]) -> str:
    names = {_normalise_joint_name(name) for name in joint_names}
    if names == {_normalise_joint_name(name) for name in COCO17_JOINTS}:
        return "coco17"
    if names == {_normalise_joint_name(name) for name in H36M17_JOINTS}:
        return "h36m17"
    return "named"


def _visibility_mask(
    skeleton: np.ndarray,
    visibility: np.ndarray | None,
    confidence: np.ndarray | None,
) -> np.ndarray:
    t_len, n_joints, _ = skeleton.shape
    valid = np.isfinite(skeleton).all(axis=-1)
    if visibility is not None:
        vis = np.asarray(visibility, dtype=bool)
        if vis.shape != (t_len, n_joints):
            raise FeatureContractError(
                f"visibility must have shape {(t_len, n_joints)}, got {vis.shape}"
            )
        valid &= vis
    if confidence is not None:
        conf = _as_float_array(confidence, name="confidence", ndim=2)
        if conf.shape != (t_len, n_joints):
            raise FeatureContractError(
                f"confidence must have shape {(t_len, n_joints)}, got {conf.shape}"
            )
        valid &= conf > 0
    return valid


def _aggregate_points(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    min_visible: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_len, _, dims = points.shape
    output = np.zeros((t_len, dims), dtype=np.float64)
    output_valid = valid.sum(axis=1) >= min_visible
    for frame in np.flatnonzero(output_valid):
        mask = valid[frame]
        if weights is None:
            output[frame] = points[frame, mask].mean(axis=0)
        else:
            frame_weights = np.asarray(weights[frame, mask], dtype=np.float64)
            frame_weights = np.maximum(frame_weights, 0.0)
            total = float(frame_weights.sum())
            if total <= 0:
                output_valid[frame] = False
            else:
                output[frame] = np.average(points[frame, mask], axis=0, weights=frame_weights)
    return output, output_valid


def _similarity_descriptor(
    skeleton: np.ndarray,
    valid: np.ndarray,
    *,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate frame-to-frame 2D translation, log-scale and rotation."""
    if skeleton.shape[-1] != 2:
        raise FeatureContractError("A8 robust similarity transform requires 2D coordinates")
    t_len = skeleton.shape[0]
    result = np.zeros((t_len, 4), dtype=np.float64)
    result_valid = np.zeros(t_len, dtype=bool)
    for frame in range(1, t_len):
        common = valid[frame] & valid[frame - 1]
        if int(common.sum()) < min_points:
            continue
        previous = skeleton[frame - 1, common]
        current = skeleton[frame, common]
        # Coordinate-wise medians and a residual trim keep one locally moving
        # wrist/elbow from turning a body-global transform into an arm feature.
        previous_center = np.median(previous, axis=0)
        current_center = np.median(current, axis=0)
        previous_zero = previous - previous_center
        current_zero = current - current_center
        previous_norm = float(np.linalg.norm(previous_zero))
        current_norm = float(np.linalg.norm(current_zero))
        if previous_norm <= 1e-12 or current_norm <= 1e-12:
            continue
        scale = current_norm / previous_norm
        dot = float(np.sum(previous_zero * current_zero))
        cross = float(np.sum(previous_zero[:, 0] * current_zero[:, 1] - previous_zero[:, 1] * current_zero[:, 0]))
        angle = np.arctan2(cross, dot)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotated = np.stack(
            [previous_zero[:, 0] * cos_angle - previous_zero[:, 1] * sin_angle,
             previous_zero[:, 0] * sin_angle + previous_zero[:, 1] * cos_angle],
            axis=-1,
        ) * scale
        residual = np.linalg.norm(current_zero - rotated, axis=-1)
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        keep = residual <= median_residual + 3.0 * max(mad, 1e-9)
        if int(keep.sum()) >= min_points and not np.all(keep):
            previous_zero = previous_zero[keep]
            current_zero = current_zero[keep]
            previous_norm = float(np.linalg.norm(previous_zero))
            current_norm = float(np.linalg.norm(current_zero))
            if previous_norm <= 1e-12 or current_norm <= 1e-12:
                continue
            scale = current_norm / previous_norm
            dot = float(np.sum(previous_zero * current_zero))
            cross = float(np.sum(previous_zero[:, 0] * current_zero[:, 1] - previous_zero[:, 1] * current_zero[:, 0]))
            angle = np.arctan2(cross, dot)
        # Median point displacement is robust to one locally moving joint and
        # is the translation quantity exposed by the global contract.
        result[frame, 0:2] = np.median(current - previous, axis=0)
        result[frame, 2] = np.log(max(scale, 1e-12))
        result[frame, 3] = angle
        result_valid[frame] = True
    return result, result_valid


def extract_global_anchors(
    skeleton: np.ndarray,
    *,
    joint_names: Sequence[str],
    visibility: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    coordinate_space: str = "unknown",
    min_visible: int = 1,
    min_transform_points: int = 3,
) -> AnchorBundle:
    """Extract A0–A8 with explicit semantic joint names.

    ``skeleton`` is ``[T, J, D]`` with D=2 or 3.  A missing required semantic
    joint is a contract error instead of silently producing a zero trajectory.
    """
    values = _as_float_array(skeleton, name="skeleton", ndim=3)
    t_len, n_joints, dims = values.shape
    if dims not in (2, 3):
        raise FeatureContractError(f"skeleton coordinate dimension must be 2 or 3, got {dims}")
    if len(joint_names) != n_joints:
        raise FeatureContractError(f"joint_names length={len(joint_names)} does not match J={n_joints}")
    if min_visible <= 0:
        raise FeatureContractError("min_visible must be positive")
    valid = _visibility_mask(values, visibility, confidence)

    def required(label: str, *names: str) -> int:
        idx = _joint_index(joint_names, names)
        if idx is None:
            raise FeatureContractError(f"joint layout lacks required {label}; tried {names}")
        return idx

    left_wrist = required("left wrist", "left_wrist", "LeftWrist", "L_Wrist", "Lwrist")
    left_elbow = required("left elbow", "left_elbow", "LeftElbow", "L_Elbow", "Lelbow")
    left_shoulder = required("left shoulder", "left_shoulder", "LeftShoulder", "L_Shoulder", "Lshoulder")
    right_shoulder = required("right shoulder", "right_shoulder", "RightShoulder", "R_Shoulder", "Rshoulder")
    left_hip = required("left hip", "left_hip", "LeftHip", "L_Hip", "Lhip")
    right_hip = required("right hip", "right_hip", "RightHip", "R_Hip", "Rhip")

    trajectories: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    auxiliary: dict[str, np.ndarray] = {}

    def single(anchor_id: str, index: int) -> None:
        trajectories[anchor_id] = values[:, index].copy()
        masks[anchor_id] = valid[:, index].copy()

    single("A0_left_wrist", left_wrist)
    single("A1_left_elbow", left_elbow)
    single("A2_left_shoulder", left_shoulder)

    shoulder_points = values[:, [left_shoulder, right_shoulder]]
    shoulder_valid = valid[:, [left_shoulder, right_shoulder]]
    trajectories["A3_shoulder_midpoint"], masks["A3_shoulder_midpoint"] = _aggregate_points(
        shoulder_points, shoulder_valid, min_visible=2
    )

    pelvis_index = _joint_index(joint_names, ("pelvis", "root", "hip", "Hip"))
    if pelvis_index is not None:
        pelvis_points = values[:, [pelvis_index]]
        pelvis_valid = valid[:, [pelvis_index]]
        trajectories["A4_pelvis"], masks["A4_pelvis"] = _aggregate_points(
            pelvis_points, pelvis_valid, min_visible=1
        )
    else:
        hip_points = values[:, [left_hip, right_hip]]
        hip_valid = valid[:, [left_hip, right_hip]]
        trajectories["A4_pelvis"], masks["A4_pelvis"] = _aggregate_points(
            hip_points, hip_valid, min_visible=2
        )

    upper_indices = [left_shoulder, right_shoulder, left_elbow, left_wrist]
    upper_points = values[:, upper_indices]
    upper_valid = valid[:, upper_indices]
    trajectories["A5_upper_centroid"], masks["A5_upper_centroid"] = _aggregate_points(
        upper_points, upper_valid, min_visible=min(2, len(upper_indices))
    )
    trajectories["A6_full_centroid"], masks["A6_full_centroid"] = _aggregate_points(
        values, valid, min_visible=min_visible
    )

    bbox_valid = valid.any(axis=1)
    bbox = np.zeros((t_len, 4), dtype=np.float64)
    for frame in np.flatnonzero(bbox_valid):
        points = values[frame, valid[frame], :2]
        low = points.min(axis=0)
        high = points.max(axis=0)
        bbox[frame] = [low[0], low[1], high[0], high[1]]
    trajectories["A7_bbox_center"] = ((bbox[:, 0:2] + bbox[:, 2:4]) / 2.0).astype(np.float64)
    masks["A7_bbox_center"] = bbox_valid
    auxiliary["A7_bbox_size"] = np.stack([bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]], axis=-1)

    trajectories["A8_robust_similarity_transform"], masks["A8_robust_similarity_transform"] = _similarity_descriptor(
        values[:, :, :2], valid, min_points=min_transform_points
    )
    return AnchorBundle(
        schema_version=SCHEMA_VERSION,
        coordinate_space=str(coordinate_space),
        joint_layout=_resolve_layout(joint_names),
        trajectories=trajectories,
        validity=masks,
        auxiliary=auxiliary,
    )


def _difference(
    values: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(values, dtype=np.float64)
    output_valid = np.zeros(len(values), dtype=bool)
    if len(values) <= 1:
        return output, output_valid
    dt = np.diff(timestamps)
    pair_valid = valid[1:] & valid[:-1] & (dt > 0)
    output[1:][pair_valid] = (values[1:] - values[:-1])[pair_valid] / dt[pair_valid, None]
    output_valid[1:] = pair_valid
    return output, output_valid


def derive_trajectory_features(
    trajectory: np.ndarray,
    timestamps: np.ndarray | Sequence[float],
    *,
    validity: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Derive position, displacement, derivatives and energy with SI-time units."""
    values = _as_float_array(trajectory, name="trajectory", ndim=2)
    ts = _validate_timestamps(timestamps, len(values))
    valid = np.isfinite(values).all(axis=-1) if validity is None else np.asarray(validity, dtype=bool)
    if valid.shape != (len(values),):
        raise FeatureContractError(f"validity must have shape {(len(values),)}, got {valid.shape}")
    if not valid.all():
        values = np.where(valid[:, None], values, 0.0)

    features: dict[str, np.ndarray] = {"position": values.copy()}
    masks: dict[str, np.ndarray] = {"position": valid.copy()}
    first = int(np.flatnonzero(valid)[0]) if valid.any() else None
    displacement = np.zeros_like(values)
    if first is not None:
        displacement[valid] = values[valid] - values[first]
    features["displacement"] = displacement
    masks["displacement"] = valid.copy()

    velocity, velocity_valid = _difference(values, ts, valid)
    acceleration, acceleration_valid = _difference(velocity, ts, velocity_valid)
    jerk, jerk_valid = _difference(acceleration, ts, acceleration_valid)
    for name, array, mask in (
        ("velocity", velocity, velocity_valid),
        ("acceleration", acceleration, acceleration_valid),
        ("jerk", jerk, jerk_valid),
    ):
        features[name] = array
        masks[name] = mask
    features["speed"] = np.linalg.norm(velocity, axis=-1)
    masks["speed"] = velocity_valid.copy()
    features["energy"] = np.sum(velocity * velocity, axis=-1)
    masks["energy"] = velocity_valid.copy()
    return features, masks


def spectral_summary(
    signal: np.ndarray,
    timestamps: np.ndarray | Sequence[float],
    *,
    validity: np.ndarray | None = None,
    band_hz: tuple[float, float] = (0.0, 4.5),
    allow_source_native: bool = False,
) -> dict[str, float | bool]:
    """Return deterministic PSD summaries on the valid contiguous samples."""
    values = _as_float_array(signal, name="signal", ndim=2)
    ts = _validate_timestamps(timestamps, len(values))
    valid = np.isfinite(values).all(axis=-1) if validity is None else np.asarray(validity, dtype=bool)
    if valid.shape != (len(values),):
        raise FeatureContractError(f"validity must have shape {(len(values),)}, got {valid.shape}")
    indices = np.flatnonzero(valid)
    if len(indices) < 4:
        return {"valid": False, "dominant_hz": 0.0, "band_energy": 0.0, "spectral_entropy": 0.0, "periodicity": 0.0}
    # A single contiguous run avoids inventing dynamics across missing frames.
    splits = np.flatnonzero(np.diff(indices) > 1)
    runs = np.split(indices, splits + 1)
    run = max(runs, key=len)
    if len(run) < 4:
        return {"valid": False, "dominant_hz": 0.0, "band_energy": 0.0, "spectral_entropy": 0.0, "periodicity": 0.0}
    x = values[run]
    dt = float(np.median(np.diff(ts[run])))
    if dt <= 0:
        raise FeatureContractError("timestamps have non-positive median dt")
    centered = x - x.mean(axis=0, keepdims=True)
    # Aggregate power across coordinate channels rather than taking a norm
    # first.  ``abs(sin)`` would otherwise double a 2 Hz component to 4 Hz.
    channel_spectra = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    spectrum = channel_spectra.sum(axis=-1)
    frequencies = np.fft.rfftfreq(len(x), d=dt)
    spectrum[0] = 0.0
    low, high = band_hz
    if low < 0 or high <= low:
        raise FeatureContractError(f"invalid frequency band {band_hz}")
    if high > 4.5 and not allow_source_native:
        raise FeatureContractError(
            "frequency band above the G10 common 4.5 Hz limit requires allow_source_native=True"
        )
    band = (frequencies >= low) & (frequencies <= high)
    band_power = spectrum[band]
    total = float(band_power.sum())
    if total <= 0:
        return {"valid": True, "dominant_hz": 0.0, "band_energy": 0.0, "spectral_entropy": 0.0, "periodicity": 0.0}
    probabilities = band_power / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / np.log(max(len(probabilities), 2)))
    principal_channel = int(np.argmax(np.var(centered, axis=0)))
    scalar = centered[:, principal_channel]
    autocorr = np.correlate(scalar, scalar, mode="full")[len(scalar) - 1:]
    periodicity = float(autocorr[1:].max() / max(autocorr[0], 1e-12)) if len(autocorr) > 1 else 0.0
    dominant_index = int(np.flatnonzero(band)[np.argmax(band_power)])
    return {
        "valid": True,
        "dominant_hz": float(frequencies[dominant_index]),
        "band_energy": total / len(scalar),
        "spectral_entropy": entropy,
        "periodicity": periodicity,
    }


def _channel_index(channel_names: Sequence[str], *candidates: str) -> int | None:
    normalised = {_normalise_joint_name(name): idx for idx, name in enumerate(channel_names)}
    for candidate in candidates:
        if _normalise_joint_name(candidate) in normalised:
            return normalised[_normalise_joint_name(candidate)]
    return None


def extract_imu_views(
    imu: np.ndarray,
    timestamps: np.ndarray | Sequence[float],
    *,
    channel_names: Sequence[str],
    sensor_location: str,
    provenance: str,
) -> dict[str, IMUView]:
    """Build explicit I0–I9-compatible IMU views from named channels."""
    values = _as_float_array(imu, name="imu", ndim=2)
    if len(channel_names) != values.shape[1]:
        raise FeatureContractError(f"channel_names length={len(channel_names)} does not match C={values.shape[1]}")
    _validate_timestamps(timestamps, len(values))
    if not str(sensor_location).strip() or not str(provenance).strip():
        raise FeatureContractError("sensor_location and provenance are required")

    def required_group(prefix: str, suffixes: Sequence[str]) -> list[int]:
        result = []
        for suffix in suffixes:
            index = _channel_index(channel_names, f"{prefix}_{suffix}", f"{prefix}{suffix}")
            if index is None:
                raise FeatureContractError(f"IMU channel group lacks {prefix}_{suffix}")
            result.append(index)
        return result

    acc_idx = required_group("acc", ("x", "y", "z"))
    acc = values[:, acc_idx]
    quat_idx: list[int] | None = None
    try:
        quat_idx = required_group("quat", ("w", "x", "y", "z"))
    except FeatureContractError:
        pass
    if quat_idx is not None:
        quat_raw = values[:, quat_idx]
        quat_norm = np.linalg.norm(quat_raw, axis=-1)
        quat_valid = np.isfinite(quat_norm) & (quat_norm > 1e-8)
        quat = np.zeros_like(quat_raw)
        quat[quat_valid] = quat_raw[quat_valid] / quat_norm[quat_valid, None]
    else:
        quat = np.zeros((len(values), 4), dtype=np.float64)
        quat_valid = np.zeros(len(values), dtype=bool)
    gyro_idx: list[int] | None = None
    try:
        gyro_idx = required_group("gyro", ("x", "y", "z"))
    except FeatureContractError:
        pass
    gyro = values[:, gyro_idx] if gyro_idx is not None else None
    common_valid = np.isfinite(acc).all(axis=-1)
    views: dict[str, IMUView] = {}

    def add(view_id: str, array: np.ndarray, channels: Sequence[str], valid: np.ndarray) -> None:
        views[view_id] = IMUView(view_id, np.asarray(array, dtype=np.float64), tuple(channels), np.asarray(valid, dtype=bool), str(sensor_location), str(provenance))

    add("I0_acc", acc, ("acc_x", "acc_y", "acc_z"), common_valid)
    magnitude = np.linalg.norm(acc, axis=-1, keepdims=True)
    centered = magnitude - magnitude[common_valid].mean(axis=0, keepdims=True) if common_valid.any() else magnitude
    add("I1_acc_magnitude", np.concatenate([magnitude, centered], axis=-1), ("acc_magnitude", "acc_magnitude_centered"), common_valid)
    acc_delta = np.zeros_like(acc)
    if len(acc) > 1:
        dt = np.diff(_validate_timestamps(timestamps, len(values)))
        pair = common_valid[1:] & common_valid[:-1]
        acc_delta[1:][pair] = (acc[1:] - acc[:-1])[pair] / dt[pair, None]
    energy = np.sum(acc_delta * acc_delta, axis=-1, keepdims=True)
    add("I2_acc_changes", np.concatenate([acc_delta, energy], axis=-1), ("acc_dx", "acc_dy", "acc_dz", "acc_change_energy"), common_valid)
    if gyro is not None:
        gyro_valid = np.isfinite(gyro).all(axis=-1)
        add("I3_gyro", gyro, ("gyro_x", "gyro_y", "gyro_z"), gyro_valid)
        add("I5_acc_gyro", np.concatenate([acc, gyro], axis=-1), ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"), common_valid & gyro_valid)
    else:
        gyro_valid = np.zeros(len(values), dtype=bool)
    if quat_idx is not None:
        add("I4_delta_quaternion", np.concatenate([quat, np.zeros((len(quat), 1))], axis=-1), ("quat_w", "quat_x", "quat_y", "quat_z", "angular_speed"), quat_valid)
    if quat_idx is not None and len(quat) > 1:
        dt = np.diff(_validate_timestamps(timestamps, len(values)))
        dot = np.abs(np.sum(quat[1:] * quat[:-1], axis=-1)).clip(0.0, 1.0)
        angular_speed = np.zeros(len(quat), dtype=np.float64)
        pair = quat_valid[1:] & quat_valid[:-1]
        angular_speed[1:][pair] = 2.0 * np.arccos(dot[pair]) / dt[pair]
        views["I4_delta_quaternion"] = IMUView("I4_delta_quaternion", np.concatenate([quat, angular_speed[:, None]], axis=-1), ("quat_w", "quat_x", "quat_y", "quat_z", "angular_speed"), quat_valid, str(sensor_location), str(provenance))
    if quat_idx is not None:
        add("I6_acc_quat", np.concatenate([acc, quat], axis=-1), ("acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"), common_valid & quat_valid)
        if gyro is not None:
            add("I7_acc_gyro_quat", np.concatenate([acc, gyro, quat], axis=-1), ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"), common_valid & gyro_valid & quat_valid)
    return views
