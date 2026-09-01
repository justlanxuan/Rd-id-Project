"""Canonical sequence transforms selected by the derived-data registry."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from preprocess.common.imu import lowpass_filter_fft, quat_to_rotmat, rotmat_to_quat_wxyz
from preprocess.common.imu_conditioning import enforce_quaternion_continuity

from .contracts import DerivedDataSpec
from .registry import DERIVED_TRANSFORM_REGISTRY

H36M_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
PROTECTED_JOINTS = frozenset({0, 7, 8, 9, 10})
DISTAL_JOINTS = np.asarray([2, 3, 5, 6, 12, 13, 15, 16])
ACC_CHANNELS = ("acc_x", "acc_y", "acc_z")
QUAT_CHANNELS = ("quat_w", "quat_x", "quat_y", "quat_z")


def _copy_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in payload.items()}


def _imu_indices(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    if "imu" not in payload:
        raise KeyError("Derived IMU transform requires an 'imu' array")
    imu = np.asarray(payload["imu"])
    if imu.ndim < 2 or imu.shape[-1] != 7:
        raise ValueError(
            "The selected derived IMU transform requires canonical 7D input "
            f"[acc3 + quat4], got shape {imu.shape}"
        )
    channels = ()
    if "imu_channels" in payload:
        channels = tuple(str(value) for value in np.asarray(payload["imu_channels"]).reshape(-1))
    if channels and len(channels) != 7:
        raise ValueError(f"Canonical 7D IMU declares {len(channels)} channels: {channels}")
    if channels:
        try:
            acc_idx = tuple(channels.index(channel) for channel in ACC_CHANNELS)
            quat_idx = tuple(channels.index(channel) for channel in QUAT_CHANNELS)
        except ValueError as exc:
            raise ValueError(f"Canonical IMU channels must contain {ACC_CHANNELS + QUAT_CHANNELS}, got {channels}") from exc
    else:
        acc_idx = (0, 1, 2)
        quat_idx = (3, 4, 5, 6)
    return imu.astype(np.float32, copy=False), acc_idx, quat_idx


def _enforce_quaternion_continuity(quat: np.ndarray) -> np.ndarray:
    return enforce_quaternion_continuity(quat)


@DERIVED_TRANSFORM_REGISTRY.register("identity")
def identity_transform(
    payload: dict[str, np.ndarray], _rng: np.random.Generator, _spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Copy canonical data without changing its values."""
    return _copy_payload(payload)


@DERIVED_TRANSFORM_REGISTRY.register("imu_acc_noise", aliases=("rc_acc_noise", "a2_2"))
def imu_acc_noise_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Add noise only to acceleration channels of canonical 7D IMU data."""
    imu, acc_idx, _ = _imu_indices(payload)
    std = float(spec.imu_acc_noise_std)
    if std < 0:
        raise ValueError("imu_acc_noise_std must be non-negative")
    output = _copy_payload(payload)
    updated = imu.copy()
    if std > 0:
        noise = rng.normal(0.0, std, size=(*imu.shape[:-1], len(acc_idx))).astype(np.float32)
        updated[..., list(acc_idx)] += noise
    output["imu"] = updated
    return output


@DERIVED_TRANSFORM_REGISTRY.register("imu_acc_lowpass", aliases=("rg_acc_lowpass", "a2_3"))
def imu_acc_lowpass_transform(
    payload: dict[str, np.ndarray], _rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Apply an offline low-pass only to acceleration channels of 7D IMU."""
    imu, acc_idx, _ = _imu_indices(payload)
    cutoff = float(spec.imu_acc_lowpass_cutoff_hz)
    if cutoff < 0 or float(spec.imu_acc_lowpass_fs_hz) <= 0:
        raise ValueError("IMU low-pass cutoff must be non-negative and sampling rate must be positive")
    output = _copy_payload(payload)
    updated = imu.copy()
    if cutoff > 0:
        acceleration = imu[..., list(acc_idx)]
        if acceleration.ndim == 2:
            filtered = lowpass_filter_fft(acceleration, cutoff, spec.imu_acc_lowpass_fs_hz)
        else:
            filtered = np.empty_like(acceleration)
            for person in range(acceleration.shape[1]):
                filtered[:, person] = lowpass_filter_fft(
                    acceleration[:, person], cutoff, spec.imu_acc_lowpass_fs_hz
                )
        updated[..., list(acc_idx)] = filtered
    output["imu"] = updated.astype(np.float32)
    return output


@DERIVED_TRANSFORM_REGISTRY.register("imu_acc_spike", aliases=("rc_acc_spike", "a2_4_spike"))
def imu_acc_spike_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Inject sparse acceleration spikes without corrupting orientation."""
    imu, acc_idx, _ = _imu_indices(payload)
    ratio = float(spec.imu_acc_spike_ratio)
    scale = float(spec.imu_acc_spike_scale)
    output = _copy_payload(payload)
    updated = imu.copy()
    acceleration = updated[..., list(acc_idx)]
    total = acceleration.size
    count = min(total, max(1, int(round(total * ratio)))) if ratio > 0 and total else 0
    if count:
        flat = acceleration.reshape(-1)
        std = max(float(np.std(flat)), 1e-8)
        indices = rng.choice(total, size=count, replace=False)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        flat[indices] += signs.astype(np.float32) * scale * std
    updated[..., list(acc_idx)] = acceleration
    output["imu"] = updated.astype(np.float32)
    return output


@DERIVED_TRANSFORM_REGISTRY.register(
    "imu_acc_dropout_hold", aliases=("rc_acc_dropout", "a2_4_dropout")
)
def imu_acc_dropout_hold_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Hold the previous acceleration sample over sparse dropout intervals."""
    imu, acc_idx, _ = _imu_indices(payload)
    output = _copy_payload(payload)
    updated = imu.copy()
    validity = np.ones(imu.shape[:2], dtype=bool) if imu.ndim == 3 else np.ones(imu.shape[:1], dtype=bool)
    if imu.ndim == 2:
        updated = updated[:, np.newaxis, :]
        validity = np.ones((imu.shape[0], 1), dtype=bool)
        squeeze_person = True
    else:
        squeeze_person = False
    if updated.shape[0] > spec.imu_acc_dropout_duration and spec.imu_acc_dropout_segments:
        for person in range(updated.shape[1]):
            for _ in range(spec.imu_acc_dropout_segments):
                start = int(rng.integers(1, updated.shape[0] - spec.imu_acc_dropout_duration + 1))
                end = start + spec.imu_acc_dropout_duration
                updated[start:end, person][:, list(acc_idx)] = updated[start - 1, person, list(acc_idx)]
                validity[start:end, person] = False
    output["imu"] = updated[:, 0] if squeeze_person else updated
    output["derived_imu_validity"] = validity
    return output


@DERIVED_TRANSFORM_REGISTRY.register("imu_quat_repair", aliases=("rg02",))
def imu_quat_repair_transform(
    payload: dict[str, np.ndarray], _rng: np.random.Generator, _spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Normalize canonical quaternions and repair their temporal sign flips."""
    imu, _acc_idx, quat_idx = _imu_indices(payload)
    output = _copy_payload(payload)
    updated = imu.copy()
    updated[..., list(quat_idx)] = enforce_quaternion_continuity(imu[..., list(quat_idx)])
    output["imu"] = updated.astype(np.float32)
    return output


@DERIVED_TRANSFORM_REGISTRY.register("imu_mount_rotation", aliases=("rc_mount_rotation", "a1"))
def imu_mount_rotation_transform(
    payload: dict[str, np.ndarray], _rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Apply RC-style mount/global orientation changes coherently to 7D IMU.

    The local acceleration is rotated by the inverse mount rotation while the
    orientation is composed as ``R_global * R_base * R_mount``.
    """
    imu, acc_idx, quat_idx = _imu_indices(payload)
    mount = Rotation.from_euler("xyz", spec.imu_mount_euler_xyz_deg, degrees=True).as_matrix().astype(np.float32)
    global_heading = Rotation.from_euler("z", spec.imu_global_yaw_deg, degrees=True).as_matrix().astype(np.float32)

    acc = imu[..., list(acc_idx)]
    quat = imu[..., list(quat_idx)]
    base_rot = quat_to_rotmat(quat)
    augmented_rot = np.einsum("ij,...jk,kl->...il", global_heading, base_rot, mount)
    augmented_quat = _enforce_quaternion_continuity(rotmat_to_quat_wxyz(augmented_rot))
    augmented_acc = np.einsum("ij,...j->...i", mount.T, acc)

    output = _copy_payload(payload)
    updated = imu.copy()
    updated[..., list(acc_idx)] = augmented_acc
    updated[..., list(quat_idx)] = augmented_quat
    output["imu"] = updated.astype(np.float32)
    return output


def _skeleton_keys(payload: dict[str, np.ndarray]) -> Iterable[str]:
    for key in ("gt_skeleton", "gt_skeleton_meters", "extract_skeleton", "skeleton"):
        if key in payload:
            yield key


def _as_people_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(skeleton, dtype=np.float32)
    if values.ndim == 3:
        values = values[:, np.newaxis, ...]
        squeeze_person = True
    elif values.ndim == 4:
        squeeze_person = False
    else:
        raise ValueError(f"Expected skeleton [T,J,3] or [T,P,J,3], got {values.shape}")
    if values.shape[-2:] != (17, 3):
        raise ValueError(f"Expected H36M-17 skeleton [..,17,3], got {values.shape}")
    return values, squeeze_person


def _restore_people_skeleton(values: np.ndarray, squeeze_person: bool) -> np.ndarray:
    return values[:, 0] if squeeze_person else values


def _person_visibility(payload: dict[str, np.ndarray], key: str, shape: tuple[int, int]) -> np.ndarray:
    visibility_key = "extract_visibility" if key.startswith("extract") else "gt_visibility"
    if visibility_key in payload:
        values = np.asarray(payload[visibility_key], dtype=bool)
        if values.shape == shape:
            return values
    return np.ones(shape, dtype=bool)


def _colored_noise(rng: np.random.Generator, shape: tuple[int, int, int], rho: float) -> np.ndarray:
    white = rng.normal(size=shape).astype(np.float32)
    output = np.empty_like(white)
    output[0] = white[0]
    innovation_scale = float(np.sqrt(max(1.0 - rho * rho, 1e-8)))
    for frame in range(1, shape[0]):
        output[frame] = rho * output[frame - 1] + innovation_scale * white[frame]
    return output


def _scale_skeleton(skeleton: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(skeleton, dtype=np.float32)
    if values.ndim == 3:
        values = values[:, np.newaxis, ...]
        squeeze_person = True
    elif values.ndim == 4:
        squeeze_person = False
    else:
        raise ValueError(f"Expected skeleton [T,J,3] or [T,P,J,3], got {values.shape}")
    if values.shape[-2:] != (17, 3):
        raise ValueError(f"Skeleton bone scaling expects H36M-17 [..,17,3], got {values.shape}")
    if scales.shape != (values.shape[1], 17):
        raise ValueError(f"Unexpected bone scale shape {scales.shape} for skeleton {values.shape}")

    output = values.copy()
    for joint, parent in enumerate(H36M_PARENTS):
        if parent < 0:
            continue
        bone = values[:, :, joint] - values[:, :, parent]
        output[:, :, joint] = output[:, :, parent] + bone * scales[None, :, joint, None]
    return output[:, 0] if squeeze_person else output


@DERIVED_TRANSFORM_REGISTRY.register("skeleton_bone_scale", aliases=("rb_s04", "s04"))
def skeleton_bone_scale_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Scale each H36M bone consistently across a sequence and person."""
    keys = tuple(_skeleton_keys(payload))
    if not keys:
        raise KeyError("Derived skeleton transform requires a skeleton array")
    reference = np.asarray(payload[keys[0]])
    if reference.ndim == 3:
        people = 1
    elif reference.ndim == 4:
        people = reference.shape[1]
    else:
        raise ValueError(f"Expected skeleton [T,J,3] or [T,P,J,3], got {reference.shape}")
    scales = rng.uniform(
        spec.skeleton_bone_scale_min,
        spec.skeleton_bone_scale_max,
        size=(people, 17),
    ).astype(np.float32)
    scales[:, 0] = 1.0

    output = _copy_payload(payload)
    for key in keys:
        output[key] = _scale_skeleton(np.asarray(payload[key]), scales)
    return output


@DERIVED_TRANSFORM_REGISTRY.register("skeleton_coord_noise", aliases=("rb_s01", "s01"))
def skeleton_coord_noise_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Add RB-style temporally correlated detector noise to skeleton XY."""
    keys = tuple(_skeleton_keys(payload))
    if not keys:
        raise KeyError("Derived skeleton transform requires a skeleton array")
    sigmas = np.full(17, spec.skeleton_coord_noise_std_mid, dtype=np.float32)
    sigmas[[1, 4, 7, 8, 9, 11, 14]] = spec.skeleton_coord_noise_std_torso
    sigmas[[2, 5, 12, 13, 15, 16]] = spec.skeleton_coord_noise_std_task
    sigmas[[3, 6]] = spec.skeleton_coord_noise_std_distal
    sigmas[0] = 0.0

    output = _copy_payload(payload)
    for key in keys:
        values, squeeze_person = _as_people_skeleton(payload[key])
        noisy = values.copy()
        valid = _person_visibility(payload, key, values.shape[:2])
        for person in range(values.shape[1]):
            noise = _colored_noise(rng, (values.shape[0], 17, 2), spec.skeleton_coord_noise_rho)
            noise *= sigmas[None, :, None]
            active = valid[:, person]
            noisy[active, person, :, :2] += noise[active]
            noisy[:, person, 0] = values[:, person, 0]
        output[key] = _restore_people_skeleton(noisy, squeeze_person).astype(np.float32)
    return output


@DERIVED_TRANSFORM_REGISTRY.register("skeleton_joint_dropout", aliases=("rb_s02", "s02"))
def skeleton_joint_dropout_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Drop short contiguous segments of distal joints and preserve provenance."""
    keys = tuple(_skeleton_keys(payload))
    if not keys:
        raise KeyError("Derived skeleton transform requires a skeleton array")
    if spec.skeleton_joint_dropout_rate <= 0:
        return _copy_payload(payload)
    reference, _ = _as_people_skeleton(payload[keys[0]])
    dropped = np.zeros((reference.shape[0], reference.shape[1], 17), dtype=bool)
    for person in range(reference.shape[1]):
        count = max(1, int(round(reference.shape[0] * spec.skeleton_joint_dropout_rate)))
        for _ in range(count):
            start = int(rng.integers(0, max(reference.shape[0], 1)))
            length = int(
                rng.integers(
                    spec.skeleton_joint_dropout_min_frames,
                    spec.skeleton_joint_dropout_max_frames + 1,
                )
            )
            joint = int(rng.choice(DISTAL_JOINTS))
            stop = min(start + length, reference.shape[0])
            dropped[start:stop, person, joint] = True
            parent = H36M_PARENTS[joint]
            if rng.random() < 0.35 and parent not in PROTECTED_JOINTS:
                dropped[start:stop, person, parent] = True

    output = _copy_payload(payload)
    for key in keys:
        values, squeeze_person = _as_people_skeleton(payload[key])
        updated = values.copy()
        updated[dropped] = 0.0
        output[key] = _restore_people_skeleton(updated, squeeze_person).astype(np.float32)
    output["derived_skeleton_joint_visibility"] = (~dropped).astype(bool)
    return output


@DERIVED_TRANSFORM_REGISTRY.register("skeleton_temporal_jitter", aliases=("rb_s03", "s03"))
def skeleton_temporal_jitter_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Apply bounded per-person temporal shifts without crossing identities."""
    keys = tuple(_skeleton_keys(payload))
    if not keys:
        raise KeyError("Derived skeleton transform requires a skeleton array")
    reference, _ = _as_people_skeleton(payload[keys[0]])
    source_indices = np.empty((reference.shape[0], reference.shape[1]), dtype=np.int64)
    for person in range(reference.shape[1]):
        shift = int(rng.choice(np.asarray([-2, -1, 1, 2])))
        source = np.clip(
            np.arange(reference.shape[0], dtype=np.int64) + shift,
            0,
            max(reference.shape[0] - 1, 0),
        )
        events = max(1, int(round(reference.shape[0] * 0.012))) if reference.shape[0] else 0
        for _ in range(events):
            if reference.shape[0] <= 1:
                break
            frame = int(rng.integers(1, reference.shape[0]))
            neighbor = frame - 1 if rng.random() < 0.5 else min(frame + 1, reference.shape[0] - 1)
            source[frame] = source[neighbor]
        source_indices[:, person] = source

    output = _copy_payload(payload)
    for key in keys:
        values, squeeze_person = _as_people_skeleton(payload[key])
        updated = np.empty_like(values)
        for person in range(values.shape[1]):
            updated[:, person] = values[source_indices[:, person], person]
        output[key] = _restore_people_skeleton(updated, squeeze_person).astype(np.float32)
    for visibility_key in ("gt_visibility", "extract_visibility"):
        if visibility_key in payload:
            visibility = np.asarray(payload[visibility_key], dtype=bool)
            if visibility.shape == source_indices.shape:
                updated = np.empty_like(visibility)
                for person in range(visibility.shape[1]):
                    updated[:, person] = visibility[source_indices[:, person], person]
                output[visibility_key] = updated
    if "gt_to_extract_map" in payload:
        mapping = np.asarray(payload["gt_to_extract_map"], dtype=np.int64)
        if mapping.shape == source_indices.shape:
            updated = np.empty_like(mapping)
            for person in range(mapping.shape[1]):
                updated[:, person] = mapping[source_indices[:, person], person]
            output["gt_to_extract_map"] = updated
    output["derived_skeleton_source_indices"] = source_indices
    return output


@DERIVED_TRANSFORM_REGISTRY.register("skeleton_track_fragmentation", aliases=("rb_s05", "s05"))
def skeleton_track_fragmentation_transform(
    payload: dict[str, np.ndarray], rng: np.random.Generator, spec: DerivedDataSpec
) -> dict[str, np.ndarray]:
    """Create bounded missing-track gaps and emit a person validity mask."""
    keys = tuple(_skeleton_keys(payload))
    if not keys:
        raise KeyError("Derived skeleton transform requires a skeleton array")
    if spec.skeleton_fragmentation_rate <= 0:
        return _copy_payload(payload)
    reference, _ = _as_people_skeleton(payload[keys[0]])
    valid = np.ones(reference.shape[:2], dtype=bool)
    recovery = np.zeros_like(valid)
    for person in range(reference.shape[1]):
        count = max(1, int(round(reference.shape[0] * spec.skeleton_fragmentation_rate)))
        for _ in range(count):
            start = int(rng.integers(0, max(reference.shape[0], 1)))
            length = int(
                rng.integers(
                    spec.skeleton_fragmentation_min_frames,
                    spec.skeleton_fragmentation_max_frames + 1,
                )
            )
            stop = min(start + length, reference.shape[0])
            valid[start:stop, person] = False
            recovery[stop : min(stop + 2, reference.shape[0]), person] = True

        # Keep one clean observation in any otherwise empty 24-frame window.
        for start in range(max(reference.shape[0] - 24 + 1, 0)):
            stop = start + 24
            active = valid[start:stop, person] & np.any(reference[start:stop, person] != 0.0, axis=(1, 2))
            if not active.any():
                candidates = np.flatnonzero(np.any(reference[start:stop, person] != 0.0, axis=(1, 2)))
                if len(candidates):
                    valid[start + int(candidates[len(candidates) // 2]), person] = True

    output = _copy_payload(payload)
    for key in keys:
        values, squeeze_person = _as_people_skeleton(payload[key])
        updated = values.copy()
        updated[~valid] = 0.0
        recovery_active = recovery & valid
        if spec.skeleton_fragmentation_recovery_noise_std > 0:
            noise = rng.normal(
                0.0,
                spec.skeleton_fragmentation_recovery_noise_std,
                size=updated[..., :2].shape,
            ).astype(np.float32)
            updated[..., :2] += noise * recovery_active[:, :, None, None]
            updated[:, :, 0] = values[:, :, 0]
        output[key] = _restore_people_skeleton(updated, squeeze_person).astype(np.float32)
    for visibility_key in ("gt_visibility", "extract_visibility"):
        if visibility_key in payload:
            visibility = np.asarray(payload[visibility_key], dtype=bool)
            if visibility.shape == valid.shape:
                output[visibility_key] = visibility & valid
    if "gt_to_extract_map" in payload:
        mapping = np.asarray(payload["gt_to_extract_map"], dtype=np.int64).copy()
        if mapping.shape == valid.shape:
            mapping[~valid] = -1
            output["gt_to_extract_map"] = mapping
    output["derived_skeleton_validity"] = valid
    return output


__all__ = [
    "ACC_CHANNELS",
    "H36M_PARENTS",
    "QUAT_CHANNELS",
    "identity_transform",
    "imu_acc_noise_transform",
    "imu_acc_lowpass_transform",
    "imu_acc_spike_transform",
    "imu_acc_dropout_hold_transform",
    "imu_mount_rotation_transform",
    "imu_quat_repair_transform",
    "skeleton_coord_noise_transform",
    "skeleton_joint_dropout_transform",
    "skeleton_bone_scale_transform",
    "skeleton_temporal_jitter_transform",
    "skeleton_track_fragmentation_transform",
]
