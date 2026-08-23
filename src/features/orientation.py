"""Extractor-first orientation contracts for G12.

This module is intentionally a pure NumPy utility.  It does not alter the
canonical ``extract_skeleton`` schema and never treats a 2-D padded z channel
as depth.  Callers must state the joint layout and coordinate-frame semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .global_motion import H36M17_JOINTS

SCHEMA_VERSION = "g12.orientation_contract.v1"


class OrientationContractError(ValueError):
    """Raised when an orientation input violates the explicit contract."""


@dataclass(frozen=True)
class OrientationTrack:
    """Frame-level orientation features plus explicit validity/provenance."""

    schema_version: str
    orientation_source: str
    orientation_kind: Literal["direct", "3d_derived", "2d_proxy"]
    coordinate_frame: str
    angle_period: float
    angle: np.ndarray
    angle_sin_cos: np.ndarray
    angle_rate: np.ndarray
    orientation_6d: np.ndarray
    direction: np.ndarray
    orientation_valid: np.ndarray
    rate_valid: np.ndarray
    degeneracy_reason: np.ndarray
    quaternion_continuous: np.ndarray | None = None


def _finite_array(value: np.ndarray | Sequence[float], *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise OrientationContractError(f"{name} must have ndim={ndim}, got {array.shape}")
    if not np.isfinite(array).all():
        raise OrientationContractError(f"{name} contains non-finite values")
    return array


def _timestamps(value: np.ndarray | Sequence[float], length: int) -> np.ndarray:
    ts = _finite_array(value, name="timestamps", ndim=1)
    if len(ts) != length:
        raise OrientationContractError(f"timestamps length={len(ts)} does not match T={length}")
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        raise OrientationContractError("timestamps must be strictly increasing")
    return ts


def _skeleton(value: np.ndarray | Sequence[float]) -> np.ndarray:
    skeleton = _finite_array(value, name="skeleton")
    if skeleton.ndim != 3 or skeleton.shape[-1] not in (2, 3):
        raise OrientationContractError(f"skeleton must be [T,J,2/3], got {skeleton.shape}")
    return skeleton


def _joint_indices(joint_names: Sequence[str], names: tuple[str, ...]) -> tuple[int, ...]:
    normalised = {"".join(ch for ch in str(name).lower() if ch.isalnum()): i for i, name in enumerate(joint_names)}
    indices = []
    for name in names:
        index = normalised.get("".join(ch for ch in name.lower() if ch.isalnum()))
        if index is None:
            raise OrientationContractError(f"joint layout lacks required joint {name!r}")
        indices.append(index)
    return tuple(indices)


def _validity(skeleton: np.ndarray, visibility: np.ndarray | None, confidence: np.ndarray | None) -> np.ndarray:
    valid = np.isfinite(skeleton).all(axis=-1)
    if visibility is not None:
        mask = np.asarray(visibility, dtype=bool)
        if mask.shape != valid.shape:
            raise OrientationContractError(f"visibility must have shape {valid.shape}, got {mask.shape}")
        valid &= mask
    if confidence is not None:
        scores = _finite_array(confidence, name="confidence", ndim=2)
        if scores.shape != valid.shape:
            raise OrientationContractError(f"confidence must have shape {valid.shape}, got {scores.shape}")
        valid &= scores > 0
    return valid


def _unit(vector: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vector, axis=-1)
    valid = norms > eps
    result = np.zeros_like(vector, dtype=np.float64)
    result[valid] = vector[valid] / norms[valid, None]
    return result, valid


def _periodic_unwrap(angle: np.ndarray, valid: np.ndarray, period: float) -> np.ndarray:
    output = np.zeros_like(angle, dtype=np.float64)
    start = 0
    while start < len(angle):
        while start < len(angle) and not valid[start]:
            start += 1
        end = start
        while end < len(angle) and valid[end]:
            end += 1
        if end > start:
            segment = angle[start:end]
            if period == np.pi:
                output[start:end] = 0.5 * np.unwrap(2.0 * segment)
            else:
                output[start:end] = np.unwrap(segment)
        start = end
    return output


def _angle_rate(angle: np.ndarray, timestamps: np.ndarray, valid: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = _periodic_unwrap(angle, valid, period)
    rate = np.zeros_like(angle, dtype=np.float64)
    rate_valid = np.zeros_like(valid, dtype=bool)
    start = 0
    while start < len(angle):
        while start < len(angle) and not valid[start]:
            start += 1
        end = start
        while end < len(angle) and valid[end]:
            end += 1
        if end - start >= 2:
            rate[start:end] = np.gradient(unwrapped[start:end], timestamps[start:end])
            rate_valid[start:end] = True
        start = end
    return rate, rate_valid


def _track(
    *,
    angle: np.ndarray,
    direction: np.ndarray,
    orientation_6d: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    source: str,
    kind: Literal["direct", "3d_derived", "2d_proxy"],
    frame: str,
    period: float,
    reasons: np.ndarray,
    quaternion_continuous: np.ndarray | None = None,
) -> OrientationTrack:
    rate, rate_valid = _angle_rate(angle, timestamps, valid, period)
    sin_cos = np.stack([np.sin(2.0 * angle), np.cos(2.0 * angle)], axis=-1) if period == np.pi else np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    sin_cos[~valid] = 0.0
    return OrientationTrack(
        schema_version=SCHEMA_VERSION,
        orientation_source=source,
        orientation_kind=kind,
        coordinate_frame=frame,
        angle_period=period,
        angle=np.where(valid, angle, 0.0),
        angle_sin_cos=sin_cos,
        angle_rate=rate,
        orientation_6d=orientation_6d,
        direction=direction,
        orientation_valid=valid,
        rate_valid=rate_valid,
        degeneracy_reason=reasons,
        quaternion_continuous=quaternion_continuous,
    )


def derive_2d_torso_proxy(
    skeleton: np.ndarray,
    timestamps: np.ndarray | Sequence[float],
    *,
    joint_names: Sequence[str] = H36M17_JOINTS,
    visibility: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    coordinate_frame: str = "image_xy",
    min_axis_length: float = 1e-6,
    orientation_source: str = "extractor_2d_torso_axis",
) -> OrientationTrack:
    """Return an unoriented shoulder-line angle (period pi) for 2-D output.

    The line is ``right_shoulder - left_shoulder``.  Because a line has no
    front/back sign, this is explicitly a proxy, not world yaw.
    """
    points = _skeleton(skeleton)
    if points.shape[-1] != 2:
        raise OrientationContractError("2D proxy requires skeleton[...,2]")
    ts = _timestamps(timestamps, len(points))
    left, right = _joint_indices(joint_names, ("left_shoulder", "right_shoulder"))
    valid_joints = _validity(points, visibility, confidence)
    axis = points[:, right] - points[:, left]
    valid = valid_joints[:, left] & valid_joints[:, right]
    direction, length_valid = _unit(axis, min_axis_length)
    valid &= length_valid
    angle = np.arctan2(direction[:, 1], direction[:, 0])
    reasons = np.full(len(points), "", dtype="U32")
    reasons[~valid_joints[:, left] | ~valid_joints[:, right]] = "missing_shoulder"
    reasons[valid_joints[:, left] & valid_joints[:, right] & ~length_valid] = "degenerate_shoulder_axis"
    return _track(
        angle=angle,
        direction=direction,
        orientation_6d=np.zeros((len(points), 6)),
        valid=valid,
        timestamps=ts,
        source=orientation_source,
        kind="2d_proxy",
        frame=coordinate_frame,
        period=np.pi,
        reasons=reasons,
    )


def derive_3d_torso_heading(
    skeleton: np.ndarray,
    timestamps: np.ndarray | Sequence[float],
    *,
    joint_names: Sequence[str] = H36M17_JOINTS,
    visibility: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    up_axis: int = 1,
    cross_order: Literal["lateral_cross_up", "up_cross_lateral"] = "lateral_cross_up",
    coordinate_frame: str = "extractor_3d_unspecified_up_y",
    min_vector_length: float = 1e-6,
    orientation_source: str = "extractor_3d_torso_heading",
) -> OrientationTrack:
    """Derive a coordinate-dependent torso heading from 3-D joints.

    ``lateral = right_shoulder-left_shoulder`` and
    ``up = thorax-pelvis``.  The caller must state the up axis and cross order;
    changing either can flip the heading and therefore changes provenance.
    """
    points = _skeleton(skeleton)
    if points.shape[-1] != 3:
        raise OrientationContractError("3D heading requires skeleton[...,3]")
    if up_axis not in (0, 1, 2):
        raise OrientationContractError("up_axis must be 0, 1, or 2")
    ts = _timestamps(timestamps, len(points))
    left, right, pelvis, thorax = _joint_indices(points_names := joint_names, ("left_shoulder", "right_shoulder", "pelvis", "thorax"))
    valid_joints = _validity(points, visibility, confidence)
    lateral = points[:, right] - points[:, left]
    up = points[:, thorax] - points[:, pelvis]
    lateral_u, lateral_valid = _unit(lateral, min_vector_length)
    up_u, up_valid = _unit(up, min_vector_length)
    if cross_order == "lateral_cross_up":
        forward = np.cross(lateral_u, up_u)
    else:
        forward = np.cross(up_u, lateral_u)
    direction, forward_valid = _unit(forward, min_vector_length)
    valid = valid_joints[:, left] & valid_joints[:, right] & valid_joints[:, pelvis] & valid_joints[:, thorax]
    valid &= lateral_valid & up_valid & forward_valid
    horizontal_axes = tuple(axis for axis in range(3) if axis != up_axis)
    projected = direction[:, horizontal_axes]
    horizontal_norm = np.linalg.norm(projected, axis=-1)
    valid &= horizontal_norm > min_vector_length
    angle = np.arctan2(projected[:, 1], projected[:, 0])
    reasons = np.full(len(points), "", dtype="U40")
    reasons[~valid_joints[:, left] | ~valid_joints[:, right]] = "missing_shoulders"
    reasons[~valid_joints[:, pelvis] | ~valid_joints[:, thorax]] = "missing_pelvis_thorax"
    reasons[valid_joints[:, left] & valid_joints[:, right] & ~lateral_valid] = "degenerate_lateral"
    reasons[valid_joints[:, pelvis] & valid_joints[:, thorax] & ~up_valid] = "degenerate_up"
    reasons[~forward_valid] = "degenerate_forward"
    reasons[valid & (horizontal_norm <= min_vector_length)] = "vertical_heading"
    return _track(
        angle=angle,
        direction=direction,
        orientation_6d=np.zeros((len(points), 6)),
        valid=valid,
        timestamps=ts,
        source=orientation_source,
        kind="3d_derived",
        frame=coordinate_frame,
        period=2.0 * np.pi,
        reasons=reasons,
    )


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = _finite_array(axis_angle, name="axis_angle", ndim=2)
    if vectors.shape[1] != 3:
        raise OrientationContractError(f"axis_angle must be [T,3], got {vectors.shape}")
    theta = np.linalg.norm(vectors, axis=1)
    result = np.broadcast_to(np.eye(3), (len(vectors), 3, 3)).copy()
    nonzero = theta > 1e-12
    unit = np.zeros_like(vectors)
    unit[nonzero] = vectors[nonzero] / theta[nonzero, None]
    x, y, z = unit.T
    K = np.zeros((len(vectors), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    sin_theta = np.sin(theta)[:, None, None]
    one_minus_cos = (1.0 - np.cos(theta))[:, None, None]
    outer = unit[:, :, None] * unit[:, None, :]
    result = np.cos(theta)[:, None, None] * np.eye(3) + sin_theta * K + one_minus_cos * outer
    result[~nonzero] = np.eye(3)
    return result


def _quaternion_to_matrix(quaternion: np.ndarray, order: Literal["wxyz", "xyzw"]) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_array(quaternion, name="quaternion", ndim=2)
    if values.shape[1] != 4:
        raise OrientationContractError(f"quaternion must be [T,4], got {values.shape}")
    if order == "xyzw":
        x, y, z, w = values.T
    else:
        w, x, y, z = values.T
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-12
    x, y, z, w = x / np.where(valid, norms, 1.0), y / np.where(valid, norms, 1.0), z / np.where(valid, norms, 1.0), w / np.where(valid, norms, 1.0)
    matrix = np.empty((len(values), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    matrix[~valid] = 0.0
    return matrix, valid


def _continuous_quaternion(quaternion: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = quaternion.copy()
    for index in range(1, len(output)):
        if valid[index] and valid[index - 1] and float(np.dot(output[index], output[index - 1])) < 0:
            output[index] *= -1.0
    return output


def direct_root_orientation(
    timestamps: np.ndarray | Sequence[float],
    *,
    axis_angle: np.ndarray | None = None,
    quaternion: np.ndarray | None = None,
    quaternion_order: Literal["wxyz", "xyzw"] = "wxyz",
    local_forward_axis: int = 2,
    local_forward_sign: float = 1.0,
    up_axis: int = 1,
    coordinate_frame: str = "direct_orientation_frame",
    orientation_source: str = "extractor_raw_root_orientation",
    min_horizontal_norm: float = 1e-6,
) -> OrientationTrack:
    """Convert direct root orientation to 6-D, yaw and yaw-rate features.

    Axis-angle is radians.  Quaternion order, local forward axis/sign, world
    up axis and coordinate frame are part of the resulting provenance.
    """
    if (axis_angle is None) == (quaternion is None):
        raise OrientationContractError("provide exactly one of axis_angle or quaternion")
    if local_forward_axis not in (0, 1, 2) or up_axis not in (0, 1, 2):
        raise OrientationContractError("local_forward_axis and up_axis must be 0, 1, or 2")
    if abs(local_forward_sign) != 1:
        raise OrientationContractError("local_forward_sign must be +1 or -1")
    if axis_angle is not None:
        matrix = _axis_angle_to_matrix(axis_angle)
        valid = np.ones(len(matrix), dtype=bool)
        quaternion_continuous = None
    else:
        raw = _finite_array(quaternion, name="quaternion", ndim=2)
        matrix, valid = _quaternion_to_matrix(raw, quaternion_order)
        quaternion_continuous = _continuous_quaternion(raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12), valid)
    ts = _timestamps(timestamps, len(matrix))
    local = np.zeros(3, dtype=np.float64)
    local[local_forward_axis] = local_forward_sign
    forward = np.einsum("tij,j->ti", matrix, local)
    horizontal_axes = tuple(axis for axis in range(3) if axis != up_axis)
    projected = forward[:, horizontal_axes]
    horizontal_norm = np.linalg.norm(projected, axis=-1)
    valid &= horizontal_norm > min_horizontal_norm
    direction = np.zeros_like(forward)
    direction[valid] = forward[valid] / np.maximum(np.linalg.norm(forward[valid], axis=-1, keepdims=True), 1e-12)
    angle = np.arctan2(projected[:, 1], projected[:, 0])
    reasons = np.full(len(matrix), "", dtype="U32")
    reasons[~valid] = "horizontal_projection_degenerate"
    sixd = matrix[:, :, :2].reshape(len(matrix), 6)
    sixd[~valid] = 0.0
    return _track(
        angle=angle,
        direction=direction,
        orientation_6d=sixd,
        valid=valid,
        timestamps=ts,
        source=orientation_source,
        kind="direct",
        frame=coordinate_frame,
        period=2.0 * np.pi,
        reasons=reasons,
        quaternion_continuous=quaternion_continuous,
    )
