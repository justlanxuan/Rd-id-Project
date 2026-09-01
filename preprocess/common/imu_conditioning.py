"""Geometry-aware IMU conditioning utilities.

The functions here operate before canonical 7D packing when a method needs
native-rate sensor channels. They are deterministic and do not depend on the
training engine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.core.registry import Registry

IMUConditioner = Callable[..., np.ndarray]
IMU_CONDITIONER_REGISTRY: Registry[IMUConditioner] = Registry("IMU conditioner")


def normalize_quaternions_wxyz(quaternions: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize quaternions in the final dimension without changing shape."""
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("Quaternion conditioning received a near-zero quaternion")
    return (values / norms).astype(np.float32)


def enforce_quaternion_continuity(quaternions: np.ndarray) -> np.ndarray:
    """Choose a continuous sign along time for q and -q equivalent rotations."""
    output = normalize_quaternions_wxyz(quaternions).astype(np.float64)
    if output.shape[0] <= 1:
        return output.astype(np.float32)
    flat = output.reshape(output.shape[0], -1, 4)
    for frame in range(1, flat.shape[0]):
        flip = np.sum(flat[frame - 1] * flat[frame], axis=-1) < 0.0
        flat[frame, flip] *= -1.0
    return output.astype(np.float32)


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def run_madgwick_imu(
    timestamps_ms: np.ndarray,
    quaternions_wxyz: np.ndarray,
    acceleration: np.ndarray,
    gyroscope_rad_s: np.ndarray,
    beta: float = 0.033,
) -> np.ndarray:
    """Run the causal Madgwick 6-axis update at native IMU timestamps.

    This is the IMU-only update from Madgwick's gradient-descent filter.  The
    supplied quaternion initializes the state, acceleration is used as a
    gravity direction, and gyro values must be in radians per second.
    """
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    initial = np.asarray(quaternions_wxyz, dtype=np.float64)
    acc = np.asarray(acceleration, dtype=np.float64)
    gyro = np.asarray(gyroscope_rad_s, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 4:
        raise ValueError(f"Expected quaternions [T,4], got {initial.shape}")
    if acc.shape != (len(initial), 3) or gyro.shape != (len(initial), 3):
        raise ValueError(
            f"Madgwick IMU expects acceleration and gyro [T,3], got {acc.shape} and {gyro.shape}"
        )
    if timestamps.shape != (len(initial),) or len(initial) == 0:
        raise ValueError("Madgwick IMU requires one timestamp per non-empty quaternion sequence")
    if beta < 0:
        raise ValueError(f"Madgwick beta must be non-negative, got {beta}")

    initial_norm = np.linalg.norm(initial[0])
    if initial_norm < 1e-12:
        raise ValueError("Madgwick IMU received a near-zero initial quaternion")
    q = initial[0] / initial_norm
    output = np.empty_like(initial, dtype=np.float64)
    output[0] = q
    deltas = np.diff(timestamps) / 1000.0
    valid_deltas = deltas[(deltas > 0.0) & (deltas < 1.0)]
    fallback_dt = float(np.median(valid_deltas)) if len(valid_deltas) else 1.0 / 30.0

    for index in range(1, len(initial)):
        dt = float(deltas[index - 1]) if 0.0 < deltas[index - 1] < 1.0 else fallback_dt
        gyr = gyro[index]
        q_dot = 0.5 * quaternion_multiply_wxyz(q, np.asarray([0.0, *gyr]))
        if np.linalg.norm(gyr) > 0.0:
            acc_norm = float(np.linalg.norm(acc[index]))
            if acc_norm > 0.0:
                ax, ay, az = acc[index] / acc_norm
                qw, qx, qy, qz = q / max(np.linalg.norm(q), 1e-12)
                objective = np.asarray(
                    [
                        2.0 * (qx * qz - qw * qy) - ax,
                        2.0 * (qw * qx + qy * qz) - ay,
                        2.0 * (0.5 - qx * qx - qy * qy) - az,
                    ],
                    dtype=np.float64,
                )
                jacobian = np.asarray(
                    [
                        [-2.0 * qy, 2.0 * qz, -2.0 * qw, 2.0 * qx],
                        [2.0 * qx, 2.0 * qw, 2.0 * qz, 2.0 * qy],
                        [0.0, -4.0 * qx, -4.0 * qy, 0.0],
                    ],
                    dtype=np.float64,
                )
                gradient = jacobian.T @ objective
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 1e-12:
                    q_dot -= float(beta) * gradient / gradient_norm
        q = q + q_dot * dt
        q /= max(np.linalg.norm(q), 1e-12)
        output[index] = q

    if not np.isfinite(output).all():
        raise ValueError("Madgwick IMU produced non-finite quaternions")
    return enforce_quaternion_continuity(output)


@IMU_CONDITIONER_REGISTRY.register("identity")
def identity_imu_conditioner(
    timestamps_ms: np.ndarray,
    quaternions_wxyz: np.ndarray,
    _acceleration: np.ndarray,
    _gyroscope_rad_s: np.ndarray | None = None,
    _beta: float = 0.033,
) -> np.ndarray:
    """Return source orientation unchanged for the legacy identity path."""
    del timestamps_ms, _acceleration, _gyroscope_rad_s, _beta
    return np.asarray(quaternions_wxyz, dtype=np.float32).copy()


@IMU_CONDITIONER_REGISTRY.register("madgwick6", aliases=("rg23", "madgwick"))
def madgwick6_imu_conditioner(
    timestamps_ms: np.ndarray,
    quaternions_wxyz: np.ndarray,
    acceleration: np.ndarray,
    gyroscope_rad_s: np.ndarray | None = None,
    beta: float = 0.033,
) -> np.ndarray:
    """Run the RG23-style causal IMU-only attitude conditioner."""
    if gyroscope_rad_s is None:
        raise ValueError("madgwick6 requires native-rate gyro values in radians per second")
    return run_madgwick_imu(
        timestamps_ms,
        quaternions_wxyz,
        acceleration,
        gyroscope_rad_s,
        beta=beta,
    )


def condition_imu(
    name: str,
    timestamps_ms: np.ndarray,
    quaternions_wxyz: np.ndarray,
    acceleration: np.ndarray,
    gyroscope_rad_s: np.ndarray | None = None,
    beta: float = 0.033,
) -> np.ndarray:
    """Resolve and apply a named raw IMU conditioner."""
    conditioner = IMU_CONDITIONER_REGISTRY.get(name)
    return conditioner(timestamps_ms, quaternions_wxyz, acceleration, gyroscope_rad_s, beta)


__all__ = [
    "IMU_CONDITIONER_REGISTRY",
    "condition_imu",
    "enforce_quaternion_continuity",
    "identity_imu_conditioner",
    "madgwick6_imu_conditioner",
    "normalize_quaternions_wxyz",
    "quaternion_multiply_wxyz",
    "run_madgwick_imu",
]
