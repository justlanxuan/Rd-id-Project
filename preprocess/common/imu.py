"""Shared IMU preprocessing utilities."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternions in wxyz order to rotation matrices."""
    q = np.asarray(q, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with last dimension 4, got {q.shape}")
    flat = q.reshape(-1, 4)
    w, x, y, z = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    r = np.zeros((len(flat), 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r.reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions in wxyz order."""
    r = np.asarray(rot, dtype=np.float32)
    if r.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending with (3, 3), got {r.shape}")
    flat = r.reshape(-1, 3, 3)
    q = np.zeros((len(flat), 4), dtype=np.float32)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]

    mask = trace > 0
    s = np.sqrt(np.clip(trace[mask] + 1.0, 1e-12, None)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (flat[mask, 2, 1] - flat[mask, 1, 2]) / s
    q[mask, 2] = (flat[mask, 0, 2] - flat[mask, 2, 0]) / s
    q[mask, 3] = (flat[mask, 1, 0] - flat[mask, 0, 1]) / s

    mask0 = (~mask) & (flat[:, 0, 0] > flat[:, 1, 1]) & (flat[:, 0, 0] > flat[:, 2, 2])
    s = np.sqrt(np.clip(1.0 + flat[mask0, 0, 0] - flat[mask0, 1, 1] - flat[mask0, 2, 2], 1e-12, None)) * 2.0
    q[mask0, 0] = (flat[mask0, 2, 1] - flat[mask0, 1, 2]) / s
    q[mask0, 1] = 0.25 * s
    q[mask0, 2] = (flat[mask0, 0, 1] + flat[mask0, 1, 0]) / s
    q[mask0, 3] = (flat[mask0, 0, 2] + flat[mask0, 2, 0]) / s

    mask1 = (~mask) & (~mask0) & (flat[:, 1, 1] > flat[:, 2, 2])
    s = np.sqrt(np.clip(1.0 + flat[mask1, 1, 1] - flat[mask1, 0, 0] - flat[mask1, 2, 2], 1e-12, None)) * 2.0
    q[mask1, 0] = (flat[mask1, 0, 2] - flat[mask1, 2, 0]) / s
    q[mask1, 1] = (flat[mask1, 0, 1] + flat[mask1, 1, 0]) / s
    q[mask1, 2] = 0.25 * s
    q[mask1, 3] = (flat[mask1, 1, 2] + flat[mask1, 2, 1]) / s

    mask2 = (~mask) & (~mask0) & (~mask1)
    s = np.sqrt(np.clip(1.0 + flat[mask2, 2, 2] - flat[mask2, 0, 0] - flat[mask2, 1, 1], 1e-12, None)) * 2.0
    q[mask2, 0] = (flat[mask2, 1, 0] - flat[mask2, 0, 1]) / s
    q[mask2, 1] = (flat[mask2, 0, 2] + flat[mask2, 2, 0]) / s
    q[mask2, 2] = (flat[mask2, 1, 2] + flat[mask2, 2, 1]) / s
    q[mask2, 3] = 0.25 * s

    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return q.reshape(*r.shape[:-2], 4).astype(np.float32)


def lowpass_filter_fft(signal: np.ndarray, cutoff_hz: float | None, fs_hz: float) -> np.ndarray:
    """Apply FFT-domain low-pass filter along the time axis."""
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal.astype(np.float32, copy=False)

    time_len = signal.shape[0]
    nyquist = fs_hz / 2.0
    effective_cutoff = float(cutoff_hz)

    if effective_cutoff >= nyquist:
        warnings.warn(
            f"Requested IMU low-pass cutoff {effective_cutoff:.3f} Hz exceeds Nyquist {nyquist:.3f} Hz; "
            f"clipping to {nyquist * 0.95:.3f} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_cutoff = max(nyquist * 0.95, 1e-6)

    freq = np.fft.rfftfreq(time_len, d=1.0 / fs_hz)
    spectrum = np.fft.rfft(signal, axis=0)
    spectrum[freq > effective_cutoff, ...] = 0.0
    return np.fft.irfft(spectrum, n=time_len, axis=0).astype(np.float32)


def parse_imu_csv(imu_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse custom IMU CSV."""
    with imu_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty IMU file: {imu_path}")

    ts_col = _find_col(["epoch_ms"], rows[0])
    q0_col = _find_col(["四元数0()"], rows[0])
    q1_col = _find_col(["四元数1()"], rows[0])
    q2_col = _find_col(["四元数2()"], rows[0])
    q3_col = _find_col(["四元数3()"], rows[0])
    ax_col = _find_col(["加速度X(g)"], rows[0])
    ay_col = _find_col(["加速度Y(g)"], rows[0])
    az_col = _find_col(["加速度Z(g)"], rows[0])

    T = len(rows)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    quat4 = np.zeros((T, 4), dtype=np.float32)
    acc3 = np.zeros((T, 3), dtype=np.float32)

    for t, row in enumerate(rows):
        timestamps_ms[t] = float(row[ts_col])
        quat4[t] = np.array([float(row[q0_col]), float(row[q1_col]), float(row[q2_col]), float(row[q3_col])], dtype=np.float32)
        acc3[t] = np.array([float(row[ax_col]), float(row[ay_col]), float(row[az_col])], dtype=np.float32) * 9.80665

    return timestamps_ms, quat4, acc3


def parse_imu_csv_with_gyro(imu_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse custom IMU CSV including gyro values in radians per second."""
    timestamps_ms, quat4, acc3 = parse_imu_csv(imu_path)
    with imu_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    gx_col = _find_col_with_prefix(("角速度X", "gyro_x", "gyrox"), rows[0])
    gy_col = _find_col_with_prefix(("角速度Y", "gyro_y", "gyroy"), rows[0])
    gz_col = _find_col_with_prefix(("角速度Z", "gyro_z", "gyroz"), rows[0])
    gyro3 = np.asarray(
        [[float(row[gx_col]), float(row[gy_col]), float(row[gz_col])] for row in rows],
        dtype=np.float32,
    )
    return timestamps_ms, quat4, acc3, np.deg2rad(gyro3).astype(np.float32)


def convert_single_imu_to_48(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    """Convert single-sensor IMU to 48D by repeating 12D four times."""
    T = quat4.shape[0]
    rot = quat_to_rotmat(quat4).reshape(T, 9)
    out = np.zeros((T, 48), dtype=np.float32)
    for i in range(4):
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc3
    return out


def convert_single_imu_to_7d(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    """Convert a custom single-IMU stream to hybrid raw 7D: acc3 + quat4."""
    return np.concatenate([acc3.astype(np.float32), quat4.astype(np.float32)], axis=-1)


def resample_imu_to_target(src_ts: np.ndarray, src_values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Linearly interpolate IMU values to target timestamps."""
    valid_start = src_ts[0]
    valid_end = src_ts[-1]

    out = np.zeros((len(target_ts), src_values.shape[1]), dtype=np.float32)
    for d in range(src_values.shape[1]):
        out[:, d] = np.interp(target_ts, src_ts, src_values[:, d])

    out[target_ts < valid_start] = np.nan
    out[target_ts > valid_end] = np.nan
    return out


def _find_col(candidates: list[str], row: dict[str, str]) -> str:
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in CSV columns: {list(row.keys())}")


def _find_col_with_prefix(prefixes: tuple[str, ...], row: dict[str, str]) -> str:
    for key in row:
        if any(key.strip().lower().startswith(prefix.lower()) for prefix in prefixes):
            return key
    raise KeyError(f"Could not find a column starting with {prefixes}; available={list(row.keys())}")
