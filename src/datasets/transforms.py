"""Dataset-side signal transforms and legacy format adapters."""

from __future__ import annotations

import warnings

import numpy as np


def lowpass_filter_fft(signal: np.ndarray, cutoff_hz: float | None, fs_hz: float) -> np.ndarray:
    """Apply a simple FFT-domain low-pass filter along the time axis.

    The input is expected to have time as axis 0, e.g. [T, D] or [T, N, D].
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal.astype(np.float32, copy=False)

    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    time_len = signal.shape[0]
    if time_len <= 1:
        return signal.astype(np.float32, copy=False)

    nyquist = fs_hz / 2.0
    effective_cutoff = float(cutoff_hz)
    if effective_cutoff >= nyquist:
        clipped = max(nyquist * 0.95, 1e-6)
        warnings.warn(
            f"Requested IMU low-pass cutoff {effective_cutoff:.3f} Hz exceeds Nyquist {nyquist:.3f} Hz; "
            f"clipping to {clipped:.3f} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_cutoff = clipped

    freq = np.fft.rfftfreq(time_len, d=1.0 / fs_hz)
    spectrum = np.fft.rfft(signal, axis=0)
    spectrum[freq > effective_cutoff, ...] = 0.0
    filtered = np.fft.irfft(spectrum, n=time_len, axis=0)
    return filtered.astype(np.float32, copy=False)


def single_sensor_to_48d(imu: np.ndarray, sensor_name: str, repeat_single_sensor: int) -> np.ndarray:
    """Expand one legacy 48D-layout IMU sensor back into the expected 48D layout."""
    order = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
    if sensor_name not in order:
        raise ValueError(f"Unsupported sensor_name={sensor_name}. Must be one of {order}")

    if repeat_single_sensor != 4:
        raise ValueError(
            "Alignment IMU encoder expects 48D input. "
            f"Use repeat_single_sensor=4 for single-sensor mode; got {repeat_single_sensor}."
        )

    k = order.index(sensor_name)
    rot = imu[:, k * 9 : (k + 1) * 9]
    acc = imu[:, 36 + k * 3 : 36 + (k + 1) * 3]

    out = np.zeros((imu.shape[0], 48), dtype=np.float32)
    for i in range(4):
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out
