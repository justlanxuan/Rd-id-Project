"""Physical turning scores and deterministic routing primitives."""

from __future__ import annotations

import numpy as np


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left)
    right = np.asarray(right, dtype=np.float64) - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-8 else 0.0


def physical_turning_score(orientation: np.ndarray, imu: np.ndarray, max_lag: int = 2) -> float:
    """Max lagged Pearson correlation of extracted turning rate and gyro norm."""
    orientation = np.asarray(orientation)
    imu = np.asarray(imu)
    if orientation.ndim != 2 or orientation.shape[1] < 3:
        raise ValueError(f"orientation must be [time,>=3], got {orientation.shape}")
    if imu.ndim != 2 or imu.shape[1] < 6:
        raise ValueError(f"imu must be [time,>=6], got {imu.shape}")
    limit = max(0, int(max_lag))
    rate = np.abs(orientation[:, 2])
    gyro = np.linalg.norm(imu[:, 3:6], axis=-1)
    scores: list[float] = []
    for lag in range(-limit, limit + 1):
        if lag < 0:
            scores.append(correlation(rate[-lag:], gyro[:lag]))
        elif lag > 0:
            scores.append(correlation(rate[:-lag], gyro[lag:]))
        else:
            scores.append(correlation(rate, gyro))
    return max(scores, default=0.0)


def turning_group_stratum(turning_count: int, threshold: float = 19.0 / 48.0) -> str:
    """Return the preregistered high/low router stratum.

    ``turning_count`` is the rounded sum of the binary activity stream across
    candidate rows; the threshold is expressed in the same count units as the
    historical E4.1 protocol.
    """
    return "high" if float(turning_count) >= float(threshold) * 48.0 else "low"


def route_turning_score(
    baseline: float,
    physical: float,
    turning_count: int,
    *,
    threshold: float = 19.0 / 48.0,
) -> float:
    """Choose the physical expert only for high-turning groups."""
    return float(physical if turning_group_stratum(turning_count, threshold) == "high" else baseline)


__all__ = ["correlation", "physical_turning_score", "route_turning_score", "turning_group_stratum"]
