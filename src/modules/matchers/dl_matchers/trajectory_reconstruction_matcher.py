"""Trajectory-reconstruction-based physics matcher.

This matcher reconstructs 1-D motion trajectories from IMU acceleration
(via cumulative integration / double integration) and compares them with
skeleton joint displacement trajectories in the time domain.

No training is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.modules.matchers.hungarian import HungarianMatcher


@dataclass
class TrajectoryConfig:
    enabled: bool = True
    fs_hz: float = 30.0
    imu_signal: str = "displacement_norm"  # acc_norm, velocity_norm, displacement_norm
    skeleton_signal: str = "joint_displacement"  # joint_speed, joint_displacement
    sensor_name: str = "mean_all"  # mean_all, L_LowLeg, R_LowLeg, L_LowArm, R_LowArm
    joint_name: str = "mean_all"  # mean_all, pelvis, left_wrist, right_wrist, left_ankle, right_ankle
    use_detrend: bool = True
    similarity_metric: str = "pearson"  # pearson, cosine, ncc
    threshold: float = 0.0


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1-D array."""
    if x.size < 2:
        return x
    t = np.arange(x.size, dtype=np.float32)
    coef = np.polyfit(t, x, 1)
    trend = np.polyval(coef, t)
    return x - trend


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    """Zero-mean and unit-variance normalization."""
    x = _as_float32(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 1e-12:
        return x - mean
    return (x - mean) / std


def _sensor_index(sensor_name: str) -> int:
    order = {"L_LowLeg": 0, "R_LowLeg": 1, "L_LowArm": 2, "R_LowArm": 3}
    if sensor_name not in order:
        raise ValueError(f"Unknown sensor_name={sensor_name}; expected one of {list(order.keys())}")
    return order[sensor_name]


def _joint_index(joint_name: str) -> int:
    # H36M 17-joint order
    order = {
        "pelvis": 0,
        "right_hip": 1,
        "right_knee": 2,
        "right_ankle": 3,
        "left_hip": 4,
        "left_knee": 5,
        "left_ankle": 6,
        "spine": 7,
        "thorax": 8,
        "neck": 9,
        "head": 10,
        "left_shoulder": 11,
        "left_elbow": 12,
        "left_wrist": 13,
        "right_shoulder": 14,
        "right_elbow": 15,
        "right_wrist": 16,
    }
    if joint_name not in order:
        raise ValueError(f"Unknown joint_name={joint_name}; expected one of {list(order.keys())}")
    return order[joint_name]


def imu_sensor_trajectory(imu_window: np.ndarray, sensor_name: str, signal: str, fs_hz: float) -> np.ndarray:
    """Reconstruct a 1-D trajectory from a single IMU sensor.

    Args:
        imu_window: [T, 48] raw IMU window.
        sensor_name: One of the four sensor names.
        signal: "acc_norm", "velocity_norm", or "displacement_norm".
        fs_hz: Sampling rate.

    Returns:
        1-D trajectory of shape [T].
    """
    imu_window = _as_float32(imu_window)
    if imu_window.ndim != 2:
        raise ValueError(f"Expected imu_window [T, 48], got {imu_window.shape}")

    k = _sensor_index(sensor_name)
    acc = imu_window[:, 36 + 3 * k : 36 + 3 * (k + 1)]

    # Use local acceleration norm (robust to unknown global coordinate frames).
    # Subtract mean to remove gravity offset before integration.
    a_norm = np.linalg.norm(acc, axis=-1)
    a_norm = a_norm - float(np.mean(a_norm))

    dt = 1.0 / max(float(fs_hz), 1e-6)

    if signal == "acc_norm":
        return a_norm

    # First integration: velocity (cumulative trapezoidal)
    vel = np.zeros_like(a_norm)
    vel[1:] = np.cumsum(0.5 * (a_norm[:-1] + a_norm[1:])) * dt

    if signal == "velocity_norm":
        return vel

    if signal == "displacement_norm":
        # Second integration: displacement
        disp = np.zeros_like(vel)
        disp[1:] = np.cumsum(0.5 * (vel[:-1] + vel[1:])) * dt
        return disp

    raise ValueError(f"Unsupported imu_signal={signal}")


def imu_window_trajectory(imu_window: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    """Extract 1-D IMU trajectory according to config."""
    if cfg.sensor_name == "mean_all":
        trajs = []
        for name in ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]:
            trajs.append(imu_sensor_trajectory(imu_window, name, cfg.imu_signal, cfg.fs_hz))
        return np.mean(trajs, axis=0)
    return imu_sensor_trajectory(imu_window, cfg.sensor_name, cfg.imu_signal, cfg.fs_hz)


def skeleton_joint_trajectory(skeleton_window: np.ndarray, joint_name: str, signal: str) -> np.ndarray:
    """Extract a 1-D trajectory for a single skeleton joint.

    Args:
        skeleton_window: [T, 17, 3] skeleton window.
        joint_name: Joint name.
        signal: "joint_speed" or "joint_displacement".

    Returns:
        1-D trajectory of shape [T].
    """
    skeleton_window = _as_float32(skeleton_window)
    if skeleton_window.ndim != 3:
        raise ValueError(f"Expected skeleton_window [T, 17, 3], got {skeleton_window.shape}")

    j = _joint_index(joint_name)
    pos = skeleton_window[:, j, :]

    if signal == "joint_speed":
        if pos.shape[0] <= 1:
            return np.zeros((pos.shape[0],), dtype=np.float32)
        diff = np.diff(pos, axis=0)
        speed = np.linalg.norm(diff, axis=-1)
        return np.concatenate([speed[:1], speed], axis=0)

    if signal == "joint_displacement":
        disp = np.linalg.norm(pos - pos[0], axis=-1)
        return disp

    raise ValueError(f"Unsupported skeleton_signal={signal}")


def skeleton_window_trajectory(skeleton_window: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    """Extract 1-D skeleton trajectory according to config."""
    if cfg.joint_name == "mean_all":
        trajs = []
        for name in ["pelvis", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]:
            trajs.append(skeleton_joint_trajectory(skeleton_window, name, cfg.skeleton_signal))
        return np.mean(trajs, axis=0)
    return skeleton_joint_trajectory(skeleton_window, cfg.joint_name, cfg.skeleton_signal)


def trajectory_similarity(a: np.ndarray, b: np.ndarray, metric: str, use_detrend: bool) -> float:
    """Compute similarity between two 1-D trajectories."""
    a = _as_float32(a).reshape(-1)
    b = _as_float32(b).reshape(-1)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return -1.0

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    if use_detrend:
        a = _detrend(a)
        b = _detrend(b)

    if metric == "pearson":
        a_n = _safe_normalize(a)
        b_n = _safe_normalize(b)
        denom = float(np.linalg.norm(a_n) * np.linalg.norm(b_n))
        if denom <= 1e-12:
            return -1.0
        return float(np.dot(a_n, b_n) / denom)

    if metric == "cosine":
        a_m = a - float(np.mean(a))
        b_m = b - float(np.mean(b))
        denom = float(np.linalg.norm(a_m) * np.linalg.norm(b_m))
        if denom <= 1e-12:
            return -1.0
        return float(np.dot(a_m, b_m) / denom)

    if metric == "ncc":
        # Normalized cross-correlation at zero lag after standardization
        a_n = _safe_normalize(a)
        b_n = _safe_normalize(b)
        denom = float(np.linalg.norm(a_n) * np.linalg.norm(b_n))
        if denom <= 1e-12:
            return -1.0
        return float(np.dot(a_n, b_n) / denom)

    raise ValueError(f"Unsupported similarity_metric={metric}")


def build_person_skeleton_series(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a [T, N_track, 17, 3] raw tracker skeleton tensor."""
    if "extract_skeleton" not in data:
        raise ValueError("No raw extracted skeletons available in NPZ (need extract_skeleton).")
    return _as_float32(data["extract_skeleton"])


def build_sequence_windows(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    seen = set()
    windows: List[Tuple[int, int]] = []
    for row in rows:
        key = (int(row["window_start"]), int(row["window_end"]))
        if key in seen:
            continue
        seen.add(key)
        windows.append(key)
    windows.sort(key=lambda item: (item[0], item[1]))
    return windows


def _interpolate_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, x.shape[0]),
        x,
    )


def compute_similarity_matrix(
    data: Dict[str, np.ndarray],
    windows: Sequence[Tuple[int, int]],
    cfg: TrajectoryConfig,
) -> np.ndarray:
    """Compute a global similarity matrix averaged over windows."""
    imu = _as_float32(data["imu"])
    if imu.ndim == 2:
        imu = imu[:, np.newaxis, :]

    skeleton_series = build_person_skeleton_series(data)
    if skeleton_series.ndim != 4:
        raise ValueError(f"Expected skeleton series [T, N, 17, 3], got {skeleton_series.shape}")

    track_visibility = np.asarray(data.get("extract_visibility"), dtype=bool)
    if track_visibility.ndim != 2:
        raise ValueError("Expected extract_visibility [T, N_track].")

    T = int(imu.shape[0])
    N_imu = int(imu.shape[1])
    N_person = int(skeleton_series.shape[1])

    sim_sum = np.zeros((N_imu, N_person), dtype=np.float32)
    sim_count = np.zeros((N_imu, N_person), dtype=np.float32)

    # Enumerate all sensor-joint combinations for cross-matching
    all_sensors = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
    all_joints = ["pelvis", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]

    for st, ed in windows:
        if st < 0 or ed > T or ed <= st:
            continue

        for i in range(N_imu):
            imu_trajs: Dict[str, np.ndarray] = {}
            for sensor in all_sensors:
                try:
                    traj = imu_sensor_trajectory(imu[st:ed, i], sensor, cfg.imu_signal, cfg.fs_hz)
                    imu_trajs[sensor] = traj
                except Exception:
                    pass
            if not imu_trajs:
                continue

            for g in range(N_person):
                vis = track_visibility[st:ed, g]
                skel_seq = skeleton_series[st:ed, g]
                if vis.any():
                    skel_seq = skel_seq[vis]
                if skel_seq.shape[0] < 2:
                    continue

                skel_trajs: Dict[str, np.ndarray] = {}
                for joint in all_joints:
                    try:
                        traj = skeleton_joint_trajectory(skel_seq, joint, cfg.skeleton_signal)
                        skel_trajs[joint] = traj
                    except Exception:
                        pass
                if not skel_trajs:
                    continue

                # Find best sensor-joint pair similarity
                best_sim = -1.0
                for sensor, traj_i in imu_trajs.items():
                    for joint, traj_g in skel_trajs.items():
                        target_len = max(traj_i.shape[0], traj_g.shape[0])
                        ti = _interpolate_to_length(traj_i, target_len)
                        tg = _interpolate_to_length(traj_g, target_len)
                        sim = trajectory_similarity(ti, tg, cfg.similarity_metric, cfg.use_detrend)
                        if np.isfinite(sim) and sim > best_sim:
                            best_sim = sim

                if best_sim > -1.0:
                    sim_sum[i, g] += best_sim
                    sim_count[i, g] += 1.0

    sim = np.full((N_imu, N_person), -1.0, dtype=np.float32)
    valid = sim_count > 0
    sim[valid] = sim_sum[valid] / sim_count[valid]
    return sim


def build_ground_truth_assignment(data: Dict[str, np.ndarray]) -> Dict[int, int]:
    imu = np.asarray(data["imu"])
    n_imu = int(imu.shape[1] if imu.ndim == 3 else 1)
    n_person = int(np.asarray(data["gt_person_ids"]).shape[0]) if "gt_person_ids" in data else 0
    return {i: i for i in range(min(n_imu, n_person))}


class TrajectoryReconstructionPhysicsMatcher:
    """Global matcher that scores reconstructed trajectory similarity and applies Hungarian assignment."""

    def __init__(self, config_dict: Dict[str, Any] | None = None) -> None:
        data = dict(config_dict or {})
        self.config = TrajectoryConfig(
            enabled=bool(data.get("enabled", True)),
            fs_hz=float(data.get("fs_hz", 30.0)),
            imu_signal=str(data.get("imu_signal", "displacement_norm")),
            skeleton_signal=str(data.get("skeleton_signal", "joint_displacement")),
            sensor_name=str(data.get("sensor_name", "mean_all")),
            joint_name=str(data.get("joint_name", "mean_all")),
            use_detrend=bool(data.get("use_detrend", True)),
            similarity_metric=str(data.get("similarity_metric", "pearson")),
            threshold=float(data.get("threshold", 0.0)),
        )
        self.hungarian = HungarianMatcher({"threshold": self.config.threshold})

    def match_sequence(
        self,
        data: Dict[str, np.ndarray],
        windows: Sequence[Tuple[int, int]],
    ) -> Dict[str, Any]:
        sim = compute_similarity_matrix(data, windows, self.config)
        imu_ids = list(range(sim.shape[0]))
        track_ids = np.asarray(data.get("extract_person_ids"), dtype=np.int64).tolist()
        if len(track_ids) != sim.shape[1]:
            track_ids = list(range(sim.shape[1]))
        result = self.hungarian.match(sim, imu_ids=imu_ids, person_ids=track_ids)
        return {
            "similarity_matrix": sim,
            "assignments": result["assignments"],
            "scores": result["scores"],
            "confidences": result["confidences"],
        }


def summarize_trajectory_result(result: Dict[str, Any]) -> str:
    import json

    payload = {
        "assignments": result.get("assignments", []),
        "scores": result.get("scores", []),
        "confidences": result.get("confidences", []),
    }
    return json.dumps(payload, ensure_ascii=False)
