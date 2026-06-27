"""Frequency-based physics matcher utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.modules.matchers.hungarian import HungarianMatcher


@dataclass
class FrequencyConfig:
    """Configuration for frequency-based matching."""

    enabled: bool = True
    fs_hz: float = 30.0
    n_fft: int = 64
    top_k: int = 0
    band_min_hz: float = 0.0
    band_max_hz: float = 0.0
    use_hann: bool = True
    use_log1p: bool = True
    normalize: str = "l2"
    threshold: float = 0.0
    imu_signal: str = "acc_norm_mean"
    skeleton_signal: str = "left_wrist_speed"


def parse_frequency_config(config_dict: Dict[str, Any] | None) -> FrequencyConfig:
    data = dict(config_dict or {})
    return FrequencyConfig(
        enabled=bool(data.get("enabled", True)),
        fs_hz=float(data.get("fs_hz", 30.0)),
        n_fft=int(data.get("n_fft", 64)),
        top_k=int(data.get("top_k", 0)),
        band_min_hz=float(data.get("band_min_hz", 0.0)),
        band_max_hz=float(data.get("band_max_hz", 0.0)),
        use_hann=bool(data.get("use_hann", True)),
        use_log1p=bool(data.get("use_log1p", True)),
        normalize=str(data.get("normalize", "l2")),
        threshold=float(data.get("threshold", 0.0)),
        imu_signal=str(data.get("imu_signal", "acc_norm_mean")),
        skeleton_signal=str(data.get("skeleton_signal", "left_wrist_speed")),
    )


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(x))
    if denom <= 1e-12:
        return x
    return x / denom


def _pick_band(spec: np.ndarray, freqs: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    if cfg.band_max_hz > cfg.band_min_hz > 0:
        mask = (freqs >= cfg.band_min_hz) & (freqs <= cfg.band_max_hz)
        return spec[mask]
    return spec


def signal_to_frequency_feature(signal: np.ndarray, cfg: FrequencyConfig) -> np.ndarray | None:
    """Convert a 1D signal into a normalized frequency descriptor."""
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.shape}")
    if signal.size == 0:
        return None

    x = signal.astype(np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = x - float(np.mean(x))
    if np.allclose(x, 0.0):
        return None

    n_fft = max(int(cfg.n_fft), int(x.shape[0]))
    if cfg.use_hann and x.shape[0] > 1:
        x = x * np.hanning(x.shape[0]).astype(np.float32)

    spec = np.abs(np.fft.rfft(x, n=n_fft)).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(cfg.fs_hz, 1e-6)).astype(np.float32)
    spec = _pick_band(spec, freqs, cfg)

    if cfg.use_log1p:
        spec = np.log1p(spec)

    if cfg.top_k and spec.size > cfg.top_k:
        idx = np.argsort(spec)[::-1][: cfg.top_k]
        idx = np.sort(idx)
        spec = spec[idx]

    if cfg.normalize == "l2":
        spec = _safe_l2_normalize(spec)
    elif cfg.normalize == "mean":
        scale = float(np.mean(np.abs(spec)))
        if scale > 1e-12:
            spec = spec / scale

    return spec.astype(np.float32)


def imu_window_signal(imu_window: np.ndarray, mode: str = "acc_norm_mean") -> np.ndarray:
    """Extract a 1D IMU signal for frequency analysis."""
    imu_window = _as_float32(imu_window)
    if imu_window.ndim != 2:
        raise ValueError(f"Expected imu_window [T, 48], got {imu_window.shape}")

    if mode in {"acc_norm_mean", "acc_norm"}:
        acc = imu_window[:, 36:48].reshape(imu_window.shape[0], 4, 3)
        return np.linalg.norm(acc, axis=-1).mean(axis=1)
    if mode == "acc_norm_all":
        acc = imu_window[:, 36:48].reshape(imu_window.shape[0], 4, 3)
        return np.linalg.norm(acc, axis=-1).reshape(-1)
    if mode == "full_mean":
        return imu_window.mean(axis=1)

    raise ValueError(f"Unsupported imu_signal mode: {mode}")


def skeleton_window_signal(skeleton_window: np.ndarray, mode: str = "left_wrist_speed") -> np.ndarray:
    """Extract a 1D skeleton signal for frequency analysis."""
    skeleton_window = _as_float32(skeleton_window)
    if skeleton_window.ndim != 3:
        raise ValueError(f"Expected skeleton_window [T, 17, 3], got {skeleton_window.shape}")

    if mode in {"joint_speed_mean", "speed_mean"}:
        if skeleton_window.shape[0] <= 1:
            return np.zeros((skeleton_window.shape[0],), dtype=np.float32)
        diff = np.diff(skeleton_window, axis=0)
        speed = np.linalg.norm(diff, axis=-1).mean(axis=1)
        return np.concatenate([speed[:1], speed], axis=0)
    if mode in {"left_wrist_speed", "lwrist_speed"}:
        if skeleton_window.shape[0] <= 1:
            return np.zeros((skeleton_window.shape[0],), dtype=np.float32)
        # H36M 17-joint order: LeftWrist is index 13.
        diff = np.diff(skeleton_window[:, 13, :], axis=0)
        speed = np.linalg.norm(diff, axis=-1)
        return np.concatenate([speed[:1], speed], axis=0)
    if mode in {"pelvis_left_wrist_speed", "base_left_wrist_speed", "hip_left_wrist_speed"}:
        if skeleton_window.shape[0] <= 1:
            return np.zeros((skeleton_window.shape[0],), dtype=np.float32)
        # H36M 17-joint order: pelvis/root index 0, LeftWrist index 13.
        diff_pelvis = np.diff(skeleton_window[:, 0, :], axis=0)
        diff_lwrist = np.diff(skeleton_window[:, 13, :], axis=0)
        speed_pelvis = np.linalg.norm(diff_pelvis, axis=-1)
        speed_lwrist = np.linalg.norm(diff_lwrist, axis=-1)
        speed = 0.5 * (speed_pelvis + speed_lwrist)
        return np.concatenate([speed[:1], speed], axis=0)
    if mode in {"pelvis_right_wrist_speed", "base_right_wrist_speed", "hip_right_wrist_speed"}:
        if skeleton_window.shape[0] <= 1:
            return np.zeros((skeleton_window.shape[0],), dtype=np.float32)
        # H36M 17-joint order: pelvis/root index 0, RightWrist index 10.
        diff_pelvis = np.diff(skeleton_window[:, 0, :], axis=0)
        diff_rwrist = np.diff(skeleton_window[:, 10, :], axis=0)
        speed_pelvis = np.linalg.norm(diff_pelvis, axis=-1)
        speed_rwrist = np.linalg.norm(diff_rwrist, axis=-1)
        speed = 0.5 * (speed_pelvis + speed_rwrist)
        return np.concatenate([speed[:1], speed], axis=0)
    if mode == "root_speed":
        if skeleton_window.shape[0] <= 1:
            return np.zeros((skeleton_window.shape[0],), dtype=np.float32)
        diff = np.diff(skeleton_window[:, 0, :], axis=0)
        speed = np.linalg.norm(diff, axis=-1)
        return np.concatenate([speed[:1], speed], axis=0)
    if mode == "joint_energy_mean":
        return np.linalg.norm(skeleton_window, axis=-1).mean(axis=1)

    raise ValueError(f"Unsupported skeleton_signal mode: {mode}")


def build_person_skeleton_series(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a [T, N_track, 17, 3] raw tracker skeleton tensor.

    This function intentionally uses only raw tracker outputs. It must not
    depend on any GT-derived alignment such as gt_to_extract_map.
    """
    if "extract_skeleton" not in data:
        raise ValueError("No raw extracted skeletons available in NPZ (need extract_skeleton).")
    return _as_float32(data["extract_skeleton"])


def build_sequence_windows(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    """Extract unique window intervals from slice CSV rows."""
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


def compute_similarity_matrix(
    data: Dict[str, np.ndarray],
    windows: Sequence[Tuple[int, int]],
    cfg: FrequencyConfig,
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

    for st, ed in windows:
        if st < 0 or ed > T or ed <= st:
            continue

        imu_feats: List[np.ndarray | None] = []
        person_feats: List[np.ndarray | None] = []

        for i in range(N_imu):
            imu_sig = imu_window_signal(imu[st:ed, i], cfg.imu_signal)
            imu_feats.append(signal_to_frequency_feature(imu_sig, cfg))

        for g in range(N_person):
            vis = track_visibility[st:ed, g]
            skel_seq = skeleton_series[st:ed, g]
            if vis.any():
                skel_seq = skel_seq[vis]
            if skel_seq.shape[0] < 2:
                person_feats.append(None)
                continue
            skel_sig = skeleton_window_signal(skel_seq, cfg.skeleton_signal)
            person_feats.append(signal_to_frequency_feature(skel_sig, cfg))

        for i in range(N_imu):
            feat_i = imu_feats[i]
            if feat_i is None:
                continue
            for g in range(N_person):
                feat_g = person_feats[g]
                if feat_g is None or feat_i.shape != feat_g.shape:
                    continue
                denom = float(np.linalg.norm(feat_i) * np.linalg.norm(feat_g))
                if denom <= 1e-12:
                    continue
                sim_sum[i, g] += float(np.dot(feat_i, feat_g) / denom)
                sim_count[i, g] += 1.0

    sim = np.full((N_imu, N_person), -1.0, dtype=np.float32)
    valid = sim_count > 0
    sim[valid] = sim_sum[valid] / sim_count[valid]
    return sim


def build_ground_truth_assignment(data: Dict[str, np.ndarray]) -> Dict[int, int]:
    """Build IMU index -> person index ground truth mapping."""
    imu = np.asarray(data["imu"])
    n_imu = int(imu.shape[1] if imu.ndim == 3 else 1)
    n_person = int(np.asarray(data["gt_person_ids"]).shape[0]) if "gt_person_ids" in data else 0
    return {i: i for i in range(min(n_imu, n_person))}


class FrequencyPhysicsMatcher:
    """Global matcher that scores frequency features and applies Hungarian assignment."""

    def __init__(self, config_dict: Dict[str, Any] | None = None) -> None:
        self.config = parse_frequency_config(config_dict)
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


def summarize_frequency_result(result: Dict[str, Any]) -> str:
    """Compact debug string for a frequency-matching result."""
    payload = {
        "assignments": result.get("assignments", []),
        "scores": result.get("scores", []),
        "confidences": result.get("confidences", []),
    }
    return json.dumps(payload, ensure_ascii=False)