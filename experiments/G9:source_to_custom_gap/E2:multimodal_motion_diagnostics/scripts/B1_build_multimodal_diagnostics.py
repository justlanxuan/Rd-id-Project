# Experiment Note: B1-multimodal-motion-diagnostics
"""Compute deterministic IMU, skeleton-motion, complexity and lag diagnostics.

The script is deliberately feature-only: it does not train or select a model. Raw
and normalized coordinate tracks remain separated, and S06 algorithm outputs use
their external baseline NPZ only for the already-verified IMU/person join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOTAL_ROOT = Path(
    "/data/fzliang/reid-project/totalcapture/preprocessed/"
    "g6_totalcapture_source/sequences"
)
EGO_ROOT = Path(
    "/data/fzliang/reid-project/egohumans/preprocessed/"
    "g6_egohumans_source/sequences"
)
CUSTOM_ROOT = Path(
    "/data/fzliang/reid-project/custom/preprocessed/"
    "hybrid_w24_session_out_rawcsv7d_swapsess"
)
S06_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
)
S06_BASELINE_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "data/interim/egohumans_repro_local/slice/sequences"
)
SENSOR_INDEX = 2  # L_LowArm in the verified legacy order L/R_LowLeg, L/R_LowArm.
LAGS = tuple(range(-8, 9))
EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))
WRISTS = (13, 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q05": None, "q25": None, "q50": None, "q75": None, "q95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        key: float(value)
        for key, value in zip(
            ("q05", "q25", "q50", "q75", "q95"),
            np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95]),
            strict=True,
        )
    }


def finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def normalize_skeleton(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4 or values.shape[-2] != 17 or values.shape[-1] not in (2, 3):
        raise ValueError(f"unsupported skeleton shape {values.shape}")
    return values


def normalize_imu(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"unsupported IMU shape {values.shape}")
    return values


def representation(skeleton: np.ndarray, external_visibility: np.ndarray | None) -> str:
    if skeleton.shape[-1] == 2:
        return "2d_xy"
    third = skeleton[..., 2]
    if external_visibility is not None and np.allclose(third, 0.0):
        return "2d_xy_zero_z_with_external_visibility"
    if np.all((third >= 0) & (third <= 1)):
        return "2d_xy_with_visibility_in_last_dim"
    return "3d_xyz"


def legacy_imu_acc(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the verified L_LowArm acceleration from either 7D or legacy 48D."""
    values = normalize_imu(array)
    if values.shape[-1] == 7:
        return values[..., :3], "7d_acc3_quat4"
    if values.shape[-1] >= 48:
        start = 36 + SENSOR_INDEX * 3
        return values[..., start : start + 3], "legacy48_L_LowArm_acc3"
    return values[..., : min(3, values.shape[-1])], f"unknown_{values.shape[-1]}d_first_channels"


def zcorr(left: np.ndarray, right: np.ndarray) -> float | None:
    a = finite_values(left)
    b = finite_values(right)
    n = min(a.size, b.size)
    if n < 5:
        return None
    a = a[:n]
    b = b[:n]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def lag_correlations(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    """Correlate skeleton speed[t+lag] with IMU acceleration[t]."""
    result: dict[str, float | None] = {}
    n = min(left.size, right.size)
    left = np.asarray(left[:n], dtype=np.float64)
    right = np.asarray(right[:n], dtype=np.float64)
    for lag in LAGS:
        if lag >= 0:
            result[str(lag)] = zcorr(left[lag:], right[: n - lag])
        else:
            result[str(lag)] = zcorr(left[: n + lag], right[-lag:])
    return result


def skeleton_features(skeleton: np.ndarray) -> list[dict[str, float]]:
    """Return one motion/complexity observation per person in a record."""
    coords = skeleton[..., :2] if representation(skeleton, None).startswith("2d_") else skeleton
    observations: list[dict[str, float]] = []
    for person in range(coords.shape[1]):
        xy = coords[:, person]
        if xy.shape[0] < 4:
            continue
        finite = np.isfinite(xy).all(axis=-1)
        xy = np.where(finite[..., None], xy, np.nan)
        centered = xy - xy[:, :1, :]
        bone_lengths = []
        for left, right in EDGES:
            bone_lengths.append(np.linalg.norm(centered[:, left] - centered[:, right], axis=-1))
        scale_values = finite_values(np.stack(bone_lengths, axis=-1))
        scale = float(np.median(scale_values[scale_values > 1e-8])) if np.any(scale_values > 1e-8) else 1.0
        velocity = np.diff(centered, axis=0)
        speed = np.linalg.norm(velocity, axis=-1) / scale
        acceleration = np.diff(velocity, axis=0)
        accel_norm = np.linalg.norm(acceleration, axis=-1) / scale if acceleration.size else np.empty(0)
        jerk = np.diff(acceleration, axis=0)
        jerk_norm = np.linalg.norm(jerk, axis=-1) / scale if jerk.size else np.empty(0)
        speed_values = finite_values(speed)
        threshold = float(np.quantile(speed_values, 0.75)) if speed_values.size else 0.0
        active = float(np.mean(speed > threshold)) if speed.size else 0.0
        simultaneous = float(np.mean(np.mean(speed > threshold, axis=-1) >= 0.5)) if speed.size else 0.0
        wrist_speed = np.mean(speed[:, WRISTS], axis=-1) if speed.shape[1] > max(WRISTS) else np.mean(speed, axis=-1)
        spectrum = np.abs(np.fft.rfft(np.nan_to_num(wrist_speed - np.nanmean(wrist_speed)))) ** 2
        spectrum = spectrum[1:]
        probability = spectrum / spectrum.sum() if spectrum.sum() > 0 else np.empty(0)
        spectral_entropy = float(-np.sum(probability * np.log(probability + 1e-12))) if probability.size else 0.0
        autocorr = lag_correlations(wrist_speed, wrist_speed)
        periodic_values = [abs(value) for lag, value in autocorr.items() if value is not None and abs(int(lag)) > 0]
        observations.append(
            {
                "motion_energy": float(np.nanmean(speed)) if speed.size else 0.0,
                "max_speed": float(np.nanmax(speed)) if speed.size else 0.0,
                "acceleration_energy": float(np.nanmean(accel_norm)) if accel_norm.size else 0.0,
                "jerk_energy": float(np.nanmean(jerk_norm)) if jerk_norm.size else 0.0,
                "active_joint_ratio": active,
                "simultaneous_motion_ratio": simultaneous,
                "spectral_entropy": spectral_entropy,
                "periodicity_peak_abs_corr": max(periodic_values) if periodic_values else 0.0,
                "scale_median": scale,
            }
        )
    return observations


def imu_features(imu: np.ndarray) -> tuple[list[dict[str, float]], np.ndarray, str]:
    acc, layout = legacy_imu_acc(imu)
    observations: list[dict[str, float]] = []
    for person in range(acc.shape[1]):
        values = acc[:, person]
        magnitude = np.linalg.norm(values, axis=-1)
        jerk = np.diff(values, n=2, axis=0)
        observations.append(
            {
                "acc_mean": float(np.mean(magnitude)),
                "acc_std": float(np.std(magnitude)),
                "acc_energy": float(np.mean(magnitude**2)),
                "acc_jerk_energy": float(np.mean(np.linalg.norm(jerk, axis=-1))) if jerk.size else 0.0,
                "acc_axis_mean_abs": float(np.mean(np.abs(values))),
            }
        )
    return observations, acc, layout


def load_record(path: Path, baseline_path: Path | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as archive:
        skeleton_key = "skeleton" if "skeleton" in archive else "gt_skeleton"
        skeleton = normalize_skeleton(archive[skeleton_key])
        visibility = archive["visibility"] if "visibility" in archive else None
        if visibility is None and skeleton.shape[-1] == 3 and np.all((skeleton[..., 2] >= 0) & (skeleton[..., 2] <= 1)):
            visibility = skeleton[..., 2]
        if "imu" in archive:
            imu = archive["imu"]
        elif baseline_path is not None and baseline_path.exists():
            with np.load(baseline_path, allow_pickle=True) as baseline:
                imu = baseline["imu"]
        else:
            imu = None
        frame_ids = archive["frame_ids"] if "frame_ids" in archive else None
        source = str(archive["source"].item()) if "source" in archive else None
    return {
        "path": str(path),
        "skeleton": skeleton,
        "visibility": visibility,
        "imu": normalize_imu(imu) if imu is not None else None,
        "frame_ids": frame_ids,
        "source": source,
    }


def update_stats(target: dict[str, list[float]], observations: Iterable[dict[str, float]]) -> None:
    for observation in observations:
        for key, value in observation.items():
            if np.isfinite(value):
                target[key].append(float(value))


def summarize(values: dict[str, list[float]]) -> dict[str, Any]:
    return {key: {"count": len(item), **quantiles(item)} for key, item in sorted(values.items())}


def source_files(source: str) -> list[tuple[Path, str, Path | None]]:
    if source == "totalcapture_gt":
        return [(path, path.stem, None) for path in sorted(TOTAL_ROOT.glob("*.npz"))]
    if source == "egohumans_canonical":
        return [(path, path.stem, None) for path in sorted(EGO_ROOT.glob("*.npz"))]
    if source == "custom_canonical":
        paths = sorted(CUSTOM_ROOT.glob("fold*/sequences/*.npz"))
        def session_group(path: Path) -> str:
            tokens = path.stem.split("_")
            return "_".join(tokens[1:3]) if len(tokens) >= 3 else path.parent.parent.name

        return [(path, session_group(path), None) for path in paths]
    if source.startswith("s06_"):
        method = source.removeprefix("s06_")
        paths = sorted((S06_ROOT / method).glob("*.npz"))
        return [(path, method, S06_BASELINE_ROOT / path.name) for path in paths]
    raise KeyError(source)


def analyse_source(source: str, max_files: int | None) -> dict[str, Any]:
    files = source_files(source)
    if max_files is not None:
        files = files[:max_files]
    skeleton_stats: dict[str, list[float]] = defaultdict(list)
    imu_stats: dict[str, list[float]] = defaultdict(list)
    lag_values: list[int] = []
    lag_corr_values: list[float] = []
    representation_counts: dict[str, int] = defaultdict(int)
    imu_layout_counts: dict[str, int] = defaultdict(int)
    frame_mismatch = 0
    no_imu = 0
    records = 0
    people = 0
    frames = 0
    group_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"records": 0, "frames": 0, "skeleton": defaultdict(list), "imu": defaultdict(list)})
    motion_observations: list[dict[str, float]] = []
    for path, group, baseline in files:
        try:
            record = load_record(path, baseline)
            skeleton = record["skeleton"]
            representation_name = representation(skeleton, record["visibility"])
            representation_counts[representation_name] += 1
            records += 1
            people += skeleton.shape[1]
            frames += skeleton.shape[0]
            skeleton_observations = skeleton_features(skeleton)
            motion_observations.extend(skeleton_observations)
            update_stats(skeleton_stats, skeleton_observations)
            bucket = group_stats[group]
            bucket["records"] += 1
            bucket["frames"] += skeleton.shape[0]
            update_stats(bucket["skeleton"], skeleton_observations)
            if record["imu"] is None:
                no_imu += 1
                continue
            imu = record["imu"]
            if imu.shape[0] != skeleton.shape[0]:
                frame_mismatch += 1
                continue
            imu_observations, acc, layout = imu_features(imu)
            imu_layout_counts[layout] += 1
            update_stats(imu_stats, imu_observations)
            update_stats(bucket["imu"], imu_observations)
            n_people = min(skeleton.shape[1], acc.shape[1])
            for person in range(n_people):
                coords = skeleton[:, person, ..., :2] if representation_name.startswith("2d_") else skeleton[:, person]
                wrist_speed = np.mean(np.linalg.norm(np.diff(coords[:, WRISTS], axis=0), axis=-1), axis=-1)
                acc_magnitude = np.linalg.norm(acc[:, person], axis=-1)
                correlations = lag_correlations(wrist_speed, acc_magnitude)
                available = [(int(lag), value) for lag, value in correlations.items() if value is not None]
                if available:
                    best_lag, best_corr = max(available, key=lambda item: abs(item[1]))
                    lag_values.append(best_lag)
                    lag_corr_values.append(best_corr)
        except (OSError, ValueError, KeyError, EOFError):
            frame_mismatch += 1
    groups = {}
    for group, values in sorted(group_stats.items()):
        groups[group] = {
            "records": values["records"],
            "frames": values["frames"],
            "skeleton_features": summarize(values["skeleton"]),
            "imu_features": summarize(values["imu"]),
        }
    motion_values = np.asarray([item["motion_energy"] for item in motion_observations], dtype=np.float64)
    if motion_values.size:
        low_cut, high_cut = np.quantile(motion_values, [1 / 3, 2 / 3])
        bins: dict[str, list[dict[str, float]]] = {"low": [], "mid": [], "high": []}
        for item in motion_observations:
            bucket = "low" if item["motion_energy"] <= low_cut else "high" if item["motion_energy"] > high_cut else "mid"
            bins[bucket].append(item)
        complexity_bins = {
            name: {"observations": len(items), "features": summarize({key: [row[key] for row in items] for key in items[0]})}
            for name, items in bins.items()
            if items
        }
    else:
        complexity_bins = {}
    return {
        "records": records,
        "people": people,
        "frames": frames,
        "representation_counts": dict(sorted(representation_counts.items())),
        "imu_layout_counts": dict(sorted(imu_layout_counts.items())),
        "records_without_imu": no_imu,
        "frame_mismatch_or_read_errors": frame_mismatch,
        "skeleton_features": summarize(skeleton_stats),
        "complexity_bins_by_source_tertile": complexity_bins,
        "imu_features": summarize(imu_stats),
        "cross_modal_lag": {
            "definition": "positive lag means skeleton wrist speed[t+lag] is compared with IMU acceleration[t]",
            "observations": len(lag_values),
            "best_lag_quantiles": quantiles([float(value) for value in lag_values]),
            "best_abs_corr_quantiles": quantiles([abs(value) for value in lag_corr_values]),
            "signed_best_corr_quantiles": quantiles(lag_corr_values),
        },
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e2_multimodal"))
    parser.add_argument("--max-files", type=int, default=None, help="optional smoke limit per source")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = [
        "totalcapture_gt",
        "egohumans_canonical",
        "custom_canonical",
        "s06_alphapose",
        "s06_fmpose3d",
        "s06_motionagformer",
        "s06_tcpformer",
        "s06_wham",
    ]
    report = {
        "schema_version": "g9-e2-multimodal-1",
        "input_audits": {
            str(path): sha256(path)
            for path in (
                Path("/data/fzliang/reid-project/g9/e1_gap_audit/semantic_audit.json"),
                Path("/data/fzliang/reid-project/g9/e1_gap_audit/gap_profile.json"),
            )
            if path.exists()
        },
        "protocol": {"window_len": 24, "stride": 16, "lags": list(LAGS), "skeleton_joint_order": "H36M17"},
        "source_roots": {
            "totalcapture_gt": str(TOTAL_ROOT),
            "egohumans_canonical": str(EGO_ROOT),
            "custom_canonical": str(CUSTOM_ROOT),
            "s06_algorithm_outputs": str(S06_ROOT),
            "s06_baseline_for_external_imu_join": str(S06_BASELINE_ROOT),
        },
        "imu_schema_and_timing_evidence": {
            "totalcapture_gt": {
                "declared_schema": "7d = acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z",
                "sensor_location": "L_LowArm",
                "source": "preprocess/datasets/totalcapture.py and configs/g6/totalcapture_source.yaml",
                "timing": "NPZ has frame_ids but no explicit sampling-rate field; raw Xsens timestamps/rate require separate provenance check.",
            },
            "egohumans_canonical": {
                "declared_schema": "7d = acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z",
                "sensor_location": "LeftWrist",
                "source": "preprocess/datasets/egohumans.py and configs/g6/egohumans_source.yaml",
                "timing": "NPZ has frame_ids but no explicit sampling-rate field; S06 manifest declares 20 fps for its camera stream.",
            },
            "custom_canonical": {
                "declared_schema": "7d from raw CSV: acceleration in g converted by 9.80665, quaternion q0..q3 interpreted as wxyz",
                "raw_timing_observation": "raw CSV median epoch_ms interval is 100 ms (10 Hz); video frame timestamp median is about 33.35 ms (30 Hz); preprocessing resamples to frame windows.",
                "source": "preprocess/common/imu.py, preprocess/datasets/custom.py and raw CSV/frame timestamp manifests",
            },
            "s06_legacy_baseline": {
                "declared_schema": "48d = four 9d rotation-matrix blocks followed by four 3d acceleration blocks",
                "selected_sensor": "L_LowArm (sensor index 2; acceleration slice 42:45)",
                "source": "preprocess/datasets/custom.py::legacy_imu48_sensor_to_7d and src/datasets/transforms.py",
                "timing": "S06 train/val manifests declare 20 fps; output frame_ids are inherited from baseline.",
            },
        },
        "sources": {source: analyse_source(source, args.max_files) for source in sources},
        "limitations": [
            "Motion magnitudes are bone-scale normalized within each record; raw coordinate-space values are not pooled.",
            "EgoHumans and AlphaPose/YOLO-Pose are treated as 2D xy tracks with visibility, not 3D merely because the array has last dimension 3.",
            "S06 IMU is joined from the verified baseline NPZ by sequence filename; algorithm output NPZ does not embed imu_ids.",
            "Lag correlation is a screening statistic, not proof of causality or a replacement for rendered frame review.",
        ],
    }
    output = args.output_root / "multimodal_motion_diagnostics.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sources": list(report["sources"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
