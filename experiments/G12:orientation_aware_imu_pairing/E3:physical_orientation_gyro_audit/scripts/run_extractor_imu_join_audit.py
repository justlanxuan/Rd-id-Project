"""Join S06 extractor orientation tracks to independent EgoHumans gyro.

Skeleton artifacts do not need to contain IMU channels.  The join key is the
shared EgoHumans sequence, sorted aria person order, and frame index.  The
realistic-IMU source supplies LeftWrist gyroscope in rad/s and target_fps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from features.orientation import derive_2d_torso_proxy, derive_3d_torso_heading  # noqa: E402

ALGORITHM_ROOT = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs")
IMU_ROOT = Path("/data/lyxie/ReID_imu_generation/outputs/datasets/egohumans/realistic/extracted_data")
OUTPUT = Path("/data/fzliang/reid-project/g12/e3_physical_audit/extractor_orientation_imu_join_corrected.json")
METHODS = {
    "yolopose_high": "2d_proxy",
    "alphapose": "2d_proxy",
    "fmpose3d": "3d_derived",
    "motionagformer": "3d_derived",
    "tcpformer": "3d_derived",
    "wham": "3d_derived",
}


def _pearson(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> float | None:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 12 or float(np.std(x[mask])) < 1e-10 or float(np.std(y[mask])) < 1e-10:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _smooth(values: np.ndarray, valid: np.ndarray, width: int = 5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    kernel = np.ones(width, dtype=float)
    if values.ndim == 1:
        num = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
        den = np.convolve(valid.astype(float), kernel, mode="same")
        out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    else:
        out = np.stack([_smooth(values[:, axis], valid, width)[0] for axis in range(values.shape[1])], axis=1)
        den = np.convolve(valid.astype(float), kernel, mode="same")
    return out, den >= max(3, width // 2 + 1)


def _association(rate: np.ndarray, rate_valid: np.ndarray, gyro: np.ndarray, fps: float) -> dict:
    n = min(len(rate), len(gyro))
    rate = np.asarray(rate[:n], dtype=float)
    gyro = np.asarray(gyro[:n], dtype=float)
    valid = np.asarray(rate_valid[:n], dtype=bool) & np.isfinite(gyro).all(axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    axis_r = [_pearson(rate, gyro[:, axis], valid) for axis in range(3)]
    finite_axis = [(abs(value), value, axis) for axis, value in enumerate(axis_r) if value is not None]
    best = max(finite_axis) if finite_axis else None
    magnitude_r = _pearson(np.abs(rate), gyro_norm, valid)

    smooth_rate, smooth_valid = _smooth(rate, valid)
    smooth_gyro, smooth_gyro_valid = _smooth(gyro, valid)
    smooth_valid &= smooth_gyro_valid
    smooth_norm = np.linalg.norm(smooth_gyro, axis=1)
    smooth_axis_r = [_pearson(smooth_rate, smooth_gyro[:, axis], smooth_valid) for axis in range(3)]
    smooth_finite = [(abs(value), value, axis) for axis, value in enumerate(smooth_axis_r) if value is not None]
    smooth_best = max(smooth_finite) if smooth_finite else None

    max_lag = int(round(fps))
    lag_rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            k = -lag
            x, y, mask = smooth_rate[:-k], smooth_gyro[k:], smooth_valid[:-k] & smooth_valid[k:]
        elif lag > 0:
            x, y, mask = smooth_rate[lag:], smooth_gyro[:-lag], smooth_valid[lag:] & smooth_valid[:-lag]
        else:
            x, y, mask = smooth_rate, smooth_gyro, smooth_valid
        values = [_pearson(x, y[:, axis], mask) for axis in range(3)]
        candidates = [(abs(value), value, axis) for axis, value in enumerate(values) if value is not None]
        if candidates:
            _, value, axis = max(candidates)
            lag_rows.append((abs(value), value, lag, axis, int(mask.sum())))
    lag_best = max(lag_rows) if lag_rows else None

    q50, q90 = np.quantile(gyro_norm[valid], [0.5, 0.9]) if valid.any() else (np.nan, np.nan)
    motion = {}
    for label, stratum in (("low", gyro_norm <= q50), ("high", gyro_norm >= q90)):
        mask = valid & stratum
        motion[label] = {
            "frames": int(mask.sum()),
            "abs_rate_vs_gyro_norm_r": _pearson(np.abs(rate), gyro_norm, mask),
        }

    rng = np.random.default_rng(20260821)
    time_null = []
    for _ in range(16):
        perm = rng.permutation(n)
        value = _pearson(np.abs(rate), gyro_norm[perm], valid)
        if value is not None:
            time_null.append(value)
    return {
        "frames": n,
        "valid_frames": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if n else 0.0,
        "raw": {
            "axis_r": axis_r,
            "best_abs_axis_r": best[1] if best else None,
            "best_axis": best[2] if best else None,
            "abs_rate_vs_gyro_norm_r": magnitude_r,
        },
        "smooth_0p25s": {
            "axis_r": smooth_axis_r,
            "best_abs_axis_r": smooth_best[1] if smooth_best else None,
            "best_axis": smooth_best[2] if smooth_best else None,
            "abs_rate_vs_gyro_norm_r": _pearson(np.abs(smooth_rate), smooth_norm, smooth_valid),
        },
        "best_lag_smooth": {
            "r": lag_best[1],
            "lag_frames": lag_best[2],
            "lag_seconds": lag_best[2] / fps,
            "axis": lag_best[3],
            "valid_frames": lag_best[4],
        } if lag_best else None,
        "motion_strata": motion,
        "time_shuffle_null": {
            "n": len(time_null),
            "abs_r95": float(np.quantile(np.abs(time_null), 0.95)) if time_null else None,
        },
    }


def _person_track(data: np.lib.npyio.NpzFile, person: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the GT-aligned person slot from an S06 algorithm output."""
    skeleton = np.asarray(data["skeleton"], dtype=float)
    visibility = np.asarray(data["visibility"], dtype=bool)
    points = skeleton[:, person]
    joint_valid = np.repeat(visibility[:, person, None], skeleton.shape[2], axis=1)
    return points, joint_valid


def _orientation(points: np.ndarray, joint_valid: np.ndarray, timestamps: np.ndarray, kind: str, method: str):
    if kind == "2d_proxy":
        return derive_2d_torso_proxy(
            points[..., :2], timestamps, visibility=joint_valid,
            coordinate_frame="extractor_image_xy", orientation_source=f"{method}_shoulder_line_proxy",
        )
    return derive_3d_torso_heading(
        points, timestamps, visibility=joint_valid, up_axis=1,
        coordinate_frame="root_centered_torso_scaled_h36m17_unspecified_camera_xyz",
        orientation_source=f"{method}_joint_derived_torso_heading",
    )


def _summary(records: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        rows = [row for row in records if row["extractor"] == method]
        paired = [row for row in rows if row["person_shuffle"] is not None]
        def values(path: tuple[str, ...], source: str = "matched", rows: list[dict] = rows) -> np.ndarray:
            result = []
            for row in rows:
                value = row[source]
                for key in path:
                    value = value.get(key) if value is not None else None
                if value is not None and np.isfinite(value):
                    result.append(abs(float(value)))
            return np.asarray(result)
        raw = values(("raw", "best_abs_axis_r"))
        smooth = values(("smooth_0p25s", "best_abs_axis_r"))
        mag = values(("smooth_0p25s", "abs_rate_vs_gyro_norm_r"))
        matched_pair = []
        shuffled_pair = []
        for row in paired:
            a = row["matched"]["smooth_0p25s"]["best_abs_axis_r"]
            b = row["person_shuffle"]["smooth_0p25s"]["best_abs_axis_r"]
            if a is not None and b is not None:
                matched_pair.append(abs(a))
                shuffled_pair.append(abs(b))
        matched_pair = np.asarray(matched_pair)
        shuffled_pair = np.asarray(shuffled_pair)
        lag_r = values(("best_lag_smooth", "r"))
        lag_s = values(("best_lag_smooth", "lag_seconds"))
        time_null = values(("time_shuffle_null", "abs_r95"))
        low_motion = values(("motion_strata", "low", "abs_rate_vs_gyro_norm_r"))
        high_motion = values(("motion_strata", "high", "abs_rate_vs_gyro_norm_r"))
        axes = [row["matched"]["smooth_0p25s"]["best_axis"] for row in rows]
        out[method] = {
            "orientation_kind": METHODS[method],
            "tracks": len(rows),
            "sequences": len({row["sequence"] for row in rows}),
            "median_valid_fraction": float(np.median([row["matched"]["valid_fraction"] for row in rows])) if rows else None,
            "median_abs_best_axis_r_raw": float(np.median(raw)) if len(raw) else None,
            "median_abs_best_axis_r_smooth_0p25s": float(np.median(smooth)) if len(smooth) else None,
            "median_abs_rate_vs_gyro_norm_r_smooth_0p25s": float(np.median(mag)) if len(mag) else None,
            "best_axis_histogram_smooth_0p25s": {str(axis): int(axes.count(axis)) for axis in (0, 1, 2)},
            "lag_control": {
                "median_abs_best_lag_r": float(np.median(lag_r)) if len(lag_r) else None,
                "median_abs_best_lag_seconds": float(np.median(lag_s)) if len(lag_s) else None,
            },
            "time_shuffle_control_median_abs_r95": float(np.median(time_null)) if len(time_null) else None,
            "motion_strata_median_abs_rate_vs_gyro_norm_r": {
                "low": float(np.median(low_motion)) if len(low_motion) else None,
                "high": float(np.median(high_motion)) if len(high_motion) else None,
            },
            "person_shuffle_control": {
                "tracks": int(len(matched_pair)),
                "matched_median_abs_r": float(np.median(matched_pair)) if len(matched_pair) else None,
                "shuffled_median_abs_r": float(np.median(shuffled_pair)) if len(shuffled_pair) else None,
                "median_matched_minus_shuffled_abs_r": float(np.median(matched_pair - shuffled_pair)) if len(matched_pair) else None,
                "matched_gt_shuffled_fraction": float(np.mean(matched_pair > shuffled_pair)) if len(matched_pair) else None,
            },
        }
    return out


def main() -> None:
    records, failures = [], []
    for method, kind in METHODS.items():
        files = sorted((ALGORITHM_ROOT / method).glob("*.npz"))
        for path in files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    sequence = str(data["sequence_id"].item()) if "sequence_id" in data.files else path.stem
                    session = sequence.removeprefix("custom_")
                    raw_paths = sorted(IMU_ROOT.glob(f"{session}_aria*.npy"))
                    n_person = len(np.asarray(data["gt_person_ids"]))
                    if len(raw_paths) != n_person:
                        raise ValueError(f"raw person count {len(raw_paths)} != canonical {n_person}")
                    raw = [np.load(raw_path, allow_pickle=True).item() for raw_path in raw_paths]
                    fps_values = [float(item["metadata"]["target_fps"]) for item in raw]
                    if len(set(fps_values)) != 1:
                        raise ValueError(f"mixed target_fps {fps_values}")
                    fps = fps_values[0]
                    frame_ids = np.asarray(data["frame_ids"], dtype=float)
                    timestamps = frame_ids / fps
                    if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) > 0):
                        raise ValueError("frame_ids are not strictly increasing")
                    for person in range(n_person):
                        points, joint_valid = _person_track(data, person)
                        track = _orientation(points, joint_valid, timestamps, kind, method)
                        gyro = np.asarray(raw[person]["gyro"][:, 0], dtype=float)
                        if len(gyro) != len(frame_ids):
                            raise ValueError(f"gyro frames {len(gyro)} != skeleton frames {len(frame_ids)}")
                        matched = _association(track.angle_rate, track.rate_valid, gyro, fps)
                        shuffled = None
                        if n_person > 1:
                            other = (person + 1) % n_person
                            shuffled = _association(
                                track.angle_rate, track.rate_valid,
                                np.asarray(raw[other]["gyro"][:, 0], dtype=float), fps,
                            )
                        records.append({
                            "extractor": method,
                            "orientation_kind": kind,
                            "sequence": sequence,
                            "person_index": person,
                            "aria_source": raw_paths[person].name,
                            "sensor": "LeftWrist",
                            "gyro_provenance": "egohumans_realistic_smpl_kinematic",
                            "fps_hz": fps,
                            "timeline": "shared frame index / raw metadata target_fps",
                            "matched": matched,
                            "person_shuffle": shuffled,
                        })
            except Exception as exc:
                failures.append({"extractor": method, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    output = {
        "schema_version": "g12.e3.extractor_orientation_imu_join.v2",
        "status": "corrected_extractor_to_external_imu_audit",
        "supersedes": "g12.e3.physical_orientation_gyro_audit.v1 extractor missing-gyro interpretation",
        "join_contract": {
            "skeleton_artifact": "S06 algorithm_outputs/<extractor>/*.npz (88 files/method; no P25/P50/P100 symlinked slices)",
            "sequence": "custom_XX_YYY skeleton -> XX_YYY_aria*.npy realistic IMU",
            "person": "sorted aria file order, matching original convert_realistic_to_pipeline.py",
            "frame": "same frame index and target_fps; exact length required",
            "skeleton_role": "orientation only",
            "imu_role": "LeftWrist gyroscope only",
            "gyro_units": "rad/s",
        },
        "summary": _summary(records),
        "records": records,
        "failures": failures,
        "no_pairing_ablation": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "records": len(records), "failures": len(failures), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
