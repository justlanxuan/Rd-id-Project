"""Read-only G12 E3 orientation/gyro physical association audit.

This is deliberately an audit, not a training or pairing experiment.  It uses
the independent measured TotalCapture AuxFields gyro and the realistic-IMU
EgoHumans gyro.  Extractor outputs are inventoried separately; WHAM's raw
direct orientation is the only extractor orientation currently sharing an
unambiguous TotalCapture timeline.  Missing extractor/gyro timestamps remain
explicit null records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from features.orientation import derive_3d_torso_heading, direct_root_orientation  # noqa: E402

TC_CANON = Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences")
EGO_CANON = Path("/data/fzliang/reid-project/egohumans/preprocessed/egohumans_realistic_hybrid_source/sequences")
TC_AUX = Path("/data/yjliu/totalcapture")
EGO_RAW = Path("/data/lyxie/ReID_imu_generation/outputs/datasets/egohumans/realistic/extracted_data")
S06_ROOT = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/data/skeleton_aug/S06_source_ablation")
METHODS = ("yolopose_high", "alphapose", "fmpose3d", "motionagformer", "tcpformer", "wham")


def parse_aux(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = iter(path.read_text(encoding="utf-8").splitlines())
    n_sensor, n_frame = map(int, next(lines).split()[:2])
    acc, gyro = [], []
    for _ in range(n_frame):
        next(lines)
        found = False
        for _ in range(n_sensor):
            p = next(lines).split()
            if p[0] == "L_LowArm":
                acc.append([float(x) for x in p[5:8]])
                gyro.append([float(x) for x in p[8:11]])
                found = True
        if not found:
            raise ValueError(f"L_LowArm missing in {path}")
    return np.asarray(acc), np.asarray(gyro)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 8 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def association(orientation_rate: np.ndarray, gyro: np.ndarray, valid: np.ndarray, fps: float, max_lag_s: float = 1.0) -> dict:
    """Summarize zero-lag and lag-screened correlations with explicit masks."""
    r = np.asarray(orientation_rate, dtype=float).reshape(-1)
    g = np.asarray(gyro, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(r) & np.isfinite(g).all(axis=1)
    if len(r) != len(g):
        n = min(len(r), len(g))
        r, g, mask = r[:n], g[:n], mask[:n]
    idx = np.flatnonzero(mask)
    out = {"frames": int(len(r)), "valid_frames": int(mask.sum()), "valid_fraction": float(mask.mean()) if len(mask) else 0.0}
    if len(idx) < 8:
        out.update({"status": "insufficient_valid", "zero_lag": None, "best_lag": None, "motion_strata": None})
        return out
    def score(a: np.ndarray, b: np.ndarray) -> tuple[float | None, int | None, int | None]:
        vals = [pearson(a, b[:, j]) for j in range(3)]
        finite = [(abs(v), v, j) for j, v in enumerate(vals) if v is not None]
        if not finite:
            return None, None, None
        _, v, j = max(finite)
        return float(v), int(j), None
    zero, axis, _ = score(r[mask], g[mask])
    lag_rows = []
    max_lag = int(round(max_lag_s * fps))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b, m = r[lag:], g[:len(r)-lag] if lag else g, mask[lag:] & (mask[:len(r)-lag] if lag else mask)
        else:
            k = -lag
            a, b, m = r[:len(r)-k], g[k:], mask[:len(r)-k] & mask[k:]
        v, j, _ = score(a[m], b[m]) if m.sum() >= 8 else (None, None, None)
        if v is not None:
            lag_rows.append((abs(v), v, lag, j, int(m.sum())))
    best = max(lag_rows) if lag_rows else None
    norms = np.linalg.norm(g, axis=1)
    q50, q90 = np.nanquantile(norms[mask], [0.5, 0.9])
    strata = {}
    for name, sm in (("low", norms <= q50), ("high", norms >= q90)):
        mm = mask & sm
        strata[name] = {"valid_frames": int(mm.sum()), "gyro_norm_median": float(np.median(norms[mm])) if mm.any() else None,
                        "r_zero_lag": pearson(r[mm], norms[mm]) if mm.sum() >= 8 else None}
    # Circular time permutation null, preserving each stream's marginal values.
    rng = np.random.default_rng(20260821)
    null = []
    for _ in range(32):
        perm = rng.permutation(len(r))
        null.append(pearson(r[mask], g[perm][mask, 0]) if mask.sum() >= 8 else None)
    null = [x for x in null if x is not None]
    out.update({"status": "ok", "zero_lag": {"r": zero, "gyro_axis": axis},
                "best_lag": {"r": best[1], "lag_frames": best[2], "lag_seconds": best[2] / fps, "gyro_axis": best[3], "valid_frames": best[4]} if best else None,
                "motion_strata": strata, "shuffle_null": {"n": len(null), "abs_r95": float(np.quantile(np.abs(null), .95)) if null else None}})
    return out


def orientation_from_h36m(skeleton: np.ndarray, visibility: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    ts = np.arange(len(skeleton), dtype=float) / fps
    visibility = np.asarray(visibility, dtype=bool)
    if visibility.ndim == 1:
        visibility = np.repeat(visibility[:, None], skeleton.shape[1], axis=1)
    b = derive_3d_torso_heading(skeleton, ts, visibility=visibility, joint_names=("pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "spine", "thorax", "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"))
    return b.angle_rate, b.rate_valid, {"valid_fraction": float(b.rate_valid.mean()), "heading": "3d_derived_h36m17"}


def wham_orientation(path: Path, fps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=True)
    pose = np.asarray(z["pose_world"], dtype=float).reshape(-1, 24, 3)[:, 0]
    ts = np.arange(len(pose), dtype=float) / fps
    b = direct_root_orientation(ts, axis_angle=pose, up_axis=1, local_forward_axis=2,
                                coordinate_frame="wham_pose_world", orientation_source="wham_raw_pose_world")
    return b.angle_rate, b.rate_valid, {"valid_fraction": float(b.rate_valid.mean()), "heading": "wham_raw_pose_world_root_orient"}


def scan_s06() -> dict:
    rows = []
    for method in METHODS:
        files = sorted((S06_ROOT / method).glob("**/sequences/*.npz"))
        n, has_ts, has_gyro = 0, 0, 0
        for f in files:
            n += 1
            with np.load(f, allow_pickle=True) as z:
                has_ts += int("timestamps_s" in z.files or "timestamps" in z.files)
                has_gyro += int("gyroscope_rads" in z.files or "gyro" in z.files)
        rows.append({"extractor": method, "files": n, "timestamp_files": has_ts, "direct_gyro_files": has_gyro,
                     "status": "missing_timestamp_and_direct_gyro" if n and not has_ts and not has_gyro else "review"})
    return {"status": "explicit_null_control", "rows": rows}


def main() -> None:
    records = []
    tc_npz = TC_CANON / "totalcapture_S2_rom1_cam1.npz"
    aux = TC_AUX / "s2/rom1/rom1_Xsens_AuxFields.sensors"
    with np.load(tc_npz, allow_pickle=True) as z:
        sk = np.asarray(z["gt_skeleton_meters"][:, 0], dtype=float)
        vis = np.asarray(z["gt_visibility"][:, 0], dtype=bool)
    acc, gyro = parse_aux(aux)
    start, end = 60, min(360, len(sk), len(gyro))
    rate, valid, meta = orientation_from_h36m(sk[start:end], vis[start:end], 60.0)
    records.append({"dataset": "totalcapture", "source": "canonical_gt_skeleton_control", "sequence": "S2_rom1_cam1", "fps_hz": 60.0,
                    "coordinate_frame": "canonical_gt_meters_unspecified_world_semantics", "association": association(rate, gyro[start:end], valid, 60.0), **meta})
    wham = Path("/data/lyxie/ReID_imu_generation/outputs/wham/recon/processed/S2_rom1_wham_0060_0360_cam1.npz")
    rate, valid, meta = wham_orientation(wham, 60.0)
    records.append({"dataset": "totalcapture", "source": "wham_raw_direct_orientation", "sequence": "S2_rom1_cam1[60:360]", "fps_hz": 60.0,
                    "coordinate_frame": "WHAM_pose_world_metadata_required", "association": association(rate, gyro[start:end], valid, 60.0), **meta})
    # EgoHumans supplies four people and therefore a genuine shuffled-person null.
    ego = EGO_CANON / "egohumans_05_003.npz"
    with np.load(ego, allow_pickle=True) as z:
        sk = np.asarray(z["gt_skeleton_meters"], dtype=float)
        vis = np.asarray(z["gt_visibility"], dtype=bool)
    gyros = []
    for p in range(1, 5):
        raw = np.load(EGO_RAW / f"05_003_aria0{p}.npy", allow_pickle=True).item()
        gyros.append(np.asarray(raw["gyro"][:, 0], dtype=float))
    for person in range(min(4, sk.shape[1])):
        rate, valid, meta = orientation_from_h36m(sk[:, person], vis[:, person], 20.0)
        records.append({"dataset": "egohumans", "source": "canonical_gt_skeleton_control", "sequence": "05_003", "person": person,
                        "pairing": "matched", "fps_hz": 20.0, "coordinate_frame": "canonical_gt_meters_unspecified_world_semantics",
                        "association": association(rate, gyros[person], valid, 20.0), **meta})
        shuffled = gyros[(person + 1) % 4]
        records.append({"dataset": "egohumans", "source": "canonical_gt_skeleton_control", "sequence": "05_003", "person": person,
                        "pairing": "shuffled_person_null", "fps_hz": 20.0, "coordinate_frame": "canonical_gt_meters_unspecified_world_semantics",
                        "association": association(rate, shuffled, valid, 20.0), **meta})
    out = {"schema_version": "g12.e3.physical_orientation_gyro_audit.v1", "status": "complete_read_only_audit",
           "protocol": {"no_pairing_ablation": True, "lag_screen_seconds": 1.0, "shuffle_null_permutations": 32,
                        "gyro_units": "rad/s", "orientation_rate_units": "rad/s", "timestamp_policy": "declared_native_fps_grid with provenance"},
           "extractor_availability": scan_s06(), "records": records}
    out_path = Path("/data/fzliang/reid-project/g12/e3_physical_audit/orientation_gyro_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "records": len(records), "extractor_rows": len(out["extractor_availability"]["rows"])}, indent=2))


if __name__ == "__main__":
    main()
