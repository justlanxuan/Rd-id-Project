# Experiment Note: C1-imu-contract
"""Compare all included IMU streams after an explicit 7D conversion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from preprocess.datasets.custom import legacy_imu48_sensor_to_7d

TOTAL_ROOT = Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences")
EGO_ROOT = Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source/sequences")
CUSTOM_ROOT = Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess")
S06_ROOT = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs")
S06_BASELINE_ROOT = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/data/interim/egohumans_repro_local/slice/sequences")


def q(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "q05": None, "q50": None, "q95": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "q05": float(np.quantile(x, 0.05)), "q50": float(np.quantile(x, 0.5)), "q95": float(np.quantile(x, 0.95))}


def files_for(source: str) -> list[tuple[Path, Path | None]]:
    if source == "totalcapture_gt":
        return [(p, None) for p in sorted(TOTAL_ROOT.glob("*.npz"))]
    if source == "egohumans_canonical":
        return [(p, None) for p in sorted(EGO_ROOT.glob("*.npz"))]
    if source == "custom_canonical":
        return [(p, None) for p in sorted(CUSTOM_ROOT.glob("fold*/sequences/*.npz"))]
    method = source.removeprefix("s06_")
    return [(p, S06_BASELINE_ROOT / p.name) for p in sorted((S06_ROOT / method).glob("*.npz"))]


def to_7d(path: Path, baseline: Path | None) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=True) as archive:
        if "imu" in archive:
            imu = np.asarray(archive["imu"], dtype=np.float32)
            layout = "embedded_7d" if imu.shape[-1] == 7 else f"embedded_{imu.shape[-1]}d"
        else:
            if baseline is None or not baseline.exists():
                raise FileNotFoundError(path)
            with np.load(baseline, allow_pickle=True) as base:
                imu = np.asarray(base["imu"], dtype=np.float32)
            layout = "baseline_legacy48"
    if imu.ndim == 2:
        imu = imu[:, None, :]
    if imu.shape[-1] == 7:
        return imu.astype(np.float64), layout
    if imu.shape[-1] >= 48:
        return legacy_imu48_sensor_to_7d(imu, "L_LowArm").astype(np.float64), layout
    raise ValueError(f"unsupported IMU shape {imu.shape} for {path}")


def analyse(source: str) -> dict[str, Any]:
    acc_energy: list[float] = []
    acc_mean: list[float] = []
    acc_std: list[float] = []
    acc_jerk: list[float] = []
    quat_norm_error: list[float] = []
    quaternion_frames = 0
    invalid_quaternion_frames = 0
    zero_quaternion_frames = 0
    records_with_invalid_quaternion = 0
    channel_means: dict[str, list[float]] = defaultdict(list)
    channel_stds: dict[str, list[float]] = defaultdict(list)
    layout_counts: dict[str, int] = defaultdict(int)
    records = 0
    persons = 0
    for path, baseline in files_for(source):
        try:
            values, layout = to_7d(path, baseline)
        except (OSError, ValueError, EOFError, FileNotFoundError):
            continue
        layout_counts[layout] += 1
        records += 1
        persons += values.shape[1]
        acc = values[..., :3]
        quat = values[..., 3:7]
        acc_norm = np.linalg.norm(acc, axis=-1)
        jerk = np.diff(acc, n=2, axis=0)
        acc_energy.extend(np.mean(acc_norm**2, axis=0).reshape(-1).tolist())
        acc_mean.extend(np.mean(acc_norm, axis=0).reshape(-1).tolist())
        acc_std.extend(np.std(acc_norm, axis=0).reshape(-1).tolist())
        if jerk.size:
            acc_jerk.extend(np.mean(np.linalg.norm(jerk, axis=-1), axis=0).reshape(-1).tolist())
        norm = np.linalg.norm(quat, axis=-1)
        quat_norm_error.extend(np.abs(norm - 1.0).reshape(-1).tolist())
        quaternion_frames += int(norm.size)
        invalid = (norm < 0.9) | (norm > 1.1)
        invalid_quaternion_frames += int(np.sum(invalid))
        zero_quaternion_frames += int(np.sum(norm < 1e-6))
        records_with_invalid_quaternion += int(np.any(invalid))
        for idx in range(7):
            channel_means[str(idx)].extend(np.mean(values[..., idx], axis=0).reshape(-1).tolist())
            channel_stds[str(idx)].extend(np.std(values[..., idx], axis=0).reshape(-1).tolist())
    return {
        "records": records,
        "persons": persons,
        "layout_counts": dict(layout_counts),
        "quaternion_quality": {
            "frames": quaternion_frames,
            "invalid_frames_lt_0.9_or_gt_1.1": invalid_quaternion_frames,
            "zero_norm_frames": zero_quaternion_frames,
            "invalid_frame_fraction": invalid_quaternion_frames / quaternion_frames if quaternion_frames else None,
            "records_with_invalid_quaternion": records_with_invalid_quaternion,
        },
        "contract": "acc_x,acc_y,acc_z,quat_w,quat_x,quat_y,quat_z",
        "features": {
            "acc_energy": q(acc_energy),
            "acc_mean": q(acc_mean),
            "acc_std": q(acc_std),
            "acc_jerk_energy": q(acc_jerk),
            "quaternion_norm_abs_error": q(quat_norm_error),
            "channel_means": {key: q(value) for key, value in sorted(channel_means.items())},
            "channel_stds": {key: q(value) for key, value in sorted(channel_stds.items())},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e2_multimodal"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = ["totalcapture_gt", "egohumans_canonical", "custom_canonical", "s06_alphapose", "s06_fmpose3d", "s06_motionagformer", "s06_tcpformer", "s06_wham"]
    report = {
        "schema_version": "g9-e2-imu-contract-1",
        "conversion": {
            "canonical_7d": "kept as embedded acc3+quat4",
            "legacy48": "converted with legacy_imu48_sensor_to_7d(sensor=L_LowArm), extracting rotation matrix block 2 and acceleration slice 42:45",
            "no_model_or_filter": True,
        },
        "sources": {source: analyse(source) for source in sources},
        "limitations": [
            "A common 7D layout does not prove equal physical units or sensor coordinate frames; those remain explicit provenance factors.",
            "S06 outputs use baseline IMU by sequence filename because algorithm output NPZ does not embed imu_ids.",
        ],
    }
    output = args.output_root / "imu_contract_comparison.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sources": list(report["sources"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
