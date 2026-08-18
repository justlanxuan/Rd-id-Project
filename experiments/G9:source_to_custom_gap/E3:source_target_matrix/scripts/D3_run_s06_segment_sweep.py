# Experiment Note: D3-S06-source-sweep
"""Prepare/evaluate S06 skeleton outputs with a fixed G6 EgoHumans model.

The sweep is diagnostic: it fixes the baseline IMU and person order, keeps the
24/16 segment protocol, and evaluates raw versus screen-calibrated coordinates.
It does not retrain or alter any source artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

S06_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
)
BASELINE_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "data/interim/egohumans_repro_local/slice/sequences"
)
CHECKPOINT = Path(
    "/data/fzliang/reid-project/g6/c9a5d3099979296a72314eba66274855e03ab1eb/"
    "artifacts/train/train__source__egohumans__seed0/best.pt"
)
METHODS = ("alphapose", "yolopose_high", "fmpose3d", "motionagformer", "tcpformer", "wham")
VARIANTS = ("raw", "screen_calibrated")


def scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def screen_calibrate(skeleton: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Map root-centered torso-scaled xy to an EgoHumans-like screen space."""
    output = np.asarray(skeleton, dtype=np.float32).copy()
    xy = output[..., :2]
    root = xy[..., 0:1, :]
    centered = xy - root
    torso = np.linalg.norm(centered[..., 8, :] - centered[..., 0, :], axis=-1)
    scale = np.full_like(torso, 171.0)
    valid = np.isfinite(torso) & (torso > 1e-6)
    np.divide(171.0, torso, out=scale, where=valid)
    xy = centered * scale[..., None, None] + np.asarray([960.0, 540.0], dtype=np.float32)
    xy = np.where(np.isfinite(xy), xy, 0.0)
    output[..., :2] = xy
    if output.shape[-1] >= 3:
        output[..., 2] = 0.0
    return output


def prepare_segments(method: str, variant: str, output_root: Path, sessions: list[str]) -> Path:
    method_root = output_root / f"{method}__{variant}"
    method_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted((S06_ROOT / method).glob("*.npz")):
        sequence = source_path.stem
        if not sequence.startswith("custom_"):
            continue
        session = sequence.removeprefix("custom_")
        if sessions and session not in sessions:
            continue
        baseline_path = BASELINE_ROOT / source_path.name
        if not baseline_path.exists():
            continue
        with np.load(source_path, allow_pickle=True) as source, np.load(baseline_path, allow_pickle=True) as baseline:
            skeleton = np.asarray(source["skeleton"], dtype=np.float32)
            visibility = np.asarray(source["visibility"], dtype=bool)
            if variant == "screen_calibrated":
                skeleton = screen_calibrate(skeleton, visibility)
            elif variant != "raw":
                raise ValueError(f"Unknown variant {variant}")
            t_len, n_people = skeleton.shape[:2]
            gt_ids = np.asarray(baseline["gt_person_ids"], dtype=np.int64)
            gt_visibility = np.asarray(baseline["gt_visibility"], dtype=bool)[:t_len, :n_people]
            mapping = np.tile(np.arange(n_people, dtype=np.int64), (t_len, 1))
            payload = {
                "sequence_id": np.asarray(f"{sequence}_seg0", dtype=object),
                "dataset": np.asarray("egohumans", dtype=object),
                "frame_ids": np.asarray(baseline["frame_ids"], dtype=np.int64)[:t_len],
                "imu": np.asarray(baseline["imu"], dtype=np.float32)[:t_len, :n_people],
                "imu_ids": gt_ids[:n_people],
                "gt_person_ids": gt_ids[:n_people],
                "gt_bboxes": np.asarray(baseline["gt_bboxes"], dtype=np.float32)[:t_len, :n_people],
                "gt_visibility": gt_visibility,
                "gt_skeleton": np.asarray(baseline["gt_skeleton"], dtype=np.float32)[:t_len, :n_people],
                "extract_person_ids": gt_ids[:n_people],
                "extract_bboxes": np.asarray(baseline["gt_bboxes"], dtype=np.float32)[:t_len, :n_people],
                "extract_visibility": visibility[:t_len, :n_people],
                "extract_skeleton": skeleton[:t_len, :n_people],
                "gt_to_extract_map": mapping,
                "extract_source": np.asarray(method, dtype=object),
                "s06_baseline_path": np.asarray(str(baseline_path), dtype=object),
            }
        np.savez_compressed(method_root / f"{sequence}_seg0.npz", **payload)
    return method_root


def evaluate_variant(method: str, variant: str, segment_root: Path, sessions: list[str], device: str, output_root: Path) -> dict[str, Any]:
    import torch

    from src.config import load_cfg
    from src.engine.evaluate import evaluate_segment_frameacc

    cfg = load_cfg("configs/g6/egohumans_source.yaml")
    cfg.defrost()
    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    frame_cfg.MODE = "segment"
    frame_cfg.SEGMENT_ROOT = str(segment_root)
    frame_cfg.SESSIONS = tuple(sessions)
    frame_cfg.WINDOW_SIZE = 24
    frame_cfg.STRIDE = 16
    frame_cfg.PER_WINDOW_FEATURES = False
    frame_cfg.CUSTOM_IMU_ROOT = ""
    frame_cfg.ENABLED = True
    cfg.TRAIN.DEVICE = device
    cfg.TEST.DEVICE = device
    cfg.freeze()
    result = evaluate_segment_frameacc(cfg, CHECKPOINT, device=torch.device(device))
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{method}__{variant}.json"
    output.write_text(json.dumps({"method": method, "variant": variant, "device": device, "checkpoint": str(CHECKPOINT), "sessions": sessions, "evaluation": result}, indent=2, ensure_ascii=False) + "\n")
    return {"method": method, "variant": variant, "output": str(output), "correct": result["correct"], "total": result["total"], "weighted_frame_acc": result["weighted_frame_acc"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/s06_segments"))
    parser.add_argument("--result-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/s06_eval"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--sessions", nargs="*", default=[], help="session suffixes such as 01_003; empty means all available")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sessions = list(args.sessions)
    if not sessions:
        sessions = sorted({path.stem.removeprefix("custom_") for path in (S06_ROOT / args.methods[0]).glob("custom_*.npz")})
    prepared = []
    for method in args.methods:
        for variant in args.variants:
            root = prepare_segments(method, variant, args.prepare_root, sessions)
            prepared.append({"method": method, "variant": variant, "segment_root": str(root)})
    if args.prepare_only:
        print(json.dumps({"prepared": prepared}, indent=2, ensure_ascii=False))
        return 0
    results = []
    for item in prepared:
        results.append(evaluate_variant(item["method"], item["variant"], Path(item["segment_root"]), sessions, args.device, args.result_root))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    summary = args.result_root / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"checkpoint": str(CHECKPOINT), "sessions": sessions, "results": results}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": str(summary), "results": len(results)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
