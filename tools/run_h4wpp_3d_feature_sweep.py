#!/usr/bin/env python3
"""Run and summarize G13 H4W++ 2-D/3-D skeleton feature profiling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ("171423", "171724", "172257", "172522")
FEATURES = (
    "hybrid",
    "h36m2d",
    "h36m3d",
    "h36m3d_no_velocity",
    "h36m3d_bone",
    "h36m3d_heading",
    "h36m3d_heading_rate",
    "h36m3d_rotinv",
    "h36m3d_zonly",
    "h36m3d_geom",
    "h36m3d_left_wrist",
    "h36m3d_right_wrist",
    "h36m3d_both_wrist",
    "h36m3d_left_wrist_rot",
)
SEEDS = (0, 42, 123)
FULL_SESSION = {suffix: f"20260211_{suffix}" for suffix in SESSIONS}
TRAIN_SESSIONS = {
    suffix: ",".join(FULL_SESSION[other] for other in SESSIONS if other != suffix)
    for suffix in SESSIONS
}


def parse_values(spec: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_seeds(spec: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return values


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[g13-3d]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def materialize_config(args: argparse.Namespace, feature: str, seed: int, suffix: str) -> Path:
    template = REPO_ROOT / "configs" / f"custom_h4wpp_fullframe_loso_{suffix}.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    run_root = args.artifact_root / "runs" / safe_name(feature) / f"seed_{seed}"
    run_name = f"test_{FULL_SESSION[suffix]}"
    config["project"] = f"g13_h4wpp_3d_{safe_name(feature)}_seed{seed}_{suffix}"
    config["train"]["model"]["hybrid_skeleton_feature_mode"] = feature
    config["train"]["seed"] = seed
    config["train"]["output"] = {"output_root": str(run_root), "run_name": run_name}
    config["test"]["output"] = {"output_root": str(run_root), "run_name": run_name}
    path = args.artifact_root / "configs" / safe_name(feature) / f"seed_{seed}" / f"fold_{suffix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def result_path(args: argparse.Namespace, feature: str, seed: int, suffix: str) -> Path:
    return (
        args.artifact_root
        / "runs"
        / safe_name(feature)
        / f"seed_{seed}"
        / f"test_{FULL_SESSION[suffix]}"
        / "results.json"
    )


def run_one(args: argparse.Namespace, gpu: str, feature: str, seed: int, suffix: str) -> None:
    config = materialize_config(args, feature, seed, suffix)
    if args.resume and result_path(args, feature, seed, suffix).is_file():
        print(f"[g13-3d] skip completed {feature=} {seed=} {suffix=}", flush=True)
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO_ROOT)
    run(
        [args.train_python, str(REPO_ROOT / "run_pipeline.py"), "--config", str(config), "--stages", "train,test"],
        env=env,
    )


def run_gpu_queue(args: argparse.Namespace, gpu: str, jobs: list[tuple[str, int, str]]) -> None:
    for feature, seed, suffix in jobs:
        run_one(args, gpu, feature, seed, suffix)


def run_all(args: argparse.Namespace) -> None:
    jobs = [(feature, seed, suffix) for feature in args.features for seed in args.seeds for suffix in SESSIONS]
    queues = [jobs[index :: len(args.gpus)] for index in range(len(args.gpus))]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_gpu_queue, args, gpu, queue) for gpu, queue in zip(args.gpus, queues, strict=True)]
        for future in futures:
            future.result()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frameacc(path: Path) -> tuple[int, int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metric = data["evaluations"]["frame_acc"]
    return int(metric["correct_assignments"]), int(metric["num_assignments"]), float(metric["frame_acc"])


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "selection_metric": "three_seed_mean_of_four_fold_macro_frameacc",
        "features": {},
    }
    for feature in args.features:
        seed_macros: list[float] = []
        seed_weighted: list[float] = []
        record: dict[str, object] = {"seeds": {}}
        for seed in args.seeds:
            fold_values: list[float] = []
            correct = total = 0
            folds: dict[str, object] = {}
            for suffix in SESSIONS:
                fold_correct, fold_total, value = load_frameacc(result_path(args, feature, seed, suffix))
                correct += fold_correct
                total += fold_total
                fold_values.append(value)
                folds[FULL_SESSION[suffix]] = {"correct": fold_correct, "total": fold_total, "frame_acc": value}
            macro = fmean(fold_values)
            weighted = correct / total
            seed_macros.append(macro)
            seed_weighted.append(weighted)
            record["seeds"][str(seed)] = {
                "folds": folds,
                "macro_frame_acc": macro,
                "weighted_frame_acc": weighted,
                "correct": correct,
                "total": total,
            }
            rows.append(
                {
                    "feature": feature,
                    "seed": seed,
                    "macro_frame_acc": macro,
                    "weighted_frame_acc": weighted,
                    "correct": correct,
                    "total": total,
                }
            )
        record["three_seed_macro_mean"] = fmean(seed_macros)
        record["three_seed_macro_std"] = pstdev(seed_macros)
        record["three_seed_weighted_mean"] = fmean(seed_weighted)
        record["three_seed_weighted_std"] = pstdev(seed_weighted)
        summary["features"][feature] = record
    best = max(args.features, key=lambda value: summary["features"][value]["three_seed_macro_mean"])
    summary["best_feature"] = best
    summary["best_three_seed_macro_mean"] = summary["features"][best]["three_seed_macro_mean"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifact_root / "summary.json"
    seed_summary_path = args.artifact_root / "seed_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with seed_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest: dict[str, object] = {
        "protocol": {
            "features": list(args.features),
            "seeds": list(args.seeds),
            "sessions": list(FULL_SESSION.values()),
            "total_runs": len(args.features) * len(args.seeds) * len(SESSIONS),
            "selection_metric": "three_seed_mean_of_four_fold_macro_frameacc",
        },
        "code": {
            "runner_sha256": sha256(REPO_ROOT / "tools" / "run_h4wpp_3d_feature_sweep.py"),
            "hybrid_encoder_sha256": sha256(REPO_ROOT / "src" / "modules" / "encoders" / "hybrid.py"),
            "stats_sha256": sha256(REPO_ROOT / "src" / "engine" / "stats.py"),
            "protocol_lock_sha256": sha256(REPO_ROOT / "experiments" / "G13:H4WPP" / "E4:3d_feature_profiling" / "protocol-lock.md"),
        },
        "source": {
            "prepared_root": "/data/fzliang/reid-project/custom/preprocessed/h4wpp_fullframe_w24",
            "artifact_root": str(args.artifact_root),
        },
        "summary_sha256": sha256(summary_path),
        "seed_summary_sha256": sha256(seed_summary_path),
        "runs": {},
    }
    for feature in args.features:
        for seed in args.seeds:
            for suffix in SESSIONS:
                key = f"{feature}/seed_{seed}/{FULL_SESSION[suffix]}"
                run_dir = result_path(args, feature, seed, suffix).parent
                config = args.artifact_root / "configs" / safe_name(feature) / f"seed_{seed}" / f"fold_{suffix}.yaml"
                manifest["runs"][key] = {
                    "config_sha256": sha256(config),
                    "checkpoint_sha256": sha256(run_dir / "best.pt"),
                    "results_sha256": sha256(run_dir / "results.json"),
                }
    (args.artifact_root / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"best_feature": best, "best_mean": summary["best_three_seed_macro_mean"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--features", type=parse_values, default=FEATURES)
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument("--gpus", type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()), default=("2", "4", "5", "6"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/fzliang/reid-project/custom/artifacts/h4wpp_3d_feature_sweep"))
    parser.add_argument("--train-python", default=sys.executable)
    args = parser.parse_args()
    args.features = tuple(args.features)
    args.seeds = tuple(args.seeds)
    args.gpus = tuple(args.gpus)
    unknown = set(args.features) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unsupported features: {sorted(unknown)}; choose from {FEATURES}")
    if args.stage in {"run", "all"}:
        run_all(args)
    if args.stage in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
