#!/usr/bin/env python3
"""Prepare, run, and summarize G13 H4W++ inference-density experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ("171423", "171724", "172257", "172522")
STRIDES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
SEEDS = (0, 42, 123)
FULL_SESSION = {suffix: f"20260211_{suffix}" for suffix in SESSIONS}
TRAIN_SESSIONS = {
    suffix: ",".join(FULL_SESSION[other] for other in SESSIONS if other != suffix)
    for suffix in SESSIONS
}


def parse_ints(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of non-negative integers")
    return values


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[g13-density]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def prepare_one(args: argparse.Namespace, stride: int, suffix: str) -> None:
    output = args.prepared_root / f"stride_{stride}" / f"loso_{FULL_SESSION[suffix]}"
    command = [
        args.h4w_python,
        str(REPO_ROOT / "tools" / "prepare_custom_h4wpp.py"),
        "--raw-root",
        str(args.raw_root),
        "--tracks-root",
        str(args.tracks_root),
        "--output",
        str(output),
        "--extract-root",
        str(args.fullframe_root),
        "--h4w-root",
        str(args.h4w_root),
        "--checkpoint",
        str(args.checkpoint),
        "--h4w-python",
        args.h4w_python,
        "--frame-stride",
        str(stride),
        "--skip-existing",
        "--window-len",
        "24",
        "--stride",
        "16",
        "--train-sessions",
        TRAIN_SESSIONS[suffix],
        "--val-sessions",
        "",
        "--test-sessions",
        FULL_SESSION[suffix],
        "--skeleton-normalize",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    run(command, env=env)


def prepare_all(args: argparse.Namespace) -> None:
    jobs = [(stride, suffix) for stride in args.strides for suffix in SESSIONS]
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(prepare_one, args, stride, suffix) for stride, suffix in jobs]
        for future in futures:
            future.result()


def materialize_config(args: argparse.Namespace, stride: int, seed: int, suffix: str) -> Path:
    template_path = REPO_ROOT / "configs" / f"custom_h4wpp_fullframe_loso_{suffix}.yaml"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    run_root = args.artifact_root / "runs" / f"stride_{stride}" / f"seed_{seed}"
    run_name = f"test_{FULL_SESSION[suffix]}"
    config["project"] = f"g13_h4wpp_density_s{stride}_seed{seed}_{suffix}"
    config["preprocess"]["prepared_root"] = str(
        args.prepared_root / f"stride_{stride}" / f"loso_{FULL_SESSION[suffix]}"
    )
    config["train"]["seed"] = seed
    config["train"]["output"] = {"output_root": str(run_root), "run_name": run_name}
    config["test"]["output"] = {"output_root": str(run_root), "run_name": run_name}
    path = args.artifact_root / "configs" / f"stride_{stride}" / f"seed_{seed}" / f"fold_{suffix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_one(args: argparse.Namespace, gpu: str, stride: int, seed: int, suffix: str) -> None:
    config = materialize_config(args, stride, seed, suffix)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(REPO_ROOT)
    run(
        [
            args.train_python,
            str(REPO_ROOT / "run_pipeline.py"),
            "--config",
            str(config),
            "--stages",
            "train,test",
        ],
        env=env,
    )


def run_gpu_queue(args: argparse.Namespace, gpu: str, jobs: list[tuple[int, int, str]]) -> None:
    for stride, seed, suffix in jobs:
        result_path = (
            args.artifact_root
            / "runs"
            / f"stride_{stride}"
            / f"seed_{seed}"
            / f"test_{FULL_SESSION[suffix]}"
            / "results.json"
        )
        if args.resume and result_path.is_file():
            print(f"[g13-density] skip completed {stride=} {seed=} {suffix=}", flush=True)
            continue
        run_one(args, gpu, stride, seed, suffix)


def run_all(args: argparse.Namespace) -> None:
    jobs = [(stride, seed, suffix) for stride in args.strides for seed in args.seeds for suffix in SESSIONS]
    queues = [jobs[index :: len(args.gpus)] for index in range(len(args.gpus))]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_gpu_queue, args, gpu, queue) for gpu, queue in zip(args.gpus, queues, strict=True)]
        for future in futures:
            future.result()


def load_frameacc(args: argparse.Namespace, stride: int, seed: int, suffix: str) -> tuple[int, int, float]:
    path = (
        args.artifact_root
        / "runs"
        / f"stride_{stride}"
        / f"seed_{seed}"
        / f"test_{FULL_SESSION[suffix]}"
        / "results.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    metric = data["evaluations"]["frame_acc"]
    return int(metric["correct_assignments"]), int(metric["num_assignments"]), float(metric["frame_acc"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"selection_metric": "three_seed_mean_of_four_fold_macro_frameacc", "strides": {}}
    for stride in args.strides:
        seed_macros = []
        seed_weighted = []
        stride_record: dict[str, object] = {"seeds": {}}
        for seed in args.seeds:
            fold_values = []
            correct = 0
            total = 0
            fold_record = {}
            for suffix in SESSIONS:
                fold_correct, fold_total, value = load_frameacc(args, stride, seed, suffix)
                correct += fold_correct
                total += fold_total
                fold_values.append(value)
                fold_record[FULL_SESSION[suffix]] = {
                    "correct": fold_correct,
                    "total": fold_total,
                    "frame_acc": value,
                }
            macro = fmean(fold_values)
            weighted = correct / total
            seed_macros.append(macro)
            seed_weighted.append(weighted)
            stride_record["seeds"][str(seed)] = {
                "folds": fold_record,
                "macro_frame_acc": macro,
                "weighted_correct": correct,
                "weighted_total": total,
                "weighted_frame_acc": weighted,
            }
            rows.append(
                {
                    "inference_stride": stride,
                    "seed": seed,
                    "macro_frame_acc": macro,
                    "weighted_frame_acc": weighted,
                    "correct": correct,
                    "total": total,
                }
            )
        stride_record["three_seed_macro_mean"] = fmean(seed_macros)
        stride_record["three_seed_macro_std"] = pstdev(seed_macros)
        stride_record["three_seed_weighted_mean"] = fmean(seed_weighted)
        stride_record["three_seed_weighted_std"] = pstdev(seed_weighted)
        summary["strides"][str(stride)] = stride_record
    best_stride = max(args.strides, key=lambda value: summary["strides"][str(value)]["three_seed_macro_mean"])
    summary["best_stride"] = best_stride
    summary["best_three_seed_macro_mean"] = summary["strides"][str(best_stride)]["three_seed_macro_mean"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifact_root / "summary.json"
    seed_summary_path = args.artifact_root / "seed_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with seed_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest: dict[str, object] = {
        "protocol": {"strides": list(args.strides), "seeds": list(args.seeds), "sessions": list(FULL_SESSION.values())},
        "summary_sha256": sha256(summary_path),
        "seed_summary_sha256": sha256(seed_summary_path),
        "fullframe_skeletons": {},
        "prepared": {},
        "runs": {},
    }
    for suffix in SESSIONS:
        skeleton = args.fullframe_root / "extracts" / FULL_SESSION[suffix] / "skeleton.json"
        manifest["fullframe_skeletons"][FULL_SESSION[suffix]] = sha256(skeleton)
    for stride in args.strides:
        for suffix in SESSIONS:
            key = f"stride_{stride}/{FULL_SESSION[suffix]}"
            prepared = args.prepared_root / f"stride_{stride}" / f"loso_{FULL_SESSION[suffix]}"
            manifest["prepared"][key] = {
                "train_csv_sha256": sha256(prepared / "windows_train.csv"),
                "test_csv_sha256": sha256(prepared / "windows_test.csv"),
            }
            for seed in args.seeds:
                run_key = f"stride_{stride}/seed_{seed}/{FULL_SESSION[suffix]}"
                run_dir = args.artifact_root / "runs" / f"stride_{stride}" / f"seed_{seed}" / f"test_{FULL_SESSION[suffix]}"
                config = args.artifact_root / "configs" / f"stride_{stride}" / f"seed_{seed}" / f"fold_{suffix}.yaml"
                manifest["runs"][run_key] = {
                    "config_sha256": sha256(config),
                    "checkpoint_sha256": sha256(run_dir / "best.pt"),
                    "results_sha256": sha256(run_dir / "results.json"),
                }
    (args.artifact_root / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_stride": best_stride, "best_mean": summary["best_three_seed_macro_mean"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "all"), default="all")
    parser.add_argument("--strides", type=parse_ints, default=STRIDES)
    parser.add_argument("--seeds", type=parse_ints, default=SEEDS)
    parser.add_argument("--gpus", type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()), default=("0", "1", "2", "3"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-root", type=Path, default=Path(os.environ.get("REID_CUSTOM_RAW_ROOT", "/data/fzliang/custom/2person")))
    parser.add_argument("--tracks-root", type=Path, default=Path(os.environ.get("REID_TRACKS_ROOT", "/data/fzliang/reid-project/custom/skeleton/alphapose")))
    parser.add_argument("--fullframe-root", type=Path, default=Path("/data/fzliang/reid-project/custom/preprocessed/h4wpp_fullframe_w24"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/data/fzliang/reid-project/custom/preprocessed/h4wpp_density_w24"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/fzliang/reid-project/custom/artifacts/h4wpp_density_sweep"))
    parser.add_argument("--h4w-root", type=Path, default=Path(os.environ.get("REID_H4WPP_ROOT", REPO_ROOT / "third-party" / "Hand4Whole-plus-plus_RELEASE")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("REID_H4WPP_CHECKPOINT", REPO_ROOT / "models" / "hand4whole_plus_plus" / "snapshot_6.pth")))
    parser.add_argument("--h4w-python", default=os.environ.get("REID_H4WPP_PYTHON", sys.executable))
    parser.add_argument("--train-python", default=sys.executable)
    args = parser.parse_args()
    args.strides = tuple(args.strides)
    args.seeds = tuple(args.seeds)
    args.gpus = tuple(args.gpus)
    if args.stage in {"prepare", "all"}:
        prepare_all(args)
    if args.stage in {"run", "all"}:
        run_all(args)
    if args.stage in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
