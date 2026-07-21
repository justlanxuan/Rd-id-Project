"""Run repository benchmarks through the public train/evaluate entrypoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML document: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    root = str(repo_root())
    env["PYTHONPATH"] = os.pathsep.join([root, str(repo_root() / "src"), env.get("PYTHONPATH", "")])
    return env


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(repo_root()), env=env_with_pythonpath())


def command_with_device(cmd: list[str], device: str) -> list[str]:
    if not device:
        return cmd
    return [*cmd, "--device", device]


def parse_devices(devices: str, fallback: str = "") -> list[str]:
    values = [x.strip() for x in devices.split(",") if x.strip()]
    if values:
        return values
    return [fallback] if fallback else [""]


def devices_from_config(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value or "")


def run_queue(
    tasks: list[tuple[str, list[str], Path]],
    devices: list[str],
    max_parallel: int,
    dry_run: bool,
) -> None:
    """Run shell-free subprocess tasks across devices."""
    max_parallel = max(1, min(int(max_parallel or len(devices)), len(devices), len(tasks)))
    if dry_run:
        for idx, (_, cmd, _) in enumerate(tasks):
            print("[RUN]", " ".join(command_with_device(cmd, devices[idx % max_parallel])))
        return

    pending = list(tasks)
    running: list[tuple[str, subprocess.Popen, object, str, Path]] = []
    failures: list[str] = []
    env = env_with_pythonpath()

    while pending or running:
        busy_devices = {device for _, _, _, device, _ in running}
        free_devices = [device for device in devices[:max_parallel] if device not in busy_devices]
        while pending and len(running) < max_parallel and free_devices:
            device = free_devices.pop(0)
            name, base_cmd, log_path = pending.pop(0)
            cmd = command_with_device(base_cmd, device)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
            log_f.write("[RUN] " + " ".join(cmd) + "\n")
            log_f.flush()
            print(f"[START] {name} on {device or 'default'} | log={log_path}")
            proc = subprocess.Popen(cmd, cwd=str(repo_root()), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((name, proc, log_f, device, log_path))

        next_running: list[tuple[str, subprocess.Popen, object, str, Path]] = []
        for name, proc, log_f, device, log_path in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((name, proc, log_f, device, log_path))
                continue
            log_f.close()
            if rc == 0:
                print(f"[DONE] {name} on {device or 'default'}")
            else:
                failures.append(f"{name} failed with exit code {rc}; see {log_path}")
                print(f"[FAIL] {failures[-1]}")
        running = next_running

        if failures:
            for _, proc, log_f, _, _ in running:
                proc.terminate()
                log_f.close()
            raise RuntimeError("\n".join(failures))

        if running:
            import time

            time.sleep(10)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in ("imu_npz_path", "imu_window_start", "imu_window_end"):
        if key not in fieldnames:
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_label_shuffle_csv(src_csv: Path, dst_csv: Path, seed: int) -> None:
    rows = read_rows(src_csv)
    if len(rows) < 2:
        raise ValueError(f"Need at least two rows for label shuffle: {src_csv}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    if np.any(perm == np.arange(len(rows))):
        perm = np.roll(perm, 1)

    out_rows: list[dict[str, str]] = []
    for row, imu_row in zip(rows, (rows[int(i)] for i in perm)):
        out = dict(row)
        out["imu_npz_path"] = imu_row["npz_path"]
        out["imu_window_start"] = imu_row["window_start"]
        out["imu_window_end"] = imu_row["window_end"]
        out["imu_idx"] = imu_row.get("imu_idx", "0")
        out_rows.append(out)
    write_rows(dst_csv, out_rows)


def select_ints(spec: str, default: Iterable[int]) -> list[int]:
    if not spec:
        return list(default)
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def select_folds(spec: str, folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spec:
        return folds
    wanted = {x.strip() for x in spec.split(",") if x.strip()}
    selected = [
        f
        for f in folds
        if str(f["name"]) in wanted or str(f["session"]) in wanted or str(f["name"]).split("_", 1)[0] in wanted
    ]
    if not selected:
        raise ValueError(f"No folds matched {sorted(wanted)}")
    return selected


def check_inputs(bench: dict[str, Any]) -> None:
    data = bench["data"]
    files = [
        Path(data["source_root"]) / "windows_train.csv",
        Path(data["source_root"]) / "windows_val.csv",
        Path(data["source_root"]) / "windows_test.csv",
    ]
    for fold in bench["runner"]["folds"]:
        fold_root = Path(data["target_folds_root"]) / str(fold["name"])
        files.extend([fold_root / "windows_train.csv", fold_root / "windows_val.csv", fold_root / "windows_test.csv"])

    missing = [str(p) for p in files if not p.exists()]
    segment_root = Path(data["target_segment_root"])
    if not segment_root.exists() or not any(segment_root.glob("custom_*.npz")):
        missing.append(f"{segment_root}/*.npz")
    if not Path(data["target_raw_imu_root"]).exists():
        missing.append(str(data["target_raw_imu_root"]))
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))
    print("[OK] Benchmark inputs are present.")


def has_source_cache(root: Path) -> bool:
    return all((root / name).exists() for name in ("windows_train.csv", "windows_val.csv", "windows_test.csv"))


def has_target_cache(bench: dict[str, Any]) -> bool:
    target_root = Path(bench["data"]["target_folds_root"])
    for fold in bench["runner"]["folds"]:
        fold_root = target_root / str(fold["name"])
        if not all((fold_root / name).exists() for name in ("windows_train.csv", "windows_val.csv", "windows_test.csv")):
            return False
    return True


def has_segment_cache(segment_root: Path) -> bool:
    return segment_root.exists() and any(segment_root.glob("custom_*.npz"))


def prepare_source(bench: dict[str, Any], output_root: Path, dry_run: bool) -> None:
    prepare = bench.get("prepare", {}).get("source", {})
    source_root = Path(bench["data"]["source_root"])
    if has_source_cache(source_root) and bool(prepare.get("skip_existing", True)):
        print(f"[SKIP] Existing source cache: {source_root}")
        return
    if not bool(prepare.get("enabled", False)):
        print("[SKIP] Source prepare is disabled; using configured source_root.")
        return

    cfg = {
        "project": f"{bench['name']}_prepare_source",
        "preprocess": {
            "dataset": "egohumans",
            **dict(prepare.get("preprocess", {})),
        },
        "slice": {
            "root": str(Path(prepare.get("preprocess_output_root", source_root)).expanduser()),
            "out_dir": str(source_root),
            **dict(prepare.get("slice", {})),
        },
    }
    cfg_path = output_root / "generated_configs" / "prepare" / "source_prepare.yaml"
    write_yaml(cfg_path, cfg)
    run_cmd([sys.executable, "-m", "src.preprocess.datasets.egohumans", "--task", "preprocess", "--config", str(cfg_path)], dry_run)
    run_cmd([sys.executable, "-m", "src.preprocess.datasets.egohumans", "--task", "pack", "--config", str(cfg_path)], dry_run)
    run_cmd([sys.executable, "-m", "src.preprocess.datasets.egohumans", "--task", "slice", "--config", str(cfg_path)], dry_run)


def prepare_target(bench: dict[str, Any], output_root: Path, dry_run: bool) -> None:
    prepare = bench.get("prepare", {}).get("target", {})
    segment_root = Path(bench["data"]["target_segment_root"])
    target_root = Path(bench["data"]["target_folds_root"])
    skip_existing = bool(prepare.get("skip_existing", True))

    cfg = {
        "project": f"{bench['name']}_prepare_target",
        "preprocess": {
            "dataset": "custom",
            **dict(prepare.get("preprocess", {})),
        },
        "extract": dict(prepare.get("extract", {})),
        "segments": {
            "output_root": str(segment_root.parent),
            **dict(prepare.get("segments", {})),
        },
        "slice": {
            "segment_root": str(segment_root),
            "output_root": str(target_root),
            "custom_imu_root": str(bench["data"]["target_raw_imu_root"]),
            **dict(prepare.get("slice", {})),
        },
    }
    cfg_path = output_root / "generated_configs" / "prepare" / "target_prepare.yaml"
    write_yaml(cfg_path, cfg)

    if has_segment_cache(segment_root) and skip_existing:
        print(f"[SKIP] Existing target skeleton segment cache: {segment_root}")
    elif bool(prepare.get("enabled", False)):
        run_cmd([sys.executable, "-m", "src.preprocess.datasets.custom", "--config", str(cfg_path)], dry_run)
        run_cmd([sys.executable, "src/pipelines/video_pipeline/dispatcher.py", "--config", str(cfg_path)], dry_run)
        run_cmd([sys.executable, "-m", "src.preprocess.datasets.custom", "--task", "pack_segments", "--config", str(cfg_path)], dry_run)
    else:
        print("[SKIP] Target skeleton prepare is disabled; using configured target_segment_root.")

    if has_target_cache(bench) and skip_existing:
        print(f"[SKIP] Existing target fold cache: {target_root}")
        return
    run_cmd([sys.executable, "-m", "src.preprocess.datasets.custom", "--task", "slice", "--config", str(cfg_path)], dry_run)


def prepare_data(bench: dict[str, Any], output_root: Path, dry_run: bool) -> None:
    prepare_source(bench, output_root, dry_run)
    prepare_target(bench, output_root, dry_run)


def source_train_config(bench: dict[str, Any], output_root: Path, seed: int) -> tuple[Path, Path]:
    source_root = Path(bench["data"]["source_root"])
    train = bench["source_train"]
    run_name = f"{bench['name']}_source_seed{seed}"
    model = dict(train.get("model", {"type": "hybrid"}))
    cfg = {
        "project": f"{bench['name']}_source",
        "preprocess": {"dataset": "egohumans", "imu": {"lowpass_cutoff_hz": None, "lowpass_fs_hz": 30.0}},
        "paths": {
            "data_root": str(source_root),
            "train_csv": str(source_root / "windows_train.csv"),
            "val_csv": str(source_root / "windows_val.csv"),
            "test_csv": str(source_root / "windows_test.csv"),
        },
        "train": {
            "output": {"output_root": str(output_root / "train" / "source"), "run_name": run_name},
            "model": model,
            "epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "num_workers": int(train["num_workers"]),
            "compute_imu_stats": False,
            "seed": seed,
            "lr_heads": float(train.get("lr_heads", 5e-4)),
            "weight_decay": float(train.get("weight_decay", 1e-3)),
            "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
            "temperature": float(train.get("temperature", 0.07)),
            "learn_temperature": bool(train.get("learn_temperature", False)),
            "imu_noise_std": 0.0,
            "imu_dropout_prob": 0.0,
            "skel_noise_std": 0.0,
            "joint_dropout_prob": 0.0,
            "early_stop_patience": int(train["early_stop_patience"]),
            "early_stop_min_delta": float(train.get("early_stop_min_delta", 0.001)),
        },
    }
    path = output_root / "generated_configs" / "source" / f"{run_name}.yaml"
    write_yaml(path, cfg)
    return path, output_root / "train" / "source" / run_name / "best.pt"


def target_config(
    bench: dict[str, Any],
    output_root: Path,
    fold: dict[str, Any],
    seed: int,
    init_ckpt: Path,
    control: str = "",
) -> tuple[Path, Path]:
    data = bench["data"]
    train = bench["target_train"]
    model = dict(train["model"])
    eval_cfg = bench["evaluation"]["frame_acc"]
    fold_root = Path(data["target_folds_root"]) / str(fold["name"])
    run_name = f"{bench['name']}_{fold['name']}_seed{seed}"
    train_csv = fold_root / "windows_train.csv"
    val_csv = fold_root / "windows_val.csv"

    if control == "label_shuffle":
        run_name = f"{run_name}_label_shuffle"
        control_root = output_root / "controls" / "label_shuffle" / str(fold["name"]) / f"seed{seed}"
        train_csv = control_root / "windows_train.csv"
        make_label_shuffle_csv(fold_root / "windows_train.csv", train_csv, seed)
        if bool(bench.get("controls", {}).get("label_shuffle", {}).get("shuffle_val", True)):
            val_csv = control_root / "windows_val.csv"
            make_label_shuffle_csv(fold_root / "windows_val.csv", val_csv, seed + 10000)

    model["init_alignment_ckpt"] = str(init_ckpt)
    cfg = {
        "project": bench["name"],
        "preprocess": {"dataset": "custom", "imu": {"lowpass_cutoff_hz": None, "lowpass_fs_hz": 20.0}},
        "slice": {
            "window_len": 24,
            "stride": 8,
            "test_sessions": [str(fold["session"])],
            "skeleton_source": "gt",
            "skeleton_normalize": False,
            "multi_person": False,
        },
        "paths": {
            "data_root": str(fold_root),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(fold_root / "windows_test.csv"),
        },
        "train": {
            "output": {"output_root": str(output_root / "train" / "target"), "run_name": run_name},
            "model": model,
            "epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "num_workers": int(train["num_workers"]),
            "compute_imu_stats": False,
            "seed": int(seed),
            "lr_heads": float(train["lr_heads"]),
            "weight_decay": float(train["weight_decay"]),
            "max_grad_norm": float(train["max_grad_norm"]),
            "temperature": float(train["temperature"]),
            "learn_temperature": bool(train["learn_temperature"]),
            "pair_loss_weight": float(train.get("pair_loss_weight", 0.0)),
            "pair_loss_target": str(train["pair_loss_target"]),
            "group_batch_by_window": bool(train["group_batch_by_window"]),
            "imu_noise_std": 0.0,
            "imu_dropout_prob": 0.0,
            "skel_noise_std": 0.0,
            "joint_dropout_prob": 0.0,
            "early_stop_patience": int(train["early_stop_patience"]),
            "early_stop_min_delta": float(train["early_stop_min_delta"]),
        },
        "test": {
            "output": {"output_root": str(output_root / "evaluate" / "target"), "run_name": run_name},
            "batch_size": int(bench["evaluation"]["batch_size"]),
            "num_workers": int(bench["evaluation"]["num_workers"]),
            "metrics": {
                "frame_acc": {
                    "enabled": True,
                    "segment_root": str(data["target_segment_root"]),
                    "custom_imu_root": str(data["target_raw_imu_root"]),
                    "custom_imu_split_mode": str(eval_cfg["custom_imu_split_mode"]),
                    "custom_imu_raw_swap_sessions": list(eval_cfg["custom_imu_raw_swap_sessions"]),
                    "sessions": [str(fold["session"])],
                    "window_size": int(eval_cfg["window_size"]),
                    "stride": int(eval_cfg["stride"]),
                    "per_window_features": bool(eval_cfg["per_window_features"]),
                    "seed": int(eval_cfg["seed"]),
                    "shuffle_match": bool(eval_cfg["shuffle_match"]),
                },
                "group_test": {"enabled": False},
            },
        },
    }
    subdir = "target" if not control else f"control_{control}"
    path = output_root / "generated_configs" / subdir / f"{run_name}.yaml"
    write_yaml(path, cfg)
    return path, output_root / "evaluate" / "target" / run_name / "results.json"


def metric_values(paths: Iterable[Path]) -> list[float]:
    values: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metric = data.get("evaluations", {}).get("frame_acc", {}).get("frame_acc")
        if metric is not None:
            values.append(float(metric))
    return values


def summarize(output_root: Path) -> None:
    result_paths = sorted((output_root / "evaluate" / "target").glob("*/results.json"))
    normal = [p for p in result_paths if "label_shuffle" not in str(p)]
    control = [p for p in result_paths if "label_shuffle" in str(p)]
    summary: dict[str, Any] = {}
    for name, paths in (("target", normal), ("label_shuffle", control)):
        values = metric_values(paths)
        summary[name] = {
            "num_runs": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "result_files": [str(p) for p in paths],
        }
    out = output_root / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an official benchmark")
    parser.add_argument("--config", default=str(repo_root() / "configs" / "benchmarks" / "cross_dataset_transfer_sota.yaml"))
    parser.add_argument("--device", default="")
    parser.add_argument("--devices", default="", help="Comma-separated devices for queued target/eval jobs, e.g. cuda:0,cuda:1.")
    parser.add_argument("--max-parallel", type=int, default=0, help="Maximum concurrent queued jobs. Defaults to number of devices.")
    parser.add_argument("--folds", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--run-target", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench = load_yaml(Path(args.config))
    runner_cfg = bench["runner"]
    if not args.devices and not args.device:
        args.devices = devices_from_config(runner_cfg.get("devices", ""))
    if not args.max_parallel:
        args.max_parallel = int(runner_cfg.get("max_parallel", 0) or 0)
    args.include_controls = bool(args.include_controls or runner_cfg.get("include_controls", False))
    args.skip_existing = bool(args.skip_existing or runner_cfg.get("skip_existing", False))
    explicit_action = any(
        [
            args.check_inputs,
            args.prepare,
            args.generate,
            args.run_source,
            args.run_target,
            args.run_eval,
            args.summarize,
            args.run_all,
        ]
    )
    default_action = str(runner_cfg.get("default_action", "") or "").strip().lower()
    if not explicit_action and default_action == "run_all":
        args.run_all = True

    output_root = Path(bench["runner"]["output_root"]).expanduser().resolve()
    folds = select_folds(args.folds, list(bench["runner"]["folds"]))
    seeds = select_ints(args.seeds, [int(x) for x in bench["runner"]["seeds"]])
    devices = parse_devices(args.devices, args.device)
    source_device = args.device or devices[0]

    if args.run_all:
        args.prepare = args.check_inputs = args.generate = args.run_source = args.run_target = args.run_eval = args.summarize = True

    source_runs = {seed: source_train_config(bench, output_root, seed) for seed in seeds}
    generated: list[tuple[Path, Path]] = []
    for fold in folds:
        for seed in seeds:
            _, source_ckpt = source_runs[seed]
            generated.append(target_config(bench, output_root, fold, seed, source_ckpt))
            if args.include_controls:
                generated.append(target_config(bench, output_root, fold, seed, source_ckpt, control="label_shuffle"))

    if args.prepare:
        prepare_data(bench, output_root, args.dry_run)
    if args.check_inputs:
        check_inputs(bench)
    if args.generate:
        print(f"[OK] Generated {len(source_runs) + len(generated)} configs under {output_root / 'generated_configs'}")
    if args.run_source:
        tasks: list[tuple[str, list[str], Path]] = []
        for seed, (source_cfg, source_ckpt) in source_runs.items():
            if args.skip_existing and source_ckpt.exists():
                print(f"[SKIP] Existing source checkpoint: {source_ckpt}")
            else:
                tasks.append((
                    source_cfg.stem,
                    [sys.executable, "-m", "src.engine.train", "--config", str(source_cfg)],
                    output_root / "logs" / "source" / f"{source_cfg.stem}.log",
                ))
        if len(tasks) == 1:
            run_cmd(command_with_device(tasks[0][1], source_device), args.dry_run)
        elif tasks:
            run_queue(tasks, devices, args.max_parallel or len(devices), args.dry_run)
    if args.run_target:
        tasks: list[tuple[str, list[str], Path]] = []
        for config_path, _ in generated:
            ckpt = output_root / "train" / "target" / config_path.stem / "best.pt"
            if args.skip_existing and ckpt.exists():
                print(f"[SKIP] Existing target checkpoint: {ckpt}")
            else:
                tasks.append((
                    config_path.stem,
                    [sys.executable, "-m", "src.engine.train", "--config", str(config_path)],
                    output_root / "logs" / "train" / f"{config_path.stem}.log",
                ))
        if tasks:
            run_queue(tasks, devices, args.max_parallel or len(devices), args.dry_run)
    if args.run_eval:
        tasks = []
        for config_path, result_path in generated:
            if args.skip_existing and result_path.exists():
                print(f"[SKIP] Existing evaluation: {result_path}")
            else:
                tasks.append((
                    config_path.stem,
                    [
                        sys.executable,
                        "-m",
                        "src.engine.evaluate",
                        "--config",
                        str(config_path),
                        "--save_json",
                        str(result_path),
                    ],
                    output_root / "logs" / "eval" / f"{config_path.stem}.log",
                ))
        if tasks:
            run_queue(tasks, devices, args.max_parallel or len(devices), args.dry_run)
    if args.summarize:
        summarize(output_root)
    if not any([args.check_inputs, args.prepare, args.generate, args.run_source, args.run_target, args.run_eval, args.summarize]):
        check_inputs(bench)
        print("No action selected. Use --run-all to train and evaluate the full benchmark.")


if __name__ == "__main__":
    main()
