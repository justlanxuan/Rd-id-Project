"""Single workflow entrypoint.

The production pipeline:

extract(optional) -> prepare(preprocess+pack+slice) -> train -> evaluate

All stage behavior is driven by the YACS config. Raw dataset paths are read-only;
derived artifacts are written under PATHS.DATA_HOME.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import resolve_config


DEFAULT_STAGES = ["extract", "prepare", "train", "evaluate"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    root = str(repo_root())
    src = str(repo_root() / "src")
    current = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = os.pathsep.join([p for p in (root, src, current) if p])
    return env


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root()), env=env_with_pythonpath())


def raw_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    return data


def stage_preprocess(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config_path)
    preprocess_cfg = cfg.get("preprocess", {})
    dataset = str(preprocess_cfg.get("dataset", "")).strip().lower()
    if dataset in {"totalcapture", "totalcapture_synthetic"}:
        from src.preprocess.datasets.totalcapture import run_preprocess

        manifest_csv = str(preprocess_cfg.get("output", "") or "")
        output_dir = str(Path(manifest_csv).parent) if manifest_csv else str(Path(cfg.get("work_dir", "")) / "preprocess")
        resolved_dir = run_preprocess(config_path, output_dir=output_dir, manifest_csv=manifest_csv or None)
        state["preprocess_dir"] = str(resolved_dir)
        return state

    modules = {
        "egohumans": "src.preprocess.datasets.egohumans",
        "custom": "src.preprocess.datasets.custom",
        "custom_plus": "src.preprocess.datasets.custom_plus",
        "custom+": "src.preprocess.datasets.custom_plus",
    }
    module = modules.get(dataset)
    if module is None:
        print(f"[INFO] Preprocess not implemented for dataset={dataset!r}; skipping.")
        return state

    manifest_csv = str(preprocess_cfg.get("output", "") or "")
    output_dir = str(Path(manifest_csv).parent) if manifest_csv else str(Path(cfg.get("work_dir", "")) / "preprocess")
    cmd = [sys.executable, "-m", module]
    if dataset == "egohumans":
        cmd.extend(["--task", "preprocess"])
    cmd.extend(["--config", str(config_path), "--output_dir", output_dir])
    if manifest_csv:
        cmd.extend(["--manifest_csv", manifest_csv])
    run_cmd(cmd)
    state["preprocess_dir"] = output_dir
    return state


def stage_extract(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    raw = raw_config(config_path)
    if not isinstance(raw.get("extract"), dict):
        print("[INFO] No explicit extract section; skipping.")
        return state

    cfg = resolve_config(config_path)
    dataset = str(cfg.get("preprocess", {}).get("dataset", "")).strip().lower()
    if dataset == "egohumans":
        run_cmd([sys.executable, "-m", "src.preprocess.datasets.egohumans", "--task", "extract", "--config", str(config_path)])
        return state

    extract_cfg = cfg.get("extract")
    if not isinstance(extract_cfg, dict):
        print("[INFO] No extract section; skipping.")
        return state

    copy_from = str(extract_cfg.get("copy_from", "") or "").strip()
    if copy_from:
        src = Path(copy_from).expanduser().resolve()
        dst = Path(extract_cfg.get("results_root", "") or Path(cfg.get("work_dir", "")) / "extract").expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"extract.copy_from not found: {src}")
        if dst.exists() and bool(extract_cfg.get("skip_existing", True)):
            print(f"[INFO] Extract copy skipped (exists): {dst}")
            return state
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[INFO] Copied extract results: {src} -> {dst}")
        return state

    run_cmd([sys.executable, str(repo_root() / "src" / "pipelines" / "video_pipeline" / "dispatcher.py"), "--config", str(config_path)])
    return state


def stage_slice(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config_path)
    dataset = str(cfg.get("preprocess", {}).get("dataset", "")).strip().lower()
    if dataset in {"totalcapture", "totalcapture_synthetic"}:
        from src.preprocess.common.slice import load_slice_cfg, run_slice_from_npz

        cfg = load_slice_cfg(str(config_path))
        root = Path(cfg.get("root", "/data/fzliang/totalcapture"))
        out_dir = Path(cfg.get("out_dir", "/data/fzliang/reid-project/totalcapture/preprocessed/default"))
        run_slice_from_npz(root, out_dir, cfg)
    elif dataset == "egohumans":
        module = "src.preprocess.datasets.egohumans"
    elif dataset in {"custom_plus", "custom+"}:
        module = "src.preprocess.datasets.custom_plus"
    elif dataset == "custom":
        module = "src.preprocess.datasets.custom"
    else:
        raise ValueError(f"Slice stage is not implemented for dataset={dataset!r}")
    if dataset not in {"totalcapture", "totalcapture_synthetic"}:
        run_cmd([sys.executable, "-m", module, "--task", "slice", "--config", str(config_path)])
    state["data_root"] = cfg.get("paths", {}).get("data_root", "")
    return state


def stage_pack(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config_path)
    dataset = str(cfg.get("preprocess", {}).get("dataset", "")).strip().lower()
    if dataset == "egohumans":
        run_cmd([sys.executable, "-m", "src.preprocess.datasets.egohumans", "--task", "pack", "--config", str(config_path)])
    else:
        print(f"[INFO] Pack stage not required for dataset={dataset!r}; skipping.")
    return state


def stage_prepare(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    state = stage_preprocess(config_path, state)
    state = stage_pack(config_path, state)
    return stage_slice(config_path, state)


def stage_train(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    run_cmd([sys.executable, "-m", "src.engine.train", "--config", str(config_path)])
    return state


def stage_evaluate(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    run_cmd([sys.executable, "-m", "src.engine.evaluate", "--config", str(config_path)])
    return state


STAGE_FUNCS = {
    "extract": stage_extract,
    "prepare": stage_prepare,
    "train": stage_train,
    "evaluate": stage_evaluate,
    # Compatibility aliases for old commands.
    "preprocess": stage_preprocess,
    "pack": stage_pack,
    "slice": stage_slice,
    "test": stage_evaluate,
}


def parse_stages(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(DEFAULT_STAGES)
    stages = [s.strip().lower() for s in spec.split(",") if s.strip()]
    available_stages = set(DEFAULT_STAGES) | {"preprocess", "pack", "slice", "test"}
    unknown = [s for s in stages if s not in available_stages]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Available: {sorted(available_stages)}")
    return stages


def run_pipeline(config_path: str | Path, stages: list[str] | None = None) -> dict[str, Any]:
    config = Path(config_path).expanduser().resolve()
    selected = stages or list(DEFAULT_STAGES)
    state: dict[str, Any] = {"config_path": config}
    print(f"[Pipeline] Config: {config}")
    print(f"[Pipeline] Stages : {selected}")
    for name in selected:
        print(f"\n========== Stage: {name} ==========")
        state = STAGE_FUNCS[name](config, state)
    print("\n========== Pipeline finished ==========")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IMU-video re-identification workflow")
    parser.add_argument("--config", required=True, help="Workflow YAML config.")
    parser.add_argument("--stages", default="all", help="all or comma-separated stage list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, parse_stages(args.stages))


if __name__ == "__main__":
    main()
