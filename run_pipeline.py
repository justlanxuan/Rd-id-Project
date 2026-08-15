#!/usr/bin/env python3
"""Top-level pipeline runner.

It only owns argument parsing and stage ordering. Concrete work lives behind
the public workflow-stage registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.workflow import build_stage

DEFAULT_STAGES = ["preprocess", "train", "test"]


def dispatch_stage(config_path: Path, stage_name: str, state: dict[str, Any]) -> dict[str, Any]:
    return build_stage(stage_name, config_path).run(state)


def stage_preprocess(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    return dispatch_stage(config_path, "preprocess", state)


def stage_train(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    return dispatch_stage(config_path, "train", state)


def stage_test(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    return dispatch_stage(config_path, "test", state)


STAGE_FUNCS = {
    "train": stage_train,
    "preprocess": stage_preprocess,
    "test": stage_test,
}


def parse_stages(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(DEFAULT_STAGES)
    requested = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not requested:
        raise ValueError("At least one stage is required.")
    unknown = [s for s in requested if s not in DEFAULT_STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Available: {DEFAULT_STAGES}")
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stage: {duplicates}")
    return requested


def run_pipeline(config_path: str | Path, stages: list[str] | None = None) -> dict[str, Any]:
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config}")
    selected = list(DEFAULT_STAGES) if stages is None else list(stages)
    if not selected:
        raise ValueError("At least one pipeline stage is required.")
    unknown = [stage for stage in selected if stage not in STAGE_FUNCS]
    if unknown:
        raise ValueError(f"Unknown canonical stage(s): {unknown}")
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
    parser.add_argument(
        "--stages",
        default="all",
        help="all or an ordered comma-separated subset of preprocess,train,test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, parse_stages(args.stages))


if __name__ == "__main__":
    main()
