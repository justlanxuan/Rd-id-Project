"""Implementations of the three public pipeline stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from preprocess.adapters import build_dataset_adapter
from preprocess.adapters.validation import validate_preprocess_output
from preprocess.common.slice import run_slice_from_npz
from preprocess.derived import derive_sequences
from src.config import load_config

from .base import PipelineStage
from .runtime import run_command


class PreprocessStage(PipelineStage):
    """Adapt raw data and, when needed, create canonical windows."""

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        next_state = {} if state is None else dict(state)
        cfg = load_config(self.config_path)
        preprocess_cfg = cfg.get("preprocess", {})
        if not isinstance(preprocess_cfg, dict):
            raise TypeError("PREPROCESS must be a mapping.")

        dataset = str(preprocess_cfg.get("dataset", "")).strip().lower()
        if not dataset:
            raise ValueError("PREPROCESS.DATASET is required for the preprocess stage.")

        manifest_csv = str(preprocess_cfg.get("output", "") or "")
        output_dir = (
            str(Path(manifest_csv).parent)
            if manifest_csv
            else str(Path(cfg.get("work_dir", "")) / "preprocess")
        )
        artifact = build_dataset_adapter(dataset, self.config_path).preprocess(
            output_dir=output_dir,
            manifest_csv=manifest_csv or None,
        )

        data_root = artifact.output_dir
        derived_cfg = preprocess_cfg.get("derived", {})
        derived_enabled = isinstance(derived_cfg, dict) and bool(derived_cfg.get("enabled", False))
        if derived_enabled:
            derived_output = str(derived_cfg.get("output", "")).strip()
            if not derived_output:
                raise ValueError("PREPROCESS.DERIVED.OUTPUT was not resolved for an enabled derived-data stage")
            data_root = derive_sequences(artifact.output_dir, derived_output, derived_cfg)
            validate_preprocess_output(dataset, data_root)

        slice_cfg = cfg.get("slice", {})
        if not isinstance(slice_cfg, dict):
            raise TypeError(f"SLICE must be a mapping, got {type(slice_cfg).__name__}.")
        if slice_cfg and (not artifact.prepared or derived_enabled):
            run_slice_from_npz(data_root, data_root, slice_cfg)

        next_state.update(
            {
                "dataset": artifact.dataset,
                "preprocess_dir": str(data_root),
                "data_root": str(data_root),
                "prepared_cache": artifact.prepared,
                "derived_data": derived_enabled,
            }
        )
        return next_state


class TrainStage(PipelineStage):
    """Train the configured model against canonical window data."""

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        next_state = {} if state is None else dict(state)
        run_command([sys.executable, "-m", "src.engine.train", "--config", str(self.config_path)])
        return next_state


class TestStage(PipelineStage):
    """Evaluate the configured checkpoint and metric protocol."""

    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        next_state = {} if state is None else dict(state)
        run_command([sys.executable, "-m", "src.engine.evaluate", "--config", str(self.config_path)])
        return next_state
