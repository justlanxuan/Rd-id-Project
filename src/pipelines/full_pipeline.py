"""Compatibility wrapper around :mod:`src.pipeline`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline import DEFAULT_STAGES, parse_stages, run_pipeline


class FullPipeline:
    """Legacy class interface; new code should call `src.pipeline` directly."""

    def __init__(self, config_path: str, stages: list[str] | None = None):
        self.config_path = Path(config_path).expanduser().resolve()
        self.stages = stages or list(DEFAULT_STAGES)
        parse_stages(",".join(self.stages))

    def run(self) -> dict[str, Any]:
        return run_pipeline(self.config_path, self.stages)
