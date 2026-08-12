"""Stable interface for top-level pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PipelineStage(ABC):
    """One independently runnable stage in the public pipeline."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {self.config_path}")

    @abstractmethod
    def run(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the stage and return state for the following stage."""
        raise NotImplementedError
