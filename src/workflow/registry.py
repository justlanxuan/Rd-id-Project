"""Registry for independently runnable public workflow stages."""

from __future__ import annotations

from pathlib import Path

from src.core.registry import Registry

from .base import PipelineStage
from .stages import PreprocessStage, TestStage, TrainStage

STAGE_REGISTRY: Registry[PipelineStage] = Registry("pipeline stage")
STAGE_REGISTRY.register("preprocess")(PreprocessStage)
STAGE_REGISTRY.register("train")(TrainStage)
STAGE_REGISTRY.register("test")(TestStage)


def build_stage(name: str, config_path: str | Path) -> PipelineStage:
    return STAGE_REGISTRY.build(name, config_path)


__all__ = ["STAGE_REGISTRY", "build_stage"]
