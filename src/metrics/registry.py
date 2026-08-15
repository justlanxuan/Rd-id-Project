"""Registry for evaluation metrics."""

from __future__ import annotations

from typing import Any

from src.core.registry import Registry

from .base import EvaluationMetric
from .window import FrameAccEvaluator, GroupTestEvaluator

METRIC_REGISTRY: Registry[EvaluationMetric] = Registry("metric")
METRIC_REGISTRY.register("frame_acc")(FrameAccEvaluator)
METRIC_REGISTRY.register("group_test")(GroupTestEvaluator)


def build_metric(name: str, **kwargs: Any) -> EvaluationMetric:
    return METRIC_REGISTRY.build(name, **kwargs)


__all__ = ["METRIC_REGISTRY", "build_metric"]
