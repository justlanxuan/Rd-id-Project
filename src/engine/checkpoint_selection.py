"""Checkpoint-selection policy shared by training and its contract tests."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_selection_metric(
    requested: str,
    capabilities: Any,
    *,
    has_validation: bool,
) -> str:
    """Resolve the metric used for best-checkpoint selection."""
    metric = str(requested).strip().lower()
    if metric == "auto":
        metric = str(capabilities.preferred_validation_metric).strip().lower() if has_validation else "train_top1"
    if metric not in {"val_loss", "val_top1", "train_top1"}:
        raise ValueError(f"Unsupported checkpoint selection metric: {metric!r}")
    if metric.startswith("val_") and not has_validation:
        raise ValueError(f"Checkpoint selection metric {metric!r} requires a non-empty validation split.")
    return metric


def selection_value_and_score(
    metric: str,
    val_metrics: Mapping[str, float],
    train_top1: float,
) -> tuple[float, float]:
    """Return the human-readable metric value and maximize-oriented score."""
    if metric == "val_loss":
        value = float(val_metrics["loss"])
        return value, -value
    if metric == "val_top1":
        value = float(val_metrics["top1"])
        return value, value
    if metric == "train_top1":
        value = float(train_top1)
        return value, value
    raise ValueError(f"Unsupported checkpoint selection metric: {metric!r}")
