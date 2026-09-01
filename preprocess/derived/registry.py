"""Registry for sequence-level derived-data transforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.core.registry import Registry

from .contracts import DerivedDataSpec

DerivedTransform = Callable[[dict[str, np.ndarray], np.random.Generator, DerivedDataSpec], dict[str, np.ndarray]]

DERIVED_TRANSFORM_REGISTRY: Registry[DerivedTransform] = Registry("derived transform")


def apply_transform(
    name: str,
    payload: dict[str, np.ndarray],
    rng: np.random.Generator,
    spec: DerivedDataSpec,
) -> dict[str, np.ndarray]:
    """Resolve and apply one registered transform."""
    transform = DERIVED_TRANSFORM_REGISTRY.get(name)
    return transform(payload, rng, spec)


__all__ = ["DERIVED_TRANSFORM_REGISTRY", "DerivedTransform", "apply_transform"]
