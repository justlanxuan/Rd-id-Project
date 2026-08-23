"""Model-domain registry and public construction function."""

from __future__ import annotations

from typing import Any

import torch

from src.core import Registry
from src.modules.matchers import IMUVideoMatcher

MODEL_REGISTRY: Registry[IMUVideoMatcher] = Registry("model")


@MODEL_REGISTRY.register("hybrid")
def _build_hybrid(cfg: Any, device: torch.device) -> IMUVideoMatcher:
    from .hybrid import build_hybrid_model

    return build_hybrid_model(cfg, device)


@MODEL_REGISTRY.register("orientation_aware")
def _build_orientation_aware(cfg: Any, device: torch.device):
    from .orientation import build_orientation_model

    return build_orientation_model(cfg, device)


def build_model(cfg: Any, device: torch.device) -> tuple[IMUVideoMatcher, str]:
    model_type = str(getattr(cfg.TRAIN.MODEL, "TYPE", "hybrid"))
    canonical_name = MODEL_REGISTRY.resolve_name(model_type)
    return MODEL_REGISTRY.build(canonical_name, cfg, device), canonical_name
