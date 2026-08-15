"""Common utilities for training and evaluation engines."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from src.models.checkpoint import load_model_checkpoint
from src.models.registry import build_model
from src.modules.matchers import IMUVideoMatcher, SymmetricInfoNCE


def _load_init_checkpoint(model: IMUVideoMatcher, model_name: str, checkpoint: str) -> None:
    report = load_model_checkpoint(
        model,
        model_name,
        checkpoint,
        allow_shape_mismatch=True,
        strict=False,
    )
    print(
        f"Loaded INIT_ALIGNMENT_CKPT: {report.checkpoint} "
        f"(missing={len(report.missing_keys)}, unexpected={len(report.unexpected_keys)}, "
        f"dropped={len(report.dropped_incompatible_keys)})"
    )


def build_alignment_model_from_cfg(
    cfg: Any,
    device: torch.device,
) -> Tuple[IMUVideoMatcher, str]:
    """Build the official hybrid IMU-video alignment model."""
    model_cfg = cfg.TRAIN.MODEL
    model, model_name = build_model(cfg, device)

    if model_cfg.INIT_ALIGNMENT_CKPT:
        _load_init_checkpoint(model, model_name, str(model_cfg.INIT_ALIGNMENT_CKPT))
    return model, model_name


def build_optimizer(
    model: IMUVideoMatcher,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build AdamW for all trainable hybrid model parameters."""
    del lr_backbone
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW([{"params": params, "lr": lr_heads}], weight_decay=weight_decay)


def build_loss_fn(
    temperature: float = 0.1,
    learn_temperature: bool = False,
    device: torch.device | None = None,
) -> SymmetricInfoNCE:
    """Build InfoNCE loss function."""
    target_device = device if device is not None else torch.device("cpu")
    return SymmetricInfoNCE(temperature=temperature, learn_temperature=learn_temperature).to(target_device)
