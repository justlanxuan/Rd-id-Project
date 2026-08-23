"""Stable model-side data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class ModelOutput(TypedDict, total=False):
    imu: torch.Tensor
    video: torch.Tensor
    skeleton: torch.Tensor
    orientation_embedding: torch.Tensor
    turning_activity_pred: torch.Tensor
    orientation_onset_logits: torch.Tensor
    gyro_onset_logits: torch.Tensor
    domain_logits: torch.Tensor


@dataclass(frozen=True)
class ModelCapabilities:
    pair_logits: bool = False
    cross_pair_logits: bool = False
    domain_logits: bool = False
    root_trajectory: bool = False
    external_imu_normalization: bool = False
    fitted_input_stats: bool = False
    full_validation_batch: bool = False
    segment_frame_acc: bool = False
    preferred_validation_metric: str = "val_top1"
    requires_orientation: bool = False
