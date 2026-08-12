"""Stable public model and checkpoint contracts.

Construction lives in :mod:`src.models.registry`; keeping package import light
prevents a cycle while the concrete matcher imports the output contracts.
"""

from .base import ModelCapabilities, ModelOutput
from .checkpoint import (
    CheckpointLoadReport,
    adapt_checkpoint_state,
    checkpoint_scalar,
    load_model_checkpoint,
    model_checkpoint_metadata,
)

__all__ = [
    "CheckpointLoadReport",
    "ModelCapabilities",
    "ModelOutput",
    "adapt_checkpoint_state",
    "checkpoint_scalar",
    "load_model_checkpoint",
    "model_checkpoint_metadata",
]
