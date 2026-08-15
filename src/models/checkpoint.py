"""Model-owned checkpoint loading and legacy migration contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.core import Registry

CheckpointAdapter = Callable[[Mapping[str, Any]], dict[str, torch.Tensor]]
MODEL_CHECKPOINT_ADAPTERS: Registry[dict[str, torch.Tensor]] = Registry(
    "model checkpoint adapter"
)
CHECKPOINT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CheckpointLoadReport:
    model_name: str
    checkpoint: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    dropped_incompatible_keys: tuple[str, ...]


def load_checkpoint_payload(checkpoint: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        return checkpoint
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}: {path}")
    return payload


def _raw_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint model state must be a mapping, got {type(state).__name__}")
    return state


@MODEL_CHECKPOINT_ADAPTERS.register("hybrid")
def adapt_hybrid_checkpoint(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Migrate historical Hybrid keys and fold legacy encoder stats into state."""
    adapted: dict[str, torch.Tensor] = {}
    for key, value in _raw_state(payload).items():
        if key == "log_temp":
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("skel."):
            adapted[f"video_encoder.{key[len('skel.'):]}"] = value
        elif key.startswith("imu."):
            adapted[f"imu_encoder.raw.{key[len('imu.'):]}"] = value
        else:
            adapted[str(key)] = value

    stats = payload.get("stats")
    if isinstance(stats, Mapping):
        stat_map = {
            "raw_mu": "video_encoder.raw_mu",
            "raw_sd": "video_encoder.raw_sd",
            "vec_mu": "video_encoder.vec_mu",
            "vec_sd": "video_encoder.vec_sd",
            "imu_mu": "imu_encoder.imu_mu",
            "imu_sd": "imu_encoder.imu_sd",
        }
        for source_key, target_key in stat_map.items():
            if source_key in stats:
                adapted[target_key] = torch.as_tensor(stats[source_key], dtype=torch.float32)
    return adapted


def adapt_checkpoint_state(
    model_name: str,
    payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return model state through a registered adapter or the generic schema."""
    if model_name in MODEL_CHECKPOINT_ADAPTERS:
        adapter = MODEL_CHECKPOINT_ADAPTERS.get(model_name)
        return adapter(payload)
    state: dict[str, torch.Tensor] = {}
    for key, value in _raw_state(payload).items():
        if isinstance(value, torch.Tensor):
            state[str(key)] = value
    return state


def load_model_checkpoint(
    model: nn.Module,
    model_name: str,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    allow_shape_mismatch: bool = False,
    strict: bool = False,
) -> CheckpointLoadReport:
    """Load a checkpoint without exposing model-specific schema to engines."""
    payload = load_checkpoint_payload(checkpoint)
    schema_version = str(payload.get("checkpoint_schema_version", "legacy"))
    if schema_version not in {"legacy", CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported checkpoint schema version: {schema_version!r}")
    stored_model_name = str(payload.get("model_name", "")).strip()
    if stored_model_name and stored_model_name != model_name:
        raise ValueError(
            f"Checkpoint model_name={stored_model_name!r} does not match requested {model_name!r}."
        )
    state = adapt_checkpoint_state(model_name, payload)
    dropped: list[str] = []
    if allow_shape_mismatch:
        model_state = model.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key not in model_state or tuple(model_state[key].shape) != tuple(value.shape):
                dropped.append(key)
                continue
            compatible[key] = value
        state = compatible

    incompatible = model.load_state_dict(state, strict=strict)
    checkpoint_name = (
        str(Path(checkpoint).expanduser().resolve())
        if isinstance(checkpoint, (str, Path))
        else "<in-memory>"
    )
    return CheckpointLoadReport(
        model_name=str(model_name),
        checkpoint=checkpoint_name,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        dropped_incompatible_keys=tuple(sorted(dropped)),
    )


def checkpoint_scalar(
    checkpoint: str | Path | Mapping[str, Any],
    key: str,
) -> torch.Tensor | None:
    """Read an auxiliary scalar retained outside the model state contract."""
    payload = load_checkpoint_payload(checkpoint)
    value = _raw_state(payload).get(key)
    return torch.as_tensor(value) if value is not None else None


def model_checkpoint_metadata(model_name: str, model: nn.Module) -> dict[str, Any]:
    """Return versioned metadata embedded in every newly written checkpoint."""
    capabilities = getattr(model, "capabilities", None)
    if capabilities is None:
        raise AttributeError(f"Model {type(model).__name__} has no capabilities declaration.")
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": str(model_name),
        "model_capabilities": asdict(capabilities),
    }


__all__ = [
    "CheckpointLoadReport",
    "MODEL_CHECKPOINT_ADAPTERS",
    "adapt_checkpoint_state",
    "checkpoint_scalar",
    "load_checkpoint_payload",
    "load_model_checkpoint",
    "model_checkpoint_metadata",
]
