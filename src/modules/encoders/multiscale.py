"""Multi-scale temporal encoders for the G11 benchmark.

The module operates on numerical skeleton/IMU features and deliberately does
not contain an image backbone or a task-specific classification head.  It
returns the fused sequence and the scale weights so downstream matching code
can audit which temporal branch was used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

DEFAULT_DILATIONS: dict[str, tuple[int, ...]] = {
    "short": (1, 2, 4),
    "middle": (3, 6, 12),
    "long": (10, 20, 40),
}


def validate_window_seconds(window_seconds: float, *, max_seconds: float = 10.0) -> float:
    """Validate an explicitly declared context duration.

    Padding or truncation is intentionally not performed here.  A caller that
    has a longer context must choose a new protocol/configuration explicitly.
    """

    value = float(window_seconds)
    limit = float(max_seconds)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"window_seconds must be finite and positive, got {window_seconds!r}")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"max_seconds must be finite and positive, got {max_seconds!r}")
    if value > limit + 1e-9:
        raise ValueError(f"window_seconds={value:g} exceeds the configured maximum of {limit:g} seconds")
    return value


def _validate_kernel_and_dilations(kernel_size: int, dilations: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    kernel = int(kernel_size)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size!r}")
    values = tuple(int(dilation) for dilation in dilations)
    if not values or any(dilation <= 0 for dilation in values):
        raise ValueError(f"dilations must contain positive integers, got {dilations!r}")
    return kernel, values


def receptive_field_samples(kernel_size: int, dilations: Sequence[int]) -> int:
    """Return the temporal receptive field of one same-padded conv per dilation."""

    kernel, values = _validate_kernel_and_dilations(kernel_size, dilations)
    return 1 + sum((kernel - 1) * dilation for dilation in values)


def receptive_field_seconds(kernel_size: int, dilations: Sequence[int], fps_hz: float, *, stride: int = 1) -> float:
    """Convert a branch receptive field from samples to seconds."""

    rate = float(fps_hz)
    step = int(stride)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"fps_hz must be finite and positive, got {fps_hz!r}")
    if step < 1:
        raise ValueError(f"stride must be positive, got {stride!r}")
    return float(receptive_field_samples(kernel_size, dilations) * step / rate)


def describe_receptive_fields(
    dilations: Mapping[str, Sequence[int]],
    *,
    kernel_size: int,
    fps_hz: float,
    stride: int = 1,
) -> dict[str, dict[str, float | int]]:
    """Return a serializable, seconds-aware branch profile."""

    return {
        str(name): {
            "samples": receptive_field_samples(kernel_size, values),
            "seconds": receptive_field_seconds(kernel_size, values, fps_hz, stride=stride),
        }
        for name, values in dilations.items()
    }


def _validate_sequence_input(x: torch.Tensor, *, name: str = "x") -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, channels], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty batch/time/channel dimensions, got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_mask(mask: torch.Tensor | None, batch: int, time: int, device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape != (batch, time):
        raise ValueError(f"mask must have shape {(batch, time)}, got {tuple(mask.shape)}")
    if mask.device != device:
        raise ValueError("mask and input must be on the same device")
    return mask.to(dtype=torch.bool)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.unsqueeze(-1).to(dtype=sequence.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


class TemporalBlock(nn.Module):
    """One residual, same-length temporal convolution block."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        kernel, values = _validate_kernel_and_dilations(kernel_size, (dilation,))
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        padding = values[0] * (kernel - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=values[0])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TemporalBlock expects [batch, channels, time], got {tuple(x.shape)}")
        y = self.dropout(self.activation(self.conv(x)))
        if y.shape != x.shape:
            raise RuntimeError(f"TemporalBlock changed shape from {tuple(x.shape)} to {tuple(y.shape)}")
        return self.activation(x + y)


class HierarchicalTemporalAttention(nn.Module):
    """Fuse scale sequences using a window-level softmax over scale weights."""

    def __init__(self, hidden_dim: int, *, mode: str = "hierarchical_attention", dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode).lower()
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if self.mode not in {"mean", "gated", "hierarchical_attention"}:
            raise ValueError(f"Unsupported multi-scale fusion mode={mode!r}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if self.mode == "gated":
            gate_hidden = max(1, self.hidden_dim // 2)
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim, gate_hidden),
                nn.Tanh(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden, 1),
            )
        elif self.mode == "hierarchical_attention":
            gate_hidden = max(1, self.hidden_dim // 2)
            self.scale_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, gate_hidden),
                nn.Tanh(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden, 3),
            )

    def forward(self, scales: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if scales.ndim != 4:
            raise ValueError(f"scales must have shape [batch, scale, time, hidden], got {tuple(scales.shape)}")
        batch, n_scales, time, hidden = scales.shape
        if n_scales != 3:
            raise ValueError(f"G11 expects exactly three temporal scales, got {n_scales}")
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden dimension {self.hidden_dim}, got {hidden}")
        if not torch.isfinite(scales).all():
            raise ValueError("scales contains NaN or Inf")
        valid = _validate_mask(mask, batch, time, scales.device)
        pooled = torch.stack([_masked_mean(scales[:, index], valid) for index in range(n_scales)], dim=1)
        if self.mode == "mean":
            weights = scales.new_full((batch, n_scales), 1.0 / n_scales)
        elif self.mode == "gated":
            weights = torch.softmax(self.gate(pooled).squeeze(-1), dim=1)
        else:
            weights = torch.softmax(self.scale_mlp(pooled.reshape(batch, -1)), dim=1)
        fused = (scales * weights[:, :, None, None]).sum(dim=1)
        if valid is not None:
            fused = fused * valid[:, :, None].to(dtype=fused.dtype)
        return fused, weights


class MultiScaleTemporalTCN(nn.Module):
    """Three-branch temporal TCN with auditable fusion weights."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        output_dim: int | None = None,
        *,
        kernel_size: int = 3,
        dilations: Mapping[str, Sequence[int]] | None = None,
        dropout: float = 0.1,
        fusion: str = "hierarchical_attention",
        max_window_seconds: float = 10.0,
        window_seconds: float | None = None,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError(f"input_dim and hidden_dim must be positive, got {input_dim!r}, {hidden_dim!r}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim) if output_dim is not None else self.hidden_dim
        self.kernel_size, _ = _validate_kernel_and_dilations(kernel_size, (1,))
        self.max_window_seconds = validate_window_seconds(max_window_seconds, max_seconds=10.0)
        self.window_seconds = None if window_seconds is None else validate_window_seconds(window_seconds, max_seconds=self.max_window_seconds)
        declared = dilations or DEFAULT_DILATIONS
        if set(declared) != {"short", "middle", "long"}:
            raise ValueError("dilations must declare exactly short, middle, and long branches")
        self.dilations = {
            name: _validate_kernel_and_dilations(self.kernel_size, values)[1]
            for name, values in declared.items()
        }
        self.scale_names = ("short", "middle", "long")
        self.input_projection = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    *(TemporalBlock(self.hidden_dim, kernel_size=self.kernel_size, dilation=dilation, dropout=dropout) for dilation in self.dilations[name])
                )
                for name in self.scale_names
            }
        )
        self.fusion = HierarchicalTemporalAttention(self.hidden_dim, mode=fusion, dropout=dropout)
        self.output_projection = nn.Identity() if self.output_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.output_dim)

    def receptive_fields(self, fps_hz: float, *, stride: int = 1) -> dict[str, dict[str, float | int]]:
        return describe_receptive_fields(self.dilations, kernel_size=self.kernel_size, fps_hz=fps_hz, stride=stride)

    def profile_spec(self, fps_hz: float, *, stride: int = 1) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "kernel_size": self.kernel_size,
            "dilations": {name: list(values) for name, values in self.dilations.items()},
            "fusion": self.fusion.mode,
            "max_window_seconds": self.max_window_seconds,
            "receptive_fields": self.receptive_fields(fps_hz, stride=stride),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        fps_hz: float | None = None,
        window_seconds: float | None = None,
    ) -> dict[str, torch.Tensor]:
        _validate_sequence_input(x)
        batch, time, _ = x.shape
        valid = _validate_mask(mask, batch, time, x.device)
        declared_window = self.window_seconds if window_seconds is None else window_seconds
        if declared_window is not None:
            validate_window_seconds(declared_window, max_seconds=self.max_window_seconds)
        if fps_hz is not None:
            rate = float(fps_hz)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"fps_hz must be finite and positive, got {fps_hz!r}")
            if declared_window is not None and time / rate > float(declared_window) + 1e-6:
                raise ValueError(
                    f"input sequence spans about {time / rate:g}s, exceeds declared window_seconds={float(declared_window):g}s"
                )
        projected = self.input_projection(x.transpose(1, 2))
        branches = []
        for name in self.scale_names:
            branch = self.branches[name](projected).transpose(1, 2)
            if valid is not None:
                branch = branch * valid[:, :, None].to(dtype=branch.dtype)
            branches.append(branch)
        stacked = torch.stack(branches, dim=1)
        fused, weights = self.fusion(stacked, valid)
        output = self.output_projection(fused)
        return {"sequence": output, "scale_weights": weights, "branches": stacked}
