"""Contracts shared by inference policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

InferenceMode = Literal["offline", "realtime"]
RealtimeScope = Literal["global", "windowed"]


@dataclass(frozen=True)
class InferenceDecision:
    """Serializable result from an inference policy."""

    mode: InferenceMode
    policy: str
    assignment: np.ndarray
    selected_segments: tuple[int, ...] = ()
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mode": self.mode,
            "policy": self.policy,
            "assignment": np.asarray(self.assignment, dtype=np.int64).tolist(),
            "selected_segments": [int(value) for value in self.selected_segments],
            "metadata": dict(self.metadata),
        }
        if self.confidence is not None:
            result["confidence"] = float(self.confidence)
        return result


@runtime_checkable
class InferencePolicy(Protocol):
    """Policy interface for application-time inference."""

    mode: InferenceMode
    policy_name: str

    def infer(self, *args: Any, **kwargs: Any) -> InferenceDecision: ...


__all__ = [
    "InferenceDecision",
    "InferenceMode",
    "InferencePolicy",
    "RealtimeScope",
]
