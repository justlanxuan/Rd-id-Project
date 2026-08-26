"""Offline inference policies."""

from __future__ import annotations

from typing import Any

from .multi_person import METHODS, MultiPersonOfflinePolicy, evaluate_scores


def build_offline_policy(name: str = "multi_person", **kwargs: Any) -> MultiPersonOfflinePolicy:
    policy = str(name).strip().lower().replace("-", "_")
    if policy in {"multi_person", "multiperson", "s08", "s08_multiperson"}:
        return MultiPersonOfflinePolicy(**kwargs)
    raise ValueError(f"Unknown offline inference policy: {name!r}")


__all__ = [
    "METHODS",
    "MultiPersonOfflinePolicy",
    "build_offline_policy",
    "evaluate_scores",
]
