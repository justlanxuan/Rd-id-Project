"""Realtime inference policies.

This package is reserved for streaming selectors that keep state across
incoming windows. The S08 merge target is offline-only for now.
"""

from __future__ import annotations

from typing import Any


def build_realtime_policy(name: str = "global", **kwargs: Any):
    raise ValueError(
        "No realtime inference policies are registered yet; "
        f"requested policy={name!r}."
    )


__all__ = ["build_realtime_policy"]
