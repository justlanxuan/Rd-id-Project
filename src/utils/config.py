"""Backward-compatible config utilities.

The implementation now lives in :mod:`src.config` and is backed by YACS. This
module keeps old imports working.
"""

from __future__ import annotations

from src.config.config import (
    cfg_to_dict,
    load_cfg,
    load_config,
    resolve_config,
    substitute_variables,
)
from src.config.defaults import get_cfg_defaults

__all__ = [
    "cfg_to_dict",
    "get_cfg_defaults",
    "load_cfg",
    "load_config",
    "resolve_config",
    "substitute_variables",
]
