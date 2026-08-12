"""Unified estimator entrypoints for pose backends.

This package consolidates the old split between extractors and pose estimators
into a single import surface while preserving compatibility with the existing
``src.modules.pose_estimators`` and ``src.modules.extractors`` modules.
"""

from src.modules.pose_estimators.alphapose_full import AlphaPoseFullConfig, AlphaPoseFullEstimator
from src.modules.pose_estimators.alphapose_sppe import AlphaPoseSPPE, AlphaPoseSPPEConfig
from src.modules.pose_estimators.wham_3d import WHAM3DConfig, WHAM3DEstimator, build_wham_3d_estimator

__all__ = [
    "AlphaPoseFullConfig",
    "AlphaPoseFullEstimator",
    "AlphaPoseSPPE",
    "AlphaPoseSPPEConfig",
    "WHAM3DConfig",
    "WHAM3DEstimator",
    "build_wham_3d_estimator",
]
