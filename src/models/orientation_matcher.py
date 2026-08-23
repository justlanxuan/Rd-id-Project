"""Stable public import for the orientation-aware matcher.

The implementation remains in ``src.g12`` for checkpoint/report compatibility;
new code should import it from this module through the official model registry.
"""

from src.g12.orientation_matcher import OrientationAwareMatcher

__all__ = ["OrientationAwareMatcher"]
