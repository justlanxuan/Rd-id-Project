"""Sequence-level derived-data transforms.

This package owns materialized data variants between canonical preprocessing
and window slicing.  It deliberately does not contain training-time random
tensor augmentation or inference policies.
"""

from .contracts import DerivedDataSpec
from .registry import DERIVED_TRANSFORM_REGISTRY
from .runner import derive_sequences

__all__ = ["DERIVED_TRANSFORM_REGISTRY", "DerivedDataSpec", "derive_sequences"]
