"""Matcher base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMatcher(ABC):
    """Abstract matcher interface."""

    @abstractmethod
    def match(
        self,
        similarity_matrix: Any,
        imu_ids: Optional[List[Any]] = None,
        person_ids: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError
