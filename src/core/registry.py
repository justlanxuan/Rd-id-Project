"""Small fail-loud registry used by one component domain at a time."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Map normalized component names to constructors.

    A registry owns one domain (for example models or extractors).  It does not
    contain component-selection policy or pipeline orchestration.
    """

    def __init__(self, domain: str) -> None:
        self.domain = str(domain).strip() or "component"
        self._constructors: dict[str, Callable[..., T]] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        key = str(name).strip().lower().replace("-", "_")
        if not key:
            raise ValueError("Registry names cannot be empty.")
        return key

    def register(
        self,
        name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        key = self._normalize(name)

        def decorator(constructor: Callable[..., T]) -> Callable[..., T]:
            if key in self._constructors or key in self._aliases:
                raise KeyError(f"Duplicate {self.domain} registration: {name!r}")
            self._constructors[key] = constructor
            for alias in aliases:
                alias_key = self._normalize(alias)
                if alias_key in self._constructors or alias_key in self._aliases:
                    raise KeyError(f"Duplicate {self.domain} alias: {alias!r}")
                self._aliases[alias_key] = key
            return constructor

        return decorator

    def resolve_name(self, name: str) -> str:
        key = self._normalize(name)
        canonical = self._aliases.get(key, key)
        if canonical not in self._constructors:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown {self.domain} {name!r}. Available: {available}")
        return canonical

    def get(self, name: str) -> Callable[..., T]:
        return self._constructors[self.resolve_name(name)]

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.resolve_name(name)
        except (KeyError, ValueError):
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
