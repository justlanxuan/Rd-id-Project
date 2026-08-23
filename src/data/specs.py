"""Public parser for raw-window experiment specifications.

The G10/G12 command-line tools historically each carried a private copy of
this parser.  Keeping the grammar here makes the dataset boundary explicit and
lets the official config-driven engine consume exactly the same specs.
"""

from __future__ import annotations

from typing import Any, Iterable


def parse_spec(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for token in str(text).split(";"):
        if "=" not in token:
            raise ValueError(f"Invalid spec token {token!r}; expected key=value")
        key, value = token.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"Invalid empty spec key in {text!r}")
        values[key] = value
    for required in ("dataset", "csv", "root", "fps_hz"):
        if required not in values:
            raise ValueError(f"Spec missing {required!r}: {text}")
    try:
        values["fps_hz"] = float(values["fps_hz"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Spec fps_hz must be numeric: {text!r}") from exc
    if values.get("gyro_sidecar_root") in {None, "", "none"}:
        values.pop("gyro_sidecar_root", None)
    return values


def load_specs(values: Iterable[str]) -> list[dict[str, Any]]:
    specs = [parse_spec(value) for value in values if str(value).strip()]
    if not specs:
        raise ValueError("At least one raw-window spec is required")
    return specs


__all__ = ["load_specs", "parse_spec"]
