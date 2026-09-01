"""Materialize canonical sequence variants before window slicing."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from preprocess.common.sequence import write_sequence_meta, write_sequence_npz

from . import transforms as _registered_transforms  # noqa: F401
from .contracts import DerivedDataSpec
from .registry import apply_transform


def _sequence_seed(seed: int, sequence_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{sequence_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Sequence metadata must be a JSON object: {path}")
    return value


def derive_sequences(
    input_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any] | DerivedDataSpec,
) -> Path:
    """Create a reproducible sequence-level derived-data root.

    The canonical input is never modified. Each sequence receives a stable
    per-sequence RNG derived from the configured seed and sequence id, so file
    ordering cannot change a generated variant.
    """
    spec = config if isinstance(config, DerivedDataSpec) else DerivedDataSpec.from_mapping(config)
    if not spec.enabled:
        raise ValueError("derive_sequences requires preprocess.derived.enabled=true")

    source = Path(input_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    sequence_dir = source / "sequences"
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Canonical sequence directory not found: {sequence_dir}")
    if source == output:
        raise ValueError("Derived output must be different from the canonical input root")

    transforms = spec.transforms or ("identity",)
    for name in transforms:
        # Resolve all names before writing any output, so a typo cannot leave
        # behind a partially generated variant.
        from .registry import DERIVED_TRANSFORM_REGISTRY

        DERIVED_TRANSFORM_REGISTRY.resolve_name(name)

    output_sequence_dir = output / "sequences"
    output_sequence_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source_path in sorted(sequence_dir.glob("*.npz")):
        with np.load(source_path, allow_pickle=True) as data:
            payload = {key: np.array(data[key], copy=True) for key in data.files}
        if "sequence_id" not in payload:
            raise ValueError(f"Canonical sequence is missing sequence_id: {source_path}")
        sequence_id = str(payload["sequence_id"].item())
        rng = np.random.default_rng(_sequence_seed(spec.seed, sequence_id))
        for transform_name in transforms:
            payload = apply_transform(transform_name, payload, rng, spec)

        output_path = output_sequence_dir / source_path.name
        write_sequence_npz(output_path, payload)
        meta = _read_meta(source_path.with_suffix(".json"))
        meta.update(
            {
                "data_layer": "derived",
                "derived_variant": spec.name,
                "parent_sequence": str(source_path),
                "derived_transforms": list(transforms),
                "derived_config": spec.to_metadata(),
                "derived_seed": _sequence_seed(spec.seed, sequence_id),
            }
        )
        write_sequence_meta(output_path.with_suffix(".json"), meta)
        records.append(
            {
                "sequence_id": sequence_id,
                "source": str(source_path),
                "output": str(output_path),
                "transforms": list(transforms),
            }
        )

    if not records:
        raise ValueError(f"No canonical sequence NPZ files found under {sequence_dir}")
    for filename in ("video_manifest.csv",):
        source_manifest = source / filename
        if source_manifest.exists():
            shutil.copy2(source_manifest, output / filename)
    output.mkdir(parents=True, exist_ok=True)
    (output / "derived_manifest.json").write_text(
        json.dumps(
            {
                "data_layer": "derived",
                "variant": spec.name,
                "config": spec.to_metadata(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


__all__ = ["derive_sequences"]
