# Experiment Note: E5-session-split-gate
"""Create deterministic, session-disjoint EgoHumans E5 train/validation manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data/fzliang/reid-project/g11/e3_raw_multiscale/manifests/egohumans_train_w0p8s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/fzliang/reid-project/g12/e5_egohumans_orientation/manifests",
    )
    parser.add_argument("--validation-sessions", type=int, default=4)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, fields = _read(input_path)
    sessions = sorted({str(row["session"]) for row in rows})
    holdout = max(1, min(int(args.validation_sessions), len(sessions) - 1))
    validation_sessions = set(sessions[-holdout:])
    train = [row for row in rows if str(row["session"]) not in validation_sessions]
    validation = [row for row in rows if str(row["session"]) in validation_sessions]
    if not train or not validation:
        raise ValueError("Session split produced an empty partition")
    for row in rows:
        native_frames = int(row["window_end"]) - int(row["window_start"])
        if native_frames != 16:
            raise ValueError(f"E5 requires 16 native frames at 20Hz, got {native_frames}: {row}")
    train_path = output_dir / "egohumans_train.csv"
    validation_path = output_dir / "egohumans_validation.csv"
    _write(train_path, train, fields)
    _write(validation_path, validation, fields)
    manifest = {
        "schema_version": "g12.e5.egohumans_session_split.v1",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "native_fps_hz": 20.0,
        "native_window_frames": 16,
        "window_seconds": 0.8,
        "validation_sessions": sorted(validation_sessions),
        "train_sessions": sorted(set(sessions) - validation_sessions),
        "train": {"csv": str(train_path), "rows": len(train), "sessions": len(set(row["session"] for row in train)), "sha256": _sha256(train_path)},
        "validation": {"csv": str(validation_path), "rows": len(validation), "sessions": len(validation_sessions), "sha256": _sha256(validation_path)},
    }
    manifest_path = output_dir / "egohumans_e5_session_split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
