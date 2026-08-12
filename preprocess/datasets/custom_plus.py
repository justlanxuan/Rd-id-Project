"""Explicitly unsupported Custom+ preprocessing boundary."""

from __future__ import annotations

from pathlib import Path


def run_preprocess(
    config_path: str | Path | None,
    output_dir: str | Path | None = None,
    manifest_csv: str | Path | None = None,
) -> Path:
    del config_path, output_dir, manifest_csv
    raise RuntimeError(
        "Custom+ preprocessing is experimental and does not yet emit the canonical "
        "sequence/window schema. Use dataset='custom' with a validated prepared cache, "
        "or implement and register a dedicated CustomPlusAdapter before production use."
    )


def main() -> None:
    run_preprocess(None)


if __name__ == "__main__":
    main()
