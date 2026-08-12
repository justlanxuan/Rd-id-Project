"""Module-local extraction dispatcher for the simplified pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.common.extract import run_video_skeleton_extraction


def run_extraction(config_path: str | Path, dry_run: bool = False) -> None:
    run_video_skeleton_extraction(str(config_path), dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified video skeleton extraction")
    parser.add_argument("--config", type=str, required=True, help="Workflow YAML config path")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extraction(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
