"""CLI wrapper for shared video skeleton extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess.common.extract import run_video_skeleton_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified video skeleton extraction")
    parser.add_argument("--config", type=str, required=True, help="Workflow YAML config path")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_video_skeleton_extraction(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
