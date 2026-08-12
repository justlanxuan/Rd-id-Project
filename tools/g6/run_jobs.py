"""Dry-run or execute the protocol-locked G6 job graph with safe resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scheduler import (
    load_job_index,
    load_protocol_record,
    run_jobs,
    summarize_execution_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--protocol-record", required=True)
    parser.add_argument("--gpus", required=True, help="Explicit comma-separated physical GPU ids.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol_record(args.protocol_record)
    _, jobs = load_job_index(args.index, protocol_record=protocol)
    statuses = run_jobs(
        jobs,
        gpus=[part.strip() for part in args.gpus.split(",") if part.strip()],
        max_parallel=args.max_parallel,
        log_root=Path(args.log_root),
        state_path=Path(args.state),
        dry_run=args.dry_run,
        expected_git_commit=str(protocol["git_commit"]),
    )
    print(json.dumps(summarize_execution_plan(statuses), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
