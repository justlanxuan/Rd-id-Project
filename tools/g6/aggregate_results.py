"""Aggregate a complete directory of G6 run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import render_results_markdown
from .results import aggregate_run_records, load_run_records
from .scheduler import load_protocol_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--protocol-record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol_record(args.protocol_record)
    records = load_run_records(args.records_root)
    result = aggregate_run_records(
        records,
        protocol_hash=str(protocol["protocol_hash"]),
        expected_git_commit=str(protocol["git_commit"]),
        expected_data_manifest_hashes={
            str(key): str(value)
            for key, value in protocol["components"]["data_manifest_hashes"].items()
        },
        verify_artifacts=True,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if args.markdown_output:
        markdown_output = Path(args.markdown_output).expanduser().resolve()
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_results_markdown(result), encoding="utf-8")
        print(markdown_output)


if __name__ == "__main__":
    main()
