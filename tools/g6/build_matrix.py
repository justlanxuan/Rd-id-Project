"""Write or inspect the G6 required-cell manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .matrix import build_required_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument(
        "--status",
        choices=("draft", "locked"),
        default="draft",
        help="A locked manifest requires --protocol-hash.",
    )
    parser.add_argument("--protocol-hash", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_hash = str(args.protocol_hash).strip()
    if args.status == "locked" and not protocol_hash:
        raise ValueError("--status locked requires --protocol-hash.")
    cells = build_required_cells()
    payload = {
        "schema_version": "1.0",
        "status": args.status,
        "protocol_hash": protocol_hash,
        "summary": {
            "training": sum(cell.job_type == "train" for cell in cells),
            "evaluation": sum(cell.job_type == "evaluate" for cell in cells),
            "total": len(cells),
        },
        "cells": [cell.to_dict() for cell in cells],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
