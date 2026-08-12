"""Create the G6 protocol hash after explicit human lock confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profiles import PROFILES
from .protocol import build_protocol_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-document", required=True)
    parser.add_argument("--data-manifest-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="g6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_protocol_record(
        args.protocol_document,
        args.data_manifest_index,
        profile_name=args.profile,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(record["protocol_hash"])


if __name__ == "__main__":
    main()
