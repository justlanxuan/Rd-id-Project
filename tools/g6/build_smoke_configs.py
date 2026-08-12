"""Write the two protocol-locked, non-formal G6 smoke configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .smoke import build_smoke_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    entries = build_smoke_configs(
        args.formal_index,
        output_dir,
        artifact_root=args.artifact_root,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    payload = {"schema_version": "1.0", "formal": False, "jobs": entries}
    index = output_dir / "index.json"
    index.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(index)


if __name__ == "__main__":
    main()
