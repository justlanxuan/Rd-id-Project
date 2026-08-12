"""Write the protocol-locked G6 train/evaluation configs and job index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configs import generate_resolved_configs
from .profiles import PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-hash", required=True)
    parser.add_argument("--data-manifest-index", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="g6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    entries = generate_resolved_configs(
        output_dir,
        protocol_hash=args.protocol_hash,
        data_manifest_index=args.data_manifest_index,
        artifact_root=args.artifact_root,
        profile_name=args.profile,
    )
    payload = {
        "schema_version": "1.0",
        "protocol_hash": args.protocol_hash,
        "profile": args.profile,
        "summary": {
            "training": sum(row["job_type"] == "train" for row in entries),
            "evaluation": sum(row["job_type"] == "evaluate" for row in entries),
            "total": len(entries),
        },
        "jobs": entries,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
