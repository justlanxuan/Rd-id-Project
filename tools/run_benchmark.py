"""Deprecated compatibility notice for the pre-G6 benchmark launcher."""

import argparse

MIGRATION = """\
tools/run_benchmark.py belongs to the archived pre-G6 protocol and has been
removed from the production execution path because it referenced deleted
src.preprocess/src.pipelines modules.

Use the protocol-locked G6 tools instead:
  python -m tools.g6.build_data_manifests --help
  python -m tools.g6.lock_protocol --help
  python -m tools.g6.build_configs --help
  python -m tools.g6.build_smoke_configs --help
  python -m tools.g6.run_jobs --help
  python -m tools.g6.aggregate_results --help

Archived experiment scripts remain under experiments/ for historical audit.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated pre-G6 benchmark launcher.",
        epilog=MIGRATION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    parse_args()
    raise SystemExit(MIGRATION)


if __name__ == "__main__":
    main()
