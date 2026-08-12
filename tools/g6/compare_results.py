"""Compare two validated aggregate result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def compare_results(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    *,
    baseline_label: str,
    variant_label: str,
) -> str:
    lines = [
        "# Benchmark result comparison",
        "",
        f"- Baseline: {baseline_label} (`{baseline['protocol_hash']}`)",
        f"- Variant: {variant_label} (`{variant['protocol_hash']}`)",
        "- Delta is variant − baseline in percentage points.",
        "",
        "## Source-domain FrameAcc",
        "",
        "| Source | Baseline | Variant | Delta |",
        "|---|---:|---:|---:|",
    ]
    for source in ("totalcapture", "egohumans"):
        left = float(baseline["source"][source]["mean"])
        right = float(variant["source"][source]["mean"])
        lines.append(
            f"| {source} | {_percent(left)} | {_percent(right)} | "
            f"{100.0 * (right - left):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Custom overall macro-session FrameAcc",
            "",
            "| Condition | Source | Baseline | Variant | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    order = (
        "zero_shot.egohumans",
        "zero_shot.totalcapture",
        "finetune.egohumans",
        "finetune.totalcapture",
        "direct.none",
    )
    for key in order:
        left_row = baseline["custom_overall"][key]
        right_row = variant["custom_overall"][key]
        left = float(left_row["macro_session"]["mean"])
        right = float(right_row["macro_session"]["mean"])
        lines.append(
            f"| {right_row['condition']} | {right_row.get('source') or 'none'} | "
            f"{_percent(left)} | {_percent(right)} | {100.0 * (right - left):+.2f} pp |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--variant-label", default="variant")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).expanduser().resolve().read_text(encoding="utf-8"))
    variant = json.loads(Path(args.variant).expanduser().resolve().read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        compare_results(
            baseline,
            variant,
            baseline_label=args.baseline_label,
            variant_label=args.variant_label,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
