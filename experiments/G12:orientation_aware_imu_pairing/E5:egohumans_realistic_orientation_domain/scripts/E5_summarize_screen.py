# Experiment Note: E5-A3-screen-summary
"""Summarize E5 screen metrics without re-running evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _summarize(root: Path) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for path in sorted(root.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        regime = name.rsplit("_seed", 1)[0]
        final = payload["final_test"]
        domains = final["per_domain"]
        record = {
            "run": name,
            "seed": int(name.rsplit("seed", 1)[1]),
            "selection": float(payload["best_selection_frame_acc"]),
            "custom23_test": float(domains["custom23_test"]["correct"] / domains["custom23_test"]["total"]),
            "custom23_strata": final.get("turning_strata", {}).get("custom23_test", {}),
            "egohumans_canonical_test_diagnostic": float(
                domains["egohumans_canonical_test_diagnostic"]["correct"]
                / domains["egohumans_canonical_test_diagnostic"]["total"]
            ),
            "controls": {
                key: float(value["correct"] / value["total"])
                for key, value in domains.items()
                if key.startswith("custom") and key not in {"custom23_test"}
            },
        }
        groups.setdefault(regime, []).append(record)
    summary: dict[str, object] = {"root": str(root), "runs": sum(len(v) for v in groups.values()), "groups": {}}
    for regime, records in sorted(groups.items()):
        group: dict[str, object] = {"runs": records}
        for key in ("selection", "custom23_test", "egohumans_canonical_test_diagnostic"):
            values = [float(record[key]) for record in records]
            group[f"{key}_mean"] = statistics.mean(values)
            group[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        controls: dict[str, float] = {}
        for key in records[0]["controls"]:
            controls[key] = statistics.mean(float(record["controls"][key]) for record in records)
        group["control_means"] = controls
        summary["groups"][regime] = group
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = {"summaries": [_summarize(Path(root).expanduser().resolve()) for root in args.root]}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "roots": len(payload["summaries"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
