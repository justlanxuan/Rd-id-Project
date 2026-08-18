# Experiment Note: D4-S06-sweep-summary
"""Summarize raw versus screen-calibrated S06 fixed-checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS = ("alphapose", "yolopose_high", "fmpose3d", "motionagformer", "tcpformer", "wham")
VARIANTS = ("raw", "screen_calibrated")


def session_of(sequence_id: str) -> str:
    tokens = sequence_id.removeprefix("custom_").split("_")
    return "_".join(tokens[:2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/s06_eval"))
    args = parser.parse_args()
    cells: dict[str, dict[str, Any]] = {}
    per_sequence: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    missing = []
    for method in METHODS:
        for variant in VARIANTS:
            path = args.result_root / f"{method}__{variant}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            report = json.loads(path.read_text())
            evaluation = report["evaluation"]
            cells[f"{method}|{variant}"] = {
                "method": method,
                "variant": variant,
                "checkpoint": report["checkpoint"],
                "sessions": report["sessions"],
                "correct": evaluation["correct"],
                "total": evaluation["total"],
                "weighted_frame_acc": evaluation["weighted_frame_acc"],
                "frame_acc": evaluation["frame_acc"],
            }
            for clip in evaluation["clips"]:
                per_sequence[method][variant + "|" + clip["sequence_id"]] = {
                    "sequence_id": clip["sequence_id"],
                    "session": session_of(clip["sequence_id"]),
                    "correct": clip["correct"],
                    "total": clip["total"],
                    "frame_acc": clip["frame_acc"],
                }
    deltas = []
    for method in METHODS:
        raw = cells.get(f"{method}|raw")
        screen = cells.get(f"{method}|screen_calibrated")
        if raw is None or screen is None:
            continue
        deltas.append(
            {
                "method": method,
                "raw_frame_acc": raw["weighted_frame_acc"],
                "screen_calibrated_frame_acc": screen["weighted_frame_acc"],
                "delta_screen_minus_raw": screen["weighted_frame_acc"] - raw["weighted_frame_acc"],
                "raw_correct_total": [raw["correct"], raw["total"]],
                "screen_correct_total": [screen["correct"], screen["total"]],
            }
        )
    sequence_deltas = []
    for method in METHODS:
        raw_clips = {key.removeprefix("raw|"): value for key, value in per_sequence[method].items() if key.startswith("raw|")}
        screen_clips = {key.removeprefix("screen_calibrated|"): value for key, value in per_sequence[method].items() if key.startswith("screen_calibrated|")}
        for sequence_id in sorted(set(raw_clips) & set(screen_clips)):
            raw = raw_clips[sequence_id]
            screen = screen_clips[sequence_id]
            sequence_deltas.append(
                {
                    "method": method,
                    "sequence_id": sequence_id,
                    "session": raw["session"],
                    "raw_frame_acc": raw["frame_acc"],
                    "screen_calibrated_frame_acc": screen["frame_acc"],
                    "delta_screen_minus_raw": screen["frame_acc"] - raw["frame_acc"],
                    "raw_correct_total": [raw["correct"], raw["total"]],
                    "screen_correct_total": [screen["correct"], screen["total"]],
                }
            )
    report = {
        "schema_version": "g9-e3-s06-sweep-summary-1",
        "protocol": {"window_size": 24, "stride": 16, "checkpoint": next(iter(cells.values()), {}).get("checkpoint")},
        "cells": list(cells.values()),
        "method_deltas": deltas,
        "sequence_deltas": sequence_deltas,
        "missing": missing,
        "interpretation": {
            "control": "same baseline IMU/person order/checkpoint and 24/16 protocol; only skeleton coordinate variant changes",
            "representation_limit": "The G6 encoder consumes xy; 3D S06 outputs are evaluated through their xy projection, not as full xyz input.",
            "causal_scope": "This is a controlled coordinate intervention on the fixed checkpoint, not a retrained source-domain performance claim.",
        },
    }
    output = args.result_root / "s06_sweep_summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "cells": len(cells), "sequence_deltas": len(sequence_deltas), "missing": missing}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
