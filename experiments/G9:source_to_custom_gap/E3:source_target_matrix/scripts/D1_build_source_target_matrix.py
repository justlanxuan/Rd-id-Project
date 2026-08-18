# Experiment Note: D1-source-target-matrix
"""Index existing G6 source→Custom results and join target-domain diagnostics.

This does not relabel the G6 result or claim that unrun S06 skeleton methods
were benchmarked. It produces an explicit availability matrix and exploratory
session joins for the next controlled sweep.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

G6_ROOT = Path("/data/fzliang/reid-project/g6/c9a5d3099979296a72314eba66274855e03ab1eb")
E1_ROOT = Path("/data/fzliang/reid-project/g9/e1_gap_audit")
E2_ROOT = Path("/data/fzliang/reid-project/g9/e2_multimodal")


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def read_runs() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((G6_ROOT / "artifacts/records").glob("evaluate__*/run_record.json")):
        record = json.loads(path.read_text())
        frame = record.get("frame_acc", {})
        if record.get("status") != "completed" or not frame.get("total"):
            continue
        rows.append(
            {
                "source": record.get("source") or "none",
                "condition": record.get("condition") or "unknown",
                "session": record.get("test_session") or "all",
                "seed": record.get("seed"),
                "correct": int(frame["correct"]),
                "total": int(frame["total"]),
                "value": float(frame["value"]),
                "protocol_hash": record.get("protocol_hash"),
                "run_record": str(path),
                "raw_results": record.get("raw_results"),
            }
        )
    return rows


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right) or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = read_runs()
    by_cell: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "correct": 0, "total": 0})
    for row in runs:
        key = "|".join((row["source"], row["condition"], row["session"]))
        cell = by_cell[key]
        cell["rows"].append(row)
        cell["correct"] += row["correct"]
        cell["total"] += row["total"]
    cells = []
    for key, value in sorted(by_cell.items()):
        source, condition, session = key.split("|", 2)
        cells.append(
            {
                "source": source,
                "condition": condition,
                "session": session,
                "seeds": sorted(row["seed"] for row in value["rows"]),
                "correct": value["correct"],
                "total": value["total"],
                "pooled_frame_acc": value["correct"] / value["total"] if value["total"] else None,
                "seed_mean": mean([row["value"] for row in value["rows"]]),
                "seed_std": float(np.std([row["value"] for row in value["rows"]], ddof=1)) if len(value["rows"]) > 1 else 0.0,
            }
        )

    semantic = json.loads((E1_ROOT / "semantic_audit.json").read_text())
    multimodal = json.loads((E2_ROOT / "multimodal_motion_diagnostics.json").read_text())
    custom_groups = multimodal["sources"]["custom_canonical"]["groups"]
    for cell in cells:
        group = custom_groups.get(cell["session"], {})
        cell["target_diagnostics"] = {
            "motion_energy_q50": group.get("skeleton_features", {}).get("motion_energy", {}).get("q50"),
            "jerk_energy_q50": group.get("skeleton_features", {}).get("jerk_energy", {}).get("q50"),
            "imu_energy_q50": group.get("imu_features", {}).get("acc_energy", {}).get("q50"),
            "complexity_observation_count": group.get("records"),
        }

    exploratory = []
    for source in sorted({cell["source"] for cell in cells}):
        subset = [cell for cell in cells if cell["source"] == source and cell["condition"] == "zero_shot"]
        exploratory.append(
            {
                "source": source,
                "n_sessions": len(subset),
                "frame_acc_vs_target_motion_energy": correlation(
                    [cell["seed_mean"] for cell in subset if cell["target_diagnostics"]["motion_energy_q50"] is not None],
                    [cell["target_diagnostics"]["motion_energy_q50"] for cell in subset if cell["target_diagnostics"]["motion_energy_q50"] is not None],
                ),
                "interpretation": "exploratory target-session association only; source and target are not randomized and n=4",
            }
        )

    availability = {}
    for name, decision in semantic["source_decisions"].items():
        availability[name] = {
            "gate_status": decision["status"],
            "benchmark_status": "existing_g6_canonical" if name in {"totalcapture_gt", "egohumans_canonical"} else "not_benchmarked_in_g6",
            "reason": decision["reason"],
        }
    report = {
        "schema_version": "g9-e3-source-target-1",
        "inputs": {
            "g6_root": str(G6_ROOT),
            "g6_protocol_hash": "b0cf097e5e4ad4554c74fe75fe0a3d7a9430e67da1dfd2e18662ad2bd87aef45",
            "semantic_audit": str(E1_ROOT / "semantic_audit.json"),
            "multimodal_diagnostics": str(E2_ROOT / "multimodal_motion_diagnostics.json"),
        },
        "availability": availability,
        "existing_g6_cells": cells,
        "exploratory_target_session_associations": exploratory,
        "missing_controlled_cells": [
            "S06 skeleton-source sweep with fixed IMU and Custom target",
            "2D versus 3D representation-controlled transfer",
            "7D IMU contract and invalid-quaternion filtered fusion control",
            "prediction-level complexity/tracklet correct-total stratification",
        ],
        "limitations": [
            "Existing G6 cells use canonical source skeletons and do not benchmark the S06 algorithm outputs.",
            "Session feature joins are descriptive; they do not establish a causal effect without controlled interventions.",
        ],
    }
    output = args.output_root / "source_target_matrix.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "g6_cells": len(cells), "runs": len(runs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
