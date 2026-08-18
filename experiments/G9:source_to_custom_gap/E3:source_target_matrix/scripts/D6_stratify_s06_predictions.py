# Experiment Note: D6-S06-prediction-strata
"""Join S06 fixed-checkpoint predictions with motion and visibility strata.

The strata are computed from the same S06 sequence artifact used by D3. Motion
complexity uses root-centered, bone-scale-normalized skeleton speed; visibility
and fragmentation use the output visibility mask. S06 outputs retain inherited
person order rather than independent tracker IDs, so fragmentation is a
visibility-run proxy and ID-switch attribution remains out of scope.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

S06_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
)
EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)
METHODS = ("alphapose", "yolopose_high", "fmpose3d", "motionagformer", "tcpformer", "wham")
VARIANTS = ("raw", "screen_calibrated")


def sequence_stem(sequence_id: str) -> str:
    return sequence_id.removeprefix("custom_").removesuffix("_seg0")


def motion_features(skeleton: np.ndarray, visibility: np.ndarray) -> dict[str, float]:
    coords = np.asarray(skeleton, dtype=np.float64)[..., :2]
    visible = np.asarray(visibility, dtype=bool)
    values: list[float] = []
    fragmentation = 0
    for person in range(coords.shape[1]):
        valid = visible[:, person] & np.isfinite(coords[:, person]).all(axis=(1, 2))
        if valid.any():
            runs = int(np.sum(valid & ~np.concatenate(([False], valid[:-1]))))
            fragmentation += runs
            xy = coords[:, person]
            root = xy[:, :1]
            centered = xy - root
            bone = np.stack(
                [np.linalg.norm(centered[:, left] - centered[:, right], axis=-1) for left, right in EDGES],
                axis=-1,
            )
            scale_values = bone[np.isfinite(bone) & (bone > 1e-8)]
            scale = float(np.median(scale_values)) if scale_values.size else 1.0
            speed = np.linalg.norm(np.diff(centered, axis=0), axis=-1) / scale
            values.extend(speed[np.isfinite(speed) & valid[1:, None]].reshape(-1).tolist())
    return {
        "motion_energy": float(np.mean(values)) if values else 0.0,
        "visibility_coverage": float(visible.mean()) if visible.size else 0.0,
        "mean_visible_people": float(visible.sum(axis=1).mean()) if visible.size else 0.0,
        "fragmentation_runs": float(fragmentation),
    }


def bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "mid"
    return "high"


def aggregate(rows: list[dict[str, Any]], field: str, thresholds: tuple[float, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("low", "mid", "high"):
        selected = [row for row in rows if bucket(float(row[field]), thresholds) == name]
        correct = sum(int(row["correct"]) for row in selected)
        total = sum(int(row["total"]) for row in selected)
        result[name] = {
            "sequences": len(selected),
            "correct": correct,
            "total": total,
            "frame_acc": float(correct / total) if total else 0.0,
            "value_mean": float(np.mean([row[field] for row in selected])) if selected else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/s06_eval/s06_sweep_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/s06_prediction_stratification.json"))
    parser.add_argument("--s06-root", type=Path, default=S06_ROOT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    sequence_rows = summary["sequence_deltas"]
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for method in METHODS:
        for row in sequence_rows:
            if row["method"] != method:
                continue
            stem = sequence_stem(row["sequence_id"])
            path = args.s06_root / method / f"custom_{stem}.npz"
            if path.exists() and stem not in metrics[method]:
                with np.load(path, allow_pickle=True) as archive:
                    metrics[method][stem] = motion_features(archive["skeleton"], archive["visibility"])
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for stem, values in metrics[method].items():
            all_rows.append({"method": method, "stem": stem, **values})
    fields = ("motion_energy", "visibility_coverage", "fragmentation_runs")
    thresholds = {
        field: tuple(float(x) for x in np.quantile([row[field] for row in all_rows], [1 / 3, 2 / 3]))
        for field in fields
    }
    output: dict[str, Any] = {
        "schema_version": "g9-e3-s06-prediction-stratification-1",
        "protocol": {
            "prediction_source": str(args.summary),
            "strata_population": "pooled six-method S06 sequences",
            "motion_definition": "root-centered xy speed divided by median positive H36M17 bone length",
            "visibility_definition": "S06 output visibility mask",
            "fragmentation_definition": "visibility true-run count; proxy only because IDs are inherited person order",
        },
        "thresholds": thresholds,
        "methods": {},
        "missing": [],
        "limitations": [
            "D3 evaluates a fixed EgoHumans checkpoint and does not retrain per S06 source.",
            "3D outputs are reduced to xy by the G6 encoder.",
            "No independent detector track IDs are present; fragmentation is not an ID-switch measurement.",
        ],
    }
    for method in METHODS:
        output["methods"][method] = {}
        rows_by_method = [row for row in sequence_rows if row["method"] == method]
        for variant in VARIANTS:
            rows: list[dict[str, Any]] = []
            for row in rows_by_method:
                stem = sequence_stem(row["sequence_id"])
                values = metrics[method].get(stem)
                key = "raw_frame_acc" if variant == "raw" else "screen_calibrated_frame_acc"
                if values is None:
                    output["missing"].append(f"{method}|{variant}|{stem}")
                    continue
                rows.append({**values, "correct": row["raw_correct_total"][0] if variant == "raw" else row["screen_correct_total"][0], "total": row["raw_correct_total"][1] if variant == "raw" else row["screen_correct_total"][1], "frame_acc": row[key]})
            output["methods"][method][variant] = {
                "sequence_count": len(rows),
                "motion_complexity": aggregate(rows, "motion_energy", thresholds["motion_energy"]),
                "visibility": aggregate(rows, "visibility_coverage", thresholds["visibility_coverage"]),
                "fragmentation_proxy": aggregate(rows, "fragmentation_runs", thresholds["fragmentation_runs"]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "methods": len(output["methods"]), "missing": len(output["missing"]), "pooled_sequences": len(all_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
