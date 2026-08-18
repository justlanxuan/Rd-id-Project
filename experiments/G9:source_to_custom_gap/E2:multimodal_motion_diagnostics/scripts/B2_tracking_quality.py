# Experiment Note: B2-tracking-quality
"""Audit visibility, tracklet fragmentation and identity provenance.

This audit distinguishes measured tracklet quality from identity information
that is simply inherited from a GT-ordered baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

S06_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
)
S06_BASELINE_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "data/interim/egohumans_repro_local/slice/sequences"
)
TOTAL_ROOT = Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences")
EGO_ROOT = Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source/sequences")
CUSTOM_ROOT = Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess")
METHODS = ("alphapose", "fmpose3d", "motionagformer", "tcpformer", "wham")


def runs(mask: np.ndarray) -> list[int]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return []
    starts = values & ~np.r_[False, values[:-1]]
    ends = values & ~np.r_[values[1:], False]
    return [int(end - start + 1) for start, end in zip(np.flatnonzero(starts), np.flatnonzero(ends), strict=True)]


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "q50": None, "q95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "q50": float(np.quantile(array, 0.5)),
        "q95": float(np.quantile(array, 0.95)),
    }


def track_stats(visibility: np.ndarray) -> dict[str, Any]:
    values = np.asarray(visibility, dtype=bool)
    if values.ndim == 1:
        values = values[:, None]
    run_lengths = [length for person in range(values.shape[1]) for length in runs(values[:, person])]
    fragmentation = [max(0, len(runs(values[:, person])) - 1) for person in range(values.shape[1])]
    candidate_group = np.sum(values, axis=1)
    return {
        "frames": int(values.shape[0]),
        "people": int(values.shape[1]),
        "coverage": float(np.mean(values)) if values.size else None,
        "candidate_group_size": dict(sorted((str(int(key)), int(value)) for key, value in Counter(candidate_group.tolist()).items())),
        "tracklet_run_length": summary([float(value) for value in run_lengths]),
        "fragmentation_per_person": summary([float(value) for value in fragmentation]),
        "all_people_visible_fraction": float(np.mean(np.all(values, axis=1))) if values.size else None,
    }


def compare_visibility(output: np.ndarray, baseline: np.ndarray | None) -> dict[str, Any]:
    result = track_stats(output)
    if baseline is None:
        result["baseline_visibility"] = "not_available"
        return result
    base = np.asarray(baseline, dtype=bool)
    out = np.asarray(output, dtype=bool)
    n0, n1 = min(base.shape[0], out.shape[0]), min(base.shape[1], out.shape[1])
    result["baseline"] = track_stats(base[:n0, :n1])
    result["vs_baseline"] = {
        "shape_overlap": [int(n0), int(n1)],
        "exact_visibility_equal": bool(np.array_equal(base[:n0, :n1], out[:n0, :n1])),
        "mean_abs_visibility_delta": float(np.mean(np.abs(base[:n0, :n1].astype(int) - out[:n0, :n1].astype(int)))),
    }
    return result


def s06_method(method: str) -> dict[str, Any]:
    records = []
    for path in sorted((S06_ROOT / method).glob("*.npz")):
        baseline_path = S06_BASELINE_ROOT / path.name
        with np.load(path, allow_pickle=True) as archive:
            visibility = archive["visibility"] if "visibility" in archive else None
            frame_ids = archive["frame_ids"] if "frame_ids" in archive else None
            person_ids = archive["gt_person_ids"] if "gt_person_ids" in archive else None
            skeleton = archive["skeleton"] if "skeleton" in archive else None
        baseline_visibility = None
        baseline_ids = None
        if baseline_path.exists():
            with np.load(baseline_path, allow_pickle=True) as baseline:
                baseline_visibility = baseline["gt_visibility"] if "gt_visibility" in baseline else None
                baseline_ids = baseline["gt_person_ids"] if "gt_person_ids" in baseline else None
        if visibility is None:
            continue
        item = compare_visibility(visibility, baseline_visibility)
        item.update(
            {
                "sequence_id": path.stem,
                "frame_ids_monotonic": bool(frame_ids is None or np.all(np.diff(frame_ids) > 0)),
                "identity_provenance": "inherited_gt_person_order",
                "embedded_person_ids": person_ids.tolist() if person_ids is not None else None,
                "baseline_person_ids": baseline_ids.tolist() if baseline_ids is not None else None,
                "skeleton_zero_fraction": float(np.mean(np.asarray(skeleton) == 0)) if skeleton is not None else None,
            }
        )
        records.append(item)
    def collect(key: str) -> list[float]:
        return [float(item[key]) for item in records if item.get(key) is not None]

    return {
        "records": len(records),
        "coverage": summary(collect("coverage")),
        "tracklet_run_length": summary([item["tracklet_run_length"]["q50"] for item in records]),
        "fragmentation_per_person": summary([item["fragmentation_per_person"]["q50"] for item in records]),
        "visibility_equal_to_baseline": sum(bool(item.get("vs_baseline", {}).get("exact_visibility_equal")) for item in records),
        "visibility_delta_q50": summary([item["vs_baseline"]["mean_abs_visibility_delta"] for item in records if "vs_baseline" in item]),
        "candidate_group_size": Counter(
            group
            for item in records
            for group, count in item["candidate_group_size"].items()
            for _ in range(count)
        ),
        "identity_switch_status": "not_identifiable_from_outputs; gt_person_ids are inherited and no independent track IDs are stored",
        "sequences": records,
    }


def canonical_summary(name: str, root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=True) as archive:
            if "gt_visibility" in archive:
                visibility = archive["gt_visibility"]
            elif "skeleton" in archive and archive["skeleton"].shape[-1] == 3:
                visibility = archive["skeleton"][..., 2].min(axis=-1) > 0
            else:
                visibility = None
            frame_ids = archive["frame_ids"] if "frame_ids" in archive else None
        if visibility is None:
            records.append({"sequence_id": path.stem, "status": "no_visibility_field"})
            continue
        item = track_stats(visibility)
        item.update(
            {
                "sequence_id": path.stem,
                "frame_ids_monotonic": bool(frame_ids is None or np.all(np.diff(frame_ids) > 0)),
                "identity_provenance": "explicit_gt_person_ids_or_window_csv",
            }
        )
        records.append(item)
    return {"records": len(records), "sequences": records}


def custom_summary() -> dict[str, Any]:
    rows = 0
    sources = Counter()
    mapping_mismatch = 0
    for csv_path in sorted(CUSTOM_ROOT.glob("fold*/windows_*.csv")):
        import csv

        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                sources[row.get("skeleton_source", "")] += 1
                mapping_mismatch += int(row.get("person_idx") != row.get("imu_idx"))
    return {
        "window_rows": rows,
        "skeleton_sources": dict(sources),
        "mapping_mismatches": mapping_mismatch,
        "visibility_status": "not_embedded_in_custom_window_npz",
        "identity_provenance": "window_csv_person_idx_and_imu_idx",
        "tracking_status": "GT windows; detector tracklet/ID-switch analysis requires raw detector outputs",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e2_multimodal"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "g9-e2-tracking-1",
        "sources": {f"s06_{method}": s06_method(method) for method in METHODS},
        "canonical": {
            "totalcapture_gt": canonical_summary("totalcapture_gt", TOTAL_ROOT),
            "egohumans_canonical": canonical_summary("egohumans_canonical", EGO_ROOT),
        },
        "custom": custom_summary(),
        "limitations": [
            "Visibility and fragmentation are measurable for S06 outputs, but independent ID switches are not: output person order is inherited from GT.",
            "Custom canonical windows are GT skeletons without embedded visibility/track IDs; raw detector caches are needed for detector-specific tracking metrics.",
        ],
    }
    output = args.output_root / "tracking_quality.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda value: value.item() if isinstance(value, np.generic) else value) + "\n")
    print(json.dumps({"output": str(output), "methods": list(report["sources"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
