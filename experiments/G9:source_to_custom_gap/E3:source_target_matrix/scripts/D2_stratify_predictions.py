# Experiment Note: D2-prediction-stratification
"""Recompute existing G6 correct/total by target motion and tracking strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

G6_ROOT = Path("/data/fzliang/reid-project/g6/c9a5d3099979296a72314eba66274855e03ab1eb")
SEGMENT_ROOT = Path("/data/fzliang/reid-project/custom/evaluation/custom_segments/sequences")


def motion_energy(skeleton: np.ndarray) -> np.ndarray:
    coords = np.asarray(skeleton, dtype=np.float64)[..., :2]
    root = coords[..., 0:1, :]
    centered = coords - root
    torso = np.linalg.norm(centered[..., 8, :] - centered[..., 0, :], axis=-1)
    scale = np.where(torso > 1e-6, torso, 1.0)
    normalized = centered / scale[..., None, None]
    velocity = np.linalg.norm(np.diff(normalized, axis=0), axis=-1)
    values = np.mean(velocity, axis=(-1, -2))
    return np.r_[values[0] if values.size else 0.0, values]


def add_count(target: dict[str, dict[str, int]], key: str, correct: int, total: int) -> None:
    target.setdefault(key, {"correct": 0, "total": 0})
    target[key]["correct"] += correct
    target[key]["total"] += total


def process_clip(clip: dict[str, Any], source: str, condition: str, seed: int, output: dict[str, Any]) -> None:
    sequence_id = clip["sequence_id"]
    path = SEGMENT_ROOT / f"{sequence_id}.npz"
    if not path.exists():
        output["missing_segments"].append(sequence_id)
        return
    with np.load(path, allow_pickle=True) as archive:
        gt_visibility = np.asarray(archive["gt_visibility"], dtype=bool)
        gt_to_extract = np.asarray(archive["gt_to_extract_map"], dtype=np.int64)
        extract_visibility = np.asarray(archive["extract_visibility"], dtype=bool)
        skeleton = np.asarray(archive["extract_skeleton"], dtype=np.float64)
        gt_ids = np.asarray(archive["gt_person_ids"], dtype=np.int64)
        extract_ids = np.asarray(archive["extract_person_ids"], dtype=np.int64)
    energy = motion_energy(skeleton)
    cuts = np.quantile(energy, [1 / 3, 2 / 3]) if energy.size else (0.0, 0.0)
    modes = {"history": clip.get("frame_assignments", []), "instantaneous": clip.get("instantaneous_frame_assignments", [])}
    for mode, assignments in modes.items():
        if not assignments:
            continue
        assignments_array = np.asarray(assignments, dtype=np.int64)
        session = "_".join(sequence_id.split("_")[1:3])
        counts = output["by_source_condition_mode"].setdefault(
            f"{source}|{condition}|{session}|{mode}",
            {"complexity": {}, "candidate_group_size": {}, "visible_people": {}, "seeds": [], "clips": 0},
        )
        if seed not in counts["seeds"]:
            counts["seeds"].append(seed)
        counts["clips"] += 1
        n_frames = min(len(assignments_array), len(gt_visibility), len(energy))
        for t in range(n_frames):
            if energy[t] <= cuts[0]:
                complexity = "low"
            elif energy[t] > cuts[1]:
                complexity = "high"
            else:
                complexity = "mid"
            candidate_group = str(int(np.sum(extract_visibility[t])))
            visible_people = str(int(np.sum(gt_visibility[t])))
            for gt_index, gt_id in enumerate(gt_ids):
                if not gt_visibility[t, gt_index]:
                    continue
                expected_track = int(gt_to_extract[t, gt_index])
                if expected_track < 0:
                    continue
                matched = np.flatnonzero(assignments_array[t] == expected_track)
                is_correct = int(len(matched) == 1 and extract_ids[int(matched[0])] == gt_id)
                add_count(counts["complexity"], complexity, is_correct, 1)
                add_count(counts["candidate_group_size"], candidate_group, is_correct, 1)
                add_count(counts["visible_people"], visible_people, is_correct, 1)
        output["clips_processed"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": "g9-e3-prediction-stratification-1", "clips_processed": 0, "missing_segments": [], "by_source_condition_mode": {}}
    for path in sorted((G6_ROOT / "artifacts/evaluate").glob("*/results.json")):
        record_path = G6_ROOT / "artifacts/records" / path.parent.name / "run_record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        if record.get("status") != "completed":
            continue
        evaluation = json.loads(path.read_text()).get("evaluations", {}).get("frame_acc", {})
        for clip in evaluation.get("clips", []):
            process_clip(clip, record.get("source") or "none", record.get("condition") or "unknown", int(record.get("seed", -1)), report)
    for value in report["by_source_condition_mode"].values():
        for stratum_name in ("complexity", "candidate_group_size", "visible_people"):
            for counts in value[stratum_name].values():
                counts["frame_acc"] = counts["correct"] / counts["total"] if counts["total"] else None
    output = args.output_root / "prediction_stratification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "clips_processed": report["clips_processed"], "missing_segments": len(report["missing_segments"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
