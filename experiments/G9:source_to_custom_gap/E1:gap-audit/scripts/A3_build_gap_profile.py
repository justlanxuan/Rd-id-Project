# Experiment Note: A3-gap-profile
"""Build a traceable, non-training gap manifest from the E1 audit reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e1_gap_audit")
    )
    args = parser.parse_args()
    inventory_path = args.audit_root / "source_inventory.json"
    semantic_path = args.audit_root / "semantic_audit.json"
    inventory = json.loads(inventory_path.read_text())
    semantic = json.loads(semantic_path.read_text())

    sample_by_name = {item["name"]: item for item in inventory["sources"]}
    canonical = semantic["canonical_samples"]
    sources = semantic["source_decisions"]

    coordinate_evidence = {}
    for name, item in canonical.items():
        sample = sample_by_name.get(name, {})
        quality = (sample.get("sample_quality") or [{}])[0]
        coordinate_evidence[name] = {
            "sample_shape": item.get("skeleton_shape"),
            "representation": item.get("representation"),
            "coordinate_dim": item.get("coordinate_dim"),
            "sample_range": item.get("sample_range"),
            "inventory_quality": {
                "mean": quality.get("mean"),
                "std": quality.get("std"),
                "min": quality.get("min"),
                "max": quality.get("max"),
                "bone_length_cv": quality.get("bone_length_cv"),
            },
            "declared_space": {
                "totalcapture_gt": "canonical gt_skeleton; gt_skeleton_meters is a separate field and must not be silently substituted",
                "egohumans_canonical": "canonical gt_skeleton; numeric scale is not declared in the NPZ and must be normalized explicitly",
                "custom_canonical": "2D canonical window skeleton; scale/normalization is defined by the Custom preprocessing contract",
            }.get(name, "unknown"),
        }

    method_evidence = {}
    for method, item in semantic["s06_methods"].items():
        method_evidence[method] = {
            "status": sources[method]["status"],
            "shape": item.get("skeleton_shape"),
            "representation": item.get("representation"),
            "sample_range": item.get("sample_range"),
            "mapping": item.get("mapping"),
            "source_metadata": item.get("source"),
            "alignment_metadata": item.get("alignment_json"),
        }

    profile = {
        "schema_version": "g9-e1-gap-profile-1",
        "inputs": {
            "source_inventory": {"path": str(inventory_path), "sha256": digest(inventory_path)},
            "semantic_audit": {"path": str(semantic_path), "sha256": digest(semantic_path)},
        },
        "gate": {
            "policy": "selective_per_source",
            "minimal_trusted_subset": semantic["minimal_trusted_subset"],
            "conditional": [name for name, item in sources.items() if item["status"] == "conditional"],
            "pending": [name for name, item in sources.items() if item["status"] == "pending"],
        },
        "coordinate_space_gap": {
            "evidence": {**coordinate_evidence, "s06_methods": method_evidence},
            "finding": "The trusted sources are finite and H36M-17-shaped, but their numeric coordinate spaces are not interchangeable by shape alone.",
            "required_control": "Compare root-centered/torso-scaled normalized skeletons in a dedicated representation track; retain raw fields for audit.",
        },
        "identity_and_time_gap": {
            "totalcapture": canonical["totalcapture_gt"]["mapping"],
            "egohumans": canonical["egohumans_canonical"]["mapping"],
            "custom": {
                "npz_embedded_mapping": canonical["custom_canonical"]["mapping"],
                "window_csv_mapping": semantic["custom_all_fold_mapping"],
                "timing_note": "Custom NPZ sample has no frame_ids; window_start/window_end/window_len are carried by the fold CSV.",
            },
            "s06_external_join": semantic["s06_baseline_mapping"],
        },
        "quality_and_independence_gap": {
            "pairwise_summary": {
                "pair_count": len(semantic["s06_pairwise"]),
                "exact_duplicate_pairs": [
                    [item["left"], item["right"]]
                    for item in semantic["s06_pairwise"]
                    if item.get("exact_equal")
                ],
                "note": "High correlation is retained as a possible shared-backbone/source effect, not labeled as duplication.",
            },
            "outlier_source": "yolopose_high",
            "outlier_action": "conditional_until_per_joint_coordinate_audit",
        },
        "gap_hypotheses": [
            {
                "id": "GAP-COORD-01",
                "hypothesis": "Coordinate normalization/representation, rather than skeleton validity, drives part of source→Custom discrepancy.",
                "evidence": "Canonical 3D/2D sources have materially different ranges while all sampled arrays are finite.",
                "next_measurement": "Root-center, torso-scale and Procrustes-controlled skeleton statistics before any model comparison.",
            },
            {
                "id": "GAP-TIME-01",
                "hypothesis": "Custom window timing is encoded outside the NPZ and can be misjoined if CSV provenance is dropped.",
                "evidence": "Custom mapping is verified in 12 CSV manifests; the NPZ itself has no frame_ids or person IDs.",
                "next_measurement": "Reconstruct timestamps from window_start/window_end and compare IMU/video lag per session.",
            },
            {
                "id": "GAP-QUALITY-01",
                "hypothesis": "YOLO-Pose high has a quality/coverage outlier that should not be mixed into the first source ranking.",
                "evidence": "Lower S06 coverage and high sample bone CV/range; no exact duplicate explanation.",
                "next_measurement": "Per-frame/per-joint range, missingness, confidence and rendered identity audit.",
            },
            {
                "id": "GAP-MODAL-01",
                "hypothesis": "After skeleton semantics are controlled, residual gap may arise from IMU marginal statistics or cross-modal relation.",
                "evidence": "Person/IMU joins are verified for the trusted subset, so this factor is testable rather than conflated with identity errors.",
                "next_measurement": "IMU distribution, wrist/forearm velocity lag, and skeleton-only versus fusion controls.",
            },
        ],
    }
    output = args.audit_root / "gap_profile.json"
    output.write_text(json.dumps(finite_or_none(profile), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "trusted_subset": profile["gate"]["minimal_trusted_subset"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
