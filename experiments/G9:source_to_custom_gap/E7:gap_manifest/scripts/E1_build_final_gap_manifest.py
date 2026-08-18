# Experiment Note: E7-final-gap-manifest
"""Aggregate the auditable G9 evidence into one gap manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/data/fzliang/reid-project/g9")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g9-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    e1 = args.g9_root / "e1_gap_audit"
    e2 = args.g9_root / "e2_multimodal"
    e3 = args.g9_root / "e3_source_target"
    inputs = {
        "source_inventory": e1 / "source_inventory.json",
        "semantic_audit": e1 / "semantic_audit.json",
        "gap_profile": e1 / "gap_profile.json",
        "coordinate_outlier_audit": e1 / "coordinate_outlier_audit.json",
        "multimodal_motion_diagnostics": e2 / "multimodal_motion_diagnostics.json",
        "tracking_quality": e2 / "tracking_quality.json",
        "imu_contract_comparison": e2 / "imu_contract_comparison.json",
        "source_target_matrix": e3 / "source_target_matrix.json",
    }
    missing = [name for name, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required G9 evidence: {missing}")
    reports = {name: load(path) for name, path in inputs.items()}
    semantic = reports["semantic_audit"]
    multimodal = reports["multimodal_motion_diagnostics"]
    tracking = reports["tracking_quality"]
    imu = reports["imu_contract_comparison"]
    outlier = reports["coordinate_outlier_audit"]
    matrix = reports["source_target_matrix"]

    manifest = {
        "schema_version": "g9-final-gap-manifest-1",
        "status": "diagnostic_complete_causal_controls_pending",
        "scope": "Source-to-Custom skeleton/IMU/temporal/complexity/tracking gap on the E1 included subset",
        "selective_gate": {
            "policy": "per_source; conditional/pending sources do not block the trusted subset",
            "included": semantic["minimal_trusted_subset"],
            "source_decisions": semantic["source_decisions"],
        },
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()
        },
        "requirements": {
            "joint_mapping": {
                "status": "verified_by_protocol_and_shape",
                "evidence": "H36M17 protocol plus explicit canonical/S06 metadata; shape alone is not treated as proof.",
            },
            "representation": {
                "status": "audited_and_split",
                "evidence": {name: value.get("representation") for name, value in semantic["canonical_samples"].items()},
                "s06": {name: value.get("representation") for name, value in semantic["s06_methods"].items()},
            },
            "coordinate_space": {
                "status": "audited_normalization_control_pending",
                "evidence": "EgoHumans raw xy is pixel-like; Custom is normalized-like; S06 3D outputs are root/torso-scaled by metadata; A4 provides root/torso controlled summaries.",
            },
            "person_imu_alignment": {
                "status": "verified_with_external_join_where_needed",
                "evidence": {
                    "custom_rows": semantic["custom_all_fold_mapping"]["rows"],
                    "custom_mismatches": semantic["custom_all_fold_mapping"]["person_imu_mismatches"],
                    "s06_baselines": semantic["s06_baseline_mapping"]["baseline_sequences_checked"],
                    "s06_mismatches": len(semantic["s06_baseline_mapping"]["mapping_mismatches"]),
                },
            },
            "time_alignment": {
                "status": "frame_join_verified_lag_screened",
                "evidence": "Canonical frame/IMU lengths match; Custom timing is in window CSV; B1 scans lag -8..8.",
            },
            "invalid_and_outlier_artifacts": {
                "status": "localized_and_reported",
                "evidence": {
                    "yolo_abs_gt_10_count": outlier["sources"]["s06_yolopose_high"]["raw_abs_gt_10_count"],
                    "yolo_files_with_extreme": outlier["sources"]["s06_yolopose_high"]["files_with_raw_abs_gt_10"],
                    "custom_invalid_quaternion_fraction": imu["sources"]["custom_canonical"]["quaternion_quality"]["invalid_frame_fraction"],
                    "custom_zero_quaternion_frames": imu["sources"]["custom_canonical"]["quaternion_quality"]["zero_norm_frames"],
                },
            },
            "duplicate_outputs": {
                "status": "no_exact_duplicate_in_s06_same_sequence_sample",
                "evidence": semantic["s06_pairwise"],
            },
            "motion_complexity": {
                "status": "screened",
                "evidence": "B1 per-source and Custom-session low/mid/high tertiles.",
            },
            "cross_modal_relation": {
                "status": "screened_lag_correlation",
                "evidence": {name: value["cross_modal_lag"] for name, value in multimodal["sources"].items()},
            },
            "tracking_identity": {
                "status": "visibility_and_tracklet_screened_id_switch_unobservable",
                "evidence": {name: value["identity_switch_status"] for name, value in tracking["sources"].items()},
            },
        },
        "gap_factors": [
            {
                "id": "GAP-REP-01",
                "factor": "2D/3D representation and coordinate normalization",
                "support": "strong observational",
                "intervention": "root/torso normalization and representation-separated sweep pending",
            },
            {
                "id": "GAP-IMU-01",
                "factor": "7D versus legacy48 layout and acceleration unit path",
                "support": "strong observational",
                "intervention": "explicit legacy48→7D conversion completed; physical frame/unit control pending",
            },
            {
                "id": "GAP-IMU-02",
                "factor": "Custom invalid quaternion tail",
                "support": "measured (0.91% frames; 28 zero norm)",
                "intervention": "filtered-versus-unfiltered fusion control pending",
            },
            {
                "id": "GAP-TIME-01",
                "factor": "Custom 10 Hz raw IMU to approximately 30 fps video resampling and residual lag",
                "support": "measured timing provenance and screened lag",
                "intervention": "timestamp-aware lag control pending",
            },
            {
                "id": "GAP-QUALITY-01",
                "factor": "YOLO-Pose high finite but extreme coordinates/coverage",
                "support": "localized to 996 coordinates in 54/88 sequences",
                "intervention": "conditional source; repair/reject policy and rerun pending",
            },
            {
                "id": "GAP-COMPLEXITY-01",
                "factor": "source/target motion complexity distribution",
                "support": "screened by bone-normalized energy, jerk, entropy and periodicity",
                "intervention": "complexity-matched prediction matrix pending",
            },
            {
                "id": "GAP-TRACK-01",
                "factor": "visibility/tracklet fragmentation and unknown independent ID switches",
                "support": "visibility and inherited-order evidence",
                "intervention": "raw detector track-ID audit pending",
            },
        ],
        "performance_evidence": {
            "existing_g6": matrix["existing_g6_cells"],
            "missing_controlled_cells": matrix["missing_controlled_cells"],
            "causal_claim_allowed": False,
        },
        "next_required_controls": [
            "Run source skeleton sweep with fixed verified IMU and Custom target, split 2D/3D.",
            "Run normalized versus raw coordinate controls without changing held-out sessions.",
            "Run IMU-only/skeleton-only/fusion with invalid-quaternion filtered denominators.",
            "Emit prediction-level complexity/visibility/tracklet correct-total tables.",
            "Add raw detector track IDs before claiming ID-switch attribution.",
        ],
    }
    output = args.g9_root / "g9_final_gap_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "status": manifest["status"], "included": manifest["selective_gate"]["included"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
