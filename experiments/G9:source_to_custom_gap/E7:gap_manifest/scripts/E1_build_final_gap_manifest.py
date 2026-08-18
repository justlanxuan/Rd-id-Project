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
        "prediction_stratification": e3 / "prediction_stratification.json",
        "s06_sweep_summary": e3 / "s06_eval" / "s06_sweep_summary.json",
        "s06_imu_filter_control": e3 / "custom_imu_filter_control.json",
        "s06_prediction_stratification": e3 / "s06_prediction_stratification.json",
        "g6_representation_boundary": e3 / "g6_representation_boundary.json",
        "custom_detector_id_audit": e2 / "custom_detector_id_audit.json",
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
    prediction = reports["prediction_stratification"]
    s06_sweep = reports["s06_sweep_summary"]
    imu_control = reports["s06_imu_filter_control"]
    s06_prediction = reports["s06_prediction_stratification"]
    representation_boundary = reports["g6_representation_boundary"]
    detector_ids = reports["custom_detector_id_audit"]
    missing_controlled_cells = [
        item for item in matrix["missing_controlled_cells"]
        if all(
            marker not in item
            for marker in (
                "S06 skeleton-source sweep",
                "7D IMU contract and invalid-quaternion filtered fusion control",
                "prediction-level complexity/tracklet correct-total stratification for the new S06 sources",
                "2D versus 3D representation-controlled transfer",
            )
        )
    ]

    manifest = {
        "schema_version": "g9-final-gap-manifest-1",
        "status": "diagnostic_complete_protocol_boundaries_explicit",
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
                "status": "audited_split_xy_only_encoder_boundary",
                "evidence": {
                    "canonical": {name: value.get("representation") for name, value in semantic["canonical_samples"].items()},
                    "s06": {name: value.get("representation") for name, value in semantic["s06_methods"].items()},
                    "g6_boundary": representation_boundary,
                },
            },
            "coordinate_space": {
                "status": "audited_fixed_checkpoint_intervention_mixed_effect",
                "evidence": "EgoHumans raw xy is pixel-like; Custom is normalized-like; S06 3D outputs are root/torso-scaled by metadata. D3/D4 screen calibration changes FrameAcc by source-specific deltas; this is not a universal repair.",
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
                "evidence": {"feature_screening": "B1 per-source and Custom-session low/mid/high tertiles.", "prediction_stratification": str(inputs["prediction_stratification"])},
            },
            "cross_modal_relation": {
                "status": "screened_lag_correlation",
                "evidence": {name: value["cross_modal_lag"] for name, value in multimodal["sources"].items()},
            },
            "imu_filter_control": {
                "status": "fixed_checkpoint_control_completed_localized_effect",
                "evidence": {
                    "path": str(inputs["s06_imu_filter_control"]),
                    "raw_history_frame_acc": imu_control["raw"]["weighted_frame_acc"],
                    "invalid_fill_history_frame_acc": imu_control["invalid_fill_only"]["weighted_frame_acc"],
                    "delta_history_frame_acc": imu_control["delta_invalid_fill_only"]["history_frame_acc"],
                    "delta_correct": imu_control["delta_invalid_fill_only"]["correct"],
                    "session_filter_stats": imu_control["filter"],
                },
            },
            "tracking_identity": {
                "status": "custom_detector_id_transitions_audited_s06_id_switch_unobservable",
                "evidence": {
                    "identity_switch": {name: value["identity_switch_status"] for name, value in tracking["sources"].items()},
                    "custom_detector_id_audit": {
                        "path": str(inputs["custom_detector_id_audit"]),
                        "sessions": detector_ids["sessions"],
                        "s06_limitation": detector_ids["s06_limitation"],
                    },
                    "s06_prediction_stratification": {
                        "path": str(inputs["s06_prediction_stratification"]),
                        "methods": len(s06_prediction["methods"]),
                        "missing": s06_prediction["missing"],
                        "limitations": s06_prediction["limitations"],
                    },
                },
            },
        },
        "gap_factors": [
            {
                "id": "GAP-REP-01",
                "factor": "2D/3D representation and coordinate normalization",
                "support": "controlled fixed-checkpoint intervention with mixed source-specific effects; D7 proves current encoder is xy-only",
                "intervention": "D3/D4 screen-calibrated sweep completed; full-xyz effect is not identifiable under current G6 protocol",
                "controlled_evidence": {
                    "path": str(inputs["s06_sweep_summary"]),
                    "cells": len(s06_sweep["cells"]),
                    "sequence_deltas": len(s06_sweep["sequence_deltas"]),
                    "method_deltas": s06_sweep["method_deltas"],
                    "interpretation": s06_sweep["interpretation"],
                    "g6_boundary": str(inputs["g6_representation_boundary"]),
                },
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
                "support": "measured and fixed-checkpoint controlled; effect localized by session",
                "intervention": "invalid-fill-only control completed; 15.26% invalid session decreased history FrameAcc by 4.50 points, aggregate by 1.04 points; policy is not a universal repair",
                "controlled_evidence": str(inputs["s06_imu_filter_control"]),
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
                "support": "screened and connected to S06 prediction-level correct/total strata",
                "intervention": "D6 complexity strata completed; complexity-matched retraining remains outside this fixed-checkpoint control",
                "controlled_evidence": str(inputs["s06_prediction_stratification"]),
            },
            {
                "id": "GAP-TRACK-01",
                "factor": "visibility/tracklet fragmentation and unknown independent ID switches",
                "support": "visibility/fragmentation prediction strata plus Custom AlphaPose raw detector-ID transitions",
                "intervention": "D6 fragmentation proxy and D8 Custom detector-ID audit completed; S06 independent IDs remain unavailable",
                "controlled_evidence": {
                    "prediction_strata": str(inputs["s06_prediction_stratification"]),
                    "custom_detector_ids": str(inputs["custom_detector_id_audit"]),
                },
            },
        ],
        "performance_evidence": {
            "existing_g6": matrix["existing_g6_cells"],
            "existing_prediction_stratification": {
                "path": str(inputs["prediction_stratification"]),
                "clips_processed": prediction["clips_processed"],
                "missing_segments": prediction["missing_segments"],
            },
            "s06_fixed_checkpoint_sweep": {
                "path": str(inputs["s06_sweep_summary"]),
                "cells": len(s06_sweep["cells"]),
                "missing": s06_sweep["missing"],
                "causal_scope": s06_sweep["interpretation"]["causal_scope"],
            },
            "s06_imu_filter_control": {
                "path": str(inputs["s06_imu_filter_control"]),
                "raw_correct_total": [imu_control["raw"]["correct"], imu_control["raw"]["total"]],
                "invalid_fill_correct_total": [imu_control["invalid_fill_only"]["correct"], imu_control["invalid_fill_only"]["total"]],
                "delta": imu_control["delta_invalid_fill_only"],
            },
            "s06_prediction_stratification": {
                "path": str(inputs["s06_prediction_stratification"]),
                "methods": len(s06_prediction["methods"]),
                "missing": s06_prediction["missing"],
            },
            "g6_representation_boundary": {
                "path": str(inputs["g6_representation_boundary"]),
                "raw_pose_max_abs_diff": representation_boundary["observed"]["raw_pose_max_abs_diff"],
                "skeleton_token_max_abs_diff": representation_boundary["observed"]["skeleton_token_max_abs_diff"],
                "conclusion": representation_boundary["conclusion"],
            },
            "custom_detector_id_audit": {
                "path": str(inputs["custom_detector_id_audit"]),
                "sessions": detector_ids["sessions"],
                "s06_limitation": detector_ids["s06_limitation"],
            },
            "missing_controlled_cells": missing_controlled_cells,
            "causal_claim_allowed": False,
        },
        "next_required_controls": [
            "For S06 ID-switch attribution, preserve independent detector IDs in a future extractor output; Custom AlphaPose transitions are now audited but are not transferable to S06.",
            "If full-xyz attribution is required, create a new xyz-compatible encoder/protocol; current G6 is proven xy-only.",
        ],
    }
    output = args.g9_root / "g9_final_gap_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "status": manifest["status"], "included": manifest["selective_gate"]["included"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
