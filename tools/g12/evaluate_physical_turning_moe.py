"""Evaluate the frozen physical turning expert with baseline fallback exactly once."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.g12.orientation_matcher import OrientationAwareMatcher
from src.g12.orientation_motion import OrientationMotionDataset
from src.metrics.turning import physical_turning_score

ROOT = Path("/data/fzliang/reid-project")
CACHE = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2"
THRESHOLD = 19.0 / 48.0
MAX_LAG = 2


def _physical_score(orientation: np.ndarray, imu: np.ndarray) -> float:
    return physical_turning_score(orientation, imu, max_lag=MAX_LAG)


def _dataset(dataset: str, csv_name: str, session: str | None = None) -> OrientationMotionDataset:
    spec: dict[str, Any] = {
        "dataset": dataset,
        "csv": str(CACHE / "manifests" / csv_name),
        "root": str(CACHE),
        "fps_hz": 30.0,
        "gyro_sidecar_root": str(ROOT / "g11/custom_complete_sidecars_v1"),
    }
    if session is not None:
        spec["session_filter"] = session
    return OrientationMotionDataset(
        [spec],
        orientation_mode="3d_heading",
        target_len=24,
        skeleton_normalize="bbox",
        imu_normalize="separate_zscore",
        window_seconds=0.8,
    )


def _model(checkpoint: dict[str, Any], dataset: OrientationMotionDataset, device: torch.device) -> OrientationAwareMatcher:
    config = checkpoint["config"]
    sample = dataset[0]
    model = OrientationAwareMatcher(
        int(sample["skeleton"].shape[-1]),
        int(sample["imu"].shape[-1]),
        hidden=int(config["hidden"]),
        embedding_dim=int(config["embedding_dim"]),
        temporal_mode=str(config["temporal_mode"]),
        multiscale_fusion=str(config["multiscale_fusion"]),
        window_seconds=float(config["window_seconds"]),
        use_orientation=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _evaluate(dataset: OrientationMotionDataset, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _model(checkpoint, dataset, device)
    items = [dataset[index] for index in range(len(dataset))]
    skeleton_embedding: list[np.ndarray] = []
    imu_embedding: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), 128):
            batch = items[start : start + 128]
            output = model(
                torch.stack([item["skeleton"] for item in batch]).to(device),
                torch.stack([item["imu"] for item in batch]).to(device),
                torch.stack([item["orientation"] for item in batch]).to(device),
            )
            skeleton_embedding.extend(output["skeleton"].cpu().numpy())
            imu_embedding.extend(output["imu"].cpu().numpy())
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[str(row["_group_key"])].append(index)
    result = {
        key: {stratum: {"correct": 0, "total": 0} for stratum in ("all", "high", "low")}
        for key in ("baseline", "physical_only", "turning_moe", "turning_moe_persistent")
    }
    turning_count_by_group = {
        group_key: int(
            sum(round(float(items[index]["orientation"][:, 4].sum().item())) for index in indices)
        )
        for group_key, indices in groups.items()
    }
    start_by_group = {
        group_key: int(dataset.rows[indices[0]]["source_window_start"])
        for group_key, indices in groups.items()
    }
    high_groups = {group_key for group_key, count in turning_count_by_group.items() if count >= 19}
    persistent_high_groups = {
        group_key
        for group_key in high_groups
        if any(
            other != group_key
            and abs(start_by_group[group_key] - start_by_group[other]) <= 24
            for other in high_groups
        )
    }
    routed_high_groups = 0
    for group_key, indices in groups.items():
        if len(indices) < 2:
            continue
        turning_count = turning_count_by_group[group_key]
        stratum = "high" if turning_count >= 19 else "low"
        routed_high_groups += int(stratum == "high")
        baseline = np.asarray(
            [[skeleton_embedding[left] @ imu_embedding[right] for right in indices] for left in indices]
        )
        physical = np.asarray(
            [
                [
                    _physical_score(
                        items[left]["orientation"].numpy(),
                        items[right]["imu"].numpy(),
                    )
                    for right in indices
                ]
                for left in indices
            ]
        )
        scores = {
            "baseline": baseline,
            "physical_only": physical,
            "turning_moe": physical if stratum == "high" else baseline,
            "turning_moe_persistent": physical if group_key in persistent_high_groups else baseline,
        }
        for method, matrix in scores.items():
            for row_index in range(len(indices)):
                correct = int(int(np.argmax(matrix[row_index])) == row_index)
                for label in ("all", stratum):
                    result[method][label]["correct"] += correct
                    result[method][label]["total"] += 1
    for method in result.values():
        for record in method.values():
            record["accuracy"] = record["correct"] / record["total"] if record["total"] else None
    return {
        "metrics": result,
        "groups": len(groups),
        "routed_high_groups": routed_high_groups,
        "persistent_routed_high_groups": len(persistent_high_groups),
    }


def main() -> int:
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    datasets = {
        "custom23_test": _dataset("custom23_test", "custom23_test.csv"),
        "custom57": _dataset("custom57", "custom23_eval.csv", "20260211_171724"),
        "custom22": _dataset("custom22", "custom23_eval.csv", "20260211_172257"),
        "custom24": _dataset("custom24", "custom23_eval.csv", "20260211_172522"),
    }
    results: dict[str, list[dict[str, Any]]] = {key: [] for key in datasets}
    for seed in range(5):
        checkpoint = ROOT / f"g12/e4_1_group_confirmation/baseline_seed{seed}/best.pt"
        for name, dataset in datasets.items():
            results[name].append({"seed": seed, **_evaluate(dataset, checkpoint, device)})
    artifact = {
        "schema_version": "g12.physical_turning_moe.v2",
        "preregistered": {
            "turning_threshold": THRESHOLD,
            "max_lag_frames": MAX_LAG,
            "physical_score": "max_lag_pearson(abs_3d_heading_rate, gyro_magnitude)",
            "routing": "physical_if_group_activity_ge_threshold_else_frozen_baseline",
            "posthoc_safety_routing": "also require another high group within 24 source frames",
        },
        "results": results,
    }
    output = ROOT / "g12/e4_1_physical_turning_moe.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "custom23_test": results["custom23_test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
