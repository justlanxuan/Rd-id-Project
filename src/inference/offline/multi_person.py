"""Offline multi-person inference policy.

This is the main RJ/S08 merge target: a pure inference-time selector that
chooses temporal evidence before the final one-to-one assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.inference.contracts import InferenceDecision
from src.inference.core import SegmentSummary, assignment_score, segment_rows
from src.modules.matchers.assignment import solve_assignment

METHODS = (
    "global_all_windows",
    "window_hungarian_all",
    "global_best_segment",
    "local_top1_segment",
    "local_topk_segment",
)


def _unique_ordered(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value not in seen:
            output.append(int(value))
            seen.add(int(value))
    return output


def _segment_summaries(segments: list[SegmentSummary]) -> list[dict[str, Any]]:
    return [segment.to_dict() for segment in segments]


def evaluate_scores(
    scores: np.ndarray,
    centers: np.ndarray,
    method: str,
    segment_frames: int,
    min_windows: int,
    top_k: int,
    gt_assignment: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one sequence or one grouped trial."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"scores must be 3D, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("scores must contain at least one window")
    if values.shape[1] != values.shape[2]:
        raise ValueError("offline multi-person inference expects square score matrices")
    centers = np.asarray(centers, dtype=np.int64)
    if len(centers) != len(values):
        raise ValueError("centers must align with scores")

    normalized_method = str(method).strip().lower()
    if normalized_method not in METHODS:
        raise ValueError(f"Unknown offline multi-person method: {method!r}")

    segments = segment_rows(values, centers, segment_frames, min_windows)
    n_people = int(values.shape[1])
    mean_matrix = np.mean(values, axis=0)

    selected_segments: list[int] = []
    row_selected_segments: list[list[int]] = []
    selected_matrix: np.ndarray
    selection_basis: str

    if normalized_method == "global_all_windows":
        selected_matrix = mean_matrix
        final_assignment = solve_assignment(selected_matrix)
        selection_basis = "mean_matrix"
    elif normalized_method == "window_hungarian_all":
        votes = np.zeros((n_people, n_people), dtype=np.float64)
        for matrix in values:
            per_window = solve_assignment(matrix)
            for row, col in enumerate(per_window):
                if col >= 0:
                    votes[row, col] += 1.0
        selected_matrix = votes
        final_assignment = solve_assignment(selected_matrix)
        selection_basis = "vote_matrix"
    elif normalized_method == "global_best_segment":
        chosen = max(segments, key=lambda segment: (segment.global_gap, -segment.segment_id))
        selected_matrix = chosen.matrix
        final_assignment = chosen.best_assignment.copy()
        selected_segments = [int(chosen.segment_id)]
        selection_basis = "segment_global_gap"
    elif normalized_method in {"local_top1_segment", "local_topk_segment"}:
        selected_matrix = np.zeros((n_people, n_people), dtype=np.float64)
        for row in range(n_people):
            ranked = sorted(
                range(len(segments)),
                key=lambda index: (
                    float(segments[index].row_values[row]),
                    -int(segments[index].segment_id),
                ),
                reverse=True,
            )
            chosen_indices = ranked[: 1 if normalized_method == "local_top1_segment" else min(top_k, len(segments))]
            row_selected_segments.append([int(segments[index].segment_id) for index in chosen_indices])
            selected_segments.extend(int(segments[index].segment_id) for index in chosen_indices)
            if normalized_method == "local_top1_segment":
                selected_matrix[row] = segments[chosen_indices[0]].matrix[row]
            else:
                values_for_row = np.asarray(
                    [segments[index].row_values[row] for index in chosen_indices],
                    dtype=np.float64,
                )
                weights = np.exp(values_for_row - np.max(values_for_row))
                weights /= np.maximum(weights.sum(), 1e-12)
                selected_matrix[row] = sum(
                    float(weight) * segments[index].matrix[row]
                    for weight, index in zip(weights, chosen_indices, strict=True)
                )
        final_assignment = solve_assignment(selected_matrix)
        selection_basis = "row_value_topk"
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown method: {method!r}")

    selected_score = assignment_score(selected_matrix, final_assignment)
    result: dict[str, Any] = {
        "mode": "offline",
        "policy": "multi_person",
        "method": normalized_method,
        "selection_basis": selection_basis,
        "n_people": n_people,
        "n_windows": int(len(values)),
        "n_segments": int(len(segments)),
        "selected_segments": _unique_ordered(selected_segments),
        "row_selected_segments": row_selected_segments,
        "selected_score": float(selected_score),
        "assignment": final_assignment.tolist(),
        "segments": _segment_summaries(segments),
    }
    if normalized_method == "global_best_segment":
        chosen = max(segments, key=lambda segment: (segment.global_gap, -segment.segment_id))
        result["selected_gap"] = float(chosen.global_gap)
    else:
        result["selected_gap"] = float("nan")
    if gt_assignment is not None:
        gt = np.asarray(gt_assignment, dtype=np.int64)
        if gt.shape != final_assignment.shape:
            raise ValueError("gt_assignment must match the final assignment shape")
        correct = final_assignment == gt
        result["correct_people"] = int(correct.sum())
        result["person_accuracy"] = float(correct.mean())
        result["exact_assignment"] = bool(correct.all())
    return result


@dataclass(frozen=True)
class MultiPersonOfflinePolicy:
    """S08-style offline selector over multi-person similarity windows."""

    method: str = "global_best_segment"
    segment_frames: int = 50
    min_windows: int = 15
    top_k: int = 2
    mode: str = "offline"
    policy_name: str = "multi_person"

    def evaluate(
        self,
        scores: np.ndarray,
        centers: np.ndarray,
        gt_assignment: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return evaluate_scores(
            scores=scores,
            centers=centers,
            method=self.method,
            segment_frames=int(self.segment_frames),
            min_windows=int(self.min_windows),
            top_k=int(self.top_k),
            gt_assignment=gt_assignment,
        )

    def infer(
        self,
        scores: np.ndarray,
        centers: np.ndarray,
        gt_assignment: np.ndarray | None = None,
    ) -> InferenceDecision:
        result = self.evaluate(scores=scores, centers=centers, gt_assignment=gt_assignment)
        return InferenceDecision(
            mode="offline",
            policy=self.policy_name,
            assignment=np.asarray(result["assignment"], dtype=np.int64),
            selected_segments=tuple(int(value) for value in result["selected_segments"]),
            metadata=result,
        )


__all__ = ["METHODS", "MultiPersonOfflinePolicy", "evaluate_scores"]
