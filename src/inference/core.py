"""Shared inference primitives used by offline and realtime policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.modules.matchers.assignment import solve_assignment


def make_windows(
    values: np.ndarray,
    window_frames: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a temporal array into overlapping windows."""
    if window_frames <= 0 or stride <= 0:
        raise ValueError("window_frames and stride must be positive")
    data = np.asarray(values)
    if len(data) < window_frames:
        return np.empty((0, window_frames, *data.shape[1:]), dtype=data.dtype), np.empty(0, dtype=np.int64)
    starts = np.arange(0, len(data) - window_frames + 1, stride, dtype=np.int64)
    windows = np.stack([data[start : start + window_frames] for start in starts], axis=0)
    return windows, starts


def assignment_score(similarity: np.ndarray, assignment: np.ndarray) -> float:
    """Sum the selected scores in an assignment vector."""
    scores = np.asarray(similarity, dtype=np.float64)
    chosen = np.asarray(assignment, dtype=np.int64)
    if scores.ndim != 2:
        raise ValueError(f"similarity must be 2D, got shape {scores.shape}")
    if chosen.ndim != 1 or chosen.shape[0] != scores.shape[0]:
        raise ValueError("assignment must be one row index per similarity row")
    rows = np.flatnonzero(chosen >= 0)
    if len(rows) == 0:
        return 0.0
    return float(scores[rows, chosen[rows]].sum())


def second_assignment_score(similarity: np.ndarray, best: np.ndarray) -> float:
    """Compute the exact second-best square assignment by edge exclusion."""
    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or scores.shape[0] <= 1:
        return float("nan")
    values: list[float] = []
    for row, col in enumerate(np.asarray(best, dtype=np.int64)):
        if col < 0:
            continue
        blocked = scores.copy()
        blocked[row, int(col)] = -1e12
        rows, columns = linear_sum_assignment(-blocked)
        if len(rows) == scores.shape[0] and np.all(blocked[rows, columns] > -1e11):
            values.append(float(scores[rows, columns].sum()))
    return max(values) if values else float("nan")


def segment_positions(
    centers: np.ndarray,
    segment_frames: int,
    min_windows: int,
) -> list[tuple[int, np.ndarray]]:
    """Group windows by coarse segment index."""
    if segment_frames <= 0 or min_windows <= 0:
        raise ValueError("segment_frames and min_windows must be positive")
    groups: dict[int, list[int]] = {}
    for index, center in enumerate(np.asarray(centers, dtype=np.int64)):
        groups.setdefault(int(center) // int(segment_frames), []).append(int(index))
    return [
        (segment_id, np.asarray(indices, dtype=np.int64))
        for segment_id, indices in sorted(groups.items())
        if len(indices) >= min_windows
    ]


def robust_value(margins: np.ndarray) -> float:
    """Median minus median absolute deviation."""
    values = np.asarray(margins, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median - mad


@dataclass(frozen=True)
class SegmentSummary:
    segment_id: int
    positions: np.ndarray
    matrix: np.ndarray
    best_assignment: np.ndarray
    best_score: float
    second_score: float
    global_gap: float
    row_values: np.ndarray
    row_winners: np.ndarray
    row_consistency: np.ndarray

    def to_dict(self, *, include_matrix: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "segment_id": int(self.segment_id),
            "positions": np.asarray(self.positions, dtype=np.int64).tolist(),
            "best_assignment": np.asarray(self.best_assignment, dtype=np.int64).tolist(),
            "best_score": float(self.best_score),
            "second_score": float(self.second_score),
            "global_gap": float(self.global_gap),
            "row_values": np.asarray(self.row_values, dtype=np.float64).tolist(),
            "row_winners": np.asarray(self.row_winners, dtype=np.int64).tolist(),
            "row_consistency": np.asarray(self.row_consistency, dtype=np.float64).tolist(),
        }
        if include_matrix:
            result["matrix"] = np.asarray(self.matrix, dtype=np.float64).tolist()
        return result


def segment_rows(
    scores: np.ndarray,
    centers: np.ndarray,
    segment_frames: int,
    min_windows: int,
) -> list[SegmentSummary]:
    """Summarize per-segment evidence for multi-person matching."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"scores must be 3D, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("scores must contain at least one window")
    n_people = int(values.shape[1])
    segments = segment_positions(centers, segment_frames, min_windows)
    if not segments:
        positions = np.arange(len(values), dtype=np.int64)
        matrix = np.mean(values, axis=0)
        best_assignment = solve_assignment(matrix)
        best_score = assignment_score(matrix, best_assignment)
        second_score = second_assignment_score(matrix, best_assignment)
        row_winners = np.argmax(matrix, axis=1).astype(np.int64)
        return [
            SegmentSummary(
                segment_id=-1,
                positions=positions,
                matrix=matrix,
                best_assignment=best_assignment,
                best_score=best_score,
                second_score=second_score,
                global_gap=(
                    (best_score - second_score) / matrix.shape[0]
                    if np.isfinite(second_score)
                    else float("nan")
                ),
                row_values=np.zeros(n_people, dtype=np.float64),
                row_winners=row_winners,
                row_consistency=np.ones(n_people, dtype=np.float64),
            )
        ]

    summaries: list[SegmentSummary] = []
    for segment_id, positions in segments:
        matrix = np.mean(values[positions], axis=0)
        best_assignment = solve_assignment(matrix)
        best_score = assignment_score(matrix, best_assignment)
        second_score = second_assignment_score(matrix, best_assignment)
        row_values: list[float] = []
        row_winners: list[int] = []
        row_consistency: list[float] = []
        for row in range(matrix.shape[0]):
            winner = int(np.argmax(matrix[row]))
            row_winners.append(winner)
            if matrix.shape[1] <= 1:
                row_values.append(float("inf"))
                row_consistency.append(1.0)
                continue
            margins = values[positions, row, winner] - np.max(
                np.delete(values[positions, row], winner, axis=1),
                axis=1,
            )
            row_values.append(robust_value(margins))
            row_consistency.append(float(np.mean(margins > 0.0)))
        summaries.append(
            SegmentSummary(
                segment_id=int(segment_id),
                positions=positions,
                matrix=matrix,
                best_assignment=best_assignment,
                best_score=best_score,
                second_score=second_score,
                global_gap=(
                    (best_score - second_score) / matrix.shape[0]
                    if np.isfinite(second_score)
                    else float("nan")
                ),
                row_values=np.asarray(row_values, dtype=np.float64),
                row_winners=np.asarray(row_winners, dtype=np.int64),
                row_consistency=np.asarray(row_consistency, dtype=np.float64),
            )
        )
    return summaries


__all__ = [
    "SegmentSummary",
    "assignment_score",
    "make_windows",
    "robust_value",
    "segment_positions",
    "segment_rows",
    "second_assignment_score",
]
