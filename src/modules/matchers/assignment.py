"""Small assignment solvers shared by inference-time matchers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_assignment(similarity: np.ndarray) -> np.ndarray:
    """Assign confident rows first, returning row-to-column indices."""
    scores = np.asarray(similarity, dtype=np.float64)
    assignment = np.full(scores.shape[0], -1, dtype=np.int64)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for row_index in range(scores.shape[0]):
        order = np.argsort(-scores[row_index], kind="stable")
        if len(order) == 0:
            continue
        best = float(scores[row_index, order[0]])
        second = float(scores[row_index, order[1]]) if len(order) > 1 else 0.0
        candidates.append((best - second, row_index, order))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    claimed: set[int] = set()
    for _, row_index, order in candidates:
        for candidate_column in order:
            column = int(candidate_column)
            if column not in claimed:
                assignment[row_index] = column
                claimed.add(column)
                break
    return assignment


def solve_assignment(similarity: np.ndarray, method: str = "hungarian") -> np.ndarray:
    """Solve a rectangular matrix and return row-to-column indices."""
    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"similarity must be 2D, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("similarity must contain only finite values")
    normalized_method = str(method).strip().lower()
    if normalized_method == "greedy":
        return greedy_assignment(scores)
    if normalized_method != "hungarian":
        raise ValueError("assignment method must be 'hungarian' or 'greedy'")
    assignment = np.full(scores.shape[0], -1, dtype=np.int64)
    if scores.shape[0] and scores.shape[1]:
        rows, columns = linear_sum_assignment(-scores)
        assignment[rows] = columns
    return assignment


__all__ = ["greedy_assignment", "solve_assignment"]
