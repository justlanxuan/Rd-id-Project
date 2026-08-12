"""Chunk-based matching utilities ported from MotionBERT grouping logic.

This module keeps chunk-window evaluation outside the base Hungarian matcher
so the standard matcher remains a pure assignment component.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalized(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def run_chunk_trials(
    imu_embs: Sequence[np.ndarray],
    vid_embs: Sequence[np.ndarray],
    chunk_windows: int,
    min_chunk_windows: int,
    num_trials: int,
    seed: int,
) -> Dict[str, object]:
    if len(imu_embs) != len(vid_embs):
        raise ValueError(f"imu_embs and vid_embs must have same length, got {len(imu_embs)} vs {len(vid_embs)}")
    if len(imu_embs) == 0:
        return {
            "num_windows": 0,
            "num_trials": 0,
            "chunk_windows": int(chunk_windows),
            "pair_prob": [],
            "mean_sim": [],
            "status": "empty inputs",
        }
    if len(imu_embs) != 2:
        return {
            "num_windows": int(min(min(len(x) for x in imu_embs), min(len(x) for x in vid_embs))),
            "num_trials": 0,
            "chunk_windows": int(chunk_windows),
            "pair_prob": None,
            "mean_sim": None,
            "status": f"chunk trials currently implemented for 2-way matching, got {len(imu_embs)} streams",
        }

    n = min(imu_embs[0].shape[0], imu_embs[1].shape[0], vid_embs[0].shape[0], vid_embs[1].shape[0])
    if n < min_chunk_windows:
        return {
            "num_windows": int(n),
            "num_trials": 0,
            "chunk_windows": int(min(chunk_windows, n)),
            "pair_prob": [[None, None], [None, None]],
            "mean_sim": [[None, None], [None, None]],
            "status": f"insufficient windows ({n} < {min_chunk_windows})",
        }

    chunk = min(chunk_windows, n)
    rng = np.random.default_rng(seed)

    imu_n = [normalized(imu_embs[0][:n]), normalized(imu_embs[1][:n])]
    vid_n = [normalized(vid_embs[0][:n]), normalized(vid_embs[1][:n])]

    counts = np.zeros((2, 2), dtype=np.int64)
    mean_sim = np.zeros((2, 2), dtype=np.float64)

    for _ in range(num_trials):
        if n == chunk:
            s = 0
        else:
            s = int(rng.integers(0, n - chunk + 1))
        e = s + chunk

        sim = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                sim[i, j] = float(np.mean(np.sum(imu_n[i][s:e] * vid_n[j][s:e], axis=1)))

        mean_sim += sim
        r, c = linear_sum_assignment(-sim)
        for rr, cc in zip(r, c, strict=True):
            counts[int(rr), int(cc)] += 1

    pair_prob = (counts / float(max(num_trials, 1))).tolist()
    mean_sim = (mean_sim / float(max(num_trials, 1))).tolist()
    return {
        "num_windows": int(n),
        "num_trials": int(num_trials),
        "chunk_windows": int(chunk),
        "pair_prob": pair_prob,
        "mean_sim": mean_sim,
        "status": "ok",
    }
