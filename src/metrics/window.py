"""Window-level FrameAcc and sampled group matching metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import EmbeddingBundle, EvaluationMetric


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def pair_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return -1.0
    similarity = cosine_matrix(a[:n], b[:n])
    return float(np.mean(np.diag(similarity)))


class FrameAccEvaluator(EvaluationMetric):
    """Compute discriminative assignment accuracy for candidate windows."""

    def __init__(
        self,
        shuffle_match: bool = True,
        seed: int = 42,
        singleton_policy: Literal["error", "exclude", "count_as_correct"] = "error",
    ) -> None:
        if singleton_policy not in {"error", "exclude", "count_as_correct"}:
            raise ValueError(f"Unsupported singleton_policy: {singleton_policy!r}")
        self.shuffle_match = shuffle_match
        self.seed = seed
        self.singleton_policy = singleton_policy

    @staticmethod
    def _group_key(row: dict[str, str]) -> tuple[str, str, int, int]:
        candidate_group_id = str(row.get("candidate_group_id", "")).strip()
        if candidate_group_id:
            return ("candidate_group_id", candidate_group_id, 0, 0)
        source_sequence = str(row.get("source_sequence", "")).strip()
        source_window_start = str(row.get("source_window_start", "")).strip()
        if source_sequence and source_window_start:
            return (
                "source_sequence",
                source_sequence,
                int(source_window_start),
                int(row["window_end"]),
            )
        return (
            str(row.get("npz_path", "")),
            str(row.get("session", "")),
            int(row["window_start"]),
            int(row["window_end"]),
        )

    @staticmethod
    def _sort_key(item: tuple[int, dict[str, str]]) -> tuple[int, int, str]:
        _, row = item
        return (
            int(row.get("candidate_index") or 0),
            int(row.get("imu_idx", 0)),
            f"{int(row.get('person_idx', 0)):08d}:{row.get('subject', '')}",
        )

    def evaluate(self, bundle: EmbeddingBundle) -> dict[str, Any]:
        grouped: dict[tuple[str, str, int, int], list[tuple[int, dict[str, str]]]] = {}
        for index, row in enumerate(bundle.rows):
            grouped.setdefault(self._group_key(row), []).append((index, row))

        rng = np.random.default_rng(self.seed)
        total = 0
        correct = 0
        window_accs: list[float] = []
        singleton = 0
        evaluated = 0
        group_sizes: list[int] = []
        assignments: list[dict[str, Any]] = []

        for key in sorted(grouped):
            items = sorted(grouped[key], key=self._sort_key)
            group_sizes.append(len(items))
            if len(items) < 2:
                singleton += 1
                if self.singleton_policy == "error":
                    raise ValueError(
                        "FrameAcc candidate group has only one item and is non-discriminative: "
                        f"group={key}. Construct multi-candidate groups or explicitly choose "
                        "another singleton policy."
                    )
                if self.singleton_policy == "count_as_correct":
                    correct += 1
                    total += 1
                    window_accs.append(1.0)
                    evaluated += 1
                assignments.append(
                    {
                        "candidate_group": list(key),
                        "status": f"singleton_{self.singleton_policy}",
                        "row_indices": [int(index) for index, _ in items],
                    }
                )
                continue

            indices = np.asarray([index for index, _ in items], dtype=np.int64)
            similarity = cosine_matrix(bundle.imu[indices], bundle.video[indices])
            if self.shuffle_match:
                permutation = rng.permutation(len(indices))
                match_similarity = similarity[permutation]
            else:
                permutation = np.arange(len(indices))
                match_similarity = similarity

            row_indices, column_indices = linear_sum_assignment(-match_similarity)
            n_correct = int(np.sum(permutation[row_indices] == column_indices))
            n_total = int(len(indices))
            correct += n_correct
            total += n_total
            window_accs.append(float(n_correct / max(n_total, 1)))
            evaluated += 1
            assignments.append(
                {
                    "candidate_group": list(key),
                    "status": "evaluated",
                    "row_indices": indices.tolist(),
                    "similarity": similarity.tolist(),
                    "imu_permutation": permutation.tolist(),
                    "hungarian_rows": row_indices.tolist(),
                    "hungarian_columns": column_indices.tolist(),
                    "matched_imu_row_indices": indices[permutation[row_indices]].tolist(),
                    "matched_video_row_indices": indices[column_indices].tolist(),
                    "correct": n_correct,
                    "total": n_total,
                }
            )

        return {
            "method": "frame_acc",
            "prediction_schema_version": "1.0",
            "num_candidate_windows": int(len(grouped)),
            "num_evaluated_windows": int(evaluated),
            "num_singleton_windows": int(singleton),
            "singleton_rate": float(singleton / len(grouped)) if grouped else 0.0,
            "candidate_group_size_min": int(min(group_sizes)) if group_sizes else 0,
            "candidate_group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
            "num_assignments": int(total),
            "correct_assignments": int(correct),
            "frame_acc": float(correct / max(total, 1)),
            "mean_window_acc": float(np.mean(window_accs)) if window_accs else 0.0,
            "std_window_acc": float(np.std(window_accs)) if window_accs else 0.0,
            "shuffle_match": bool(self.shuffle_match),
            "singleton_policy": self.singleton_policy,
            "assignments": assignments,
        }


class GroupTestEvaluator(EvaluationMetric):
    """Compute sampled group matching over sequence chunks."""

    def __init__(
        self,
        group_sizes: list[int],
        num_trials: int = 50,
        chunk_windows: int = 30,
        min_chunk_windows: int = 15,
        seed: int = 42,
        shuffle_match: bool = True,
        per_subject_split: bool = False,
    ) -> None:
        self.group_sizes = group_sizes
        self.num_trials = num_trials
        self.chunk_windows = chunk_windows
        self.min_chunk_windows = min_chunk_windows
        self.seed = seed
        self.shuffle_match = shuffle_match
        self.per_subject_split = per_subject_split

    def _build_units(self, bundle: EmbeddingBundle) -> list[dict[str, Any]]:
        sequence_map: dict[str, dict[str, list[np.ndarray]]] = {}
        for index, row in enumerate(bundle.rows):
            sequence_name = f"{row.get('subject', '')}_{row.get('session', '')}"
            if sequence_name not in sequence_map:
                sequence_map[sequence_name] = {"imu": [], "video": []}
            sequence_map[sequence_name]["imu"].append(bundle.imu[index])
            sequence_map[sequence_name]["video"].append(bundle.video[index])

        units: list[dict[str, Any]] = []
        for sequence_name, sequence_data in sorted(sequence_map.items()):
            imu_embeddings = np.stack(sequence_data["imu"], axis=0)
            video_embeddings = np.stack(sequence_data["video"], axis=0)
            length = min(len(imu_embeddings), len(video_embeddings))
            if length < self.min_chunk_windows:
                continue
            start = 0
            chunk_id = 0
            while start < length:
                end = min(start + self.chunk_windows, length)
                if end - start >= self.min_chunk_windows:
                    units.append(
                        {
                            "unit_id": f"{sequence_name}_c{chunk_id:03d}",
                            "seq_name": sequence_name,
                            "subject": sequence_name.split("_")[0],
                            "imu_emb": imu_embeddings[start:end],
                            "video_emb": video_embeddings[start:end],
                        }
                    )
                    chunk_id += 1
                start += self.chunk_windows
        return units

    def _eval_group(
        self,
        units: list[dict[str, Any]],
        group_size: int,
        seed: int,
    ) -> dict[str, Any]:
        if len(units) < group_size:
            return {
                "group_size": int(group_size),
                "num_units": int(len(units)),
                "num_trials": 0,
                "mean_acc": None,
                "std_acc": None,
                "mean_diag_sim": None,
                "mean_offdiag_sim": None,
                "note": f"insufficient units ({len(units)} < {group_size})",
            }

        rng = np.random.default_rng(seed)
        trial_acc: list[float] = []
        trial_diag: list[float] = []
        trial_offdiag: list[float] = []
        for _ in range(self.num_trials):
            indices = rng.choice(len(units), size=group_size, replace=False)
            selected = [units[index] for index in indices]
            if self.shuffle_match and group_size > 1:
                permutation = rng.permutation(group_size)
                imu_selected = [selected[permutation[index]] for index in range(group_size)]
            else:
                permutation = np.arange(group_size)
                imu_selected = selected

            similarity = np.zeros((group_size, group_size), dtype=np.float32)
            for row_index in range(group_size):
                for column_index in range(group_size):
                    similarity[row_index, column_index] = pair_similarity(
                        imu_selected[row_index]["imu_emb"],
                        selected[column_index]["video_emb"],
                    )
            row_indices, column_indices = linear_sum_assignment(-similarity)
            if self.shuffle_match:
                correct = np.sum(permutation[row_indices] == column_indices)
            else:
                correct = np.sum(row_indices == column_indices)
            trial_acc.append(float(correct) / float(group_size))
            trial_diag.append(float(np.mean(np.diag(similarity))))
            if group_size > 1:
                trial_offdiag.append(
                    float(np.mean(similarity[~np.eye(group_size, dtype=bool)]))
                )

        return {
            "group_size": int(group_size),
            "num_units": int(len(units)),
            "num_trials": int(self.num_trials),
            "mean_acc": float(np.mean(trial_acc)),
            "std_acc": float(np.std(trial_acc)),
            "mean_diag_sim": float(np.mean(trial_diag)),
            "mean_offdiag_sim": float(np.mean(trial_offdiag)) if trial_offdiag else None,
        }

    def evaluate(self, bundle: EmbeddingBundle) -> dict[str, Any]:
        units = self._build_units(bundle)
        if self.per_subject_split:
            by_subject: dict[str, list[dict[str, Any]]] = {}
            for unit in units:
                by_subject.setdefault(str(unit["subject"]), []).append(unit)
            sampled_units: list[dict[str, Any]] = []
            rng = np.random.default_rng(self.seed)
            for subject in sorted(by_subject):
                subject_units = by_subject[subject]
                if subject_units:
                    sampled_units.append(subject_units[int(rng.integers(0, len(subject_units)))])
            eval_units = sampled_units
        else:
            eval_units = units

        results = [
            self._eval_group(eval_units, group_size, self.seed + group_size)
            for group_size in self.group_sizes
        ]
        return {
            "method": "group_test",
            "num_units": int(len(units)),
            "num_eval_units": int(len(eval_units)),
            "chunk_windows": int(self.chunk_windows),
            "min_chunk_windows": int(self.min_chunk_windows),
            "num_trials": int(self.num_trials),
            "shuffle_match": bool(self.shuffle_match),
            "per_subject_split": bool(self.per_subject_split),
            "results": results,
        }
