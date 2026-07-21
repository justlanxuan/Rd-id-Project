"""Dataset-aware batch samplers."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class SameWindowBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps all people from the same temporal window together."""

    def __init__(self, rows: list[dict[str, str]], batch_size: int, seed: int, drop_last: bool = True) -> None:
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        groups: dict[tuple[str, str, str], list[int]] = {}
        for idx, row in enumerate(rows):
            key = (
                str(row.get("source_sequence") or row.get("session") or ""),
                str(row.get("source_window_start") or row.get("window_start") or ""),
                str(row.get("window_end") or ""),
            )
            groups.setdefault(key, []).append(idx)
        self.groups = [sorted(v, key=lambda i: str(rows[i].get("subject", ""))) for v in groups.values()]
        if not self.groups:
            raise ValueError("SameWindowBatchSampler received no groups.")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(self.groups))
        batch: list[int] = []
        for group_idx in order:
            group = list(self.groups[int(group_idx)])
            if len(group) > self.batch_size:
                group = group[: self.batch_size]
            if batch and len(batch) + len(group) > self.batch_size:
                if (not self.drop_last) or len(batch) == self.batch_size:
                    yield batch
                batch = []
            batch.extend(group)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(g) for g in self.groups)
        if self.drop_last:
            return total // self.batch_size
        return int(np.ceil(total / max(self.batch_size, 1)))
