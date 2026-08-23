"""Dataset-aware batch samplers."""

from __future__ import annotations

import numpy as np
from torch.utils.data import BatchSampler, Sampler


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


class DomainBalancedGroupBatchSampler(BatchSampler):
    """Sample one row from temporal groups while balancing dataset domains."""

    def __init__(self, dataset, batch_size: int, seed: int, steps: int | None = None) -> None:
        if int(batch_size) < 2:
            raise ValueError("batch_size must be >=2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        grouped = dataset.group_indices()
        by_domain: dict[str, list[list[int]]] = {}
        for (domain, _), indices in grouped.items():
            by_domain.setdefault(str(domain), []).append(list(indices))
        self.by_domain = {name: groups for name, groups in by_domain.items() if groups}
        if not self.by_domain:
            raise ValueError("No temporal groups available")
        self.domains = sorted(self.by_domain)
        default_steps = min(len(groups) for groups in self.by_domain.values())
        self.steps = int(steps) if steps and int(steps) > 0 else max(1, default_steps * len(self.domains) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pools = {domain: rng.permutation(len(groups)).tolist() for domain, groups in self.by_domain.items()}
        cursors = {domain: 0 for domain in self.domains}
        for _ in range(self.steps):
            batch: list[int] = []
            for offset in range(self.batch_size):
                domain = self.domains[offset % len(self.domains)]
                if cursors[domain] >= len(pools[domain]):
                    pools[domain] = rng.permutation(len(self.by_domain[domain])).tolist()
                    cursors[domain] = 0
                group = self.by_domain[domain][pools[domain][cursors[domain]]]
                cursors[domain] += 1
                batch.append(int(group[int(rng.integers(0, len(group)))]))
            yield batch

    def __len__(self) -> int:
        return self.steps


class OrientationHardNegativeBatchSampler(BatchSampler):
    """Prefer same-action windows with nearby turning activity as negatives."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int,
        steps: int,
        pool_multiplier: int = 4,
        hard_fraction: float = 1.0,
    ) -> None:
        if int(batch_size) < 2 or int(pool_multiplier) < 1 or not 0.0 < float(hard_fraction) <= 1.0:
            raise ValueError("batch_size >=2, pool_multiplier >=1 and hard_fraction in (0,1] are required")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.steps = int(steps)
        self.pool_multiplier = int(pool_multiplier)
        self.hard_fraction = float(hard_fraction)
        buckets: dict[tuple[str, str], list[tuple[list[int], float]]] = {}
        for (domain, _), indices in dataset.group_indices().items():
            row = dataset.rows[indices[0]]
            action = str(row.get("session") or row.get("source_sequence") or "unknown")
            activity = float(np.mean([dataset[index]["orientation"][:, 4].mean().item() for index in indices]))
            buckets.setdefault((str(domain), action), []).append((list(indices), activity))
        self.buckets = {key: value for key, value in buckets.items() if len(value) >= 2}
        self.by_domain: dict[str, list[tuple[list[int], float]]] = {}
        for (domain, _), records in self.buckets.items():
            self.by_domain.setdefault(domain, []).extend(records)
        if not self.buckets:
            raise ValueError("No action bucket contains at least two hard-negative groups")
        self.bucket_keys = sorted(self.buckets)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for step in range(self.steps):
            key = self.bucket_keys[step % len(self.bucket_keys)]
            records = self.buckets[key]
            anchor = records[int(rng.integers(0, len(records)))]
            ranked = sorted(records, key=lambda item: abs(item[1] - anchor[1]))
            hard_count = max(2, min(self.batch_size, round(self.batch_size * self.hard_fraction)))
            pool_size = min(len(ranked), max(hard_count, hard_count * self.pool_multiplier))
            pool = ranked[:pool_size]
            positions = rng.choice(len(pool), size=hard_count, replace=len(pool) < hard_count)
            selected = [pool[int(position)] for position in positions]
            selected_first = {record[0][0] for record in selected}
            random_pool = [record for record in self.by_domain[key[0]] if record[0][0] not in selected_first]
            random_count = self.batch_size - hard_count
            if random_count:
                if not random_pool:
                    random_pool = selected
                positions = rng.choice(len(random_pool), size=random_count, replace=len(random_pool) < random_count)
                selected.extend(random_pool[int(position)] for position in positions)
            rng.shuffle(selected)
            yield [int(record[0][int(rng.integers(0, len(record[0])))]) for record in selected]

    def __len__(self) -> int:
        return self.steps
