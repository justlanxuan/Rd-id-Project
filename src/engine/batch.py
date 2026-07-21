"""Batch metadata helpers for training and validation."""

from __future__ import annotations

from typing import Dict

import torch

from src.datasets import WindowAlignmentDataset


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {
        "imu": batch["imu"].to(device),
        "skeleton": batch["skeleton"].to(device),
    }
    if "root_trajectory" in batch:
        result["root_trajectory"] = batch["root_trajectory"].to(device)
    if "domain" in batch:
        result["domain"] = batch["domain"]
    return result


def parse_domain_label_map(spec: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid domain_label_map item: {item}")
        k, v = item.split(":", 1)
        mapping[k.strip().lower()] = int(v.strip())
    return mapping


def domain_labels_from_batch(batch_domains, domain_map: Dict[str, int], device: torch.device):
    if batch_domains is None:
        return None

    if isinstance(batch_domains, (list, tuple)):
        domains = [str(x).strip().lower() for x in batch_domains]
    else:
        domains = [str(batch_domains).strip().lower()]

    labels = [domain_map.get(d, 0) for d in domains]
    return torch.tensor(labels, dtype=torch.long, device=device)


def build_subject_label_map(dataset: WindowAlignmentDataset) -> Dict[str, int]:
    """Build a deterministic subject-id map from the training split."""
    subjects = sorted({str(row.get("subject", "")) for row in dataset.rows if str(row.get("subject", ""))})
    return {subject: i for i, subject in enumerate(subjects)}


def subject_labels_from_batch(batch_subjects, subject_map: Dict[str, int], device: torch.device):
    if not subject_map or batch_subjects is None:
        return None
    if isinstance(batch_subjects, (list, tuple)):
        subjects = [str(x) for x in batch_subjects]
    else:
        subjects = [str(batch_subjects)]
    if any(s not in subject_map for s in subjects):
        return None
    return torch.tensor([subject_map[s] for s in subjects], dtype=torch.long, device=device)


def group_labels_from_batch(batch_groups, device: torch.device):
    if batch_groups is None:
        return None
    if isinstance(batch_groups, (list, tuple)):
        groups = [str(x) for x in batch_groups]
    else:
        groups = [str(batch_groups)]
    group_map = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    return torch.tensor([group_map[g] for g in groups], dtype=torch.long, device=device)
