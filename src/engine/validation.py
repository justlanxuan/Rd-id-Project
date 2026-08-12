"""Validation helpers used by the training engine."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.engine.batch import (
    domain_labels_from_batch,
    group_labels_from_batch,
    move_to_device,
    subject_labels_from_batch,
)
from src.engine.losses import (
    cross_pair_window_ce_loss,
    pair_anti_tie_loss_from_model,
    pair_bce_loss,
    retrieval_top1,
    subject_contrastive_loss,
    subject_retrieval_top1,
    window_contrastive_loss,
    window_contrastive_top1,
)


def evaluate_epoch(
    model,
    data_loader,
    loss_fn,
    device,
    domain_loss_weight: float = 0.0,
    domain_map: Dict[str, int] | None = None,
    subject_map: Dict[str, int] | None = None,
    contrastive_target: str = "pair",
    pair_loss_weight: float = 0.0,
    pair_loss_target: str = "subject",
    pair_anti_tie_weight: float = 0.0,
) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": 0.0, "top1": 0.0, "main_loss": 0.0, "domain_loss": 0.0}

    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_domain_loss = 0.0
    total_pair_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    use_domain = domain_loss_weight > 0 and getattr(model, "domain_classifier", None) is not None

    with torch.no_grad():
        for batch in data_loader:
            b = move_to_device(batch, device)
            forward_kwargs = {"imu": b["imu"], "skeleton": b["skeleton"]}
            if "root_trajectory" in b:
                forward_kwargs["root_trajectory"] = b["root_trajectory"]
            out = model(**forward_kwargs)
            subject_labels = subject_labels_from_batch(batch.get("subject"), subject_map or {}, device)
            group_labels = group_labels_from_batch(batch.get("group_key"), device)
            if contrastive_target == "subject" and subject_labels is not None:
                main_loss = subject_contrastive_loss(out["imu"], out["video"], subject_labels, loss_fn)
                acc = subject_retrieval_top1(out["imu"], out["video"], subject_labels)
            elif contrastive_target == "pair":
                main_loss = loss_fn(out["imu"], out["video"])
                acc = retrieval_top1(out["imu"], out["video"])
            elif contrastive_target == "window":
                main_loss = window_contrastive_loss(out["imu"], out["video"], group_labels, loss_fn)
                acc = window_contrastive_top1(out["imu"], out["video"], group_labels)
            else:
                raise ValueError(f"Unsupported contrastive_target={contrastive_target!r}")

            domain_loss = torch.zeros((), device=main_loss.device)
            if use_domain:
                domain_labels = domain_labels_from_batch(b.get("domain"), domain_map or {}, device)
                if domain_labels is not None and domain_labels.shape[0] == out["imu"].shape[0]:
                    domain_logits = model.domain_classifier(out["imu"])
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)

            pair_loss = torch.zeros((), device=main_loss.device)
            if pair_loss_weight > 0:
                if getattr(model, "cross_pair_head", None) is not None:
                    pair_loss = cross_pair_window_ce_loss(model, b["imu"], b["skeleton"], group_labels)
                else:
                    pair_loss = pair_bce_loss(
                        model,
                        out["imu"],
                        out["video"],
                        subject_labels,
                        group_labels,
                        pair_loss_target,
                    )
            anti_tie_loss = torch.zeros((), device=main_loss.device)
            if pair_anti_tie_weight > 0:
                anti_tie_loss = pair_anti_tie_loss_from_model(model, out["imu"], out["video"], group_labels)

            loss = (
                main_loss
                + domain_loss_weight * domain_loss
                + pair_loss_weight * pair_loss
                + pair_anti_tie_weight * anti_tie_loss
            )
            total_loss += float(loss.item())
            total_main_loss += float(main_loss.item())
            total_domain_loss += float(domain_loss.item())
            total_pair_loss += float(pair_loss.item())
            total_acc += acc
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "top1": 0.0}
    return {
        "loss": total_loss / total_batches,
        "main_loss": total_main_loss / total_batches,
        "domain_loss": total_domain_loss / total_batches,
        "pair_loss": total_pair_loss / total_batches,
        "top1": total_acc / total_batches,
    }
