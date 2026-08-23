"""Training-specific losses and batch retrieval metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval_top1(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels_a: torch.Tensor | None = None,
    labels_b: torch.Tensor | None = None,
) -> float:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    sims = torch.matmul(z_a, z_b.t())

    if labels_a is None:
        labels_a = torch.arange(sims.shape[0], device=sims.device)
    if labels_b is None:
        labels_b = torch.arange(sims.shape[1], device=sims.device)

    acc_ab = (sims.argmax(dim=1) == labels_a).float().mean()
    acc_ba = (sims.argmax(dim=0) == labels_b).float().mean()
    return float((0.5 * (acc_ab + acc_ba)).item())


def weighted_info_nce(
    z_imu: torch.Tensor,
    z_video: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE with an auditable per-window weight."""
    if z_imu.shape != z_video.shape or z_imu.ndim != 2:
        raise ValueError("weighted_info_nce expects matching [batch, embedding] tensors")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(z_imu, dim=-1, eps=1e-6) @ F.normalize(z_video, dim=-1, eps=1e-6).T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    weight = sample_weight.to(logits).flatten().clamp_min(0.0)
    if weight.shape[0] != logits.shape[0]:
        raise ValueError("sample_weight length must match batch")
    weight = weight / weight.mean().clamp_min(1e-6)
    loss = 0.5 * (
        (F.cross_entropy(logits, labels, reduction="none") * weight).mean()
        + (F.cross_entropy(logits.T, labels, reduction="none") * weight).mean()
    )
    accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, accuracy


def turning_alignment_loss(prediction: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
    """Align predicted turning activity with gyro magnitude without labels."""
    gyro_magnitude = torch.linalg.vector_norm(imu[:, :, 3:6], dim=-1).mean(dim=1)
    target = (gyro_magnitude - gyro_magnitude.mean()) / gyro_magnitude.std().clamp_min(1e-4)
    pred = (prediction - prediction.mean()) / prediction.std().clamp_min(1e-4)
    return F.smooth_l1_loss(pred, target)


def turning_onset_loss(output: dict[str, torch.Tensor], orientation: torch.Tensor) -> torch.Tensor:
    """Match skeleton-derived and gyro-specific onset heads."""
    if "gyro_onset_logits" not in output or "orientation_onset_logits" not in output:
        return orientation.new_zeros(())
    bins = output["orientation_onset_logits"].shape[-1]
    target = F.adaptive_avg_pool1d(orientation[:, :, 4].unsqueeze(1), bins).squeeze(1)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(output["orientation_onset_logits"], target)
        + F.binary_cross_entropy_with_logits(output["gyro_onset_logits"], target)
    )


def _loss_temperature(loss_fn) -> torch.Tensor:
    if hasattr(loss_fn, "log_temperature"):
        return torch.exp(loss_fn.log_temperature).clamp(0.02, 0.5)
    return torch.clamp(loss_fn.temperature, min=1e-6)


def subject_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    subject_labels: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    """Cross-modal supervised contrastive loss using subject identity positives."""
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError(f"Expected matching [B,D] embeddings, got {tuple(z_a.shape)} vs {tuple(z_b.shape)}")
    if subject_labels.shape[0] != z_a.shape[0]:
        raise ValueError(f"Subject label length mismatch: {subject_labels.shape[0]} vs {z_a.shape[0]}")

    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = torch.matmul(z_a, z_b.t()) / _loss_temperature(loss_fn)
    pos = subject_labels[:, None].eq(subject_labels[None, :]).float()
    pos_count = pos.sum(dim=1).clamp_min(1.0)

    log_prob_ab = F.log_softmax(logits, dim=1)
    log_prob_ba = F.log_softmax(logits.t(), dim=1)
    loss_ab = -((pos * log_prob_ab).sum(dim=1) / pos_count).mean()
    loss_ba = -((pos.t() * log_prob_ba).sum(dim=1) / pos_count).mean()
    return 0.5 * (loss_ab + loss_ba)


def subject_retrieval_top1(z_a: torch.Tensor, z_b: torch.Tensor, subject_labels: torch.Tensor) -> float:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    sims = torch.matmul(z_a, z_b.t())
    pred_ab = subject_labels[sims.argmax(dim=1)]
    pred_ba = subject_labels[sims.argmax(dim=0)]
    acc_ab = pred_ab.eq(subject_labels).float().mean()
    acc_ba = pred_ba.eq(subject_labels).float().mean()
    return float((0.5 * (acc_ab + acc_ba)).item())


def window_contrastive_loss(
    z_imu: torch.Tensor,
    z_video: torch.Tensor,
    group_labels: torch.Tensor | None,
    loss_fn,
) -> torch.Tensor:
    if group_labels is None:
        return loss_fn(z_imu, z_video)
    z_imu = F.normalize(z_imu, dim=-1)
    z_video = F.normalize(z_video, dim=-1)
    if hasattr(loss_fn, "log_temperature"):
        temperature = torch.exp(loss_fn.log_temperature).clamp(0.02, 0.5)
    else:
        temperature = torch.clamp(loss_fn.temperature, min=1e-6)

    losses = []
    for group in torch.unique(group_labels):
        idx = torch.nonzero(group_labels.eq(group), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        logits = torch.matmul(z_imu.index_select(0, idx), z_video.index_select(0, idx).t()) / temperature
        targets = torch.arange(idx.numel(), device=z_imu.device)
        losses.append(0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)))
    if not losses:
        return loss_fn(z_imu, z_video)
    return torch.stack(losses).mean()


def window_contrastive_top1(
    z_imu: torch.Tensor,
    z_video: torch.Tensor,
    group_labels: torch.Tensor | None,
) -> float:
    if group_labels is None:
        return retrieval_top1(z_imu, z_video)
    z_imu = F.normalize(z_imu, dim=-1)
    z_video = F.normalize(z_video, dim=-1)
    accs = []
    for group in torch.unique(group_labels):
        idx = torch.nonzero(group_labels.eq(group), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        sims = torch.matmul(z_imu.index_select(0, idx), z_video.index_select(0, idx).t())
        targets = torch.arange(idx.numel(), device=z_imu.device)
        accs.append(
            0.5
            * (
                (sims.argmax(dim=1) == targets).float().mean()
                + (sims.argmax(dim=0) == targets).float().mean()
            )
        )
    if not accs:
        return retrieval_top1(z_imu, z_video)
    return float(torch.stack(accs).mean().item())


def pair_bce_loss(
    model,
    z_imu: torch.Tensor,
    z_video: torch.Tensor,
    subject_labels: torch.Tensor | None,
    group_labels: torch.Tensor | None = None,
    target: str = "subject",
) -> torch.Tensor:
    if getattr(model, "pair_head", None) is None:
        return torch.zeros((), device=z_imu.device)
    logits = model.pair_logits(z_imu, z_video)
    if target == "window_ce":
        if group_labels is None:
            return torch.zeros((), device=z_imu.device)
        losses = []
        for group in torch.unique(group_labels):
            idx = torch.nonzero(group_labels.eq(group), as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            group_logits = logits.index_select(0, idx).index_select(1, idx)
            targets = torch.arange(idx.numel(), device=logits.device)
            losses.append(
                0.5
                * (
                    F.cross_entropy(group_logits, targets)
                    + F.cross_entropy(group_logits.t(), targets)
                )
            )
        if not losses:
            return torch.zeros((), device=z_imu.device)
        return torch.stack(losses).mean()
    if target == "subject":
        if subject_labels is None:
            return torch.zeros((), device=z_imu.device)
        labels = subject_labels[:, None].eq(subject_labels[None, :]).float()
    elif target == "pair":
        labels = torch.eye(logits.shape[0], logits.shape[1], device=logits.device)
    elif target == "subject_pair":
        if subject_labels is None:
            return torch.zeros((), device=z_imu.device)
        subject_targets = subject_labels[:, None].eq(subject_labels[None, :]).float()
        pair_targets = torch.eye(logits.shape[0], logits.shape[1], device=logits.device)
        subject_loss = F.binary_cross_entropy_with_logits(logits, subject_targets)
        pair_loss = F.binary_cross_entropy_with_logits(logits, pair_targets)
        return 0.5 * (subject_loss + pair_loss)
    elif target == "window":
        if group_labels is None:
            return torch.zeros((), device=z_imu.device)
        mask = group_labels[:, None].eq(group_labels[None, :])
        if not bool(mask.any()):
            return torch.zeros((), device=z_imu.device)
        labels = torch.eye(logits.shape[0], logits.shape[1], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])
    else:
        raise ValueError(f"Unsupported pair loss target: {target}")
    return F.binary_cross_entropy_with_logits(logits, labels)


def cross_pair_window_ce_loss(
    model,
    imu: torch.Tensor,
    skeleton: torch.Tensor,
    group_labels: torch.Tensor | None,
) -> torch.Tensor:
    if getattr(model, "cross_pair_head", None) is None:
        return torch.zeros((), device=imu.device)
    if group_labels is None:
        return torch.zeros((), device=imu.device)
    losses = []
    for group in torch.unique(group_labels):
        idx = torch.nonzero(group_labels.eq(group), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        logits = model.cross_pair_logits(imu.index_select(0, idx), skeleton.index_select(0, idx))
        targets = torch.arange(idx.numel(), device=imu.device)
        losses.append(0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)))
    if not losses:
        return torch.zeros((), device=imu.device)
    return torch.stack(losses).mean()


def pair_anti_tie_loss(
    logits: torch.Tensor,
    group_labels: torch.Tensor | None,
    margin: float = 0.05,
) -> torch.Tensor:
    if group_labels is None:
        return torch.zeros((), device=logits.device)
    losses = []
    for group in torch.unique(group_labels):
        idx = torch.nonzero(group_labels.eq(group), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        sub = logits.index_select(0, idx).index_select(1, idx)
        if sub.shape[0] == 2 and sub.shape[1] == 2:
            diag = sub[0, 0] + sub[1, 1]
            off = sub[0, 1] + sub[1, 0]
            losses.append(F.relu(float(margin) - torch.abs(diag - off)))
        row_gap = torch.pdist(sub, p=2).mean() if sub.shape[0] > 1 else sub.new_tensor(0.0)
        col_gap = torch.pdist(sub.t(), p=2).mean() if sub.shape[1] > 1 else sub.new_tensor(0.0)
        losses.append(F.relu(float(margin) - row_gap))
        losses.append(F.relu(float(margin) - col_gap))
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()


def pair_anti_tie_loss_from_model(
    model,
    z_imu: torch.Tensor,
    z_video: torch.Tensor,
    group_labels: torch.Tensor | None,
) -> torch.Tensor:
    if getattr(model, "pair_head", None) is None:
        return torch.zeros((), device=z_imu.device)
    return pair_anti_tie_loss(model.pair_logits(z_imu, z_video), group_labels)
