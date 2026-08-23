"""Train/evaluate the G10 global-only sequence matcher.

Examples use repeated ``--train-spec``/``--eval-spec`` arguments of the form:
``dataset=totalcapture;csv=/...;root=/...;fps_hz=60``.  The script writes a
checkpoint, metrics and protocol JSON under one run directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from src.data.specs import load_specs, parse_spec
from src.datasets.samplers import DomainBalancedGroupBatchSampler
from src.g10.global_encoder import GlobalMotionDataset, GlobalMotionMatcher, evaluate_global_matcher


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.weight * gradient, None


def _gradient_reverse(values: torch.Tensor, weight: float) -> torch.Tensor:
    return _GradientReverse.apply(values, weight)


def _coral_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape[0] < 2 or second.shape[0] < 2:
        return first.new_zeros(())
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    covariance_first = first.T @ first / max(first.shape[0] - 1, 1)
    covariance_second = second.T @ second / max(second.shape[0] - 1, 1)
    return (covariance_first - covariance_second).pow(2).mean()


def _mmd_rbf_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Unbiased-ish RBF MMD on paired-domain embeddings.

    The median pairwise distance is detached so the bandwidth does not create
    an additional gradient path.  The diagonal is retained for numerical
    stability; this is a small screening regularizer, not a calibrated domain
    discrepancy estimate.
    """
    if first.shape[0] < 2 or second.shape[0] < 2:
        return first.new_zeros(())

    def kernel(left: torch.Tensor, right: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(left, right).pow(2)
        return torch.exp(-distances / torch.clamp(2.0 * bandwidth.pow(2), min=1e-6))

    cross = torch.cdist(first, second).detach()
    bandwidth = torch.median(cross).clamp_min(1e-3)
    within_first = kernel(first, first, bandwidth).mean()
    within_second = kernel(second, second, bandwidth).mean()
    between = kernel(first, second, bandwidth).mean()
    return (within_first + within_second - 2.0 * between).clamp_min(0.0)


_parse_spec = parse_spec


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "skeleton": torch.stack([item["skeleton"] for item in batch]),
        "imu": torch.stack([item["imu"] for item in batch]),
        "domain": [item["domain"] for item in batch],
        "group_key": [item["group_key"] for item in batch],
        "identity": [item["identity"] for item in batch],
    }


def _info_nce(out: dict[str, torch.Tensor], temperature: float) -> tuple[torch.Tensor, float]:
    logits = out["imu"] @ out["skeleton"].T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    acc = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, acc


def _load_specs(values: list[str]) -> list[dict[str, Any]]:
    return load_specs(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-spec", action="append", required=True)
    parser.add_argument("--eval-spec", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-id", default="A2_left_shoulder")
    parser.add_argument("--skeleton-feature", default="speed")
    parser.add_argument("--imu-view", default="I1_acc_magnitude")
    parser.add_argument("--imu-feature", default="magnitude")
    parser.add_argument("--target-len", type=int, default=24)
    parser.add_argument("--normalize", choices=("zscore", "none"), default="zscore")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temporal-mode", choices=("gru", "mean", "attn", "transformer", "tcn", "multiscale"), default="gru")
    parser.add_argument("--multiscale-fusion", choices=("mean", "gated", "hierarchical_attention"), default="hierarchical_attention")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-domain", default="custom23", help="Primary checkpoint-selection domain; defaults to target-development session 23.")
    parser.add_argument("--domain-method", choices=("erm", "coral", "mmd", "dann", "groupdro"), default="erm")
    parser.add_argument("--domain-weight", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.target_len <= 1 or args.temperature <= 0:
        parser.error("epochs, target-len and temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_specs = _load_specs(args.train_spec)
    eval_specs = _load_specs(args.eval_spec)
    feature_kwargs = {
        "anchor_id": args.anchor_id,
        "skeleton_feature": args.skeleton_feature,
        "imu_view": args.imu_view,
        "imu_feature": args.imu_feature,
        "target_len": args.target_len,
        "normalize": args.normalize,
    }
    train_ds = GlobalMotionDataset(train_specs, **feature_kwargs)
    eval_ds = GlobalMotionDataset(eval_specs, **feature_kwargs)
    sample = train_ds[0]
    model = GlobalMotionMatcher(
        skeleton_dim=int(sample["skeleton"].shape[-1]),
        imu_dim=int(sample["imu"].shape[-1]),
        hidden=args.hidden,
        embedding_dim=args.embedding_dim,
        temporal_mode=args.temporal_mode,
        multiscale_fusion=args.multiscale_fusion,
        window_seconds=args.window_seconds,
    ).to(device)
    domain_names = train_ds.domains
    domain_map = {name: index for index, name in enumerate(domain_names)}
    domain_classifier = nn.Linear(args.embedding_dim, len(domain_names)).to(device) if args.domain_method == "dann" and len(domain_names) > 1 else None
    sampler: Sampler[list[int]] = DomainBalancedGroupBatchSampler(train_ds, args.batch_size, args.seed, args.steps_per_epoch)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=_collate, num_workers=args.num_workers)
    optimizer_parameters = list(model.parameters()) + (list(domain_classifier.parameters()) if domain_classifier is not None else [])
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        train_accs: list[float] = []
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            imu = batch["imu"].to(device)
            result = model(skeleton, imu)
            loss, acc = _info_nce(result, args.temperature)
            domain_loss = loss.new_zeros(())
            if args.domain_method == "groupdro" and len(domain_names) > 1:
                labels = torch.tensor([domain_map[value] for value in batch["domain"]], device=device)
                domain_losses = []
                for domain_index in range(len(domain_names)):
                    mask = labels == domain_index
                    if int(mask.sum()) >= 2:
                        domain_losses.append(_info_nce({"skeleton": result["skeleton"][mask], "imu": result["imu"][mask]}, args.temperature)[0])
                if domain_losses:
                    loss = torch.stack(domain_losses).max()
            elif args.domain_method == "coral" and len(domain_names) > 1:
                labels = torch.tensor([domain_map[value] for value in batch["domain"]], device=device)
                domain_embeddings = 0.5 * (result["skeleton"] + result["imu"])
                first = domain_embeddings[labels == 0]
                second = domain_embeddings[labels == 1]
                domain_loss = _coral_loss(first, second)
            elif args.domain_method == "mmd" and len(domain_names) > 1:
                labels = torch.tensor([domain_map[value] for value in batch["domain"]], device=device)
                domain_embeddings = 0.5 * (result["skeleton"] + result["imu"])
                first = domain_embeddings[labels == 0]
                second = domain_embeddings[labels == 1]
                domain_loss = _mmd_rbf_loss(first, second)
            elif args.domain_method == "dann" and domain_classifier is not None:
                labels = torch.tensor([domain_map[value] for value in batch["domain"]], device=device)
                domain_embeddings = 0.5 * (result["skeleton"] + result["imu"])
                domain_logits = domain_classifier(_gradient_reverse(domain_embeddings, args.domain_weight))
                domain_loss = F.cross_entropy(domain_logits, labels)
            loss = loss + args.domain_weight * domain_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite G10 global loss at epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            train_accs.append(acc)
        eval_metrics = evaluate_global_matcher(model, eval_ds, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_retrieval_top1": float(np.mean(train_accs)) if train_accs else None,
            "domain_method": args.domain_method,
            "domain_loss": float(domain_loss.item()) if "domain_loss" in locals() else 0.0,
            "eval": eval_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        payload = {
            "schema_version": "g10.global_encoder_checkpoint.v1",
            "config": vars(args),
            "feature_kwargs": feature_kwargs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "domain_classifier": domain_classifier.state_dict() if domain_classifier is not None else None,
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(payload, output / "last.pt")
        selected_metrics = eval_metrics.get("per_domain", {}).get(args.selection_domain, {})
        selected_acc = selected_metrics.get("frame_acc") if selected_metrics else eval_metrics["frame_acc"]
        if (selected_acc or -1.0) > best_acc:
            best_acc = selected_acc or -1.0
            torch.save(payload, output / "best.pt")
    summary = {
        "schema_version": "g10.global_encoder_run.v1",
        "config": vars(args),
        "feature_kwargs": feature_kwargs,
        "train_rows": len(train_ds),
        "train_domains": train_ds.domains,
        "eval_rows": len(eval_ds),
        "history": history,
        "best_eval_frame_acc": best_acc if best_acc >= 0 else None,
        "selection_domain": args.selection_domain,
        "device": str(device),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
