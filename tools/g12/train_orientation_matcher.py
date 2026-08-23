"""Train the controlled G12 orientation-aware matcher ablation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.specs import load_specs
from src.datasets.samplers import DomainBalancedGroupBatchSampler, OrientationHardNegativeBatchSampler
from src.engine.losses import turning_alignment_loss, turning_onset_loss, weighted_info_nce
from src.g12.orientation_matcher import OrientationAwareMatcher
from src.g12.orientation_motion import ORIENTATION_DIM, ORIENTATION_SCHEMA, OrientationMotionDataset


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "skeleton": torch.stack([item["skeleton"] for item in batch]),
        "imu": torch.stack([item["imu"] for item in batch]),
        "orientation": torch.stack([item["orientation"] for item in batch]),
        "domain": [item["domain"] for item in batch],
        "group_key": [item["group_key"] for item in batch],
        "identity": [item["identity"] for item in batch],
    }


def _turning_alignment_loss(prediction: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
    """Train the orientation branch to predict window-level gyro activity."""
    return turning_alignment_loss(prediction, imu)


def _turn_onset_loss(out: dict[str, torch.Tensor], orientation: torch.Tensor) -> torch.Tensor:
    """Predict the skeleton-derived turn-onset timeline from each isolated tower."""
    return turning_onset_loss(out, orientation)


def _weighted_info_nce(
    out: dict[str, torch.Tensor], temperature: float, sample_weight: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE with auditable per-window turning weights."""
    return weighted_info_nce(out["imu"], out["skeleton"], sample_weight, temperature)


@torch.no_grad()
def evaluate(
    model: OrientationAwareMatcher,
    dataset: OrientationMotionDataset,
    device: torch.device,
    turning_threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    skeleton_embeddings: dict[int, np.ndarray] = {}
    imu_embeddings: dict[int, np.ndarray] = {}
    activity_by_index: dict[int, float] = {}
    gates: list[float] = []
    for start in range(0, len(dataset), 128):
        indices = list(range(start, min(start + 128, len(dataset))))
        # Keep this explicit rather than relying on DataLoader worker ordering.
        item = _collate([dataset[index] for index in indices])
        out = model(item["skeleton"].to(device), item["imu"].to(device), item["orientation"].to(device))
        if "turning_gate" in out:
            gates.extend(out["turning_gate"].flatten().detach().cpu().numpy().tolist())
        activity = item["orientation"][:, :, 4].mean(dim=1).detach().cpu().numpy()
        for offset, index in enumerate(indices):
            skeleton_embeddings[index] = out["skeleton"][offset].cpu().numpy()
            imu_embeddings[index] = out["imu"][offset].cpu().numpy()
            activity_by_index[index] = float(activity[offset])
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(dataset.rows):
        groups.setdefault((str(row["_dataset"]), str(row["_group_key"])), []).append(index)
    group_activity: dict[tuple[str, str], float] = {
        key: float(np.mean([activity_by_index[index] for index in indices])) for key, indices in groups.items()
    }
    activity_thresholds: dict[str, float] = {}
    for domain in sorted({key[0] for key in groups}):
        values = [activity for (item_domain, _), activity in group_activity.items() if item_domain == domain]
        activity_thresholds[domain] = (
            float(turning_threshold)
            if turning_threshold is not None
            else (float(np.quantile(values, 0.75)) if values else 0.0)
        )
    per_domain: dict[str, dict[str, int]] = {}
    turning_strata: dict[str, dict[str, dict[str, int]]] = {}
    margins: dict[str, list[float]] = {}
    correct = total = singleton = 0
    for (domain, group_key), indices in groups.items():
        record = per_domain.setdefault(domain, {"correct": 0, "total": 0, "singleton": 0, "groups": 0})
        if len(indices) < 2:
            singleton += 1
            record["singleton"] += 1
            continue
        record["groups"] += 1
        scores = np.asarray([[skeleton_embeddings[i] @ imu_embeddings[j] for j in indices] for i in indices])
        for row_idx, index in enumerate(indices):
            prediction = indices[int(np.argmax(scores[row_idx]))]
            ok = str(dataset.rows[prediction]["_identity"]) == str(dataset.rows[index]["_identity"])
            correct += int(ok)
            total += 1
            record["correct"] += int(ok)
            record["total"] += 1
            stratum = (
                "high"
                if group_activity[(domain, group_key)] + 1e-7 >= activity_thresholds[domain]
                else "low"
            )
            stratum_record = turning_strata.setdefault(domain, {}).setdefault(stratum, {"correct": 0, "total": 0})
            stratum_record["correct"] += int(ok)
            stratum_record["total"] += 1
            ordered = np.sort(scores[row_idx])
            margins.setdefault(domain, []).append(float(ordered[-1] - ordered[-2]))
    result = {
        "correct": correct,
        "total": total,
        "frame_acc": float(correct / total) if total else 0.0,
        "singleton_groups": singleton,
        "per_domain": per_domain,
        "mean_margin": {domain: float(np.mean(values)) for domain, values in margins.items()},
        "turning_strata": turning_strata,
        "turning_activity_threshold": activity_thresholds,
    }
    if gates:
        result["turning_gate_mean"] = float(np.mean(gates))
        result["turning_gate_std"] = float(np.std(gates))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-spec", action="append", required=True)
    parser.add_argument("--eval-spec", action="append", required=True)
    parser.add_argument("--test-spec", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "turning_gate",
            "turning_concat",
            "turning_gyro",
            "turning_cross",
            "turning_conditional_cross",
            "turning_residual",
        ),
        default="turning_gate",
    )
    parser.add_argument("--orientation-mode", choices=("proxy", "3d_heading", "none"), default="proxy")
    parser.add_argument("--orientation-profile", choices=("full", "rate"), default="full")
    parser.add_argument("--selection-stratum", choices=("all", "high", "low"), default="all")
    parser.add_argument("--selection-domain", default="custom23")
    parser.add_argument("--turning-threshold", type=float)
    parser.add_argument("--target-len", type=int, default=24)
    parser.add_argument("--target-fps-hz", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=0.8)
    parser.add_argument("--skeleton-normalize", choices=("none", "bbox"), default="bbox")
    parser.add_argument("--imu-normalize", choices=("none", "separate_zscore"), default="separate_zscore")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temporal-mode", choices=("gru", "multiscale"), default="multiscale")
    parser.add_argument("--multiscale-fusion", choices=("mean", "gated", "hierarchical_attention"), default="hierarchical_attention")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--aux-turning-weight", type=float, default=0.0)
    parser.add_argument("--turning-loss-weight", type=float, default=0.0)
    parser.add_argument("--turn-onset-weight", type=float, default=0.0)
    parser.add_argument("--sampler", choices=("domain_balanced", "orientation_hard"), default="domain_balanced")
    parser.add_argument("--hard-pool-multiplier", type=int, default=4)
    parser.add_argument("--hard-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:6")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_specs = load_specs(args.train_spec)
    eval_specs = load_specs(args.eval_spec)
    test_specs = load_specs(args.test_spec) if args.test_spec else []
    train_dataset = OrientationMotionDataset(
        train_specs, orientation_mode=args.orientation_mode, orientation_profile=args.orientation_profile, target_len=args.target_len,
        skeleton_normalize=args.skeleton_normalize, imu_normalize=args.imu_normalize,
        window_seconds=args.window_seconds,
    )
    eval_dataset = OrientationMotionDataset(
        eval_specs, orientation_mode=args.orientation_mode, orientation_profile=args.orientation_profile, target_len=args.target_len,
        skeleton_normalize=args.skeleton_normalize, imu_normalize=args.imu_normalize,
        window_seconds=args.window_seconds,
    )
    test_dataset = (
        OrientationMotionDataset(
            test_specs,
            orientation_mode=args.orientation_mode,
            orientation_profile=args.orientation_profile,
            target_len=args.target_len,
            skeleton_normalize=args.skeleton_normalize,
            imu_normalize=args.imu_normalize,
            window_seconds=args.window_seconds,
        )
        if test_specs
        else None
    )
    sample = train_dataset[0]
    model = OrientationAwareMatcher(
        int(sample["skeleton"].shape[-1]), int(sample["imu"].shape[-1]), ORIENTATION_DIM,
        hidden=args.hidden, embedding_dim=args.embedding_dim, temporal_mode=args.temporal_mode,
        multiscale_fusion=args.multiscale_fusion, window_seconds=args.window_seconds,
        use_orientation=args.variant != "baseline",
        fusion_mode=(
            "gyro_focus"
            if args.variant == "turning_gyro"
            else (
                "cross"
                if args.variant == "turning_cross"
                else (
                    "conditional_cross"
                    if args.variant == "turning_conditional_cross"
                    else (
                        "residual"
                        if args.variant == "turning_residual"
                        else ("concat" if args.variant == "turning_concat" else "gate")
                    )
                )
            )
        ),
    ).to(device)
    if args.sampler == "orientation_hard":
        sampler = OrientationHardNegativeBatchSampler(
            train_dataset,
            args.batch_size,
            args.seed,
            args.steps_per_epoch,
            pool_multiplier=args.hard_pool_multiplier,
            hard_fraction=args.hard_fraction,
        )
    else:
        sampler = DomainBalancedGroupBatchSampler(train_dataset, args.batch_size, args.seed, args.steps_per_epoch)
    loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=_collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, retrieval, gradients = [], [], []
        for batch in loader:
            out = model(batch["skeleton"].to(device), batch["imu"].to(device), batch["orientation"].to(device))
            if args.turning_loss_weight > 0.0 and args.variant != "baseline":
                activity = batch["orientation"][:, :, 4].mean(dim=1).to(device)
                loss, accuracy = _weighted_info_nce(
                    out, args.temperature, 1.0 + args.turning_loss_weight * activity
                )
            else:
                loss, accuracy = weighted_info_nce(
                    out["imu"], out["skeleton"], torch.ones(out["imu"].shape[0], device=device), args.temperature
                )
            if args.aux_turning_weight > 0.0 and args.variant != "baseline":
                loss = loss + args.aux_turning_weight * _turning_alignment_loss(
                    out["turning_activity_pred"], batch["imu"].to(device)
                )
            if args.turn_onset_weight > 0.0 and args.variant != "baseline":
                loss = loss + args.turn_onset_weight * _turn_onset_loss(
                    out, batch["orientation"].to(device)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite orientation-aware loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError("non-finite orientation-aware gradient")
            optimizer.step()
            losses.append(float(loss.item()))
            retrieval.append(accuracy)
            gradients.append(float(gradient.item()))
        metrics = evaluate(model, eval_dataset, device, turning_threshold=args.turning_threshold)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_retrieval_top1": float(np.mean(retrieval)), "mean_gradient_norm": float(np.mean(gradients)), "eval": metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        payload = {"schema_version": "g12.orientation_matcher_checkpoint.v1", "config": vars(args), "orientation_schema": ORIENTATION_DIM, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": record}
        torch.save(payload, output / "last.pt")
        if args.selection_stratum == "all":
            selected_record = metrics.get("per_domain", {}).get(args.selection_domain, {})
        else:
            selected_record = metrics.get("turning_strata", {}).get(args.selection_domain, {}).get(args.selection_stratum, {})
        selected = selected_record.get("correct", 0) / max(selected_record.get("total", 1), 1)
        if selected > best:
            best = selected
            torch.save(payload, output / "best.pt")
    final_test = None
    if test_dataset is not None:
        best_payload = torch.load(output / "best.pt", map_location=device)
        model.load_state_dict(best_payload["model"])
        final_test = evaluate(model, test_dataset, device, turning_threshold=args.turning_threshold)
    summary = {
        "schema_version": "g12.orientation_matcher_run.v2",
        "config": vars(args),
        "orientation_contract": ORIENTATION_SCHEMA,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "test_rows": len(test_dataset) if test_dataset is not None else 0,
        "best_selection_frame_acc": best if best >= 0 else None,
        "history": history,
        "final_test": final_test,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
