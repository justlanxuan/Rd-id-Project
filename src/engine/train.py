"""Train IMU-video alignment model (MotionBERT-style)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset, lowpass_filter_fft
from src.engine.common import build_alignment_model, build_optimizer, build_loss_fn
from src.modules.domain import dann_alpha_schedule
from src.modules.matchers import retrieval_top1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMU-Video alignment")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None, help="Root for npz relative paths")

    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--imu_ckpt", type=str, default="")
    parser.add_argument("--init_alignment_ckpt", type=str, default="")

    parser.add_argument("--embed_dim", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learn_temperature", action="store_true")
    parser.add_argument("--compute_imu_stats", action="store_true")
    parser.add_argument("--imu_stats_json", type=str, default="")
    parser.add_argument("--per_session_stats_dir", type=str, default="", help="Directory containing per-session imu_stats.json files (e.g., <session>_imu_stats.json)")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    parser.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None, help="FFT low-pass cutoff for IMU windows in Hz; set <= 0 to disable.")
    parser.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0, help="Sampling rate used by the IMU low-pass filter.")

    parser.add_argument("--imu_noise_std", type=float, default=0.01)
    parser.add_argument("--imu_dropout_prob", type=float, default=0.05)
    parser.add_argument("--skel_noise_std", type=float, default=0.005)
    parser.add_argument("--joint_dropout_prob", type=float, default=0.05)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Early stopping patience (0 = disabled)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Minimum improvement for early stopping")

    # IMU adapter options
    parser.add_argument("--adapter_type", type=str, default=None, choices=["none", "affine", "physics", "temporal_conv"],
                        help="IMU input adapter type. 'none' disables adapter.")
    parser.add_argument("--adapter_train_only", action="store_true",
                        help="Freeze all pretrained parameters (backbone, IMU encoder, video encoder) and train only the adapter.")

    # Global motion encoder options
    parser.add_argument("--use_global_motion", action="store_true")
    parser.add_argument("--global_motion_input_dim", type=int, default=2)
    parser.add_argument("--global_motion_hidden_dim", type=int, default=64)
    parser.add_argument("--global_motion_num_layers", type=int, default=2)
    parser.add_argument("--global_motion_dropout", type=float, default=0.1)
    parser.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    parser.add_argument("--global_motion_fusion_type", type=str, default="concat")
    parser.add_argument("--global_motion_fusion_proj", action="store_true")
    parser.add_argument("--global_motion_root_source", type=str, default="auto")
    parser.add_argument(
        "--global_motion_aux_weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary InfoNCE loss between IMU and global video embeddings.",
    )
    parser.add_argument(
        "--global_motion_train_only",
        action="store_true",
        help="Freeze IMU/local/fusion and train only the global motion encoder against IMU.",
    )

    # Physics encoder options
    parser.add_argument("--imu_encoder_type", type=str, default="lstm", choices=["lstm", "physics"])
    parser.add_argument("--physics_d_model", type=int, default=128)
    parser.add_argument("--physics_n_heads", type=int, default=4)
    parser.add_argument("--physics_num_layers", type=int, default=3)
    parser.add_argument("--physics_fs_hz", type=float, default=30.0)
    parser.add_argument("--physics_n_fft", type=int, default=64)
    parser.add_argument("--physics_dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_root",
        type=str,
        default="artifacts",
        help="Root folder to store all training artifacts and checkpoints.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name; if omitted, timestamp is used.",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_every_epoch", action="store_true", help="Save a checkpoint after every epoch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--shuffle_video_in_batch",
        action="store_true",
        help="Randomly shuffle video embeddings within each batch during training to break position bias."
    )

    # Domain adversarial options
    parser.add_argument("--num_domains", type=int, default=0,
                        help="Number of domains for adversarial training. 0 disables DANN.")
    parser.add_argument("--domain_loss_weight", type=float, default=0.0,
                        help="Weight for domain classification loss.")
    parser.add_argument("--domain_hidden_dim", type=int, default=256,
                        help="Hidden dim of the domain classifier MLP.")
    parser.add_argument("--domain_alpha", type=float, default=1.0,
                        help="Fixed Gradient Reversal Layer alpha.")
    parser.add_argument("--domain_schedule", action="store_true",
                        help="Use increasing alpha schedule: 2/(1+exp(-10*p))-1.")
    parser.add_argument("--domain_label_map", type=str, default="egohumans:0,custom:1",
                        help="Domain label mapping in CSV domain column, e.g. 'egohumans:0,custom:1'.")

    return parser.parse_args()


def resolve_save_dir(args: argparse.Namespace) -> Path:
    """Resolve output directory and force all artifacts under output_root."""
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() if args.run_name else ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")

    return (output_root / run_name).resolve()


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


def read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_imu_stats_from_train_csv(
    train_csv: str,
    data_root: str | None,
    imu_lowpass_cutoff_hz: float | None = None,
    imu_lowpass_fs_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(train_csv)
    base = Path(data_root) if data_root else Path(train_csv).resolve().parent

    # Aggregate stats per (npz_path, imu_idx)
    from collections import defaultdict
    per_source: dict[tuple[str, int], list] = defaultdict(lambda: [None, None, 0])

    for row in rows:
        rel = row["npz_path"]
        imu_idx = int(row.get("imu_idx", 0))
        key = (rel, imu_idx)
        if per_source[key][2] > 0:
            continue  # already accumulated for this source

        data = np.load((base / rel).resolve(), allow_pickle=True)
        imu = data["imu"].astype(np.float64)
        if imu.ndim == 3:
            imu = imu[:, imu_idx, :]

        if imu_lowpass_cutoff_hz is not None and imu_lowpass_cutoff_hz > 0:
            imu = lowpass_filter_fft(imu, imu_lowpass_cutoff_hz, imu_lowpass_fs_hz)

        per_source[key][0] = imu.sum(axis=0)
        per_source[key][1] = (imu * imu).sum(axis=0)
        per_source[key][2] = imu.shape[0]

    total_count = sum(v[2] for v in per_source.values())
    if total_count == 0:
        raise ValueError("No IMU frames found while computing stats.")

    sums = np.zeros_like(list(per_source.values())[0][0])
    sq_sums = np.zeros_like(list(per_source.values())[0][1])
    for s, sq, c in per_source.values():
        sums += s
        sq_sums += sq

    mean = sums / total_count
    var = np.maximum(sq_sums / total_count - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def maybe_augment_inputs(imu: torch.Tensor, skeleton: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    if args.imu_noise_std > 0:
        imu = imu + torch.randn_like(imu) * args.imu_noise_std

    if args.imu_dropout_prob > 0:
        feat_keep = (torch.rand(imu.shape[0], 1, imu.shape[2], device=imu.device) > args.imu_dropout_prob).float()
        imu = imu * feat_keep

    if args.skel_noise_std > 0:
        skeleton = skeleton + torch.randn_like(skeleton) * args.skel_noise_std

    if args.joint_dropout_prob > 0:
        joint_keep = (
            torch.rand(skeleton.shape[0], 1, skeleton.shape[2], 1, device=skeleton.device) > args.joint_dropout_prob
        ).float()
        skeleton = skeleton * joint_keep

    return imu, skeleton


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_epoch(model, data_loader, loss_fn, device, domain_loss_weight: float = 0.0, domain_map: Dict[str, int] | None = None) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": 0.0, "top1": 0.0, "main_loss": 0.0, "aux_loss": 0.0, "domain_loss": 0.0}

    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_domain_loss = 0.0
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
            main_loss = loss_fn(out["imu"], out["video"])
            aux_weight = float(getattr(model, "global_motion_aux_weight", 0.0))
            aux_loss = torch.zeros((), device=main_loss.device)
            if aux_weight > 0.0 and "video_global" in out:
                aux_loss = loss_fn(out["imu"], out["video_global"])

            domain_loss = torch.zeros((), device=main_loss.device)
            if use_domain:
                domain_labels = domain_labels_from_batch(b.get("domain"), domain_map or {}, device)
                if domain_labels is not None and domain_labels.shape[0] == out["imu"].shape[0]:
                    domain_logits = model.domain_classifier(out["imu"])
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)

            loss = main_loss + aux_weight * aux_loss + domain_loss_weight * domain_loss
            acc = retrieval_top1(out["imu"], out["video"])
            total_loss += float(loss.item())
            total_main_loss += float(main_loss.item())
            total_aux_loss += float(aux_loss.item())
            total_domain_loss += float(domain_loss.item())
            total_acc += acc
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "top1": 0.0}
    return {
        "loss": total_loss / total_batches,
        "main_loss": total_main_loss / total_batches,
        "aux_loss": total_aux_loss / total_batches,
        "domain_loss": total_domain_loss / total_batches,
        "top1": total_acc / total_batches,
    }


def main() -> None:
    args = parse_args()
    # Set random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior may slow down training; enable only if needed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
    elif args.compute_imu_stats:
        imu_mean, imu_std = compute_imu_stats_from_train_csv(
            args.train_csv,
            args.data_root,
            args.imu_lowpass_cutoff_hz,
            args.imu_lowpass_fs_hz,
        )

    per_session_stats = None
    if args.per_session_stats_dir:
        per_session_stats = {}
        psd = Path(args.per_session_stats_dir)
        for p in psd.glob("*_imu_stats.json"):
            session_id = p.stem.replace("_imu_stats", "")
            stats = json.loads(p.read_text())
            per_session_stats[session_id] = (
                np.asarray(stats["imu_mean"], dtype=np.float32),
                np.asarray(stats["imu_std"], dtype=np.float32),
            )
        print(f"[INFO] Loaded per-session stats for {len(per_session_stats)} sessions: {sorted(per_session_stats.keys())}")

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    return_root = getattr(args, "use_global_motion", False)
    root_source = getattr(args, "global_motion_root_source", "auto")
    train_ds = WindowAlignmentDataset(
        args.train_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
        return_root_trajectory=return_root,
        root_source=root_source,
        per_session_stats=per_session_stats,
    )
    try:
        val_ds = WindowAlignmentDataset(
            args.val_csv,
            root_dir=args.data_root,
            imu_mean=imu_mean,
            imu_std=imu_std,
            imu_sensor=imu_sensor,
            repeat_single_sensor=args.repeat_single_sensor,
            imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
            imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
            return_root_trajectory=return_root,
            root_source=root_source,
            per_session_stats=per_session_stats,
        )
    except ValueError as e:
        if "No rows found" in str(e):
            print(f"[WARN] Validation CSV is empty: {args.val_csv}. Validation will be skipped.")
            val_ds = None
        else:
            raise

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model, cfg_name = build_alignment_model(args, device, embed_dim=args.embed_dim)
    model.global_motion_aux_weight = float(args.global_motion_aux_weight)
    domain_map = parse_domain_label_map(args.domain_label_map)

    # If adapter_train_only: freeze everything except adapter parameters
    adapter_param_count = 0
    if args.adapter_train_only:
        for name, p in model.named_parameters():
            if "adapter" not in name:
                p.requires_grad = False
            else:
                adapter_param_count += p.numel()
        print(f"[INFO] adapter_train_only: frozen all non-adapter params. Adapter params: {adapter_param_count:,}")

    optimizer = build_optimizer(
        model,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )
    loss_fn = build_loss_fn(
        temperature=args.temperature,
        learn_temperature=args.learn_temperature,
        device=device,
    )

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    if imu_mean is not None and imu_std is not None:
        (save_dir / "imu_stats.json").write_text(
            json.dumps({"imu_mean": imu_mean.tolist(), "imu_std": imu_std.tolist()}, indent=2)
        )

    val_count = len(val_ds) if val_ds is not None else 0
    print(f"Train windows: {len(train_ds)}, Val windows: {val_count}")
    print(f"Trainable params: {count_trainable_params(model):,}")
    print(f"Backbone cfg name: {cfg_name}")
    print(f"Artifacts directory: {save_dir}")

    epoch_logs = []

    best_val = -1.0
    epochs_no_improve = 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        from src.engine.common import _get_backbone_from_video_encoder
        backbone = _get_backbone_from_video_encoder(model.video_encoder)
        for p in backbone.parameters():
            # If adapter_train_only, backbone stays frozen regardless of freeze_backbone_epochs
            if not args.adapter_train_only:
                p.requires_grad = not freeze_backbone
            else:
                p.requires_grad = False

        if getattr(model, "domain_classifier", None) is not None:
            if args.domain_schedule:
                progress = (epoch - 1) / max(args.epochs - 1, 1)
                alpha = dann_alpha_schedule(progress)
                model.domain_classifier.set_alpha(alpha)
                print(f"[INFO] Epoch {epoch}: domain alpha = {alpha:.4f} (progress={progress:.3f})")
            else:
                model.domain_classifier.set_alpha(args.domain_alpha)

        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux_loss = 0.0
        running_domain_loss = 0.0
        running_acc = 0.0
        steps = 0
        use_domain = args.domain_loss_weight > 0 and getattr(model, "domain_classifier", None) is not None

        for step, batch in enumerate(train_loader, start=1):
            b = move_to_device(batch, device)
            b["imu"], b["skeleton"] = maybe_augment_inputs(b["imu"], b["skeleton"], args)
            forward_kwargs = {"imu": b["imu"], "skeleton": b["skeleton"]}
            if "root_trajectory" in b:
                forward_kwargs["root_trajectory"] = b["root_trajectory"]
            out = model(**forward_kwargs)

            # Optionally shuffle video embeddings to break position bias
            if args.shuffle_video_in_batch:
                B = out["imu"].shape[0]
                perm = torch.randperm(B, device=device)
                perm_inv = torch.empty_like(perm)
                perm_inv[perm] = torch.arange(B, device=device)

                video_for_loss = out["video"][perm]
                video_global_for_loss = out["video_global"][perm] if "video_global" in out else None
                labels_a = perm_inv
                labels_b = perm
            else:
                video_for_loss = out["video"]
                video_global_for_loss = out.get("video_global")
                labels_a = None
                labels_b = None

            main_loss = loss_fn(out["imu"], video_for_loss, labels_a=labels_a, labels_b=labels_b)
            aux_loss = torch.zeros((), device=main_loss.device)
            if args.global_motion_aux_weight > 0.0 and video_global_for_loss is not None:
                aux_loss = loss_fn(out["imu"], video_global_for_loss, labels_a=labels_a, labels_b=labels_b)

            domain_loss = torch.zeros((), device=main_loss.device)
            if use_domain:
                domain_labels = domain_labels_from_batch(b.get("domain"), domain_map, device)
                if domain_labels is not None and domain_labels.shape[0] == out["imu"].shape[0]:
                    domain_logits = model.domain_classifier(out["imu"])
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)

            loss = main_loss + args.global_motion_aux_weight * aux_loss + args.domain_loss_weight * domain_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            acc = retrieval_top1(out["imu"], video_for_loss, labels_a=labels_a, labels_b=labels_b)
            running_loss += float(loss.item())
            running_main_loss += float(main_loss.item())
            running_aux_loss += float(aux_loss.item())
            running_domain_loss += float(domain_loss.item())
            running_acc += acc
            steps += 1

            if step == 1 and args.shuffle_video_in_batch:
                print(f"[DIAG] Shuffle ON | perm head={perm[:5].tolist()} | top1={acc:.4f} | random_exp={1.0/B:.4f}")
            elif step == 1 and not args.shuffle_video_in_batch:
                print(f"[DIAG] Shuffle OFF | top1={acc:.4f}")

            if step % args.log_interval == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / steps:.4f} main={running_main_loss / steps:.4f} "
                    f"aux={running_aux_loss / steps:.4f} domain={running_domain_loss / steps:.4f} top1={running_acc / steps:.4f}"
                )

        val_metrics = evaluate_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            domain_loss_weight=args.domain_loss_weight,
            domain_map=domain_map,
        )
        train_loss = running_loss / max(steps, 1)
        train_main_loss = running_main_loss / max(steps, 1)
        train_aux_loss = running_aux_loss / max(steps, 1)
        train_domain_loss = running_domain_loss / max(steps, 1)
        train_top1 = running_acc / max(steps, 1)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_main={train_main_loss:.4f} "
            f"train_aux={train_aux_loss:.4f} train_domain={train_domain_loss:.4f} train_top1={train_top1:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_main={val_metrics['main_loss']:.4f} "
            f"val_aux={val_metrics['aux_loss']:.4f} val_domain={val_metrics['domain_loss']:.4f} val_top1={val_metrics['top1']:.4f}"
        )

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_main_loss": train_main_loss,
                "train_aux_loss": train_aux_loss,
                "train_domain_loss": train_domain_loss,
                "train_top1": train_top1,
                "val_loss": val_metrics["loss"],
                "val_main_loss": val_metrics["main_loss"],
                "val_aux_loss": val_metrics["aux_loss"],
                "val_domain_loss": val_metrics["domain_loss"],
                "val_top1": val_metrics["top1"],
            }
        )
        mode = "w" if epoch == 1 else "a"
        with (save_dir / "epoch_metrics.jsonl").open(mode, encoding="utf-8") as f:
            f.write(json.dumps(epoch_logs[-1], ensure_ascii=True) + "\n")

        payload = {
            "epoch": epoch,
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_top1": val_metrics["top1"],
        }
        torch.save(payload, save_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(payload, save_dir / f"epoch_{epoch:03d}.pt")

        score_for_best = val_metrics["top1"] if val_loader is not None else train_top1
        if score_for_best > best_val + args.early_stop_min_delta:
            best_val = score_for_best
            epochs_no_improve = 0
            torch.save(payload, save_dir / "best.pt")
        else:
            epochs_no_improve += 1

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
            stopped_epoch = epoch
            break

    metrics = {"best_val_top1": best_val, "stopped_epoch": stopped_epoch, "save_dir": str(save_dir)}
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Best val top1={best_val:.4f} (stopped at epoch {stopped_epoch})")


if __name__ == "__main__":
    main()
