"""Train IMU-video alignment with two IMU positives per skeleton."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import lowpass_filter_fft
from src.datasets.alignment_dataset_multi import WindowAlignmentDatasetMultiIMU
from src.engine.common import build_alignment_model, build_optimizer, build_loss_fn
from src.engine.train import compute_imu_stats_from_train_csv
from src.modules.matchers import retrieval_top1
from src.modules.domain import dann_alpha_schedule
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMU-video alignment with two IMU sources per skeleton")

    # Data: two paired IMU sources
    parser.add_argument("--train_csv_a", type=str, required=True)
    parser.add_argument("--train_csv_b", type=str, required=True)
    parser.add_argument("--val_csv_a", type=str, default="")
    parser.add_argument("--val_csv_b", type=str, default="")
    parser.add_argument("--data_root_a", type=str, default=None)
    parser.add_argument("--data_root_b", type=str, default=None)
    parser.add_argument("--imu_stats_json_a", type=str, default="",
                        help="Optional IMU stats JSON for source A. If empty, computed from train CSV.")
    parser.add_argument("--imu_stats_json_b", type=str, default="",
                        help="Optional IMU stats JSON for source B. If empty, computed from train CSV.")

    # Model
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--imu_ckpt", type=str, default="")
    parser.add_argument("--init_alignment_ckpt", type=str, default="")
    parser.add_argument("--embed_dim", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learn_temperature", action="store_true")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    parser.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None)
    parser.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0)

    parser.add_argument("--imu_noise_std", type=float, default=0.01)
    parser.add_argument("--imu_dropout_prob", type=float, default=0.05)
    parser.add_argument("--skel_noise_std", type=float, default=0.005)
    parser.add_argument("--joint_dropout_prob", type=float, default=0.05)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)

    # IMU adapter options
    parser.add_argument("--adapter_type", type=str, default=None, choices=["none", "affine", "physics", "temporal_conv"])
    parser.add_argument("--adapter_train_only", action="store_true")

    # Domain adversarial (DANN) options
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

    # Physics encoder options
    parser.add_argument("--imu_encoder_type", type=str, default="lstm", choices=["lstm", "physics"])
    parser.add_argument("--physics_d_model", type=int, default=128)
    parser.add_argument("--physics_n_heads", type=int, default=4)
    parser.add_argument("--physics_num_layers", type=int, default=3)
    parser.add_argument("--physics_fs_hz", type=float, default=30.0)
    parser.add_argument("--physics_n_fft", type=int, default=64)
    parser.add_argument("--physics_dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="artifacts")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_save_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() if args.run_name else ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    return (output_root / run_name).resolve()


def load_stats(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    stats = json.loads(Path(json_path).read_text())
    mean = np.asarray(stats["imu_mean"], dtype=np.float32)
    std = np.asarray(stats["imu_std"], dtype=np.float32)
    return mean, std


def maybe_augment_inputs(
    imu_a: torch.Tensor,
    imu_b: torch.Tensor,
    skeleton: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.imu_noise_std > 0:
        imu_a = imu_a + torch.randn_like(imu_a) * args.imu_noise_std
        imu_b = imu_b + torch.randn_like(imu_b) * args.imu_noise_std

    if args.imu_dropout_prob > 0:
        feat_keep_a = (torch.rand(imu_a.shape[0], 1, imu_a.shape[2], device=imu_a.device) > args.imu_dropout_prob).float()
        imu_a = imu_a * feat_keep_a
        feat_keep_b = (torch.rand(imu_b.shape[0], 1, imu_b.shape[2], device=imu_b.device) > args.imu_dropout_prob).float()
        imu_b = imu_b * feat_keep_b

    if args.skel_noise_std > 0:
        skeleton = skeleton + torch.randn_like(skeleton) * args.skel_noise_std

    if args.joint_dropout_prob > 0:
        joint_keep = (
            torch.rand(skeleton.shape[0], 1, skeleton.shape[2], 1, device=skeleton.device) > args.joint_dropout_prob
        ).float()
        skeleton = skeleton * joint_keep

    return imu_a, imu_b, skeleton


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {
        "imu_a": batch["imu_a"].to(device),
        "imu_b": batch["imu_b"].to(device),
        "skeleton": batch["skeleton"].to(device),
    }
    if "root_trajectory" in batch:
        result["root_trajectory"] = batch["root_trajectory"].to(device)
    return result


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_epoch(model, data_loader, loss_fn, device, domain_loss_weight: float = 0.0) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": 0.0, "main_loss": 0.0, "aux_loss": 0.0, "top1": 0.0, "domain_loss": 0.0}

    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    total_domain_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    use_domain = domain_loss_weight > 0 and getattr(model, "domain_classifier", None) is not None

    with torch.no_grad():
        for batch in data_loader:
            b = move_to_device(batch, device)
            z_vid = model.video_encoder(b["skeleton"])
            z_imu_a = model.imu_encoder(b["imu_a"])
            z_imu_b = model.imu_encoder(b["imu_b"])
            z_imu = torch.cat([z_imu_a, z_imu_b], dim=0)
            z_vid_rep = z_vid.repeat(2, 1)

            main_loss = loss_fn(z_imu, z_vid_rep)
            acc = retrieval_top1(z_imu, z_vid_rep)

            loss = main_loss
            domain_loss = torch.tensor(0.0, device=device)
            if use_domain:
                B = z_imu_a.shape[0]
                domain_logits = model.domain_classifier(z_imu)
                domain_labels = torch.cat([
                    torch.zeros(B, device=device, dtype=torch.long),
                    torch.ones(B, device=device, dtype=torch.long),
                ])
                domain_loss = F.cross_entropy(domain_logits, domain_labels)
                loss = main_loss + domain_loss_weight * domain_loss

            total_loss += float(loss.item())
            total_main_loss += float(main_loss.item())
            total_domain_loss += float(domain_loss.item())
            total_acc += acc
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "main_loss": 0.0, "aux_loss": 0.0, "top1": 0.0, "domain_loss": 0.0}
    return {
        "loss": total_loss / total_batches,
        "main_loss": total_main_loss / total_batches,
        "aux_loss": 0.0,
        "top1": total_acc / total_batches,
        "domain_loss": total_domain_loss / total_batches,
    }


def main() -> None:
    args = parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if getattr(args, "use_global_motion", False):
        raise NotImplementedError("Multi-IMU training does not yet support global motion.")

    # Load or compute per-source IMU stats
    if args.imu_stats_json_a:
        imu_mean_a, imu_std_a = load_stats(args.imu_stats_json_a)
    else:
        imu_mean_a, imu_std_a = compute_imu_stats_from_train_csv(
            args.train_csv_a, args.data_root_a, args.imu_lowpass_cutoff_hz, args.imu_lowpass_fs_hz
        )

    if args.imu_stats_json_b:
        imu_mean_b, imu_std_b = load_stats(args.imu_stats_json_b)
    else:
        imu_mean_b, imu_std_b = compute_imu_stats_from_train_csv(
            args.train_csv_b, args.data_root_b, args.imu_lowpass_cutoff_hz, args.imu_lowpass_fs_hz
        )

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    return_root = getattr(args, "use_global_motion", False)
    root_source = getattr(args, "global_motion_root_source", "auto")

    train_ds = WindowAlignmentDatasetMultiIMU(
        csv_paths=[args.train_csv_a, args.train_csv_b],
        root_dirs=[args.data_root_a, args.data_root_b],
        imu_stats=[(imu_mean_a, imu_std_a), (imu_mean_b, imu_std_b)],
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
        return_root_trajectory=return_root,
        root_source=root_source,
    )

    val_ds = None
    if args.val_csv_a and args.val_csv_b:
        val_ds = WindowAlignmentDatasetMultiIMU(
            csv_paths=[args.val_csv_a, args.val_csv_b],
            root_dirs=[args.data_root_a, args.data_root_b],
            imu_stats=[(imu_mean_a, imu_std_a), (imu_mean_b, imu_std_b)],
            imu_sensor=imu_sensor,
            repeat_single_sensor=args.repeat_single_sensor,
            imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
            imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
            return_root_trajectory=return_root,
            root_source=root_source,
        )

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

    if args.adapter_train_only:
        adapter_param_count = 0
        for name, p in model.named_parameters():
            if "adapter" not in name:
                p.requires_grad = False
            else:
                adapter_param_count += p.numel()
        print(f"[INFO] adapter_train_only: frozen all non-adapter params. Adapter params: {adapter_param_count:,}")

    optimizer = build_optimizer(model, lr_backbone=args.lr_backbone, lr_heads=args.lr_heads, weight_decay=args.weight_decay)
    loss_fn = build_loss_fn(temperature=args.temperature, learn_temperature=args.learn_temperature, device=device)

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "imu_stats_a.json").write_text(
        json.dumps({"imu_mean": imu_mean_a.tolist(), "imu_std": imu_std_a.tolist()}, indent=2)
    )
    (save_dir / "imu_stats_b.json").write_text(
        json.dumps({"imu_mean": imu_mean_b.tolist(), "imu_std": imu_std_b.tolist()}, indent=2)
    )

    val_count = len(val_ds) if val_ds is not None else 0
    print(f"Train windows (pairs): {len(train_ds)}, Val windows (pairs): {val_count}")
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
            if not args.adapter_train_only:
                p.requires_grad = not freeze_backbone
            else:
                p.requires_grad = False

        # Update Gradient Reversal Layer alpha
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
        running_domain_loss = 0.0
        running_acc = 0.0
        steps = 0

        use_domain = args.domain_loss_weight > 0 and getattr(model, "domain_classifier", None) is not None

        for step, batch in enumerate(train_loader, start=1):
            b = move_to_device(batch, device)
            b["imu_a"], b["imu_b"], b["skeleton"] = maybe_augment_inputs(
                b["imu_a"], b["imu_b"], b["skeleton"], args
            )

            z_vid = model.video_encoder(b["skeleton"])
            z_imu_a = model.imu_encoder(b["imu_a"])
            z_imu_b = model.imu_encoder(b["imu_b"])
            z_imu = torch.cat([z_imu_a, z_imu_b], dim=0)
            z_vid_rep = z_vid.repeat(2, 1)

            main_loss = loss_fn(z_imu, z_vid_rep)
            loss = main_loss
            domain_loss = torch.tensor(0.0, device=device)

            if use_domain:
                B = z_imu_a.shape[0]
                domain_logits = model.domain_classifier(z_imu)
                domain_labels = torch.cat([
                    torch.zeros(B, device=device, dtype=torch.long),
                    torch.ones(B, device=device, dtype=torch.long),
                ])
                domain_loss = F.cross_entropy(domain_logits, domain_labels)
                loss = main_loss + args.domain_loss_weight * domain_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            acc = retrieval_top1(z_imu, z_vid_rep)
            running_loss += float(loss.item())
            running_main_loss += float(main_loss.item())
            running_domain_loss += float(domain_loss.item())
            running_acc += acc
            steps += 1

            if step == 1:
                diag_extra = ""
                if use_domain:
                    diag_extra = f" domain_loss={domain_loss.item():.4f}"
                print(f"[DIAG] Multi-IMU | top1={acc:.4f} | random_exp={1.0 / z_imu.shape[0]:.4f}{diag_extra}")

            if step % args.log_interval == 0:
                log_str = (
                    f"[Epoch {epoch}/{args.epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / steps:.4f} main={running_main_loss / steps:.4f}"
                )
                if use_domain:
                    log_str += f" domain={running_domain_loss / steps:.4f}"
                log_str += f" top1={running_acc / steps:.4f}"
                print(log_str)

        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device, domain_loss_weight=args.domain_loss_weight)
        train_loss = running_loss / max(steps, 1)
        train_main_loss = running_main_loss / max(steps, 1)
        train_domain_loss = running_domain_loss / max(steps, 1)
        train_top1 = running_acc / max(steps, 1)

        epoch_print = (
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_main={train_main_loss:.4f} "
            f"train_top1={train_top1:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_main={val_metrics['main_loss']:.4f} val_top1={val_metrics['top1']:.4f}"
        )
        if use_domain:
            epoch_print += (
                f" train_domain={train_domain_loss:.4f} val_domain={val_metrics['domain_loss']:.4f}"
            )
        print(epoch_print)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_main_loss": train_main_loss,
            "train_top1": train_top1,
            "val_loss": val_metrics["loss"],
            "val_main_loss": val_metrics["main_loss"],
            "val_top1": val_metrics["top1"],
        }
        if use_domain:
            epoch_log["train_domain_loss"] = train_domain_loss
            epoch_log["val_domain_loss"] = val_metrics["domain_loss"]
        epoch_logs.append(epoch_log)
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
            print(f"Early stopping triggered at epoch {epoch}")
            stopped_epoch = epoch
            break

    metrics = {"best_val_top1": best_val, "stopped_epoch": stopped_epoch, "save_dir": str(save_dir)}
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Best val top1={best_val:.4f} (stopped at epoch {stopped_epoch})")


if __name__ == "__main__":
    main()
