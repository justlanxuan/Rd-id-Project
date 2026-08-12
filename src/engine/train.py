"""Train the official hybrid IMU-video alignment model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_cfg
from src.datasets import SameWindowBatchSampler, WindowAlignmentDataset
from src.engine.augmentation import maybe_augment_inputs
from src.engine.batch import (
    build_subject_label_map,
    domain_labels_from_batch,
    group_labels_from_batch,
    move_to_device,
    parse_domain_label_map,
    subject_labels_from_batch,
)
from src.engine.checkpoint_selection import resolve_selection_metric, selection_value_and_score
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
from src.engine.stats import compute_imu_stats_from_train_csv, count_trainable_params, fit_hybrid_encoder_stats
from src.engine.validation import evaluate_epoch
from src.models.checkpoint import checkpoint_scalar, model_checkpoint_metadata
from src.modules.domain import dann_alpha_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMU-Video alignment")
    parser.add_argument("--config", type=str, required=True, help="Workflow YAML.")
    parser.add_argument("--device", type=str, default="", help="Optional runtime device override.")
    return parser.parse_args()


def resolve_save_dir(cfg) -> Path:
    """Resolve output directory and force all artifacts under output_root."""
    output_root = Path(cfg.TRAIN.OUTPUT.OUTPUT_ROOT).expanduser().resolve()
    run_name = cfg.TRAIN.OUTPUT.RUN_NAME.strip() if cfg.TRAIN.OUTPUT.RUN_NAME else ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")

    return (output_root / run_name).resolve()


def main() -> None:
    cli_args = parse_args()
    cfg = load_cfg(cli_args.config)
    if cli_args.device:
        cfg.defrost()
        cfg.TRAIN.DEVICE = cli_args.device
        cfg.freeze()
    T = cfg.TRAIN
    P = cfg.PATHS
    IMU_PRE = cfg.PREPROCESS.IMU
    # Set random seeds
    seed = int(T.SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior may slow down training; enable only if needed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    device = torch.device(T.DEVICE if torch.cuda.is_available() else "cpu")

    from src.engine.common import build_alignment_model_from_cfg, build_loss_fn, build_optimizer

    model, cfg_name = build_alignment_model_from_cfg(cfg, device)
    capabilities = model.capabilities

    imu_mean = None
    imu_std = None
    if capabilities.external_imu_normalization and T.IMU_STATS_JSON:
        stats = json.loads(Path(T.IMU_STATS_JSON).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
    elif capabilities.external_imu_normalization and T.COMPUTE_IMU_STATS:
        imu_mean, imu_std = compute_imu_stats_from_train_csv(
            P.TRAIN_CSV,
            P.DATA_ROOT,
            IMU_PRE.LOWPASS_CUTOFF_HZ,
            IMU_PRE.LOWPASS_FS_HZ,
        )

    per_session_stats = None
    if T.PER_SESSION_STATS_DIR:
        per_session_stats = {}
        psd = Path(T.PER_SESSION_STATS_DIR)
        for p in psd.glob("*_imu_stats.json"):
            session_id = p.stem.replace("_imu_stats", "")
            stats = json.loads(p.read_text())
            per_session_stats[session_id] = (
                np.asarray(stats["imu_mean"], dtype=np.float32),
                np.asarray(stats["imu_std"], dtype=np.float32),
            )
        print(f"[INFO] Loaded per-session stats for {len(per_session_stats)} sessions: {sorted(per_session_stats.keys())}")

    imu_sensor = T.IMU_SENSOR.strip() if T.IMU_SENSOR else None
    train_ds = WindowAlignmentDataset(
        P.TRAIN_CSV,
        root_dir=P.DATA_ROOT,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=T.REPEAT_SINGLE_SENSOR,
        imu_lowpass_cutoff_hz=IMU_PRE.LOWPASS_CUTOFF_HZ,
        imu_lowpass_fs_hz=IMU_PRE.LOWPASS_FS_HZ,
        return_root_trajectory=False,
        root_source="auto",
        per_session_stats=per_session_stats,
    )
    try:
        val_ds = WindowAlignmentDataset(
            P.VAL_CSV,
            root_dir=P.DATA_ROOT,
            imu_mean=imu_mean,
            imu_std=imu_std,
            imu_sensor=imu_sensor,
            repeat_single_sensor=T.REPEAT_SINGLE_SENSOR,
            imu_lowpass_cutoff_hz=IMU_PRE.LOWPASS_CUTOFF_HZ,
            imu_lowpass_fs_hz=IMU_PRE.LOWPASS_FS_HZ,
            return_root_trajectory=False,
            root_source="auto",
            per_session_stats=per_session_stats,
        )
    except ValueError as e:
        if "No rows found" in str(e):
            print(f"[WARN] Validation CSV is empty: {P.VAL_CSV}. Validation will be skipped.")
            val_ds = None
        else:
            raise

    if bool(getattr(T, "GROUP_BATCH_BY_WINDOW", False)):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=SameWindowBatchSampler(
                train_ds.rows,
                batch_size=int(T.BATCH_SIZE),
                seed=int(T.SEED),
                drop_last=True,
            ),
            num_workers=T.NUM_WORKERS,
            pin_memory=True,
        )
        print("[INFO] Using same-window grouped batch sampler for hard negatives.")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=T.BATCH_SIZE,
            shuffle=True,
            num_workers=T.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
    val_loader = None
    if val_ds is not None:
        val_batch_size = len(val_ds) if capabilities.full_validation_batch else T.BATCH_SIZE
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=T.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    if bool(getattr(T, "CROSS_PAIR_TRAIN_ONLY", False)):
        if getattr(model, "cross_pair_head", None) is None:
            raise ValueError("TRAIN.CROSS_PAIR_TRAIN_ONLY requires TRAIN.MODEL.CROSS_PAIR_HEAD=true.")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.cross_pair_head.parameters():
            param.requires_grad = True
        print("[INFO] Training only cross_pair_head; IMU/video encoders are frozen.")
    domain_map = parse_domain_label_map(T.DOMAIN_LABEL_MAP)
    contrastive_target = str(getattr(T, "CONTRASTIVE_TARGET", "pair")).lower()
    if contrastive_target not in {"pair", "subject", "window"}:
        raise ValueError(f"Unsupported TRAIN.CONTRASTIVE_TARGET={T.CONTRASTIVE_TARGET!r}")
    pair_loss_target = str(getattr(T, "PAIR_LOSS_TARGET", "subject")).lower()
    if pair_loss_target not in {"subject", "pair", "subject_pair", "window", "window_ce"}:
        raise ValueError(f"Unsupported TRAIN.PAIR_LOSS_TARGET={T.PAIR_LOSS_TARGET!r}")
    if contrastive_target == "subject" and T.SHUFFLE_VIDEO_IN_BATCH:
        raise ValueError("SHUFFLE_VIDEO_IN_BATCH is incompatible with subject contrastive training.")
    subject_map = build_subject_label_map(train_ds)
    id_head_imu = None
    id_head_video = None
    id_params = []
    if float(T.ID_LOSS_WEIGHT) > 0:
        num_subjects = len(subject_map)
        if num_subjects < 2:
            print("[WARN] ID_LOSS_WEIGHT > 0 but fewer than two subjects were found; disabling ID auxiliary loss.")
        else:
            hidden_dim = int(getattr(model.imu_encoder, "hidden_size", T.MODEL.HYBRID_HIDDEN))
            id_head_imu = nn.Linear(hidden_dim, num_subjects).to(device)
            id_head_video = nn.Linear(hidden_dim, num_subjects).to(device)
            id_params = list(id_head_imu.parameters()) + list(id_head_video.parameters())
            print(f"[INFO] Enabled ID auxiliary loss for {num_subjects} subjects with weight={float(T.ID_LOSS_WEIGHT):.4f}.")

    # If adapter_train_only: freeze everything except adapter parameters
    adapter_param_count = 0
    if T.ADAPTER_TRAIN_ONLY:
        for name, p in model.named_parameters():
            if "adapter" not in name:
                p.requires_grad = False
            else:
                adapter_param_count += p.numel()
        print(f"[INFO] adapter_train_only: frozen all non-adapter params. Adapter params: {adapter_param_count:,}")

    loss_fn = build_loss_fn(
        temperature=T.TEMPERATURE,
        learn_temperature=T.LEARN_TEMPERATURE,
        device=device,
    )
    if T.LEARN_TEMPERATURE and T.MODEL.INIT_ALIGNMENT_CKPT:
        init_log_temp = checkpoint_scalar(str(T.MODEL.INIT_ALIGNMENT_CKPT), "log_temp")
        if init_log_temp is not None and hasattr(loss_fn, "log_temperature"):
            init_log_temp = init_log_temp.to(dtype=torch.float32)
            with torch.no_grad():
                loss_fn.log_temperature.copy_(init_log_temp.to(loss_fn.log_temperature.device))
            print(f"[INFO] Initialized learnable loss log-temperature from checkpoint: {float(init_log_temp):.6f}")
    optimizer = build_optimizer(
        model,
        lr_backbone=T.LR_BACKBONE,
        lr_heads=T.LR_HEADS,
        weight_decay=T.WEIGHT_DECAY,
    )
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        optimizer.add_param_group({"params": loss_params, "lr": T.LR_HEADS, "weight_decay": T.WEIGHT_DECAY})
        print(f"[INFO] Added {sum(p.numel() for p in loss_params)} trainable loss parameter(s) to optimizer.")
    if id_params:
        optimizer.add_param_group({"params": id_params, "lr": T.LR_HEADS, "weight_decay": T.WEIGHT_DECAY})
        print(f"[INFO] Added {sum(p.numel() for p in id_params)} ID-head parameter(s) to optimizer.")

    save_dir = resolve_save_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    if capabilities.external_imu_normalization and imu_mean is not None and imu_std is not None:
        (save_dir / "imu_stats.json").write_text(
            json.dumps({"imu_mean": imu_mean.tolist(), "imu_std": imu_std.tolist()}, indent=2)
        )

    if capabilities.fitted_input_stats:
        fit_hybrid_encoder_stats(model, train_ds, batch_size=min(int(T.BATCH_SIZE), 64))

    val_count = len(val_ds) if val_ds is not None else 0
    print(f"Train windows: {len(train_ds)}, Val windows: {val_count}")
    print(f"Trainable params: {count_trainable_params(model):,}")
    print(f"Backbone cfg name: {cfg_name}")
    print(f"Artifacts directory: {save_dir}")

    epoch_logs = []

    selection_metric = resolve_selection_metric(
        str(getattr(T, "BEST_METRIC", "auto")),
        capabilities,
        has_validation=val_loader is not None,
    )
    best_selection_score = float("-inf")
    best_metric_value = None
    epochs_no_improve = 0
    stopped_epoch = T.EPOCHS
    for epoch in range(1, T.EPOCHS + 1):
        if getattr(model, "domain_classifier", None) is not None:
            if T.DOMAIN_SCHEDULE:
                progress = (epoch - 1) / max(T.EPOCHS - 1, 1)
                alpha = dann_alpha_schedule(progress)
                model.domain_classifier.set_alpha(alpha)
                print(f"[INFO] Epoch {epoch}: domain alpha = {alpha:.4f} (progress={progress:.3f})")
            else:
                model.domain_classifier.set_alpha(T.DOMAIN_ALPHA)

        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_domain_loss = 0.0
        running_id_loss = 0.0
        running_pair_loss = 0.0
        running_acc = 0.0
        steps = 0
        use_domain = T.DOMAIN_LOSS_WEIGHT > 0 and getattr(model, "domain_classifier", None) is not None

        for step, batch in enumerate(train_loader, start=1):
            b = move_to_device(batch, device)
            b["imu"], b["skeleton"] = maybe_augment_inputs(b["imu"], b["skeleton"], cfg)
            forward_kwargs = {"imu": b["imu"], "skeleton": b["skeleton"]}
            if "root_trajectory" in b:
                forward_kwargs["root_trajectory"] = b["root_trajectory"]
            out = model(**forward_kwargs)

            # Optionally shuffle video embeddings to break position bias
            if T.SHUFFLE_VIDEO_IN_BATCH:
                B = out["imu"].shape[0]
                perm = torch.randperm(B, device=device)
                perm_inv = torch.empty_like(perm)
                perm_inv[perm] = torch.arange(B, device=device)

                video_for_loss = out["video"][perm]
                labels_a = perm_inv
                labels_b = perm
            else:
                video_for_loss = out["video"]
                labels_a = None
                labels_b = None

            subject_labels_for_main = subject_labels_from_batch(batch.get("subject"), subject_map, device)
            group_labels_for_main = group_labels_from_batch(batch.get("group_key"), device)
            if contrastive_target == "subject" and subject_labels_for_main is not None:
                main_loss = subject_contrastive_loss(out["imu"], video_for_loss, subject_labels_for_main, loss_fn)
            elif contrastive_target == "window":
                if T.SHUFFLE_VIDEO_IN_BATCH:
                    raise ValueError("SHUFFLE_VIDEO_IN_BATCH is incompatible with window contrastive training.")
                main_loss = window_contrastive_loss(out["imu"], out["video"], group_labels_for_main, loss_fn)
            else:
                main_loss = loss_fn(out["imu"], video_for_loss, labels_a=labels_a, labels_b=labels_b)

            domain_loss = torch.zeros((), device=main_loss.device)
            if use_domain:
                domain_labels = domain_labels_from_batch(b.get("domain"), domain_map, device)
                if domain_labels is not None and domain_labels.shape[0] == out["imu"].shape[0]:
                    domain_logits = model.domain_classifier(out["imu"])
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)

            id_loss = torch.zeros((), device=main_loss.device)
            if id_head_imu is not None and id_head_video is not None:
                subject_labels = subject_labels_from_batch(batch.get("subject"), subject_map, device)
                if subject_labels is not None and subject_labels.shape[0] == out["imu"].shape[0]:
                    id_loss = 0.5 * (
                        F.cross_entropy(id_head_imu(out["imu"]), subject_labels)
                        + F.cross_entropy(id_head_video(out["video"]), subject_labels)
                    )

            pair_loss = torch.zeros((), device=main_loss.device)
            if float(T.PAIR_LOSS_WEIGHT) > 0:
                if getattr(model, "cross_pair_head", None) is not None:
                    pair_loss = cross_pair_window_ce_loss(model, b["imu"], b["skeleton"], group_labels_for_main)
                else:
                    pair_loss = pair_bce_loss(
                        model,
                        out["imu"],
                        out["video"],
                        subject_labels_for_main,
                        group_labels_for_main,
                        pair_loss_target,
                    )
            anti_tie_loss = torch.zeros((), device=main_loss.device)
            if float(getattr(T, "PAIR_ANTI_TIE_WEIGHT", 0.0)) > 0:
                anti_tie_loss = pair_anti_tie_loss_from_model(model, out["imu"], out["video"], group_labels_for_main)

            loss = (
                main_loss
                + T.DOMAIN_LOSS_WEIGHT * domain_loss
                + float(T.ID_LOSS_WEIGHT) * id_loss
                + float(T.PAIR_LOSS_WEIGHT) * pair_loss
                + float(getattr(T, "PAIR_ANTI_TIE_WEIGHT", 0.0)) * anti_tie_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if T.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(model.parameters()) + loss_params + id_params if p.requires_grad],
                    T.MAX_GRAD_NORM,
                )
            optimizer.step()

            if contrastive_target == "subject" and subject_labels_for_main is not None:
                acc = subject_retrieval_top1(out["imu"], video_for_loss, subject_labels_for_main)
            elif contrastive_target == "window":
                acc = window_contrastive_top1(out["imu"], out["video"], group_labels_for_main)
            else:
                acc = retrieval_top1(out["imu"], video_for_loss, labels_a=labels_a, labels_b=labels_b)
            running_loss += float(loss.item())
            running_main_loss += float(main_loss.item())
            running_domain_loss += float(domain_loss.item())
            running_id_loss += float(id_loss.item())
            running_pair_loss += float(pair_loss.item())
            running_acc += acc
            steps += 1

            if step == 1 and T.SHUFFLE_VIDEO_IN_BATCH:
                print(f"[DIAG] Shuffle ON | perm head={perm[:5].tolist()} | top1={acc:.4f} | random_exp={1.0/B:.4f}")
            elif step == 1 and not T.SHUFFLE_VIDEO_IN_BATCH:
                print(f"[DIAG] Shuffle OFF | top1={acc:.4f}")

            if step % T.LOG_INTERVAL == 0:
                print(
                    f"[Epoch {epoch}/{T.EPOCHS}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / steps:.4f} main={running_main_loss / steps:.4f} "
                    f"domain={running_domain_loss / steps:.4f} id={running_id_loss / steps:.4f} "
                    f"pair={running_pair_loss / steps:.4f} "
                    f"top1={running_acc / steps:.4f}"
                )
            if T.MAX_STEPS_PER_EPOCH > 0 and step >= T.MAX_STEPS_PER_EPOCH:
                print(
                    f"[SMOKE] Stopping epoch after {step} steps "
                    f"(TRAIN.MAX_STEPS_PER_EPOCH={T.MAX_STEPS_PER_EPOCH})."
                )
                break

        val_metrics = evaluate_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            domain_loss_weight=T.DOMAIN_LOSS_WEIGHT,
            domain_map=domain_map,
            subject_map=subject_map,
            contrastive_target=contrastive_target,
            pair_loss_weight=float(T.PAIR_LOSS_WEIGHT),
            pair_loss_target=pair_loss_target,
            pair_anti_tie_weight=float(getattr(T, "PAIR_ANTI_TIE_WEIGHT", 0.0)),
        )
        train_loss = running_loss / max(steps, 1)
        train_main_loss = running_main_loss / max(steps, 1)
        train_domain_loss = running_domain_loss / max(steps, 1)
        train_id_loss = running_id_loss / max(steps, 1)
        train_pair_loss = running_pair_loss / max(steps, 1)
        train_top1 = running_acc / max(steps, 1)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_main={train_main_loss:.4f} "
            f"train_domain={train_domain_loss:.4f} train_id={train_id_loss:.4f} "
            f"train_pair={train_pair_loss:.4f} train_top1={train_top1:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_main={val_metrics['main_loss']:.4f} "
            f"val_domain={val_metrics['domain_loss']:.4f} val_pair={val_metrics.get('pair_loss', 0.0):.4f} "
            f"val_top1={val_metrics['top1']:.4f}"
        )

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_main_loss": train_main_loss,
                "train_domain_loss": train_domain_loss,
                "train_id_loss": train_id_loss,
                "train_pair_loss": train_pair_loss,
                "train_top1": train_top1,
                "val_loss": val_metrics["loss"],
                "val_main_loss": val_metrics["main_loss"],
                "val_domain_loss": val_metrics["domain_loss"],
                "val_pair_loss": val_metrics.get("pair_loss", 0.0),
                "val_top1": val_metrics["top1"],
            }
        )
        mode = "w" if epoch == 1 else "a"
        with (save_dir / "epoch_metrics.jsonl").open(mode, encoding="utf-8") as f:
            f.write(json.dumps(epoch_logs[-1], ensure_ascii=True) + "\n")

        metric_value, selection_score = selection_value_and_score(selection_metric, val_metrics, train_top1)
        payload = {
            **model_checkpoint_metadata(cfg_name, model),
            "epoch": epoch,
            "config": cfg.dump(sort_keys=False),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_top1": val_metrics["top1"],
            "selection_metric": selection_metric,
            "selection_value": metric_value,
            "selection_score": selection_score,
        }
        torch.save(payload, save_dir / "last.pt")
        if T.SAVE_EVERY_EPOCH:
            torch.save(payload, save_dir / f"epoch_{epoch:03d}.pt")

        if selection_score > best_selection_score + T.EARLY_STOP_MIN_DELTA:
            best_selection_score = selection_score
            best_metric_value = metric_value
            epochs_no_improve = 0
            torch.save(payload, save_dir / "best.pt")
        else:
            epochs_no_improve += 1

        if T.EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= T.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
            stopped_epoch = epoch
            break

    metrics = {
        "selection_metric": selection_metric,
        "best_metric_value": best_metric_value,
        "best_selection_score": best_selection_score,
        "stopped_epoch": stopped_epoch,
        "save_dir": str(save_dir),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"Training complete. Best {selection_metric}={best_metric_value:.4f} "
        f"(selection score={best_selection_score:.4f}, stopped at epoch {stopped_epoch})"
    )


if __name__ == "__main__":
    main()
