#!/usr/bin/env python3
"""Diagnose why accuracy remains high even with shuffle_video_in_batch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.engine.common import build_alignment_model
from src.modules.matchers import retrieval_top1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/interim/totalcapture_vicon/slice/windows_train.csv")
    p.add_argument("--data_root", type=str, default="data/interim/totalcapture_vicon/slice")
    p.add_argument("--motionbert_root", type=str, default="/home/fzliang/MotionBERT")
    p.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    p.add_argument("--motionbert_ckpt", type=str, default="checkpoint/pretrain/MB_lite_models.bin")
    p.add_argument("--imu_ckpt", type=str, default="/home/fzliang/despite/pretrained_models/v2/SIE_v2.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imu_sensor", type=str, default="R_LowArm")
    p.add_argument("--repeat_single_sensor", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=3, help="How many batches to diagnose")
    return p.parse_args()


def diagnose_batch(batch, model, device, batch_idx):
    """Run a single batch through diagnosis."""
    imu = batch["imu"].to(device)
    skeleton = batch["skeleton"].to(device)
    out = model(imu=imu, skeleton=skeleton)

    B = out["imu"].shape[0]

    # 1. Unshuffled (diagonal) accuracy
    acc_unshuffled = retrieval_top1(out["imu"], out["video"])

    # 2. Shuffled accuracy
    perm = torch.randperm(B, device=device)
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(B, device=device)
    video_shuffled = out["video"][perm]
    acc_shuffled = retrieval_top1(out["imu"], video_shuffled, labels_a=perm_inv, labels_b=perm)

    # 3. Random baseline (expected accuracy for pure guess)
    random_baseline = 100.0 / B  # percentage

    # 4. Check if batch is homogeneous (all same subject/session)
    subjects = batch.get("subject", [])
    sessions = batch.get("session", [])
    unique_subjects = set(subjects)
    unique_sessions = set(sessions)

    # 5. Embedding similarity analysis
    z_imu = F.normalize(out["imu"], dim=-1).detach().cpu()
    z_vid = F.normalize(out["video"], dim=-1).detach().cpu()
    sims = torch.matmul(z_imu, z_vid.t())  # [B, B]

    diag_sim = torch.diag(sims).mean().item()
    offdiag_mask = ~torch.eye(B, dtype=torch.bool)
    offdiag_sim = sims[offdiag_mask].mean().item()
    separation = diag_sim - offdiag_sim

    # 6. Top-1 without diagonal (check if model uses position bias)
    # If we zero out the diagonal, does accuracy crash?
    sims_nodiag = sims.clone()
    sims_nodiag.fill_diagonal_(-999)
    top1_nodiag = (sims_nodiag.argmax(dim=1) == torch.arange(B)).float().mean().item()

    print(f"\n{'='*70}")
    print(f"Batch {batch_idx} | B={B}")
    print(f"Subjects: {unique_subjects} | Sessions: {unique_sessions}")
    print(f"Unshuffled top1: {acc_unshuffled*100:.1f}%")
    print(f"Shuffled top1:   {acc_shuffled*100:.1f}%")
    print(f"Random baseline: {random_baseline:.1f}%")
    print(f"No-diagonal top1:{top1_nodiag*100:.1f}%")
    print(f"Diag sim: {diag_sim:.4f} | Off-diag sim: {offdiag_sim:.4f} | Separation: {separation:.4f}")

    if batch_idx == 1:
        print("\n[Interpretation]")
        if acc_unshuffled > 0.8 and acc_shuffled < 0.3:
            print("→ Model has SEVERE position bias (relies on diagonal).")
        elif acc_unshuffled > 0.8 and acc_shuffled > 0.8:
            print("→ Model has NO position bias. High accuracy is from feature quality.")
        elif acc_unshuffled > 0.8 and top1_nodiag < 0.3:
            print("→ Model secretly uses diagonal even when shuffled (bug or hidden bias).")
        elif separation < 0.1:
            print("→ Embeddings are NOT well-separated. High top1 may be spurious.")
        else:
            print("→ Embeddings are well-separated. Model learned real alignment.")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset (same as training)
    ds = WindowAlignmentDataset(
        args.train_csv,
        root_dir=args.data_root,
        imu_sensor=args.imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_mean=None,
        imu_std=None,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model with pretrained weights (same as training)
    model, _ = build_alignment_model(args, device)
    model.eval()

    print(f"Dataset: {len(ds)} windows")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("\nDiagnosing pretrained model BEFORE any training...")

    for i, batch in enumerate(loader, 1):
        if i > args.num_batches:
            break
        diagnose_batch(batch, model, device, i)

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
