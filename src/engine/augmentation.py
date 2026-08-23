"""Training-time input augmentation helpers."""

from __future__ import annotations

import torch


def maybe_augment_inputs(imu: torch.Tensor, skeleton: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.TRAIN.IMU_NOISE_STD > 0:
        imu = imu + torch.randn_like(imu) * cfg.TRAIN.IMU_NOISE_STD

    if cfg.TRAIN.IMU_DROPOUT_PROB > 0:
        feat_keep = (torch.rand(imu.shape[0], 1, imu.shape[2], device=imu.device) > cfg.TRAIN.IMU_DROPOUT_PROB).float()
        imu = imu * feat_keep

    if cfg.TRAIN.SKEL_NOISE_STD > 0:
        skeleton = skeleton + torch.randn_like(skeleton) * cfg.TRAIN.SKEL_NOISE_STD

    if cfg.TRAIN.JOINT_DROPOUT_PROB > 0:
        if skeleton.ndim == 4:
            joint_keep = (
                torch.rand(skeleton.shape[0], 1, skeleton.shape[2], 1, device=skeleton.device)
                > cfg.TRAIN.JOINT_DROPOUT_PROB
            ).float()
        elif skeleton.ndim == 3 and skeleton.shape[-1] % 3 == 0:
            joints = skeleton.shape[-1] // 3
            joint_keep = (
                torch.rand(skeleton.shape[0], 1, joints, 1, device=skeleton.device)
                > cfg.TRAIN.JOINT_DROPOUT_PROB
            ).float().repeat_interleave(3, dim=-1).reshape(skeleton.shape[0], 1, -1)
        else:
            raise ValueError(
                "Joint dropout expects skeleton [B,T,J,3] or flattened [B,T,J*3], "
                f"got {tuple(skeleton.shape)}"
            )
        skeleton = skeleton * joint_keep

    return imu, skeleton
