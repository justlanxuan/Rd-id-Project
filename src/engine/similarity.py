"""Shared model-to-similarity helpers for evaluation engines."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def model_similarity_matrix(
    model,
    imu_emb: torch.Tensor,
    video_emb: torch.Tensor,
    cosine_weight: float = 0.0,
    pair_logit_weight: float = 1.0,
) -> np.ndarray:
    cosine = imu_emb @ video_emb.t()
    if getattr(model, "pair_head", None) is None:
        return cosine.detach().cpu().numpy()
    pair_logits = model.pair_logits(imu_emb, video_emb)
    combined = float(cosine_weight) * cosine + float(pair_logit_weight) * pair_logits
    return combined.detach().cpu().numpy()


def encode_hybrid_precomputed(
    model,
    raw: torch.Tensor,
    vec: torch.Tensor,
    imu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_encoder = model.video_encoder
    imu_encoder = model.imu_encoder
    raw = (raw - video_encoder.raw_mu) / video_encoder.raw_sd.clamp_min(1e-6)
    vec = (vec - video_encoder.vec_mu) / video_encoder.vec_sd.clamp_min(1e-6)
    imu = (imu - imu_encoder.imu_mu) / imu_encoder.imu_sd.clamp_min(1e-6)
    video = video_encoder.fuse(torch.cat([video_encoder.raw(raw), video_encoder.vec(vec)], dim=1))
    video = F.normalize(video, dim=1)
    imu_emb = F.normalize(imu_encoder.raw(imu), dim=1)
    return imu_emb, video


__all__ = ["encode_hybrid_precomputed", "model_similarity_matrix"]
