"""Common utilities for training and evaluation engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.config import get_cfg_defaults
from src.modules.encoders import HybridIMUEncoder, HybridSkeletonEncoder
from src.modules.matchers import IMUVideoMatcher, SymmetricInfoNCE


def _adapt_legacy_hybrid_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy hybrid experiment checkpoint keys to production matcher keys."""
    adapted: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "log_temp":
            continue
        if key.startswith("skel."):
            adapted[f"video_encoder.{key[len('skel.'):]}"] = value
        elif key.startswith("imu."):
            adapted[f"imu_encoder.raw.{key[len('imu.'):]}"] = value
        else:
            adapted[key] = value
    return adapted


def _load_init_checkpoint(model: IMUVideoMatcher, checkpoint: str) -> None:
    init_path = Path(checkpoint).expanduser()
    if not init_path.exists():
        raise FileNotFoundError(f"INIT_ALIGNMENT_CKPT not found: {init_path}")
    raw = torch.load(str(init_path), map_location="cpu")
    init_state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    init_state = _adapt_legacy_hybrid_state_dict(init_state)
    if isinstance(raw, dict) and isinstance(raw.get("stats"), dict):
        stats = raw["stats"]
        stat_map = {
            "raw_mu": "video_encoder.raw_mu",
            "raw_sd": "video_encoder.raw_sd",
            "vec_mu": "video_encoder.vec_mu",
            "vec_sd": "video_encoder.vec_sd",
            "imu_mu": "imu_encoder.imu_mu",
            "imu_sd": "imu_encoder.imu_sd",
        }
        for src_key, dst_key in stat_map.items():
            if src_key in stats:
                init_state[dst_key] = torch.as_tensor(stats[src_key], dtype=torch.float32)
    model_state = model.state_dict()
    init_state = {
        key: value
        for key, value in init_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    print(
        f"Loaded INIT_ALIGNMENT_CKPT: {init_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def build_alignment_model_from_cfg(
    cfg: Any,
    device: torch.device,
) -> Tuple[IMUVideoMatcher, str]:
    """Build the official hybrid IMU-video alignment model."""
    model_cfg = cfg.TRAIN.MODEL
    model_type = str(getattr(model_cfg, "TYPE", "hybrid")).lower()
    if model_type != "hybrid":
        raise ValueError(
            f"Unsupported TRAIN.MODEL.TYPE={model_cfg.TYPE!r}. "
            "The production codebase currently supports only the hybrid encoder."
        )

    hidden = int(model_cfg.HYBRID_HIDDEN)
    imu_encoder = HybridIMUEncoder(
        hidden_size=hidden,
        imu_smooth_kernel=int(model_cfg.HYBRID_IMU_SMOOTH),
        feature_mode=str(model_cfg.HYBRID_IMU_FEATURE_MODE),
        temporal_layers=int(model_cfg.HYBRID_TEMPORAL_LAYERS),
        temporal_kernel=int(model_cfg.HYBRID_TEMPORAL_KERNEL),
        temporal_mode=str(model_cfg.HYBRID_TEMPORAL_MODE),
        dropout=float(model_cfg.HYBRID_DROPOUT),
    )
    video_encoder = HybridSkeletonEncoder(
        hidden_size=hidden,
        skeleton_smooth_kernel=int(model_cfg.HYBRID_SKELETON_SMOOTH),
        image_height=float(model_cfg.HYBRID_IMAGE_HEIGHT),
        image_width=float(model_cfg.HYBRID_IMAGE_WIDTH),
        token_layers=int(model_cfg.HYBRID_TOKEN_LAYERS),
        token_heads=int(model_cfg.HYBRID_TOKEN_HEADS),
        temporal_layers=int(model_cfg.HYBRID_TEMPORAL_LAYERS),
        temporal_kernel=int(model_cfg.HYBRID_TEMPORAL_KERNEL),
        temporal_mode=str(model_cfg.HYBRID_TEMPORAL_MODE),
        feature_mode=str(model_cfg.HYBRID_SKELETON_FEATURE_MODE),
        dropout=float(model_cfg.HYBRID_DROPOUT),
    )
    model = IMUVideoMatcher(
        imu_encoder=imu_encoder,
        video_encoder=video_encoder,
        num_domains=int(model_cfg.NUM_DOMAINS),
        domain_hidden_dim=int(model_cfg.DOMAIN_HIDDEN_DIM),
        pair_head=bool(model_cfg.PAIR_HEAD),
        pair_hidden_dim=int(model_cfg.PAIR_HIDDEN_DIM),
        cross_pair_head=bool(model_cfg.CROSS_PAIR_HEAD),
        cross_pair_hidden_dim=int(model_cfg.CROSS_PAIR_HIDDEN_DIM),
    ).to(device)

    if model_cfg.INIT_ALIGNMENT_CKPT:
        _load_init_checkpoint(model, str(model_cfg.INIT_ALIGNMENT_CKPT))
    return model, "hybrid"


def build_alignment_model(
    args: Any,
    device: torch.device,
    embed_dim: int = 128,
) -> Tuple[IMUVideoMatcher, str]:
    """Compatibility wrapper for legacy scripts; still builds only hybrid."""
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.MODEL.TYPE = "hybrid"
    cfg.TRAIN.MODEL.HYBRID_HIDDEN = int(getattr(args, "hybrid_hidden", embed_dim))
    cfg.TRAIN.MODEL.NUM_DOMAINS = int(getattr(args, "num_domains", 0))
    cfg.TRAIN.MODEL.DOMAIN_HIDDEN_DIM = int(getattr(args, "domain_hidden_dim", 256))
    init_ckpt = str(getattr(args, "init_alignment_ckpt", "") or "")
    if init_ckpt:
        cfg.TRAIN.MODEL.INIT_ALIGNMENT_CKPT = init_ckpt
    cfg.freeze()
    return build_alignment_model_from_cfg(cfg, device)


def _get_backbone_from_video_encoder(video_encoder):
    """Hybrid encoders do not expose a separately frozen backbone."""
    if isinstance(video_encoder, HybridSkeletonEncoder):
        return None
    raise AttributeError(f"Unsupported video encoder: {type(video_encoder).__name__}")


def build_optimizer(
    model: IMUVideoMatcher,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build AdamW for all trainable hybrid model parameters."""
    del lr_backbone
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW([{"params": params, "lr": lr_heads}], weight_decay=weight_decay)


def build_loss_fn(
    temperature: float = 0.1,
    learn_temperature: bool = False,
    device: torch.device = torch.device("cpu"),
) -> SymmetricInfoNCE:
    """Build InfoNCE loss function."""
    return SymmetricInfoNCE(temperature=temperature, learn_temperature=learn_temperature).to(device)
