"""Common utilities for training and evaluation engines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.modules.encoders import (
    IMUEncoder, PhysicsIMUEncoder, VideoEncoder,
    GlobalMotionEncoder, GlobalVideoEncoder,
)
from src.modules.encoders.utils import (
    build_motionbert_backbone,
    load_motionbert_checkpoint,
    load_despite_imu_weights,
)
from src.modules.matchers import IMUVideoMatcher, GlobalIMUVideoMatcher, SymmetricInfoNCE


def build_alignment_model(
    args: Any,
    device: torch.device,
    embed_dim: int = 512,
) -> Tuple[IMUVideoMatcher, str]:
    """Build IMU-Video alignment model from CLI/config args.

    Returns:
        model: Assembled IMUVideoMatcher on device
        cfg_name: MotionBERT config name for logging
    """
    motionbert_root = Path(args.motionbert_root).expanduser().resolve()
    if str(motionbert_root) not in sys.path:
        sys.path.insert(0, str(motionbert_root))

    config_path = Path(args.motionbert_config)
    if not config_path.is_absolute():
        config_path = motionbert_root / config_path

    ckpt_path = Path(args.motionbert_ckpt) if getattr(args, "motionbert_ckpt", "") else None
    if ckpt_path is not None and not ckpt_path.is_absolute():
        ckpt_path = motionbert_root / ckpt_path

    backbone, cfg = build_motionbert_backbone(str(config_path))
    skip_motionbert_ckpt = getattr(args, "skip_motionbert_ckpt", False)
    if not skip_motionbert_ckpt:
        if ckpt_path is None:
            raise ValueError("--motionbert_ckpt is required unless --skip_motionbert_ckpt is set.")
        load_motionbert_checkpoint(backbone, str(ckpt_path), strict=True)
    else:
        print("[WARN] skip_motionbert_ckpt enabled: using randomly initialized MotionBERT backbone.")

    imu_encoder_type = getattr(args, "imu_encoder_type", "lstm")
    if imu_encoder_type == "physics":
        imu_encoder = PhysicsIMUEncoder(
            d_model=getattr(args, "physics_d_model", 128),
            n_heads=getattr(args, "physics_n_heads", 4),
            num_layers=getattr(args, "physics_num_layers", 3),
            embed_dim=embed_dim,
            fs_hz=getattr(args, "physics_fs_hz", 30.0),
            n_fft=getattr(args, "physics_n_fft", 64),
            dropout=getattr(args, "physics_dropout", 0.1),
        )
        print(f"[INFO] Using PhysicsIMUEncoder (d_model={getattr(args, 'physics_d_model', 128)}, "
              f"layers={getattr(args, 'physics_num_layers', 3)})")
    else:
        adapter_type = getattr(args, "adapter_type", None)
        imu_encoder = IMUEncoder(
            input_size=48, hidden_size=embed_dim, num_layers=2,
            device=str(device), adapter_type=adapter_type,
        )
        imu_ckpt = getattr(args, "imu_ckpt", "")
        if imu_ckpt:
            imu_ckpt_path = Path(imu_ckpt).expanduser()
            if imu_ckpt_path.exists():
                load_despite_imu_weights(imu_encoder, str(imu_ckpt_path), strict=False)
            else:
                print(f"[WARN] IMU checkpoint not found at {imu_ckpt_path}; using random init.")

    use_global_motion = getattr(args, "use_global_motion", False)
    if use_global_motion:
        global_only_mode = bool(getattr(args, "global_motion_train_only", False))
        local_encoder = VideoEncoder(backbone=backbone, rep_dim=embed_dim, temporal_layers=2)
        global_encoder = GlobalMotionEncoder(
            input_dim=getattr(args, "global_motion_input_dim", 2),
            hidden_dim=getattr(args, "global_motion_hidden_dim", 64),
            num_layers=getattr(args, "global_motion_num_layers", 2),
            embed_dim=embed_dim,
            dropout=getattr(args, "global_motion_dropout", 0.1),
            input_type=getattr(args, "global_motion_input_type", "diff_raw"),
            fusion_proj=True if global_only_mode else getattr(args, "global_motion_fusion_proj", True),
        )
        video_encoder = GlobalVideoEncoder(
            local_encoder=local_encoder,
            global_encoder=global_encoder,
            embed_dim=embed_dim,
            fusion_type=getattr(args, "global_motion_fusion_type", "concat"),
        )
        model = GlobalIMUVideoMatcher(imu_encoder=imu_encoder, video_encoder=video_encoder).to(device)
        if global_only_mode:
            model.global_only = True
            for p in model.imu_encoder.parameters():
                p.requires_grad = False
            for p in model.video_encoder.local_encoder.parameters():
                p.requires_grad = False
            for p in model.video_encoder.fusion.parameters():
                p.requires_grad = False
            for p in model.video_encoder.global_encoder.parameters():
                p.requires_grad = True
        print(
            f"[INFO] Global motion enabled: input_dim={global_encoder.input_dim}, "
            f"hidden_dim={getattr(args, 'global_motion_hidden_dim', 64)}, "
            f"fusion={getattr(args, 'global_motion_fusion_type', 'concat')}, "
            f"global_only={getattr(args, 'global_motion_train_only', False)}"
        )
    else:
        video_encoder = VideoEncoder(backbone=backbone, rep_dim=embed_dim, temporal_layers=2)
        model = IMUVideoMatcher(imu_encoder=imu_encoder, video_encoder=video_encoder).to(device)

    init_alignment_ckpt = getattr(args, "init_alignment_ckpt", "")
    if init_alignment_ckpt:
        init_path = Path(init_alignment_ckpt).expanduser()
        if not init_path.exists():
            raise FileNotFoundError(f"init_alignment_ckpt not found: {init_path}")
        raw = torch.load(str(init_path), map_location="cpu")
        init_state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            f"Loaded init_alignment_ckpt: {init_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    return model, getattr(cfg, "name", "unknown")


def _get_backbone_from_video_encoder(video_encoder):
    """Extract backbone from VideoEncoder or GlobalVideoEncoder."""
    if hasattr(video_encoder, "backbone"):
        return video_encoder.backbone
    elif hasattr(video_encoder, "local_encoder") and hasattr(video_encoder.local_encoder, "backbone"):
        return video_encoder.local_encoder.backbone
    else:
        raise AttributeError(f"Cannot find backbone in {type(video_encoder).__name__}")


def build_optimizer(
    model: IMUVideoMatcher,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with separate LRs for backbone and heads."""
    backbone = _get_backbone_from_video_encoder(model.video_encoder)
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = []
    head_params += [p for p in model.imu_encoder.parameters() if p.requires_grad]
    if hasattr(model.video_encoder, "joint_compress"):
        head_params += [p for p in model.video_encoder.joint_compress.parameters() if p.requires_grad]
        head_params += [p for p in model.video_encoder.temporal_lstm.parameters() if p.requires_grad]
    elif hasattr(model.video_encoder, "local_encoder"):
        head_params += [p for p in model.video_encoder.local_encoder.joint_compress.parameters() if p.requires_grad]
        head_params += [p for p in model.video_encoder.local_encoder.temporal_lstm.parameters() if p.requires_grad]
        head_params += [p for p in model.video_encoder.global_encoder.parameters() if p.requires_grad]
        if hasattr(model.video_encoder, "fusion"):
            head_params += [p for p in model.video_encoder.fusion.parameters() if p.requires_grad]
        if hasattr(model.video_encoder, "film"):
            head_params += [p for p in model.video_encoder.film.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ],
        weight_decay=weight_decay,
    )


def build_loss_fn(
    temperature: float = 0.1,
    learn_temperature: bool = False,
    device: torch.device = torch.device("cpu"),
) -> SymmetricInfoNCE:
    """Build InfoNCE loss function."""
    return SymmetricInfoNCE(temperature=temperature, learn_temperature=learn_temperature).to(device)
