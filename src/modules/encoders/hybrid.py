"""Hybrid shoulder-vector encoders.

This encoder pair represents video as raw shoulder-relative pose plus
shoulder-local arm-vector tokens, and IMU as a named, configurable sequence of
channels. The IMU encoder and video encoder live together because they are a
matched representation pair.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.imu import CANONICAL_7D_CHANNELS

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
LEFT_WRIST = 9
LEFT_HIP = 11
RIGHT_HIP = 12

RAW_POSE_DIM = 72
SKELETON_TOKEN_DIM = 12
SKELETON_N_TOKENS = 4
H36M_MOTION_DIM = 17 * 3 * 2
H36M_HIP = 0
H36M_LEFT_SHOULDER = 11
H36M_RIGHT_SHOULDER = 14
H36M_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
H36M_FEATURE_MODES = (
    "h36m2d",
    "h36m3d",
    "h36m3d_no_velocity",
    "h36m3d_bone",
    "h36m3d_heading",
    "h36m3d_heading_rate",
    "h36m3d_rotinv",
    "h36m3d_zonly",
    "h36m3d_geom",
    "h36m3d_left_wrist",
    "h36m3d_right_wrist",
    "h36m3d_both_wrist",
    "h36m3d_left_wrist_rot",
)


def _moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    b, t = x.shape[:2]
    rest = x.shape[2:]
    flat = x.reshape(b, t, -1).transpose(1, 2)
    pad = kernel // 2
    flat = F.pad(flat, (pad, pad), mode="replicate")
    out = F.avg_pool1d(flat, kernel_size=kernel, stride=1)
    return out.transpose(1, 2).reshape(b, t, *rest)


def _diff_same(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    y = x
    for _ in range(order):
        y = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1:] - y[:, :-1]], dim=1)
    return y


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / _norm(x).clamp_min(1e-8).unsqueeze(-1)


def _pose_to_btj2(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim != 4:
        raise ValueError(f"Expected skeleton [B,T,J,C] or [B,T,C,J], got {tuple(pose.shape)}")
    if pose.shape[-2:] == (17, 2) or pose.shape[-2:] == (17, 3):
        return pose[..., :2].float()
    if pose.shape[-2:] == (2, 17) or pose.shape[-2:] == (3, 17):
        return pose[:, :, :2, :].transpose(2, 3).float()
    raise ValueError(f"Unsupported skeleton shape {tuple(pose.shape)}")


def _normalize_pixels(xy: torch.Tensor, image_height: float, image_width: float) -> torch.Tensor:
    scale = xy.new_tensor([image_width, image_height])
    return xy / scale


def _unwrap_time(x: torch.Tensor) -> torch.Tensor:
    """Match numpy.unwrap along the temporal axis."""
    if x.shape[1] <= 1:
        return x
    delta = x[:, 1:] - x[:, :-1]
    two_pi = 2.0 * torch.pi
    correction = torch.zeros_like(delta)
    correction = torch.where(delta > torch.pi, correction - two_pi, correction)
    correction = torch.where(delta < -torch.pi, correction + two_pi, correction)
    return torch.cat([x[:, :1], x[:, 1:] + torch.cumsum(correction, dim=1)], dim=1)


def skeleton_tokens(
    pose: torch.Tensor,
    smooth_kernel: int = 9,
    image_height: float = 1080.0,
    image_width: float = 1920.0,
) -> torch.Tensor:
    xy = _normalize_pixels(_pose_to_btj2(pose).double(), image_height, image_width)
    xy = _moving_average(xy, smooth_kernel)

    lsho = xy[:, :, LEFT_SHOULDER]
    rsho = xy[:, :, RIGHT_SHOULDER]
    elbow = xy[:, :, LEFT_ELBOW]
    wrist = xy[:, :, LEFT_WRIST]
    hip_center = (xy[:, :, LEFT_HIP] + xy[:, :, RIGHT_HIP]) / 2.0
    shoulder_center = (lsho + rsho) / 2.0
    body_center = (hip_center + shoulder_center) / 2.0

    shoulder_axis = rsho - lsho
    scale = _norm(shoulder_axis).clamp_min(0.05)
    ex = shoulder_axis / _norm(shoulder_axis).clamp_min(1e-8).unsqueeze(-1)
    ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)

    def local(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([(v * ex).sum(dim=-1), (v * ey).sum(dim=-1)], dim=-1)

    upper = local(elbow - lsho) / scale.unsqueeze(-1)
    fore = local(wrist - elbow) / scale.unsqueeze(-1)
    wrist_rel = local(wrist - lsho) / scale.unsqueeze(-1)
    center_rel = local(body_center - lsho) / scale.unsqueeze(-1)
    shoulder_abs = lsho
    center_vel = _diff_same(center_rel, 1)

    upper_len = _norm(upper)
    fore_len = _norm(fore)
    upper_ang = _unwrap_time(torch.atan2(upper[..., 1], upper[..., 0]))
    fore_ang = _unwrap_time(torch.atan2(fore[..., 1], fore[..., 0]))
    rel_rot = _unwrap_time(fore_ang - upper_ang)
    rel_rot_vel = _diff_same(rel_rot.unsqueeze(-1), 1).squeeze(-1)

    b, t = pose.shape[:2]
    tokens = xy.new_zeros((b, t, SKELETON_N_TOKENS, SKELETON_TOKEN_DIM))

    def fill_bone(slot: int, vec: torch.Tensor, rel: torch.Tensor | None) -> None:
        vel = _diff_same(vec, 1)
        acc = _diff_same(vec, 2)
        tokens[:, :, slot, 0:2] = _unit(vec)
        tokens[:, :, slot, 2] = _norm(vec)
        tokens[:, :, slot, 3:5] = vel
        tokens[:, :, slot, 5] = _norm(acc)
        if rel is not None:
            tokens[:, :, slot, 6] = rel
        tokens[:, :, slot, 7] = 1.0

    fill_bone(0, upper, None)
    fill_bone(1, fore, rel_rot_vel)

    upper_u = _unit(upper)
    fore_u = _unit(fore)
    dot = (upper_u * fore_u).sum(dim=-1)
    cross = upper_u[..., 0] * fore_u[..., 1] - upper_u[..., 1] * fore_u[..., 0]
    closure = wrist_rel - (upper + fore)
    tokens[:, :, 2, 0] = dot
    tokens[:, :, 2, 1] = cross
    tokens[:, :, 2, 2] = torch.cos(rel_rot)
    tokens[:, :, 2, 3] = torch.sin(rel_rot)
    # Joint dropout and missing detections can collapse one bone while leaving
    # the other nonzero.  The raw ratio then reaches millions and may keep the
    # forward pass finite while producing NaN gradients in attention backward.
    # A log-ratio is symmetric, informative, and bounded for degenerate bones.
    tokens[:, :, 2, 4] = torch.log(
        (upper_len.clamp_min(1e-4) / fore_len.clamp_min(1e-4)).clamp(1e-3, 1e3)
    )
    tokens[:, :, 2, 5:7] = closure
    tokens[:, :, 2, 7] = _norm(closure)
    tokens[:, :, 2, 8:10] = wrist_rel
    tokens[:, :, 2, 10] = rel_rot_vel.abs()
    tokens[:, :, 2, 11] = 1.0

    tokens[:, :, 3, 0:2] = shoulder_abs
    tokens[:, :, 3, 2:4] = center_rel
    tokens[:, :, 3, 4:6] = center_vel
    tokens[:, :, 3, 6] = scale
    tokens[:, :, 3, 7] = 1.0
    return tokens.float()


def raw_pose_sequence(
    pose: torch.Tensor,
    smooth_kernel: int = 9,
    image_height: float = 1080.0,
    image_width: float = 1920.0,
) -> torch.Tensor:
    xy = _normalize_pixels(_pose_to_btj2(pose).double(), image_height, image_width)
    xy = _moving_average(xy, smooth_kernel)
    shoulder = xy[:, :, LEFT_SHOULDER]
    hip_center = (xy[:, :, LEFT_HIP] + xy[:, :, RIGHT_HIP]) / 2.0
    rel = xy - shoulder[:, :, None, :]
    rel_vel = _diff_same(rel, 1)
    context = torch.cat([shoulder, hip_center - shoulder], dim=-1)
    out = torch.cat([rel.reshape(rel.shape[0], rel.shape[1], -1), rel_vel.reshape(rel.shape[0], rel.shape[1], -1), context], dim=-1)
    if out.shape[-1] != RAW_POSE_DIM:
        raise ValueError(f"Unexpected raw pose dim {out.shape}")
    return out.float()


def h36m_motion_sequence(
    pose: torch.Tensor,
    smooth_kernel: int = 9,
    *,
    include_depth: bool = True,
) -> torch.Tensor:
    """Build scale-normalized H36M-17 positions and velocities.

    ``h36m3d`` and ``h36m2d`` deliberately share this 102-D contract and the
    same temporal tower.  The 2-D control only zeros depth before centering;
    this keeps architecture/capacity fixed while testing the 3-D coordinate.
    """
    if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
        raise ValueError(f"Expected H36M skeleton [B,T,17,3], got {tuple(pose.shape)}")
    xyz = pose.double()
    if not include_depth:
        xyz = xyz.clone()
        xyz[..., 2] = 0.0
    xyz = xyz - xyz[:, :, H36M_HIP : H36M_HIP + 1]
    shoulder_axis = (
        xyz[:, :, H36M_RIGHT_SHOULDER] - xyz[:, :, H36M_LEFT_SHOULDER]
    )
    scale = _norm(shoulder_axis).clamp_min(1e-4)
    xyz = xyz / scale[:, :, None, None]
    xyz = _moving_average(xyz, smooth_kernel)
    velocity = _diff_same(xyz, 1)
    output = torch.cat(
        [xyz.reshape(*xyz.shape[:2], -1), velocity.reshape(*velocity.shape[:2], -1)],
        dim=-1,
    )
    if output.shape[-1] != H36M_MOTION_DIM:
        raise ValueError(f"Unexpected H36M motion feature shape {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("H36M motion features contain non-finite values")
    return output.float()


def h36m_feature_dim(feature_mode: str) -> int:
    """Return the fixed input width for a profiled H36M feature family."""
    mode = str(feature_mode).lower()
    if mode in {"h36m2d", "h36m3d", "h36m3d_no_velocity", "h36m3d_bone", "h36m3d_rotinv", "h36m3d_zonly"}:
        return H36M_MOTION_DIM
    if mode == "h36m3d_heading_rate":
        return H36M_MOTION_DIM + 2
    if mode == "h36m3d_heading":
        return H36M_MOTION_DIM + 4
    if mode == "h36m3d_geom":
        return H36M_MOTION_DIM + 17
    if mode in {"h36m3d_left_wrist", "h36m3d_right_wrist", "h36m3d_left_wrist_rot"}:
        return H36M_MOTION_DIM + 6
    if mode == "h36m3d_both_wrist":
        return H36M_MOTION_DIM + 12
    raise ValueError(f"Unsupported H36M feature mode={feature_mode!r}")


def _h36m_normalized_xyz(
    pose: torch.Tensor,
    smooth_kernel: int,
    *,
    include_depth: bool,
) -> torch.Tensor:
    if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
        raise ValueError(f"Expected H36M skeleton [B,T,17,3], got {tuple(pose.shape)}")
    xyz = pose.double()
    if not include_depth:
        xyz = xyz.clone()
        xyz[..., 2] = 0.0
    xyz = xyz - xyz[:, :, H36M_HIP : H36M_HIP + 1]
    shoulder_axis = xyz[:, :, H36M_RIGHT_SHOULDER] - xyz[:, :, H36M_LEFT_SHOULDER]
    scale = _norm(shoulder_axis).clamp_min(1e-4)
    xyz = xyz / scale[:, :, None, None]
    return _moving_average(xyz, smooth_kernel)


def _h36m_heading_features(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return periodic torso heading and a validity mask in the x/z plane."""
    lateral = xyz[:, :, H36M_RIGHT_SHOULDER] - xyz[:, :, H36M_LEFT_SHOULDER]
    up = xyz[:, :, 8] - xyz[:, :, H36M_HIP]
    forward = torch.cross(lateral, up, dim=-1)
    horizontal = forward[..., (0, 2)]
    valid = _norm(horizontal) > 1e-5
    heading = torch.atan2(horizontal[..., 1], horizontal[..., 0])
    heading = torch.where(valid, heading, torch.zeros_like(heading))
    unwrapped = _unwrap_time(heading)
    rate = _diff_same(unwrapped.unsqueeze(-1), 1).squeeze(-1)
    sin_cos = torch.stack([torch.sin(heading), torch.cos(heading)], dim=-1)
    sin_cos = torch.where(valid.unsqueeze(-1), sin_cos, torch.zeros_like(sin_cos))
    return torch.cat([sin_cos, rate.unsqueeze(-1)], dim=-1), valid.float().unsqueeze(-1)


def _h36m_bone_vectors(xyz: torch.Tensor) -> torch.Tensor:
    vectors = torch.zeros_like(xyz)
    for joint, parent in enumerate(H36M_PARENTS):
        if parent >= 0:
            vectors[:, :, joint] = xyz[:, :, joint] - xyz[:, :, parent]
    return vectors


def _h36m_wrist_features(xyz: torch.Tensor, *, side: str, torso_relative: bool = False) -> torch.Tensor:
    """Return a 6-D forearm direction/rotation proxy for one wrist.

    H4W++'s current exported cache contains H36M-17 body joints, not MANO
    finger joints.  This therefore measures 3-D forearm/wrist direction and
    frame-to-frame rotation, not a full wrist-local hand pose.
    """
    if side == "left":
        elbow, wrist = 12, 13
    elif side == "right":
        elbow, wrist = 15, 16
    else:
        raise ValueError(f"Unsupported wrist side={side!r}")
    vector = xyz[:, :, wrist] - xyz[:, :, elbow]
    direction = _unit(vector)
    if torso_relative:
        lateral = _unit(xyz[:, :, H36M_RIGHT_SHOULDER] - xyz[:, :, H36M_LEFT_SHOULDER])
        up = _unit(xyz[:, :, 8] - xyz[:, :, H36M_HIP])
        forward = _unit(torch.cross(lateral, up, dim=-1))
        direction = torch.stack(
            [
                (direction * lateral).sum(dim=-1),
                (direction * up).sum(dim=-1),
                (direction * forward).sum(dim=-1),
            ],
            dim=-1,
        )
    previous = torch.cat([direction[:, :1], direction[:, :-1]], dim=1)
    rotation = torch.cross(previous, direction, dim=-1)
    return torch.cat([direction, rotation], dim=-1)


def h36m_feature_sequence(
    pose: torch.Tensor,
    smooth_kernel: int = 9,
    feature_mode: str = "h36m3d",
) -> torch.Tensor:
    """Build one controlled 3-D skeleton feature family for G13 profiling.

    All position/velocity families use the same H36M-17 layout, pelvis origin,
    shoulder-width scale and temporal smoothing.  ``h36m2d`` zeros depth before
    feature construction; the remaining modes isolate velocity, bone vectors,
    global heading, heading rate, heading removal and static geometry.
    """
    mode = str(feature_mode).lower()
    xyz = _h36m_normalized_xyz(
        pose,
        smooth_kernel,
        include_depth=mode != "h36m2d",
    )
    if mode == "h36m3d_zonly":
        xyz = xyz.clone()
        xyz[..., :2] = 0.0
    if mode == "h36m3d_rotinv":
        heading, valid = _h36m_heading_features(xyz)
        angle = torch.atan2(heading[..., 0], heading[..., 1])
        c, s = torch.cos(angle), torch.sin(angle)
        x, z = xyz[..., 0].clone(), xyz[..., 2].clone()
        xyz = xyz.clone()
        xyz[..., 0] = c.unsqueeze(-1) * x + s.unsqueeze(-1) * z
        xyz[..., 2] = -s.unsqueeze(-1) * x + c.unsqueeze(-1) * z
        xyz = torch.where(valid.unsqueeze(-1).bool(), xyz, _h36m_normalized_xyz(pose, smooth_kernel, include_depth=True))
    if mode == "h36m3d_bone":
        xyz = _h36m_bone_vectors(xyz)
    velocity = _diff_same(xyz, 1)
    if mode == "h36m3d_no_velocity":
        velocity = torch.zeros_like(velocity)
    output = [xyz.reshape(*xyz.shape[:2], -1), velocity.reshape(*velocity.shape[:2], -1)]
    if mode == "h36m3d_heading":
        heading, valid = _h36m_heading_features(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True))
        output.append(torch.cat([heading, valid], dim=-1))
    elif mode == "h36m3d_heading_rate":
        heading, valid = _h36m_heading_features(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True))
        output.append(torch.cat([heading[..., 2:3], valid], dim=-1))
    elif mode == "h36m3d_geom":
        bones = _h36m_bone_vectors(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True))
        output.append(_norm(bones, dim=-1))
    elif mode == "h36m3d_left_wrist":
        output.append(_h36m_wrist_features(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True), side="left"))
    elif mode == "h36m3d_right_wrist":
        output.append(_h36m_wrist_features(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True), side="right"))
    elif mode == "h36m3d_both_wrist":
        base = _h36m_normalized_xyz(pose, smooth_kernel, include_depth=True)
        output.append(torch.cat([_h36m_wrist_features(base, side="left"), _h36m_wrist_features(base, side="right")], dim=-1))
    elif mode == "h36m3d_left_wrist_rot":
        output.append(_h36m_wrist_features(_h36m_normalized_xyz(pose, smooth_kernel, include_depth=True), side="left", torso_relative=True))
    result = torch.cat(output, dim=-1)
    expected = h36m_feature_dim(mode)
    if result.shape[-1] != expected:
        raise ValueError(f"Unexpected H36M feature shape {tuple(result.shape)} for {mode}; expected={expected}")
    if not torch.isfinite(result).all():
        raise ValueError(f"H36M features contain non-finite values for mode={mode}")
    return result.float()


def _canonical_7d_indices(
    channel_names: Sequence[str] | None, width: int
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]] | None:
    if width != len(CANONICAL_7D_CHANNELS):
        return None
    names = CANONICAL_7D_CHANNELS if channel_names is None else tuple(channel_names)
    if set(names) != set(CANONICAL_7D_CHANNELS) or len(names) != len(CANONICAL_7D_CHANNELS):
        return None
    positions = {name: index for index, name in enumerate(names)}
    return (
        tuple(positions[name] for name in CANONICAL_7D_CHANNELS[:3]),
        tuple(positions[name] for name in CANONICAL_7D_CHANNELS[3:]),
    )


def _is_canonical_7d(channel_names: Sequence[str] | None, width: int) -> bool:
    return _canonical_7d_indices(channel_names, width) is not None


def _prepare_named_imu(imu: torch.Tensor, smooth_kernel: int, channel_names: Sequence[str] | None) -> torch.Tensor:
    values = imu.double()
    if channel_names is None:
        channel_names = CANONICAL_7D_CHANNELS if values.shape[-1] == 7 else None
    if channel_names is not None and len(channel_names) != values.shape[-1]:
        raise ValueError(
            f"IMU channel count={len(channel_names)} does not match tensor width={values.shape[-1]}"
        )
    values = _moving_average(values, smooth_kernel)
    if channel_names is None:
        return values.float()
    indices = {name: index for index, name in enumerate(channel_names)}
    quaternion_names = ("quat_w", "quat_x", "quat_y", "quat_z")
    if all(name in indices for name in quaternion_names):
        quaternion = values[..., [indices[name] for name in quaternion_names]]
        quaternion = quaternion / _norm(quaternion).clamp_min(1e-8).unsqueeze(-1)
        values = values.clone()
        values[..., [indices[name] for name in quaternion_names]] = quaternion
    return values.float()


def raw_imu_sequence(
    imu: torch.Tensor,
    smooth_kernel: int = 5,
    channel_names: Sequence[str] | None = None,
) -> torch.Tensor:
    if imu.ndim < 2 or imu.shape[-1] <= 0:
        raise ValueError(f"Hybrid IMU encoder expects [B,T,C] with C>0, got {tuple(imu.shape)}")
    return _prepare_named_imu(imu, smooth_kernel, channel_names)


def imu_sequence_features(
    imu: torch.Tensor,
    smooth_kernel: int = 5,
    mode: str = "raw",
    channel_names: Sequence[str] | None = None,
) -> torch.Tensor:
    raw = raw_imu_sequence(imu, smooth_kernel, channel_names)
    mode = str(mode).lower()
    if mode == "raw":
        return raw
    if mode != "dynamic":
        raise ValueError(f"Unsupported hybrid IMU feature mode: {mode}")

    canonical_indices = _canonical_7d_indices(channel_names, raw.shape[-1])
    if canonical_indices is not None:
        acc_idx, quat_idx = canonical_indices
        acc = raw[..., list(acc_idx)]
        quat = raw[..., list(quat_idx)]
        acc_centered = acc - acc.mean(dim=1, keepdim=True)
        acc_delta = _diff_same(acc, 1)
        dots = (quat[:, 1:] * quat[:, :-1]).sum(dim=-1).abs().clamp(0.0, 1.0)
        ang = 2.0 * torch.acos(dots)
        ang_speed = torch.cat([torch.zeros_like(ang[:, :1]), ang], dim=1).unsqueeze(-1)
        return torch.cat([raw, acc_centered, acc_delta, ang_speed], dim=-1).float()

    centered = raw - raw.mean(dim=1, keepdim=True)
    delta = _diff_same(raw, 1)
    return torch.cat([raw, centered, delta], dim=-1).float()


class TemporalConv(nn.Module):
    def __init__(self, hidden: int, layers: int = 2, kernel_size: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        pad = kernel_size // 2
        blocks = []
        for _ in range(layers):
            blocks.append(nn.Conv1d(hidden, hidden, kernel_size, padding=pad))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return x + y


class TokenEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        n_tokens: int,
        hidden: int,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(token_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.type_emb = nn.Parameter(torch.randn(n_tokens, hidden) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, k, _ = x.shape
        h = self.proj(x) + self.type_emb[None, None, :, :]
        cls = self.cls.expand(b * t, -1, -1)
        h = torch.cat([cls, h.reshape(b * t, k, -1)], dim=1)
        h = self.attn(h)[:, 0]
        return h.reshape(b, t, -1)


class RawTower(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        temporal_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.05,
        temporal_mode: str = "gru",
    ) -> None:
        super().__init__()
        self.temporal_mode = str(temporal_mode).lower()
        if self.temporal_mode not in {"gru", "attn", "mean"}:
            raise ValueError(f"Unsupported temporal_mode={temporal_mode!r}")
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.temporal = TemporalConv(hidden, layers=temporal_layers, kernel_size=kernel_size, dropout=dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        if self.temporal_mode == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )
        self.hidden_size = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.pre(x))
        h, last = self.gru(h)
        if self.temporal_mode == "gru":
            return last[-1]
        if self.temporal_mode == "mean":
            return h.mean(dim=1)
        weight = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
        return torch.sum(h * weight.unsqueeze(-1), dim=1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.pre(x))
        h, _ = self.gru(h)
        return h


class VectorTower(nn.Module):
    def __init__(
        self,
        hidden: int,
        temporal_layers: int = 2,
        kernel_size: int = 5,
        token_layers: int = 1,
        token_heads: int = 4,
        dropout: float = 0.1,
        temporal_mode: str = "gru",
    ) -> None:
        super().__init__()
        self.temporal_mode = str(temporal_mode).lower()
        if self.temporal_mode not in {"gru", "attn", "mean"}:
            raise ValueError(f"Unsupported temporal_mode={temporal_mode!r}")
        self.tokens = TokenEncoder(
            SKELETON_TOKEN_DIM,
            SKELETON_N_TOKENS,
            hidden,
            n_heads=token_heads,
            n_layers=token_layers,
            dropout=dropout,
        )
        self.temporal = TemporalConv(hidden, layers=temporal_layers, kernel_size=kernel_size, dropout=dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        if self.temporal_mode == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )
        self.hidden_size = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.tokens(x))
        h, last = self.gru(h)
        if self.temporal_mode == "gru":
            return last[-1]
        if self.temporal_mode == "mean":
            return h.mean(dim=1)
        weight = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
        return torch.sum(h * weight.unsqueeze(-1), dim=1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.tokens(x))
        h, _ = self.gru(h)
        return h


class HybridSkeletonEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        skeleton_smooth_kernel: int = 9,
        image_height: float = 1080.0,
        image_width: float = 1920.0,
        token_layers: int = 1,
        token_heads: int = 4,
        temporal_layers: int = 2,
        temporal_kernel: int = 5,
        temporal_mode: str = "gru",
        feature_mode: str = "hybrid",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.skeleton_smooth_kernel = skeleton_smooth_kernel
        self.image_height = image_height
        self.image_width = image_width
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in {
            "hybrid",
            "raw",
            "vector",
            "hybrid_zero_raw",
            "hybrid_zero_vector",
            *H36M_FEATURE_MODES,
        }:
            raise ValueError(f"Unsupported skeleton feature_mode={feature_mode!r}")
        if self.feature_mode in H36M_FEATURE_MODES:
            h36m_dim = h36m_feature_dim(self.feature_mode)
            self.register_buffer(
                "h36m_mu", torch.zeros(1, 1, h36m_dim), persistent=True
            )
            self.register_buffer(
                "h36m_sd", torch.ones(1, 1, h36m_dim), persistent=True
            )
            self.h36m = RawTower(
                h36m_dim,
                hidden_size,
                temporal_layers=temporal_layers,
                kernel_size=temporal_kernel,
                dropout=dropout,
                temporal_mode=temporal_mode,
            )
            self.raw = None
            self.vec = None
            self.fuse = None
        else:
            self.register_buffer(
                "raw_mu", torch.zeros(1, 1, RAW_POSE_DIM), persistent=True
            )
            self.register_buffer(
                "raw_sd", torch.ones(1, 1, RAW_POSE_DIM), persistent=True
            )
            self.register_buffer(
                "vec_mu",
                torch.zeros(1, 1, 1, SKELETON_TOKEN_DIM),
                persistent=True,
            )
            self.register_buffer(
                "vec_sd",
                torch.ones(1, 1, 1, SKELETON_TOKEN_DIM),
                persistent=True,
            )
            self.raw = RawTower(
                RAW_POSE_DIM,
                hidden_size,
                temporal_layers=temporal_layers,
                kernel_size=temporal_kernel,
                dropout=dropout,
                temporal_mode=temporal_mode,
            )
            self.vec = VectorTower(
                hidden_size,
                temporal_layers=temporal_layers,
                kernel_size=temporal_kernel,
                token_layers=token_layers,
                token_heads=token_heads,
                dropout=dropout,
                temporal_mode=temporal_mode,
            )
            self.fuse = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            self.h36m = None

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        if self.feature_mode in H36M_FEATURE_MODES:
            assert self.h36m is not None
            features = h36m_motion_sequence(
                skeleton, self.skeleton_smooth_kernel, include_depth=self.feature_mode == "h36m3d"
            )
            if self.feature_mode not in {"h36m3d", "h36m2d"}:
                features = h36m_feature_sequence(skeleton, self.skeleton_smooth_kernel, self.feature_mode)
            features = (features - self.h36m_mu) / self.h36m_sd.clamp_min(1e-6)
            return F.normalize(self.h36m(features), dim=1)
        assert self.raw is not None and self.vec is not None and self.fuse is not None
        raw = raw_pose_sequence(skeleton, self.skeleton_smooth_kernel, self.image_height, self.image_width)
        vec = skeleton_tokens(skeleton, self.skeleton_smooth_kernel, self.image_height, self.image_width)
        raw = (raw - self.raw_mu) / self.raw_sd.clamp_min(1e-6)
        vec = (vec - self.vec_mu) / self.vec_sd.clamp_min(1e-6)
        raw_h = self.raw(raw)
        vec_h = self.vec(vec)
        if self.feature_mode == "raw":
            h = raw_h
        elif self.feature_mode == "vector":
            h = vec_h
        elif self.feature_mode == "hybrid_zero_raw":
            h = self.fuse(torch.cat([torch.zeros_like(raw_h), vec_h], dim=1))
        elif self.feature_mode == "hybrid_zero_vector":
            h = self.fuse(torch.cat([raw_h, torch.zeros_like(vec_h)], dim=1))
        else:
            h = self.fuse(torch.cat([raw_h, vec_h], dim=1))
        return F.normalize(h, dim=1)

    def forward_sequence(self, skeleton: torch.Tensor) -> torch.Tensor:
        if self.feature_mode in H36M_FEATURE_MODES:
            assert self.h36m is not None
            if self.feature_mode in {"h36m3d", "h36m2d"}:
                features = h36m_motion_sequence(
                    skeleton,
                    self.skeleton_smooth_kernel,
                    include_depth=self.feature_mode == "h36m3d",
                )
            else:
                features = h36m_feature_sequence(skeleton, self.skeleton_smooth_kernel, self.feature_mode)
            features = (features - self.h36m_mu) / self.h36m_sd.clamp_min(1e-6)
            return self.h36m.forward_sequence(features)
        assert self.raw is not None and self.vec is not None and self.fuse is not None
        raw = raw_pose_sequence(skeleton, self.skeleton_smooth_kernel, self.image_height, self.image_width)
        vec = skeleton_tokens(skeleton, self.skeleton_smooth_kernel, self.image_height, self.image_width)
        raw = (raw - self.raw_mu) / self.raw_sd.clamp_min(1e-6)
        vec = (vec - self.vec_mu) / self.vec_sd.clamp_min(1e-6)
        raw_h = self.raw.forward_sequence(raw)
        vec_h = self.vec.forward_sequence(vec)
        if self.feature_mode == "raw":
            return raw_h
        if self.feature_mode == "vector":
            return vec_h
        if self.feature_mode == "hybrid_zero_raw":
            raw_h = torch.zeros_like(raw_h)
        elif self.feature_mode == "hybrid_zero_vector":
            vec_h = torch.zeros_like(vec_h)
        b, t, h = raw_h.shape
        fused = self.fuse(torch.cat([raw_h, vec_h], dim=-1).reshape(b * t, h * 2))
        return fused.reshape(b, t, h)


class HybridIMUEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        imu_smooth_kernel: int = 5,
        feature_mode: str = "raw",
        input_dim: int = 7,
        input_channels: Sequence[str] | None = None,
        temporal_layers: int = 2,
        temporal_kernel: int = 5,
        temporal_mode: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.imu_smooth_kernel = imu_smooth_kernel
        self.feature_mode = str(feature_mode).lower()
        if int(input_dim) <= 0:
            raise ValueError(f"Hybrid IMU input_dim must be positive, got {input_dim}")
        self.base_input_dim = int(input_dim)
        self.input_channels = tuple(input_channels) if input_channels is not None else None
        if self.input_channels is not None and len(self.input_channels) != self.base_input_dim:
            raise ValueError(
                f"Hybrid IMU input channel count={len(self.input_channels)} "
                f"does not match input_dim={self.base_input_dim}"
            )
        canonical_dynamic = _is_canonical_7d(self.input_channels, self.base_input_dim)
        self.feature_dim = 14 if self.feature_mode == "dynamic" and canonical_dynamic else (
            self.base_input_dim * 3 if self.feature_mode == "dynamic" else self.base_input_dim
        )
        self.register_buffer("imu_mu", torch.zeros(1, 1, self.feature_dim), persistent=True)
        self.register_buffer("imu_sd", torch.ones(1, 1, self.feature_dim), persistent=True)
        self.raw = RawTower(
            self.feature_dim,
            hidden_size,
            temporal_layers=temporal_layers,
            kernel_size=temporal_kernel,
            dropout=dropout,
            temporal_mode=temporal_mode,
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        x = imu_sequence_features(imu, self.imu_smooth_kernel, self.feature_mode, self.input_channels)
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Hybrid IMU feature width={x.shape[-1]} does not match configured width={self.feature_dim}; "
                f"input_shape={tuple(imu.shape)}, channels={self.input_channels}"
            )
        x = (x - self.imu_mu) / self.imu_sd.clamp_min(1e-6)
        return F.normalize(self.raw(x), dim=1)

    def forward_sequence(self, imu: torch.Tensor) -> torch.Tensor:
        x = imu_sequence_features(imu, self.imu_smooth_kernel, self.feature_mode, self.input_channels)
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Hybrid IMU feature width={x.shape[-1]} does not match configured width={self.feature_dim}; "
                f"input_shape={tuple(imu.shape)}, channels={self.input_channels}"
            )
        x = (x - self.imu_mu) / self.imu_sd.clamp_min(1e-6)
        return self.raw.forward_sequence(x)
