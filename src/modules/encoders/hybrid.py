"""Hybrid shoulder-vector encoders.

This encoder pair represents video as raw shoulder-relative pose plus
shoulder-local arm-vector tokens, and IMU as raw 7D acceleration/quaternion
sequence. The IMU encoder and video encoder live together because they are a
matched representation pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
LEFT_WRIST = 9
LEFT_HIP = 11
RIGHT_HIP = 12

RAW_POSE_DIM = 72
SKELETON_TOKEN_DIM = 12
SKELETON_N_TOKENS = 4


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


def _signed_angle_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = b - a
    return torch.atan2(torch.sin(delta), torch.cos(delta))


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


def raw_imu_sequence(imu: torch.Tensor, smooth_kernel: int = 5) -> torch.Tensor:
    if imu.shape[-1] < 7:
        raise ValueError(f"Hybrid raw IMU encoder expects at least 7 channels, got {tuple(imu.shape)}")
    imu = imu[..., :7].double()
    acc = _moving_average(imu[..., :3], smooth_kernel)
    quat = _moving_average(imu[..., 3:7], smooth_kernel)
    quat = quat / _norm(quat).clamp_min(1e-8).unsqueeze(-1)
    return torch.cat([acc, quat], dim=-1).float()


def imu_sequence_features(imu: torch.Tensor, smooth_kernel: int = 5, mode: str = "raw") -> torch.Tensor:
    raw = raw_imu_sequence(imu, smooth_kernel)
    mode = str(mode).lower()
    if mode == "raw":
        return raw
    if mode != "dynamic":
        raise ValueError(f"Unsupported hybrid IMU feature mode: {mode}")

    acc = raw[..., :3]
    quat = raw[..., 3:7]
    acc_centered = acc - acc.mean(dim=1, keepdim=True)
    acc_delta = _diff_same(acc, 1)
    dots = (quat[:, 1:] * quat[:, :-1]).sum(dim=-1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dots)
    ang_speed = torch.cat([torch.zeros_like(ang[:, :1]), ang], dim=1).unsqueeze(-1)
    return torch.cat([raw, acc_centered, acc_delta, ang_speed], dim=-1).float()


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
        if self.feature_mode not in {"hybrid", "raw", "vector", "hybrid_zero_raw", "hybrid_zero_vector"}:
            raise ValueError(f"Unsupported skeleton feature_mode={feature_mode!r}")
        self.register_buffer("raw_mu", torch.zeros(1, 1, RAW_POSE_DIM), persistent=True)
        self.register_buffer("raw_sd", torch.ones(1, 1, RAW_POSE_DIM), persistent=True)
        self.register_buffer("vec_mu", torch.zeros(1, 1, 1, SKELETON_TOKEN_DIM), persistent=True)
        self.register_buffer("vec_sd", torch.ones(1, 1, 1, SKELETON_TOKEN_DIM), persistent=True)
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

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
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
        temporal_layers: int = 2,
        temporal_kernel: int = 5,
        temporal_mode: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.imu_smooth_kernel = imu_smooth_kernel
        self.feature_mode = str(feature_mode).lower()
        input_dim = 14 if self.feature_mode == "dynamic" else 7
        self.register_buffer("imu_mu", torch.zeros(1, 1, input_dim), persistent=True)
        self.register_buffer("imu_sd", torch.ones(1, 1, input_dim), persistent=True)
        self.raw = RawTower(
            input_dim,
            hidden_size,
            temporal_layers=temporal_layers,
            kernel_size=temporal_kernel,
            dropout=dropout,
            temporal_mode=temporal_mode,
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        x = imu_sequence_features(imu, self.imu_smooth_kernel, self.feature_mode)
        x = (x - self.imu_mu) / self.imu_sd.clamp_min(1e-6)
        return F.normalize(self.raw(x), dim=1)

    def forward_sequence(self, imu: torch.Tensor) -> torch.Tensor:
        x = imu_sequence_features(imu, self.imu_smooth_kernel, self.feature_mode)
        x = (x - self.imu_mu) / self.imu_sd.clamp_min(1e-6)
        return self.raw.forward_sequence(x)
