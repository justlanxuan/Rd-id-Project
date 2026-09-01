"""Contracts for reproducible sequence-level data variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _as_transform_names(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


@dataclass(frozen=True)
class DerivedDataSpec:
    """Resolved configuration for one materialized derived-data variant."""

    enabled: bool = False
    name: str = "derived"
    output: str = ""
    transforms: tuple[str, ...] = ()
    seed: int = 42
    imu_acc_noise_std: float = 0.0
    imu_acc_lowpass_cutoff_hz: float = 0.0
    imu_acc_lowpass_fs_hz: float = 30.0
    imu_acc_spike_ratio: float = 0.0
    imu_acc_spike_scale: float = 5.0
    imu_acc_dropout_duration: int = 3
    imu_acc_dropout_segments: int = 1
    imu_mount_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_global_yaw_deg: float = 0.0
    skeleton_bone_scale_min: float = 0.9
    skeleton_bone_scale_max: float = 1.1
    skeleton_coord_noise_rho: float = 0.85
    skeleton_coord_noise_std_torso: float = 0.006
    skeleton_coord_noise_std_mid: float = 0.012
    skeleton_coord_noise_std_task: float = 0.015
    skeleton_coord_noise_std_distal: float = 0.022
    skeleton_joint_dropout_rate: float = 1.0 / 90.0
    skeleton_joint_dropout_min_frames: int = 2
    skeleton_joint_dropout_max_frames: int = 6
    skeleton_fragmentation_rate: float = 1.0 / 180.0
    skeleton_fragmentation_min_frames: int = 2
    skeleton_fragmentation_max_frames: int = 8
    skeleton_fragmentation_recovery_noise_std: float = 0.01

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DerivedDataSpec":
        data = value if isinstance(value, Mapping) else {}
        euler = tuple(float(item) for item in data.get("imu_mount_euler_xyz_deg", (0.0, 0.0, 0.0)))
        if len(euler) != 3:
            raise ValueError("preprocess.derived.imu_mount_euler_xyz_deg must contain exactly three values")
        scale_min = float(data.get("skeleton_bone_scale_min", 0.9))
        scale_max = float(data.get("skeleton_bone_scale_max", 1.1))
        if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
            raise ValueError("Skeleton bone scale bounds must be positive and min <= max")
        rho = float(data.get("skeleton_coord_noise_rho", 0.85))
        if not 0.0 <= rho < 1.0:
            raise ValueError("skeleton_coord_noise_rho must be in [0, 1)")
        joint_min = int(data.get("skeleton_joint_dropout_min_frames", 2))
        joint_max = int(data.get("skeleton_joint_dropout_max_frames", 6))
        fragment_min = int(data.get("skeleton_fragmentation_min_frames", 2))
        fragment_max = int(data.get("skeleton_fragmentation_max_frames", 8))
        if joint_min <= 0 or joint_min > joint_max:
            raise ValueError("Joint dropout frame bounds must be positive and min <= max")
        if fragment_min <= 0 or fragment_min > fragment_max:
            raise ValueError("Track fragmentation frame bounds must be positive and min <= max")
        if float(data.get("skeleton_joint_dropout_rate", 1.0 / 90.0)) < 0:
            raise ValueError("skeleton_joint_dropout_rate must be non-negative")
        if float(data.get("skeleton_fragmentation_rate", 1.0 / 180.0)) < 0:
            raise ValueError("skeleton_fragmentation_rate must be non-negative")
        acc_noise_std = float(data.get("imu_acc_noise_std", 0.0))
        acc_lowpass_cutoff = float(data.get("imu_acc_lowpass_cutoff_hz", 0.0))
        acc_lowpass_fs = float(data.get("imu_acc_lowpass_fs_hz", 30.0))
        acc_spike_ratio = float(data.get("imu_acc_spike_ratio", 0.0))
        acc_spike_scale = float(data.get("imu_acc_spike_scale", 5.0))
        acc_dropout_duration = int(data.get("imu_acc_dropout_duration", 3))
        acc_dropout_segments = int(data.get("imu_acc_dropout_segments", 1))
        if acc_noise_std < 0 or acc_lowpass_cutoff < 0 or acc_lowpass_fs <= 0:
            raise ValueError("IMU noise/cutoff must be non-negative and IMU low-pass sampling rate must be positive")
        if not 0.0 <= acc_spike_ratio <= 1.0 or acc_spike_scale < 0:
            raise ValueError("IMU spike ratio must be in [0, 1] and spike scale must be non-negative")
        if acc_dropout_duration <= 0 or acc_dropout_segments < 0:
            raise ValueError("IMU dropout duration must be positive and segment count must be non-negative")
        return cls(
            enabled=bool(data.get("enabled", False)),
            name=str(data.get("name", "derived")).strip() or "derived",
            output=str(data.get("output", "")).strip(),
            transforms=_as_transform_names(data.get("transforms", ())),
            seed=int(data.get("seed", 42)),
            imu_acc_noise_std=acc_noise_std,
            imu_acc_lowpass_cutoff_hz=acc_lowpass_cutoff,
            imu_acc_lowpass_fs_hz=acc_lowpass_fs,
            imu_acc_spike_ratio=acc_spike_ratio,
            imu_acc_spike_scale=acc_spike_scale,
            imu_acc_dropout_duration=acc_dropout_duration,
            imu_acc_dropout_segments=acc_dropout_segments,
            imu_mount_euler_xyz_deg=euler,
            imu_global_yaw_deg=float(data.get("imu_global_yaw_deg", 0.0)),
            skeleton_bone_scale_min=scale_min,
            skeleton_bone_scale_max=scale_max,
            skeleton_coord_noise_rho=rho,
            skeleton_coord_noise_std_torso=float(data.get("skeleton_coord_noise_std_torso", 0.006)),
            skeleton_coord_noise_std_mid=float(data.get("skeleton_coord_noise_std_mid", 0.012)),
            skeleton_coord_noise_std_task=float(data.get("skeleton_coord_noise_std_task", 0.015)),
            skeleton_coord_noise_std_distal=float(data.get("skeleton_coord_noise_std_distal", 0.022)),
            skeleton_joint_dropout_rate=float(data.get("skeleton_joint_dropout_rate", 1.0 / 90.0)),
            skeleton_joint_dropout_min_frames=joint_min,
            skeleton_joint_dropout_max_frames=joint_max,
            skeleton_fragmentation_rate=float(data.get("skeleton_fragmentation_rate", 1.0 / 180.0)),
            skeleton_fragmentation_min_frames=fragment_min,
            skeleton_fragmentation_max_frames=fragment_max,
            skeleton_fragmentation_recovery_noise_std=float(
                data.get("skeleton_fragmentation_recovery_noise_std", 0.01)
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "name": self.name,
            "transforms": list(self.transforms),
            "seed": self.seed,
            "imu_acc_noise_std": self.imu_acc_noise_std,
            "imu_acc_lowpass_cutoff_hz": self.imu_acc_lowpass_cutoff_hz,
            "imu_acc_lowpass_fs_hz": self.imu_acc_lowpass_fs_hz,
            "imu_acc_spike_ratio": self.imu_acc_spike_ratio,
            "imu_acc_spike_scale": self.imu_acc_spike_scale,
            "imu_acc_dropout_duration": self.imu_acc_dropout_duration,
            "imu_acc_dropout_segments": self.imu_acc_dropout_segments,
            "imu_mount_euler_xyz_deg": list(self.imu_mount_euler_xyz_deg),
            "imu_global_yaw_deg": self.imu_global_yaw_deg,
            "skeleton_bone_scale_min": self.skeleton_bone_scale_min,
            "skeleton_bone_scale_max": self.skeleton_bone_scale_max,
            "skeleton_coord_noise_rho": self.skeleton_coord_noise_rho,
            "skeleton_coord_noise_std_torso": self.skeleton_coord_noise_std_torso,
            "skeleton_coord_noise_std_mid": self.skeleton_coord_noise_std_mid,
            "skeleton_coord_noise_std_task": self.skeleton_coord_noise_std_task,
            "skeleton_coord_noise_std_distal": self.skeleton_coord_noise_std_distal,
            "skeleton_joint_dropout_rate": self.skeleton_joint_dropout_rate,
            "skeleton_joint_dropout_min_frames": self.skeleton_joint_dropout_min_frames,
            "skeleton_joint_dropout_max_frames": self.skeleton_joint_dropout_max_frames,
            "skeleton_fragmentation_rate": self.skeleton_fragmentation_rate,
            "skeleton_fragmentation_min_frames": self.skeleton_fragmentation_min_frames,
            "skeleton_fragmentation_max_frames": self.skeleton_fragmentation_max_frames,
            "skeleton_fragmentation_recovery_noise_std": self.skeleton_fragmentation_recovery_noise_std,
        }
