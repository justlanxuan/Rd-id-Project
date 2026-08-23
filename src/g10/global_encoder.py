"""Global-only sequence encoder and auditable matching dataset for G10.

The module intentionally keeps the learned benchmark independent of the legacy
hybrid pose encoder.  A sample contains one explicit global skeleton feature
and one explicit IMU view; no bone vectors or local-pose tokens are generated.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.features.global_motion import (
    H36M17_JOINTS,
    derive_trajectory_features,
    extract_global_anchors,
    extract_imu_views,
    spectral_summary,
)
from src.metrics import EmbeddingBundle, FrameAccEvaluator
from src.modules.encoders.multiscale import MultiScaleTemporalTCN


def _safe_normalize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize embeddings with finite gradients even for a zero vector."""
    denominator = torch.sqrt(values.square().sum(dim=-1, keepdim=True).clamp_min(float(eps) ** 2))
    return values / denominator


def _interp_columns(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == target_len:
        return values.copy()
    if len(values) < 2:
        return np.repeat(values[:1], target_len, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(target, source, values[:, col]) for col in range(values.shape[1])], axis=-1).astype(np.float32)


def _window_key(row: dict[str, str]) -> str:
    explicit = str(row.get("candidate_group_id", "")).strip()
    if explicit:
        return f"candidate|{explicit}"
    sequence = row.get("source_sequence") or row.get("sequence_id") or row.get("npz_path", "")
    start = row.get("source_window_start") or row.get("window_start", "")
    end = row.get("source_window_end") or row.get("window_end", "")
    return "|".join((str(sequence), str(start), str(end)))


def _identity(row: dict[str, str]) -> str:
    if str(row.get("candidate_group_id", "")).strip():
        return str(row.get("candidate_index") or row.get("source_sequence") or row.get("npz_path") or "0")
    return str(row.get("source_person") or row.get("imu_idx") or row.get("person_idx") or "0")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return ((values - center) / np.maximum(scale, 1e-5)).astype(np.float32)


def _scalar_feature(values: dict[str, np.ndarray], masks: dict[str, np.ndarray], feature: str) -> np.ndarray:
    if feature not in values:
        raise ValueError(f"Unknown feature {feature!r}; available={sorted(values)}")
    array = np.asarray(values[feature], dtype=np.float32)
    valid = np.asarray(masks[feature], dtype=bool)
    if array.ndim == 1:
        output = array[:, None]
    else:
        output = np.linalg.norm(array, axis=-1, keepdims=True)
    output[~valid] = 0.0
    return output


_SPECTRAL_FEATURE_KEYS = {
    "dominant_frequency": "dominant_hz",
    "band_energy": "band_energy",
    "spectral_entropy": "spectral_entropy",
    "periodicity": "periodicity",
}


def _frequency_profile(signal: np.ndarray, timestamps: np.ndarray, validity: np.ndarray, target_len: int) -> np.ndarray:
    """Return a common 0–4.5 Hz log-power profile repeated per window row."""
    values = np.asarray(signal, dtype=np.float32)
    valid_indices = np.flatnonzero(np.asarray(validity, dtype=bool))
    if len(valid_indices) < 4:
        return np.zeros((len(values), target_len), dtype=np.float32)
    splits = np.flatnonzero(np.diff(valid_indices) > 1)
    runs = np.split(valid_indices, splits + 1)
    run = max(runs, key=len)
    if len(run) < 4:
        return np.zeros((len(values), target_len), dtype=np.float32)
    dt = float(np.median(np.diff(np.asarray(timestamps)[run])))
    if dt <= 0:
        return np.zeros((len(values), target_len), dtype=np.float32)
    centered = values[run] - values[run].mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power = np.asarray(power.sum(axis=-1), dtype=np.float32)
    frequencies = np.fft.rfftfreq(len(run), d=dt)
    band = frequencies <= 4.5
    if int(band.sum()) < 2:
        return np.zeros((len(values), target_len), dtype=np.float32)
    target_frequencies = np.linspace(0.0, 4.5, int(target_len), dtype=np.float32)
    profile = np.interp(target_frequencies, frequencies[band], np.log1p(power[band])).astype(np.float32)
    profile -= profile.mean()
    profile /= max(float(profile.std()), 1e-5)
    return np.repeat(profile[None, :], len(values), axis=0)


def _window_spectral_feature(
    signal: np.ndarray,
    timestamps: np.ndarray,
    validity: np.ndarray,
    feature: str,
) -> np.ndarray:
    key = _SPECTRAL_FEATURE_KEYS.get(feature)
    if key is None:
        raise ValueError(f"Unknown spectral feature {feature!r}; available={sorted(_SPECTRAL_FEATURE_KEYS)}")
    summary = spectral_summary(signal, timestamps, validity=validity)
    value = float(summary[key]) if bool(summary.get("valid", False)) else 0.0
    return np.full((len(signal), 1), value, dtype=np.float32)


class GlobalMotionDataset(Dataset):
    """Window dataset supporting one or multiple explicitly declared domains.

    ``specs`` entries contain ``dataset``, ``csv``, ``root`` and ``fps_hz``;
    optional ``gyro_sidecar_root`` enables the 10D acc+gyro+quat contract.  A
    folded Custom window sidecar is preferred by exact NPZ filename, while
    source sidecars use the full sequence id.
    """

    def __init__(
        self,
        specs: Sequence[dict[str, Any]],
        *,
        anchor_id: str,
        skeleton_feature: str,
        imu_view: str,
        imu_feature: str,
        target_len: int = 24,
        normalize: str = "zscore",
        joint_names: Sequence[str] = H36M17_JOINTS,
    ) -> None:
        if not specs:
            raise ValueError("GlobalMotionDataset requires at least one spec")
        self.anchor_id = anchor_id
        self.skeleton_feature = skeleton_feature
        self.imu_view = imu_view
        self.imu_feature = imu_feature
        self.target_len = int(target_len)
        self.normalize = str(normalize)
        self.joint_names = tuple(joint_names)
        self.rows: list[dict[str, Any]] = []
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._feature_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
        for spec in specs:
            csv_path = Path(spec["csv"]).expanduser().resolve()
            root = Path(spec["root"]).expanduser().resolve()
            sidecar = Path(spec["gyro_sidecar_root"]).expanduser().resolve() if spec.get("gyro_sidecar_root") else None
            fps_hz = float(spec["fps_hz"])
            dataset = str(spec["dataset"])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if spec.get("session_filter"):
                rows = [row for row in rows if row.get("session") == str(spec["session_filter"])]
            if spec.get("skip_missing_sidecar") and sidecar is not None:
                kept: list[dict[str, str]] = []
                for row in rows:
                    candidate = sidecar / Path(row["npz_path"]).name
                    if candidate.is_file():
                        kept.append(row)
                rows = kept
            if not rows:
                raise ValueError(f"No rows found in {csv_path}")
            for row in rows:
                copied = dict(row)
                copied["_root"] = str(root)
                copied["_sidecar"] = str(sidecar) if sidecar else ""
                copied["_fps_hz"] = fps_hz
                copied["_dataset"] = dataset
                copied["_group_key"] = _window_key(row)
                copied["_identity"] = _identity(row)
                self.rows.append(copied)
        if not self.rows:
            raise ValueError("GlobalMotionDataset contains no rows")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({str(row["_dataset"]) for row in self.rows}))

    def group_indices(self) -> dict[tuple[str, str], list[int]]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, row in enumerate(self.rows):
            groups[(str(row["_dataset"]), str(row["_group_key"]))].append(idx)
        return dict(groups)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._cache:
            with np.load(path, allow_pickle=True) as data:
                self._cache[path] = {key: data[key] for key in data.files}
        return self._cache[path]

    @staticmethod
    def _slice(data: dict[str, np.ndarray], key: str, st: int, ed: int, person: int) -> np.ndarray:
        values = np.asarray(data[key])
        local = ed > values.shape[0]
        if local:
            st, ed = 0, ed - st
        if values.ndim >= 3 and key in {"gt_skeleton", "extract_skeleton"}:
            return values[st:ed, person]
        if values.ndim >= 3 and key == "imu":
            return values[st:ed, person]
        return values[st:ed]

    def _imu_values(self, row: dict[str, Any], data: dict[str, np.ndarray], st: int, ed: int, person: int, path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
        values = np.asarray(data["imu"])
        local = ed > values.shape[0]
        local_st, local_ed = (0, ed - st) if local else (st, ed)
        if values.ndim == 3:
            values = values[local_st:local_ed, int(row.get("imu_idx", person))]
        else:
            values = values[local_st:local_ed]
        channels = tuple(str(x) for x in np.asarray(data["imu_channels"]).reshape(-1)) if "imu_channels" in data else (
            "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"
        )
        sidecar_root = Path(row["_sidecar"]) if row.get("_sidecar") else None
        if sidecar_root is not None:
            sequence_name = str(data["sequence_id"].item()) if "sequence_id" in data else ""
            sidecar_path = sidecar_root / path.name
            if not sidecar_path.is_file() and sequence_name:
                sidecar_path = sidecar_root / f"{sequence_name}.npz"
            if not sidecar_path.is_file():
                raise FileNotFoundError(f"Declared gyro sidecar is missing for {path.name}: {sidecar_path}")
            if sidecar_path.is_file():
                with np.load(sidecar_path, allow_pickle=True) as side:
                    acc = np.asarray(side["acceleration_mps2"])
                    gyro = np.asarray(side["gyroscope_rads"])
                    acc = acc if acc.ndim == 2 else acc[:, int(row.get("imu_idx", person))]
                    gyro = gyro if gyro.ndim == 2 else gyro[:, int(row.get("imu_idx", person))]
                    acc = acc[local_st:local_ed]
                    gyro = gyro[local_st:local_ed]
                if len(acc) != len(values):
                    raise ValueError(f"Sidecar/window mismatch for {path}: {acc.shape} vs {values.shape}")
                if values.shape[-1] < 7:
                    raise ValueError(f"Canonical IMU lacks quaternion channels for {path}")
                values = np.concatenate([acc, gyro, values[:, 3:7]], axis=-1)
                channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z")
        return values.astype(np.float64), channels

    def _features(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[index]
        path = (Path(row["_root"]) / row["npz_path"]).resolve()
        cache_key = (path, row["window_start"], row["window_end"], row.get("person_idx", "0"), row.get("imu_idx", "0"))
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        data = self._load(path)
        st, ed = int(row["window_start"]), int(row["window_end"])
        person = int(row.get("person_idx", 0))
        skeleton_key = "gt_skeleton" if "gt_skeleton" in data else "skeleton"
        skeleton = self._slice(data, skeleton_key, st, ed, person)
        if skeleton.shape[-1] < 2:
            raise ValueError(f"Skeleton must have at least xy coordinates: {path} {skeleton.shape}")
        timestamps = np.arange(len(skeleton), dtype=np.float64) / float(row["_fps_hz"])
        anchors = extract_global_anchors(skeleton, joint_names=self.joint_names, coordinate_space="screen_or_world")
        trajectories = anchors.trajectories[self.anchor_id]
        masks = anchors.validity[self.anchor_id]
        skeleton_values, skeleton_masks = derive_trajectory_features(trajectories, timestamps, validity=masks)
        if self.skeleton_feature == "frequency_profile":
            skeleton_feature = _frequency_profile(trajectories, timestamps, masks, self.target_len)
        elif self.skeleton_feature in _SPECTRAL_FEATURE_KEYS:
            skeleton_feature = _window_spectral_feature(trajectories, timestamps, masks, self.skeleton_feature)
        else:
            skeleton_feature = _scalar_feature(skeleton_values, skeleton_masks, self.skeleton_feature)
        imu, channels = self._imu_values(row, data, st, ed, person, path)
        imu_views = extract_imu_views(
            imu,
            timestamps,
            channel_names=channels,
            sensor_location="declared",
            provenance="declared",
        )
        if self.imu_view not in imu_views:
            raise ValueError(f"IMU view {self.imu_view!r} unavailable for channels={channels}")
        imu_view = imu_views[self.imu_view]
        imu_values = np.asarray(imu_view.values, dtype=np.float32)
        if self.imu_feature == "frequency_profile":
            imu_feature = _frequency_profile(imu_values, timestamps, imu_view.validity, self.target_len)
        elif self.imu_feature in _SPECTRAL_FEATURE_KEYS:
            imu_feature = _window_spectral_feature(imu_values, timestamps, imu_view.validity, self.imu_feature)
        elif self.imu_feature == "raw":
            imu_feature = imu_values
        elif self.imu_view == "I1_acc_magnitude":
            component = 0 if self.imu_feature in {"magnitude", "energy", "speed"} else 1
            imu_feature = imu_values[:, component : component + 1]
        elif self.imu_view == "I2_acc_changes":
            imu_feature = imu_values[:, 3:4] if self.imu_feature in {"energy", "change_energy"} else np.linalg.norm(imu_values[:, :3], axis=-1, keepdims=True)
        elif self.imu_view == "I4_delta_quaternion":
            imu_feature = imu_values[:, 4:5]
        elif self.imu_view in {"I0_acc", "I3_gyro", "I5_acc_gyro", "I6_acc_quat", "I7_acc_gyro_quat"}:
            imu_feature = imu_values if self.imu_feature == "raw" else np.linalg.norm(imu_values, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported IMU view for global encoder: {self.imu_view}")
        skeleton_feature = _interp_columns(skeleton_feature, self.target_len)
        imu_feature = _interp_columns(np.asarray(imu_feature, dtype=np.float32), self.target_len)
        if self.normalize == "zscore":
            skeleton_feature = _zscore(skeleton_feature)
            imu_feature = _zscore(imu_feature)
        elif self.normalize != "none":
            raise ValueError(f"Unsupported global feature normalization {self.normalize!r}")
        skeleton_feature = np.nan_to_num(skeleton_feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        imu_feature = np.nan_to_num(imu_feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._feature_cache[cache_key] = (skeleton_feature, imu_feature)
        return skeleton_feature, imu_feature

    def __getitem__(self, index: int) -> dict[str, Any]:
        skeleton, imu = self._features(index)
        row = self.rows[index]
        return {
            "skeleton": torch.from_numpy(skeleton),
            "imu": torch.from_numpy(imu),
            "index": int(index),
            "domain": str(row["_dataset"]),
            "group_key": str(row["_group_key"]),
            "identity": str(row["_identity"]),
        }


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 96,
        output_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        mode: str = "gru",
        *,
        multiscale_fusion: str = "hierarchical_attention",
        window_seconds: float | None = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"gru", "mean", "attn", "transformer", "tcn", "multiscale"}:
            raise ValueError(f"Unsupported temporal encoder mode={mode!r}")
        if self.mode == "multiscale":
            self.multiscale = MultiScaleTemporalTCN(
                input_dim,
                hidden_dim=hidden,
                output_dim=hidden,
                dropout=dropout,
                fusion=multiscale_fusion,
                window_seconds=window_seconds,
            )
            self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
            return
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden) if use_layer_norm else nn.Identity(),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for _ in range(int(layers)):
            blocks.extend((nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(dropout)))
        self.temporal = nn.Sequential(*blocks)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        if self.mode == "transformer":
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer: nn.Module | None = nn.TransformerEncoder(transformer_layer, num_layers=2)
        else:
            self.transformer = None
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, output_dim))

    def forward_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "multiscale":
            output = self.multiscale(x)
            pooled = output["sequence"].mean(dim=1)
            return _safe_normalize(self.out(pooled)), output["scale_weights"]
        h = self.proj(x)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        if self.mode == "tcn":
            return _safe_normalize(self.out(h.mean(dim=1))), None
        if self.mode == "transformer":
            assert self.transformer is not None
            h = self.transformer(h)
            pooled = h.mean(dim=1)
            return _safe_normalize(self.out(pooled)), None
        h, last = self.gru(h)
        if self.mode == "gru":
            pooled = last[-1]
        elif self.mode == "mean":
            pooled = h.mean(dim=1)
        else:
            weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
            pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return _safe_normalize(self.out(pooled)), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.forward_with_diagnostics(x)
        return embedding


class GlobalMotionMatcher(nn.Module):
    def __init__(
        self,
        skeleton_dim: int = 1,
        imu_dim: int = 1,
        hidden: int = 96,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        temporal_mode: str = "gru",
        *,
        multiscale_fusion: str = "hierarchical_attention",
        window_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.temporal_mode = str(temporal_mode)
        self.skeleton_encoder = TemporalEncoder(
            skeleton_dim,
            hidden,
            embedding_dim,
            dropout=dropout,
            mode=temporal_mode,
            multiscale_fusion=multiscale_fusion,
            window_seconds=window_seconds,
        )
        self.imu_encoder = TemporalEncoder(
            imu_dim,
            hidden,
            embedding_dim,
            dropout=dropout,
            mode=temporal_mode,
            multiscale_fusion=multiscale_fusion,
            window_seconds=window_seconds,
        )

    def forward(self, skeleton: torch.Tensor, imu: torch.Tensor) -> dict[str, torch.Tensor]:
        skeleton_embedding, skeleton_weights = self.skeleton_encoder.forward_with_diagnostics(skeleton)
        imu_embedding, imu_weights = self.imu_encoder.forward_with_diagnostics(imu)
        output = {"skeleton": skeleton_embedding, "imu": imu_embedding}
        if skeleton_weights is not None and imu_weights is not None:
            output["skeleton_scale_weights"] = skeleton_weights
            output["imu_scale_weights"] = imu_weights
        return output


@torch.no_grad()
def evaluate_global_matcher(model: GlobalMotionMatcher, dataset: GlobalMotionDataset, device: torch.device, batch_size: int = 128) -> dict[str, Any]:
    model.eval()
    skeleton_embeddings: dict[int, np.ndarray] = {}
    imu_embeddings: dict[int, np.ndarray] = {}
    skeleton_scale_weights: list[np.ndarray] = []
    imu_scale_weights: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = [dataset[index] for index in indices]
        skel = torch.stack([item["skeleton"] for item in batch]).to(device)
        imu = torch.stack([item["imu"] for item in batch]).to(device)
        out = model(skel, imu)
        if "skeleton_scale_weights" in out:
            skeleton_scale_weights.append(out["skeleton_scale_weights"].detach().cpu().numpy())
            imu_scale_weights.append(out["imu_scale_weights"].detach().cpu().numpy())
        for offset, index in enumerate(indices):
            skeleton_embeddings[index] = out["skeleton"][offset].cpu().numpy()
            imu_embeddings[index] = out["imu"][offset].cpu().numpy()
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[(str(row["_dataset"]), str(row["_group_key"]))].append(index)
    correct = total = 0
    singleton = 0
    margins: list[float] = []
    confidences: list[float] = []
    entropies: list[float] = []
    ties: list[bool] = []
    outcomes: list[bool] = []
    per_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "singleton": 0, "groups": 0})
    diagnostics: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"margin": [], "confidence": [], "entropy": [], "tie": [], "outcome": []})
    for (domain, _), indices in groups.items():
        if len(indices) < 2:
            singleton += 1
            per_domain[domain]["singleton"] += 1
            continue
        per_domain[domain]["groups"] += 1
        scores = np.asarray([[float(skeleton_embeddings[i] @ imu_embeddings[j]) for j in indices] for i in indices])
        for row_idx, index in enumerate(indices):
            prediction = indices[int(np.argmax(scores[row_idx]))]
            ok = str(dataset.rows[prediction]["_identity"]) == str(dataset.rows[index]["_identity"])
            correct += int(ok)
            total += 1
            per_domain[domain]["correct"] += int(ok)
            per_domain[domain]["total"] += 1
            ordered = np.sort(scores[row_idx])
            margin = float(ordered[-1] - ordered[-2])
            logits = scores[row_idx] - scores[row_idx].max()
            probabilities = np.exp(logits) / np.exp(logits).sum()
            confidence = float(probabilities.max())
            entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / np.log(len(probabilities)))
            tie = margin <= 1e-6
            margins.append(margin)
            confidences.append(confidence)
            entropies.append(entropy)
            ties.append(tie)
            outcomes.append(bool(ok))
            diagnostics[domain]["margin"].append(margin)
            diagnostics[domain]["confidence"].append(confidence)
            diagnostics[domain]["entropy"].append(entropy)
            diagnostics[domain]["tie"].append(tie)
            diagnostics[domain]["outcome"].append(bool(ok))

    def calibration_error(confidence_values: list[float], correct_values: list[bool], bins: int = 10) -> float | None:
        if not confidence_values:
            return None
        confidence_array = np.asarray(confidence_values)
        correct_array = np.asarray(correct_values, dtype=np.float64)
        error = 0.0
        for low in np.linspace(0.0, 1.0, bins, endpoint=False):
            high = low + 1.0 / bins
            mask = (confidence_array >= low) & (confidence_array < high if high < 1.0 else confidence_array <= high)
            if mask.any():
                error += float(mask.mean()) * abs(float(confidence_array[mask].mean()) - float(correct_array[mask].mean()))
        return error

    def abstention(margin_values: list[float], correct_values: list[bool]) -> dict[str, dict[str, float | int]]:
        if not margin_values:
            return {}
        order = np.argsort(np.asarray(margin_values))[::-1]
        correct_array = np.asarray(correct_values, dtype=np.float64)
        result: dict[str, dict[str, float | int]] = {}
        for coverage in (1.0, 0.8, 0.5):
            kept = max(1, int(round(len(order) * coverage)))
            indices = order[:kept]
            result[f"coverage_{coverage:.1f}"] = {"kept": int(kept), "accuracy": float(correct_array[indices].mean())}
        return result

    per_domain_result = {}
    for key, value in per_domain.items():
        stats = diagnostics[key]
        per_domain_result[key] = {
            **value,
            "frame_acc": value["correct"] / value["total"] if value["total"] else None,
            "chance_frame_acc": value["groups"] / value["total"] if value["total"] else None,
            "mean_margin": float(np.mean(stats["margin"])) if stats["margin"] else None,
            "mean_assignment_entropy": float(np.mean(stats["entropy"])) if stats["entropy"] else None,
            "tie_rate": float(np.mean(stats["tie"])) if stats["tie"] else None,
            "expected_calibration_error": calibration_error(stats["confidence"], stats["outcome"]),
            "margin_abstention": abstention(stats["margin"], stats["outcome"]),
        }
    result = {
        "correct": correct,
        "total": total,
        "frame_acc": float(correct / total) if total else None,
        "candidate_groups": len(groups),
        "singleton_groups_skipped": singleton,
        "chance_frame_acc": (len(groups) - singleton) / total if total else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "mean_assignment_entropy": float(np.mean(entropies)) if entropies else None,
        "tie_rate": float(np.mean(ties)) if ties else None,
        "expected_calibration_error": calibration_error(confidences, outcomes),
        "margin_abstention": abstention(margins, outcomes),
        "per_domain": per_domain_result,
    }
    # Also route the same embeddings through the repository's canonical
    # FrameAcc implementation.  The learned matcher keeps its richer domain
    # diagnostics above, while this field proves that G10/G11 reports use the
    # same candidate-group/Hungarian contract as the native evaluator.
    official_rows: list[dict[str, str]] = []
    for row in dataset.rows:
        official_rows.append(
            {
                "candidate_group_id": f"{row['_dataset']}:{row['_group_key']}",
                "candidate_index": str(row["_identity"]),
                "source_person": str(row["_identity"]),
                "imu_idx": "0",
                "person_idx": "0",
                "subject": str(row["_identity"]),
                "session": str(row["_dataset"]),
                "npz_path": str(row.get("npz_path", "")),
                "window_start": "0",
                "window_end": str(dataset.target_len),
            }
        )
    official_bundle = EmbeddingBundle(
        rows=official_rows,
        imu=np.stack([imu_embeddings[index] for index in range(len(dataset))]),
        video=np.stack([skeleton_embeddings[index] for index in range(len(dataset))]),
    )
    result["reid_project_frame_acc"] = FrameAccEvaluator(
        shuffle_match=False,
        singleton_policy="exclude",
    ).evaluate(official_bundle)
    if skeleton_scale_weights:
        result["scale_weights"] = {
            "skeleton_mean": np.concatenate(skeleton_scale_weights, axis=0).mean(axis=0).tolist(),
            "imu_mean": np.concatenate(imu_scale_weights, axis=0).mean(axis=0).tolist(),
        }
    return result
