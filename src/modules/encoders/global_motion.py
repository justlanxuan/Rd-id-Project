"""Global motion encoder: encodes root trajectory / bbox motion as embeddings.

This encoder is designed to be pluggable into the alignment pipeline.
It extracts global positional information that is lost during skeleton
normalization (root-centering + scale normalization).

Supported input sources:
    - bbox_center: 2D bounding box center trajectory from video
    - root_3d: 3D root joint position (e.g., from Vicon GT in meters)
    - velocity_2d: 2D velocity computed from bbox center differences

Author: auto-generated for global motion experiment
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from src.modules.encoders.base import BaseEncoder


class GlobalMotionEncoder(BaseEncoder):
    """Encode global root/bbox trajectory into an embedding vector.

    Args:
        input_dim: Dimension of input trajectory (2 for bbox center, 3 for 3D root)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        embed_dim: Output embedding dimension
        dropout: Dropout probability
        input_type: Type of preprocessing applied to input
            - "raw": use trajectory as-is
            - "diff": use first-order differences (velocity)
            - "diff_raw": concatenate raw + diff
        fusion_proj: If True, project LSTM output to embed_dim via MLP.
            If False, LSTM hidden_dim must equal embed_dim.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.1,
        input_type: Literal["raw", "diff", "diff_raw"] = "diff_raw",
        fusion_proj: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_type = input_type
        self.fusion_proj = fusion_proj

        # Determine actual input dim after preprocessing
        if input_type == "diff_raw":
            lstm_input_dim = input_dim * 2
        else:
            lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if fusion_proj:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            if hidden_dim != embed_dim:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must equal embed_dim ({embed_dim}) "
                    "when fusion_proj=False"
                )
            self.output_proj = nn.Identity()

    def _preprocess(self, traj: torch.Tensor) -> torch.Tensor:
        """Apply input preprocessing: raw, diff, or diff_raw.

        Args:
            traj: [B, T, input_dim]
        Returns:
            [B, T, processed_dim]
        """
        if self.input_type == "raw":
            return traj
        elif self.input_type == "diff":
            # Pad first frame with zeros
            diff = torch.zeros_like(traj)
            diff[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
            return diff
        elif self.input_type == "diff_raw":
            diff = torch.zeros_like(traj)
            diff[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
            return torch.cat([traj, diff], dim=-1)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.forward(trajectory)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            trajectory: [B, T, input_dim] root/bbox trajectory
        Returns:
            [B, embed_dim] global motion embedding
        """
        x = self._preprocess(trajectory)  # [B, T, processed_dim]
        out, _ = self.lstm(x)  # [B, T, hidden_dim]
        last = out[:, -1, :]  # [B, hidden_dim]
        return self.output_proj(last)  # [B, embed_dim]


def extract_root_trajectory_from_data(
    data: dict,
    st: int,
    ed: int,
    skeleton_source: str = "extract",
    person_idx: int = 0,
    source_type: Literal["bbox_center", "bbox_full", "bbox_vel", "root_3d", "auto"] = "auto",
) -> Optional[torch.Tensor]:
    """Extract root trajectory from NPZ data dict.

    Args:
        data: NPZ data dict
        st, ed: window start/end
        skeleton_source: "gt" or "extract"
        person_idx: person index
        source_type:
            - "bbox_center": use extract_bboxes center (2D)
            - "bbox_full":   use extract_bboxes center + size (4D)
            - "bbox_vel":    use extract_bboxes center + velocity (4D)
            - "root_3d": use gt_skeleton_meters root (3D)
            - "auto": try root_3d first, fall back to bbox_center

    Returns:
        [T, traj_dim] trajectory tensor, or None if unavailable
    """
    if source_type in ("root_3d", "auto"):
        key = "gt_skeleton_meters" if skeleton_source == "gt" else "gt_skeleton_meters"
        if key in data:
            skel = data[key][st:ed, person_idx]  # [T, 17, 3]
            root = skel[:, 0, :]  # [T, 3]
            return torch.from_numpy(root.astype("float32"))

    bbox_key = "gt_bboxes" if skeleton_source == "gt" else "extract_bboxes"
    if bbox_key in data:
        bboxes = data[bbox_key][st:ed, person_idx]  # [T, 4] (xmin, ymin, xmax, ymax)
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        cx_t = torch.from_numpy(cx).float()
        cy_t = torch.from_numpy(cy).float()

        if source_type in ("bbox_center", "auto"):
            return torch.stack([cx_t, cy_t], dim=-1)

        if source_type == "bbox_full":
            w = torch.from_numpy(np.abs(bboxes[:, 2] - bboxes[:, 0])).float()
            h = torch.from_numpy(np.abs(bboxes[:, 3] - bboxes[:, 1])).float()
            return torch.stack([cx_t, cy_t, w, h], dim=-1)

        if source_type == "bbox_vel":
            vx = np.zeros_like(cx, dtype=np.float32)
            vy = np.zeros_like(cy, dtype=np.float32)
            vx[1:] = cx[1:] - cx[:-1]
            vy[1:] = cy[1:] - cy[:-1]
            return torch.stack([
                cx_t, cy_t,
                torch.from_numpy(vx), torch.from_numpy(vy)
            ], dim=-1)

    return None


class GlobalVideoEncoder(nn.Module):
    """Video encoder that fuses local skeleton features with global motion features.

    This is a drop-in replacement for VideoEncoder when global motion is available.
    """

    def __init__(
        self,
        local_encoder: nn.Module,
        global_encoder: GlobalMotionEncoder,
        embed_dim: int = 512,
        fusion_type: Literal["concat", "add", "gated", "gated_residual", "film"] = "concat",
    ) -> None:
        super().__init__()
        self.local_encoder = local_encoder
        self.global_encoder = global_encoder
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim + embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
            )
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "add":
            # Requires global_encoder output_dim == embed_dim
            self.fusion = nn.Identity()
        elif fusion_type == "film":
            self.film = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim * 2, embed_dim * 2),
            )
        elif fusion_type == "gated_residual":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        skeleton: torch.Tensor,
        root_trajectory: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            skeleton: [B, T, 17, 3] normalized skeleton
            root_trajectory: [B, T, traj_dim] global trajectory (optional)
            return_components: Whether to return intermediate local/global embeddings
        Returns:
            [B, embed_dim] fused video embedding, or tuple when return_components=True
        """
        z_local = self.local_encoder(skeleton)  # [B, embed_dim]

        if root_trajectory is None:
            # Fallback: return local only (for compatibility)
            if return_components:
                return z_local, z_local, None
            return z_local

        z_global = self.global_encoder(root_trajectory)  # [B, embed_dim]

        if self.fusion_type == "add":
            fused = z_local + z_global
            if return_components:
                return fused, z_local, z_global
            return fused

        if self.fusion_type == "film":
            gamma, beta = self.film(z_global).chunk(2, dim=-1)
            out = z_local * (1.0 + gamma) + beta
            if return_components:
                return out, z_local, z_global
            return out

        combined = torch.cat([z_local, z_global], dim=-1)  # [B, embed_dim*2]

        if self.fusion_type == "gated":
            gate = self.gate(combined)
            fused = self.fusion(combined)
            out = gate * z_local + (1 - gate) * fused
            if return_components:
                return out, z_local, z_global
            return out

        if self.fusion_type == "gated_residual":
            gate = self.gate(combined)
            fused = self.fusion(combined)
            out = z_local + gate * fused
            if return_components:
                return out, z_local, z_global
            return out

        out = self.fusion(combined)  # concat
        if return_components:
            return out, z_local, z_global
        return out
