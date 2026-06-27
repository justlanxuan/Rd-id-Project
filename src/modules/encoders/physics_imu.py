"""Physics-feature-based IMU encoder with Transformer aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.encoders.base import BaseEncoder
from src.modules.encoders.physics_features import IMUPhysicsFeatureExtractor


class PhysicsIMUEncoder(BaseEncoder):
    """IMU encoder that extracts physics features and aggregates with Transformer.

    Input: raw IMU [B, T, 48] (single-sensor repeated format)
    Output: embedding [B, embed_dim] (aligned with video encoder)

    Internally extracts 6 physics tokens:
      - acc_x stats, acc_y stats, acc_z stats, acc_norm stats,
      - acc_norm freq features, rotation stats
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        embed_dim: int = 512,
        fs_hz: float = 30.0,
        n_fft: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        self.feature_extractor = IMUPhysicsFeatureExtractor(fs_hz=fs_hz, n_fft=n_fft)

        # Each of the 6 tokens has different raw dimension; project to d_model
        self.token_projs = nn.ModuleList([
            nn.Linear(8, d_model),   # acc_x stats
            nn.Linear(8, d_model),   # acc_y stats
            nn.Linear(8, d_model),   # acc_z stats
            nn.Linear(8, d_model),   # acc_norm stats
            nn.Linear(6, d_model),   # acc_norm freq
            nn.Linear(18, d_model),  # rot stats
        ])

        # Learnable positional encoding for 6 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, T, 48]. Returns: [B, embed_dim]."""
        feat_list = self.feature_extractor(x)  # list of [B, D]

        # Project each token to d_model and stack
        tokens = []
        for i, feat in enumerate(feat_list):
            tokens.append(self.token_projs[i](feat))  # [B, d_model]

        h = torch.stack(tokens, dim=1)  # [B, 6, d_model]
        h = h + self.pos_embed          # add positional encoding
        h = self.transformer(h)         # [B, 6, d_model]
        h = h.mean(dim=1)               # mean pool over tokens → [B, d_model]
        return self.output_proj(h)      # [B, embed_dim]
