"""Domain-adversarial model components."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 2) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(x))

    def set_alpha(self, alpha: float) -> None:
        self.grl.alpha = float(alpha)


def dann_alpha_schedule(progress: float) -> float:
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {progress}")
    return float(2.0 / (1.0 + math.exp(-10.0 * float(progress))) - 1.0)
