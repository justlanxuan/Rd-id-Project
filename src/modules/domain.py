"""Domain adversarial components (Gradient Reversal Layer + Domain Classifier)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Forward identity, backward scales gradient by -alpha."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrap GradientReversalFunction as a nn.Module with mutable alpha."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """Simple MLP domain classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        return self.net(x)

    def set_alpha(self, alpha: float) -> None:
        self.grl.alpha = alpha


def dann_alpha_schedule(progress: float) -> float:
    """Standard DANN alpha scheduling.

    Args:
        progress: Training progress in [0, 1].

    Returns:
        alpha value in [0, 1].
    """
    import math

    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
