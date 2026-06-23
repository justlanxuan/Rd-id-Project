"""Matching losses for cross-modal alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """Batch-wise symmetric InfoNCE with implicit in-batch negatives.
    
    This loss aligns two modalities (e.g., IMU and video) by maximizing
    the similarity between positive pairs while pushing apart negatives
    within the batch.
    
    Args:
        temperature: Temperature parameter for softmax scaling
        learn_temperature: Whether to learn the temperature parameter
        
    Example:
        >>> loss_fn = SymmetricInfoNCE(temperature=0.1)
        >>> z_imu = torch.randn(32, 512)  # IMU embeddings
        >>> z_vid = torch.randn(32, 512)  # Video embeddings
        >>> loss = loss_fn(z_imu, z_vid)
    """

    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False) -> None:
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(
                torch.tensor(float(temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "temperature",
                torch.tensor(float(temperature), dtype=torch.float32),
            )

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        labels_a: torch.Tensor | None = None,
        labels_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.
        
        Args:
            z_a: Embeddings from modality A [B, D]
            z_b: Embeddings from modality B [B, D]
            labels_a: Optional labels for z_a -> z_b direction. If None, uses diagonal.
            labels_b: Optional labels for z_b -> z_a direction. If None, uses diagonal.
            
        Returns:
            Scalar loss value
        """
        if z_a.ndim != 2 or z_b.ndim != 2:
            raise ValueError("Expected z_a and z_b to be [B, D].")
        if z_a.shape != z_b.shape:
            raise ValueError(f"Shape mismatch: {z_a.shape} vs {z_b.shape}")

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        t = torch.clamp(self.temperature, min=1e-6)
        logits = torch.matmul(z_a, z_b.t()) / t

        if labels_a is None:
            labels_a = torch.arange(logits.shape[0], device=logits.device)
        if labels_b is None:
            labels_b = torch.arange(logits.shape[1], device=logits.device)

        loss_ab = F.cross_entropy(logits, labels_a)
        loss_ba = F.cross_entropy(logits.t(), labels_b)
        return 0.5 * (loss_ab + loss_ba)


def retrieval_top1(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels_a: torch.Tensor | None = None,
    labels_b: torch.Tensor | None = None,
) -> float:
    """Top-1 paired retrieval accuracy over a batch.
    
    Computes the accuracy of retrieving the correct paired sample
    from one modality given the other modality.
    
    Args:
        z_a: Embeddings from modality A [B, D]
        z_b: Embeddings from modality B [B, D]
        labels_a: Optional labels for z_a -> z_b direction. If None, uses diagonal.
        labels_b: Optional labels for z_b -> z_a direction. If None, uses diagonal.
        
    Returns:
        Top-1 retrieval accuracy (0.0 - 1.0)
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    sims = torch.matmul(z_a, z_b.t())

    if labels_a is None:
        labels_a = torch.arange(sims.shape[0], device=sims.device)
    if labels_b is None:
        labels_b = torch.arange(sims.shape[1], device=sims.device)

    acc_ab = (sims.argmax(dim=1) == labels_a).float().mean()
    acc_ba = (sims.argmax(dim=0) == labels_b).float().mean()
    return float((0.5 * (acc_ab + acc_ba)).item())
