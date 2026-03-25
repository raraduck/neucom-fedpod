"""Evaluation metrics."""
import torch


def dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice coefficient.

    Args:
        pred:   binary tensor [B, C, ...]
        target: binary tensor [B, C, ...]
    Returns:
        Dice per (batch, channel) [B, C]
    """
    axes = tuple(range(2, pred.dim()))
    inter = (pred * target).sum(dim=axes)
    union = pred.sum(dim=axes) + target.sum(dim=axes)
    return (2 * inter + eps) / (union + eps)
