"""
Loss functions for class-incremental learning.
"""

import torch
import torch.nn.functional as F


def class_balanced_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss with per-batch inverse-frequency class weighting.

    For each training step the class distribution in the combined
    (current + replay) batch is typically skewed: current-task classes
    dominate.  This function computes per-sample weights so that every
    class present in the batch contributes **equally** to the total loss,
    regardless of how many samples it has.

    Weight for a sample of class c = (total_samples) / (num_classes * count_c)

    This is equivalent to computing the mean loss *per class* first, then
    averaging across classes.
    """
    per_sample_loss = F.cross_entropy(logits, targets, reduction="none")

    classes_in_batch, counts = targets.unique(return_counts=True)
    num_classes = classes_in_batch.numel()
    total = targets.numel()

    count_map = torch.zeros(
        targets.max() + 1, device=targets.device, dtype=torch.float32,
    )
    count_map[classes_in_batch] = counts.float()

    sample_counts = count_map[targets]
    weights = total / (num_classes * sample_counts)

    return (per_sample_loss * weights).mean()
