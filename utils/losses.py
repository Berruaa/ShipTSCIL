"""
Loss functions for class-incremental learning.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    new_logits: torch.Tensor,
    old_logits: torch.Tensor,
    old_classes: list[int],
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Knowledge-distillation loss (LwF) over old-class logits.

    Computes KL divergence between temperature-scaled softmax distributions
    of the *old* (frozen snapshot) and *new* (current) head, restricted to
    the subset of outputs that correspond to previously learned classes.

    Returns 0 when ``old_classes`` is empty (first task).
    """
    if not old_classes:
        return new_logits.new_tensor(0.0)

    old_idx = torch.tensor(old_classes, dtype=torch.long, device=new_logits.device)

    soft_targets = F.softmax(old_logits[:, old_idx] / temperature, dim=1)
    log_soft_preds = F.log_softmax(new_logits[:, old_idx] / temperature, dim=1)

    return F.kl_div(log_soft_preds, soft_targets, reduction="batchmean") * (temperature ** 2)


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
