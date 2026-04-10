"""
Class-incremental learning with Learning without Forgetting (LwF).

Frozen MOMENT encoder + trainable linear head.  At each task boundary
the current head is snapshotted; during training the loss is:

    loss = CE + λ · KD

where CE is the classification loss on current-task data and KD is a
KL-divergence distillation term that encourages the head's outputs on
*old* classes to stay close to the snapshot's outputs.

No replay buffer is used — knowledge of previous tasks is preserved
entirely through distillation.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.linear_probe import LinearProbeMethod
from utils.losses import class_balanced_ce_loss, distillation_loss


class CILLwFMethod(LinearProbeMethod):

    def __init__(
        self,
        model_name,
        num_classes,
        train_dataset,
        device="cpu",
        lr=1e-3,
        distill_temperature=2.0,
        distill_weight=1.0,
        balanced_loss=True,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self.balanced_loss = balanced_loss

        self._old_head = None
        self._old_classes: list[int] = []

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        """Snapshot the head before learning a new task."""
        if old_classes:
            self._old_head = deepcopy(self.model.head)
            self._old_head.eval()
            for p in self._old_head.parameters():
                p.requires_grad = False
            self._old_classes = list(old_classes)
        else:
            self._old_head = None
            self._old_classes = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        self.model.train()
        self.model.encoder.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_mask, batch_y in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Train",
            leave=False,
            disable=True,
        ):
            batch_x = batch_x.to(self.device).float()
            batch_mask = batch_mask.to(self.device)
            batch_y = batch_y.to(self.device)

            with torch.no_grad():
                embeddings = self.model.encoder(batch_x, batch_mask)

            logits = self.model.head(embeddings)

            if self.balanced_loss:
                ce_loss = class_balanced_ce_loss(logits, batch_y)
            else:
                ce_loss = F.cross_entropy(logits, batch_y)

            if self._old_head is not None:
                with torch.no_grad():
                    old_logits = self._old_head(embeddings)
                kd_loss = distillation_loss(
                    new_logits=logits,
                    old_logits=old_logits,
                    old_classes=self._old_classes,
                    temperature=self.distill_temperature,
                )
                loss = ce_loss + self.distill_weight * kd_loss
            else:
                kd_loss = logits.new_tensor(0.0)
                loss = ce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            n = batch_y.size(0)
            total_loss += loss.item() * n
            total_ce += ce_loss.item() * n
            total_kd += kd_loss.item() * n
            total_correct += (preds == batch_y).sum().item()
            total_samples += n

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "ce_loss": total_ce / total_samples,
            "kd_loss": total_kd / total_samples,
        }
