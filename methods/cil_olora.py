"""
Class-incremental learning with Orthogonal LoRA (O-LoRA).

Implements the method from Wang et al. (2023) "Orthogonal Subspace Learning
for Language Model Continual Learning" (EMNLP 2023 Findings), adapted for
time-series classification with the MOMENT encoder.

Each new task gets its own LoRA adapters (A_t, B_t) injected into the
encoder's attention projections.  Previous adapters are frozen, and an
orthogonality loss  L_orth = sum_{i<t} ||A_i^T A_t||_F^2  is added to
the task loss so the current update subspace stays orthogonal to all
past subspaces.  This prevents catastrophic forgetting without needing
replay buffers or distillation.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.base import BaseMethod
from models.head import LinearClassifier
from models.model import MomentModel
from models.olora import OLoRAMomentEncoder
from utils.losses import class_balanced_ce_loss


class CILOLoRAMethod(BaseMethod):
    """O-LoRA: orthogonal subspace continual learning for time series."""

    def __init__(
        self,
        model_name,
        num_classes,
        train_dataset,
        device="cpu",
        lr=1e-3,
        olora_config=None,
        balanced_loss=True,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.device = device
        self.lr = lr
        self.balanced_loss = balanced_loss

        olora_config = olora_config or {}
        self._olora_lambda = olora_config.get("olora_lambda", 1.0)
        self.lora_lr = olora_config.get("lr", lr * 0.2)

        self._lora_enabled = True

        self.encoder = OLoRAMomentEncoder(
            model_name=model_name,
            lora_rank=olora_config.get("rank", 8),
            lora_alpha=olora_config.get("alpha", 16),
            lora_target_modules=olora_config.get("target_modules"),
            lora_dropout=olora_config.get("dropout", 0.05),
        ).to(device)

        self.embedding_dim = self._infer_embedding_dim(train_dataset)
        self.head = LinearClassifier(
            in_dim=self.embedding_dim,
            num_classes=num_classes,
        ).to(device)
        self.model = MomentModel(encoder=self.encoder, head=self.head).to(device)

        self.optimizer = None
        self._current_task_id = 0

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        """Add new orthogonal LoRA adapters and rebuild the optimizer."""
        self._current_task_id = task_id
        self.encoder.add_task()

        param_groups = [
            {"params": list(self.model.head.parameters()), "lr": self.lr},
            {"params": self.encoder.current_lora_parameters(), "lr": self.lora_lr},
        ]
        self.optimizer = torch.optim.Adam(param_groups)

    def end_task(self, task_id, seen_classes):
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        self.model.train()
        self.model.encoder.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_orth = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Train",
            leave=False,
            disable=True,
        ):
            if len(batch) == 2:
                embeddings, batch_y = batch
                embeddings = embeddings.to(self.device).float()
            else:
                batch_x, batch_mask, batch_y = batch
                batch_x = batch_x.to(self.device).float()
                batch_mask = batch_mask.to(self.device)
                embeddings = self.model.encoder(batch_x, batch_mask)
            batch_y = batch_y.to(self.device)

            logits = self.model.head(embeddings)

            if self.balanced_loss:
                ce_loss = class_balanced_ce_loss(logits, batch_y)
            else:
                ce_loss = F.cross_entropy(logits, batch_y)

            orth_loss = self.encoder.orthogonality_loss()
            loss = ce_loss + self._olora_lambda * orth_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            n = batch_y.size(0)
            total_loss += loss.item() * n
            total_ce += ce_loss.item() * n
            total_orth += orth_loss.item() * n
            total_correct += (preds == batch_y).sum().item()
            total_samples += n

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "ce_loss": total_ce / total_samples,
            "orth_loss": total_orth / total_samples,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Eval",
            leave=False,
            disable=True,
        ):
            if len(batch) == 2:
                embeddings, batch_y = batch
                embeddings = embeddings.to(self.device).float()
            else:
                batch_x, batch_mask, batch_y = batch
                batch_x = batch_x.to(self.device).float()
                batch_mask = batch_mask.to(self.device)
                embeddings = self.model.encoder(batch_x, batch_mask)
            batch_y = batch_y.to(self.device)

            logits = self.model.head(embeddings)
            loss = F.cross_entropy(logits, batch_y)

            preds = torch.argmax(logits, dim=1)
            n = batch_y.size(0)
            total_loss += loss.item() * n
            total_correct += (preds == batch_y).sum().item()
            total_samples += n

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": avg_acc}
