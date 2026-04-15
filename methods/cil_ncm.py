"""
FastICARL-style Nearest-Class-Mean (NCM) class-incremental learning.

A class prototype (mean embedding) is maintained for every seen class.
At inference, a sample is assigned to the class whose prototype is nearest
in L2 distance — no linear head is needed and no gradient updates are
performed.

With a frozen MOMENT encoder the prototype for a class is perfectly stable
once computed: there is no catastrophic forgetting of class knowledge.

Key properties
--------------
* **Zero forgetting** — old-class prototypes are stored explicitly and
  never modified after the task that introduced them.
* **No hyperparameters** — no learning rate, no replay buffer, no KD weight.
* **Single-epoch equivalence** — multiple epochs repeat the same computation
  (frozen encoder → identical embeddings).  Set ``epochs=1`` in config to
  avoid redundant passes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.base import BaseMethod


class CILNCMMethod(BaseMethod):
    """
    NCM-based CIL method (FastICARL-style).

    Maintains ``self.prototypes``: a ``{class_label: mean_embedding}`` dict
    (CPU tensors).  Each new task's prototypes are computed from the task's
    training data.  Old-class prototypes are unchanged.

    ``collect_predictions`` in the evaluation pipeline detects the ``predict``
    method and routes through NCM automatically, bypassing the unused linear
    head.
    """

    def __init__(self, model_name, num_classes, train_dataset, device="cpu", lr=1e-3):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )
        # {class_label (int): mean embedding (CPU float32 Tensor, shape (D,))}
        self.prototypes: dict[int, torch.Tensor] = {}
        self._current_task_classes: list[int] = []

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        """Note which classes are being introduced; clear their stale prototypes."""
        self._current_task_classes = list(task_classes)
        for c in task_classes:
            self.prototypes.pop(c, None)

    def end_task(self, task_id, seen_classes):
        print(f"  [NCM] Prototypes computed for classes: "
              f"{sorted(self.prototypes.keys())}")

    # ------------------------------------------------------------------
    # Embedding collection helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_embeddings(
        self, dataloader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-pass the whole dataloader and return CPU tensors.

        Returns
        -------
        all_embs   : (N, D) float32 on CPU
        all_labels : (N,)   int64  on CPU
        """
        self.model.eval()
        emb_parts = []
        label_parts = []

        for batch in tqdm(dataloader, desc="NCM embed", leave=False, disable=True):
            if len(batch) == 2:
                emb, y = batch
                emb = emb.to(self.device).float()
            else:
                x, mask, y = batch
                emb = self.model.encoder(
                    x.to(self.device).float(), mask.to(self.device)
                )
            emb_parts.append(emb.cpu())
            label_parts.append(y.cpu())

        return torch.cat(emb_parts, dim=0), torch.cat(label_parts, dim=0)

    # ------------------------------------------------------------------
    # NCM core
    # ------------------------------------------------------------------

    def _proto_matrix(self) -> tuple[list[int], torch.Tensor]:
        """Return (sorted_labels, (K, D) prototype matrix) on self.device."""
        labels = sorted(self.prototypes.keys())
        matrix = torch.stack(
            [self.prototypes[l] for l in labels], dim=0
        ).to(self.device)
        return labels, matrix

    @torch.no_grad()
    def _ncm_predict(
        self, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Classify a batch of embeddings by nearest prototype (L2 distance).

        Parameters
        ----------
        emb : (N, D) float tensor on self.device

        Returns
        -------
        preds       : (N,) predicted class labels in the true label space
        neg_dists   : (N, K) negative L2 distances (usable as logits for CE)
        proto_labels: list[int] of length K — column ordering of neg_dists
        """
        proto_labels, proto_matrix = self._proto_matrix()
        dists = torch.cdist(emb.float(), proto_matrix.float())     # (N, K)
        best_local = torch.argmin(dists, dim=1)                    # (N,)
        preds = torch.tensor(
            [proto_labels[i] for i in best_local.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        return preds, -dists, proto_labels

    def _ncm_loss_acc(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute CE loss and accuracy for a set of embeddings + true labels."""
        preds, neg_dists, proto_labels = self._ncm_predict(emb)
        label_to_col = {l: i for i, l in enumerate(proto_labels)}

        # Filter to samples whose true class has a prototype (should be all).
        valid_mask = torch.tensor(
            [l.item() in label_to_col for l in labels],
            dtype=torch.bool,
            device=self.device,
        )
        if not valid_mask.any():
            return 0.0, 0.0

        y_local = torch.tensor(
            [label_to_col[l.item()] for l in labels[valid_mask]],
            dtype=torch.long,
            device=self.device,
        )
        loss = F.cross_entropy(neg_dists[valid_mask], y_local).item()
        acc = (preds[valid_mask] == labels[valid_mask]).float().mean().item()
        return loss, acc

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader) -> dict:
        """
        Compute / refresh prototypes for current-task classes, then report
        NCM accuracy and loss on the full training batch as diagnostics.

        No gradient updates are performed.
        """
        all_embs, all_labels = self._collect_embeddings(dataloader)

        # Update prototypes for the current task's classes.
        for label in self._current_task_classes:
            class_mask = all_labels == label
            if class_mask.any():
                self.prototypes[label] = all_embs[class_mask].mean(dim=0)

        if not self.prototypes:
            return {"loss": 0.0, "acc": 0.0}

        loss, acc = self._ncm_loss_acc(
            all_embs.to(self.device), all_labels.to(self.device)
        )
        return {"loss": loss, "acc": acc}

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, dataloader) -> dict:
        """NCM evaluation on any dataloader (seen or all classes)."""
        if not self.prototypes:
            return {"loss": 0.0, "acc": 0.0}

        all_embs, all_labels = self._collect_embeddings(dataloader)
        loss, acc = self._ncm_loss_acc(
            all_embs.to(self.device), all_labels.to(self.device)
        )
        return {"loss": loss, "acc": acc}

    # ------------------------------------------------------------------
    # predict  —  used by collect_predictions() in pipelines/evaluation.py
    # ------------------------------------------------------------------

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (y_true, y_pred) as numpy arrays using NCM classification.

        ``collect_predictions`` in the evaluation pipeline detects this method
        and calls it in preference to the linear-head fallback.
        """
        all_embs, all_labels = self._collect_embeddings(dataloader)
        preds, _, _ = self._ncm_predict(all_embs.to(self.device))
        return all_labels.numpy(), preds.cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint: persist prototypes alongside model state
    # ------------------------------------------------------------------

    def save(self, checkpoint_path, label_classes, dataset_name, extra_config=None):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "head_state_dict":  self.model.head.state_dict(),
            "label_classes":    label_classes,
            "embedding_dim":    self.embedding_dim,
            "num_classes":      self.num_classes,
            "model_name":       self.model_name,
            "dataset_name":     dataset_name,
            "extra_config":     extra_config or {},
            # NCM-specific
            "prototypes": {k: v.clone() for k, v in self.prototypes.items()},
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "prototypes" in checkpoint:
            self.prototypes = {
                k: v.cpu() for k, v in checkpoint["prototypes"].items()
            }
        return checkpoint
