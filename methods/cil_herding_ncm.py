"""
Version B FastICARL: Herding exemplar selection + NCM inference.

This method combines two components built independently in this codebase:

* **Herding** (iCaRL-style) — at the end of each task, greedy herding
  selects the *k* exemplars per class whose running mean best approximates
  the true class-mean embedding.

* **NCM classifier** — at inference, a sample is assigned to the class
  whose exemplar mean is nearest in L2 distance.  No linear head is trained
  or used.

Comparison with the two simpler methods:

+----------------------+----------------------------+----------------------------+
| Property             | cil_ncm (Version A)        | cil_herding_ncm (Version B)|
+======================+============================+============================+
| Prototype source     | mean of ALL training embs  | mean of herding exemplars  |
| Memory after task    | zero (only μ stored)       | k embeddings per class     |
| Herding used?        | no                         | yes                        |
| Linear head trained? | no                         | no                         |
+----------------------+----------------------------+----------------------------+

During training ``_get_prototypes()`` blends two sources so evaluation
metrics are always meaningful:
* Old classes  → mean of committed herding exemplars.
* Current task → mean of staged (pre-rebuild) embeddings.

After ``end_task`` / rebuild, staging is cleared and all prototypes come
from herding exemplars.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.base import BaseMethod
from utils.replay_buffers import HerdingBuffer


class CILHerdingNCMMethod(BaseMethod):
    """
    FastICARL Version B: herding exemplar selection + NCM inference.

    Parameters
    ----------
    replay_buffer_size : total exemplar budget across all classes.
                         Per-class budget = replay_buffer_size // num_seen_classes,
                         recomputed and trimmed after each task.
    """

    def __init__(
        self,
        model_name,
        num_classes,
        train_dataset,
        device="cpu",
        lr=1e-3,
        replay_buffer_size=1000,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )
        self.replay_buffer_size = int(replay_buffer_size)
        self._buffer = HerdingBuffer(self.replay_buffer_size)
        self._current_task_classes: list[int] = []

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        self._current_task_classes = list(task_classes)
        self._buffer.begin_task()

    def end_task(self, task_id, seen_classes):
        max_per_class = max(1, self.replay_buffer_size // len(seen_classes))
        self._buffer.rebuild(max_per_class)
        dist = self._buffer.class_distribution()
        print(f"  [HerdingNCM] Exemplars rebuilt — {sum(dist.values())} total "
              f"across {len(dist)} classes  (max {max_per_class}/class): {dist}")

    # ------------------------------------------------------------------
    # Prototype helpers
    # ------------------------------------------------------------------

    def _get_prototypes(self) -> dict[int, torch.Tensor]:
        """
        Build the best available prototype dict (CPU tensors).

        Priority
        --------
        * Committed exemplars (post-rebuild) — used for old classes and for
          the current task after ``end_task`` has run.
        * Staging means (pre-rebuild)        — used for current-task classes
          while training is still in progress.

        This makes ``evaluate`` meaningful at every point in the loop.
        """
        prototypes: dict[int, torch.Tensor] = {}

        # Committed exemplars (old classes + current task after rebuild).
        for label, exemplars in self._buffer._exemplars.items():
            if exemplars:
                embs = torch.stack([e[0] for e in exemplars])
                prototypes[label] = embs.mean(dim=0)

        # Staging (current task, mid-training — overrides if already set above,
        # but _exemplars won't have current-task labels until after rebuild).
        for label, staged in self._buffer._staging.items():
            if staged and label not in prototypes:
                embs = torch.stack([s[0] for s in staged])
                prototypes[label] = embs.mean(dim=0)

        return prototypes

    # ------------------------------------------------------------------
    # NCM core  (mirrors CILNCMMethod)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ncm_predict(
        self,
        emb: torch.Tensor,
        prototypes: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Classify *emb* (N, D) by nearest prototype.

        Returns
        -------
        preds       : (N,) predicted labels in the true label space
        neg_dists   : (N, K) negative L2 distances (usable as CE logits)
        proto_labels: column ordering of neg_dists
        """
        proto_labels = sorted(prototypes.keys())
        proto_matrix = torch.stack(
            [prototypes[l] for l in proto_labels], dim=0
        ).to(self.device)                                          # (K, D)

        dists = torch.cdist(emb.float(), proto_matrix.float())    # (N, K)
        best_local = torch.argmin(dists, dim=1)                   # (N,)
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
        prototypes: dict[int, torch.Tensor],
    ) -> tuple[float, float]:
        """Return (CE loss, accuracy) using NCM classification."""
        preds, neg_dists, proto_labels = self._ncm_predict(emb, prototypes)
        label_to_col = {l: i for i, l in enumerate(proto_labels)}

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
    # Embedding collection helper  (mirrors CILNCMMethod)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_embeddings(
        self, dataloader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (N, D) and (N,) CPU tensors from a full dataloader pass."""
        self.model.eval()
        emb_parts, label_parts = [], []

        for batch in tqdm(dataloader, desc="embed", leave=False, disable=True):
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
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader) -> dict:
        """
        Accumulate embeddings into the herding buffer's staging area, then
        report NCM accuracy/loss as a diagnostic.  No gradient updates.
        """
        all_embs, all_labels = self._collect_embeddings(dataloader)

        # Stage current-task samples for herding (rebuild happens in end_task).
        samples = [(all_embs[i], int(all_labels[i].item()))
                   for i in range(all_embs.size(0))]
        self._buffer.add_batch(samples, [s[1] for s in samples])

        prototypes = self._get_prototypes()
        if not prototypes:
            return {"loss": 0.0, "acc": 0.0}

        loss, acc = self._ncm_loss_acc(
            all_embs.to(self.device), all_labels.to(self.device), prototypes
        )
        return {"loss": loss, "acc": acc}

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, dataloader) -> dict:
        prototypes = self._get_prototypes()
        if not prototypes:
            return {"loss": 0.0, "acc": 0.0}

        all_embs, all_labels = self._collect_embeddings(dataloader)
        loss, acc = self._ncm_loss_acc(
            all_embs.to(self.device), all_labels.to(self.device), prototypes
        )
        return {"loss": loss, "acc": acc}

    # ------------------------------------------------------------------
    # predict  —  hook used by collect_predictions() in evaluation.py
    # ------------------------------------------------------------------

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (y_true, y_pred) via NCM.  Called by ``collect_predictions``
        before the linear-head fallback.
        """
        prototypes = self._get_prototypes()
        all_embs, all_labels = self._collect_embeddings(dataloader)
        if not prototypes:
            return all_labels.numpy(), np.zeros_like(all_labels.numpy())

        preds, _, _ = self._ncm_predict(all_embs.to(self.device), prototypes)
        return all_labels.numpy(), preds.cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint: persist exemplars so prototypes survive save/load
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
            # Herding NCM-specific: store the full exemplar dict.
            "exemplars": {
                label: [(emb.clone(), lbl) for emb, lbl in exemplars]
                for label, exemplars in self._buffer._exemplars.items()
            },
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "exemplars" in checkpoint:
            self._buffer._exemplars = {
                k: [(emb.cpu(), lbl) for emb, lbl in v]
                for k, v in checkpoint["exemplars"].items()
            }
            self._buffer._staging.clear()
        return checkpoint
