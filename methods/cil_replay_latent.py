from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.linear_probe import LinearProbeMethod
from utils.losses import class_balanced_ce_loss, distillation_loss
from utils.replay_buffers import build_replay_buffer


class CILReplayLatentMethod(LinearProbeMethod):
    """
    Class-incremental learning with latent (embedding) replay.

    Stores encoder embeddings (z, y) in a replay buffer.  The buffer can be
    either class-balanced (equal storage per class) or a plain reservoir
    (class-agnostic), controlled by the ``balanced_replay`` flag.

    Because the MOMENT encoder is frozen, stored embeddings are identical to
    freshly computed ones, so replaying through the head alone is lossless
    while saving both memory and compute.

    When ``use_distillation`` is True, an LwF-style knowledge-distillation
    term is added: at each task boundary the head is snapshotted, and during
    training the KL divergence between the old and new head's outputs on
    old-class logits is added to the loss.
    """

    def __init__(
        self,
        model_name,
        num_classes,
        train_dataset,
        device="cpu",
        lr=1e-3,
        replay_buffer_size=1000,
        replay_batch_size=32,
        balanced_replay=True,
        balanced_loss=True,
        use_distillation=False,
        distill_temperature=2.0,
        distill_weight=1.0,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_batch_size = int(replay_batch_size)
        self.balanced_replay = balanced_replay
        self.balanced_loss = balanced_loss

        self.use_distillation = use_distillation
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self._old_head = None
        self._old_classes: list[int] = []

        self._buffer = build_replay_buffer(
            total_size=self.replay_buffer_size, balanced=self.balanced_replay,
        )

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        if self.use_distillation and old_classes:
            self._old_head = deepcopy(self.model.head)
            self._old_head.eval()
            for p in self._old_head.parameters():
                p.requires_grad = False
            self._old_classes = list(old_classes)
        else:
            self._old_head = None
            self._old_classes = []

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _add_to_replay(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Store (embedding, label) pairs into the class-balanced buffer."""
        if self.replay_buffer_size <= 0:
            return

        emb_cpu = embeddings.detach().cpu()
        lab_cpu = labels.detach().cpu()

        samples = [(emb_cpu[i].clone(), int(lab_cpu[i].item()))
                    for i in range(emb_cpu.size(0))]
        labels_list = [s[1] for s in samples]
        self._buffer.add_batch(samples, labels_list)

    def _sample_replay_batch(self):
        """Return a class-balanced (embeddings, labels) mini-batch."""
        if len(self._buffer) == 0 or self.replay_batch_size <= 0:
            return None

        selected = self._buffer.sample(self.replay_batch_size)
        if not selected:
            return None

        replay_z = torch.stack([s[0] for s in selected], dim=0).to(self.device).float()
        replay_y = torch.tensor(
            [s[1] for s in selected],
            dtype=torch.long,
            device=self.device,
        )
        return replay_z, replay_y

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
                current_emb = self.model.encoder(batch_x, batch_mask)

            n_current = current_emb.size(0)

            replay_batch = self._sample_replay_batch()
            if replay_batch is not None:
                replay_z, replay_y = replay_batch
                train_emb = torch.cat([current_emb, replay_z], dim=0)
                train_y = torch.cat([batch_y, replay_y], dim=0)
            else:
                train_emb = current_emb
                train_y = batch_y

            logits = self.model.head(train_emb)

            if self.balanced_loss:
                ce_loss = class_balanced_ce_loss(logits, train_y)
            else:
                ce_loss = F.cross_entropy(logits, train_y)

            # KD on current-task samples only: replay already has hard labels,
            # distillation matters on new-class inputs where old-class logits
            # would otherwise drift unchecked.
            if self._old_head is not None:
                current_logits = logits[:n_current]
                with torch.no_grad():
                    old_logits = self._old_head(current_emb)
                kd_loss = distillation_loss(
                    new_logits=current_logits,
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
            n = train_y.size(0)
            total_loss += loss.item() * n
            total_ce += ce_loss.item() * n
            total_kd += kd_loss.item() * n
            total_correct += (preds == train_y).sum().item()
            total_samples += n

            self._add_to_replay(current_emb, batch_y)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        result = {"loss": avg_loss, "acc": avg_acc}
        if self.use_distillation:
            result["ce_loss"] = total_ce / total_samples
            result["kd_loss"] = total_kd / total_samples
        return result
