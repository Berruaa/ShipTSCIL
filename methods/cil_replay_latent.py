import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.linear_probe import LinearProbeMethod
from utils.losses import class_balanced_ce_loss
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

        self._buffer = build_replay_buffer(
            total_size=self.replay_buffer_size, balanced=self.balanced_replay,
        )

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

            replay_batch = self._sample_replay_batch()
            if replay_batch is not None:
                replay_z, replay_y = replay_batch
                train_emb = torch.cat([current_emb, replay_z], dim=0)
                train_y = torch.cat([batch_y, replay_y], dim=0)
            else:
                train_emb = current_emb
                train_y = batch_y

            self.optimizer.zero_grad()
            logits = self.model.head(train_emb)
            if self.balanced_loss:
                loss = class_balanced_ce_loss(logits, train_y)
            else:
                loss = F.cross_entropy(logits, train_y)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size = train_y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == train_y).sum().item()
            total_samples += batch_size

            self._add_to_replay(current_emb, batch_y)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": avg_acc}
