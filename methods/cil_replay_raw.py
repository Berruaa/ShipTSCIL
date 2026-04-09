import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.linear_probe import LinearProbeMethod


class CILReplayRawMethod(LinearProbeMethod):
    """
    Class-incremental learning with raw sample replay.

    Keeps a fixed-size memory buffer of raw (x, mask, y) samples using
    reservoir sampling and mixes replay samples into every training step.
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

        self._replay_buffer = []
        self._replay_seen_samples = 0

    def _add_to_replay(self, batch_x, batch_mask, batch_y):
        """
        Add current batch to replay memory via reservoir sampling.
        """
        if self.replay_buffer_size <= 0:
            return

        batch_x_cpu = batch_x.detach().cpu()
        batch_mask_cpu = batch_mask.detach().cpu()
        batch_y_cpu = batch_y.detach().cpu()

        for i in range(batch_x_cpu.size(0)):
            sample = (batch_x_cpu[i].clone(), batch_mask_cpu[i].clone(), int(batch_y_cpu[i].item()))
            self._replay_seen_samples += 1

            if len(self._replay_buffer) < self.replay_buffer_size:
                self._replay_buffer.append(sample)
                continue

            replace_idx = random.randint(0, self._replay_seen_samples - 1)
            if replace_idx < self.replay_buffer_size:
                self._replay_buffer[replace_idx] = sample

    def _sample_replay_batch(self):
        if not self._replay_buffer or self.replay_batch_size <= 0:
            return None

        sample_size = min(self.replay_batch_size, len(self._replay_buffer))
        replay_samples = random.sample(self._replay_buffer, k=sample_size)

        replay_x = torch.stack([s[0] for s in replay_samples], dim=0).to(self.device).float()
        replay_mask = torch.stack([s[1] for s in replay_samples], dim=0).to(self.device)
        replay_y = torch.tensor([s[2] for s in replay_samples], dtype=torch.long, device=self.device)
        return replay_x, replay_mask, replay_y

    def train_epoch(self, dataloader):
        self.model.train()
        self.model.encoder.eval()  # frozen encoder

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

            replay_batch = self._sample_replay_batch()
            if replay_batch is not None:
                replay_x, replay_mask, replay_y = replay_batch
                train_x = torch.cat([batch_x, replay_x], dim=0)
                train_mask = torch.cat([batch_mask, replay_mask], dim=0)
                train_y = torch.cat([batch_y, replay_y], dim=0)
            else:
                train_x, train_mask, train_y = batch_x, batch_mask, batch_y

            self.optimizer.zero_grad()
            logits, _ = self.model(train_x, train_mask)
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size = train_y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == train_y).sum().item()
            total_samples += batch_size

            self._add_to_replay(batch_x, batch_mask, batch_y)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": avg_acc}
