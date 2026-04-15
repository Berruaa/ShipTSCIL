import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.linear_probe import LinearProbeMethod
from utils.losses import class_balanced_ce_loss
from utils.replay_buffers import HerdingBuffer, build_replay_buffer


class CILReplayRawMethod(LinearProbeMethod):
    """
    Class-incremental learning with raw sample replay.

    Keeps a fixed-size memory buffer of samples.  When the dataloader
    provides precomputed embeddings the buffer stores (embedding, label)
    pairs (equivalent to latent replay); when it provides raw time-series
    the buffer stores (x, mask, label) tuples and re-encodes them each
    step.  The buffer can be class-balanced or a plain reservoir,
    controlled by the ``balanced_replay`` flag.
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
        herding_replay=False,
        lora_config=None,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
            lora_config=lora_config,
        )
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_batch_size = int(replay_batch_size)
        self.balanced_replay = balanced_replay
        self.balanced_loss = balanced_loss
        self.herding_replay = herding_replay

        self._buffer = build_replay_buffer(
            total_size=self.replay_buffer_size,
            balanced=self.balanced_replay,
            herding=self.herding_replay,
        )
        self._buffer_is_embeddings = False

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        if self.herding_replay:
            self._buffer.begin_task()

    def end_task(self, task_id, seen_classes):
        if self.herding_replay and self._buffer_is_embeddings:
            max_per_class = max(1, self.replay_buffer_size // len(seen_classes))
            self._buffer.rebuild(max_per_class)
            dist = self._buffer.class_distribution()
            print(f"  [Herding] Buffer rebuilt — {sum(dist.values())} exemplars "
                  f"across {len(dist)} classes  (max {max_per_class}/class): {dist}")
        elif self.herding_replay and not self._buffer_is_embeddings:
            print("  [Herding] Skipped: herding requires precomputed embeddings "
                  "(raw time-series path detected).")

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _add_to_replay_emb(self, embeddings, labels):
        if self.replay_buffer_size <= 0:
            return
        emb_cpu = embeddings.detach().cpu()
        lab_cpu = labels.detach().cpu()
        samples = [(emb_cpu[i].clone(), int(lab_cpu[i].item()))
                    for i in range(emb_cpu.size(0))]
        self._buffer.add_batch(samples, [s[1] for s in samples])

    def _add_to_replay_raw(self, batch_x, batch_mask, batch_y):
        if self.replay_buffer_size <= 0:
            return
        bx = batch_x.detach().cpu()
        bm = batch_mask.detach().cpu()
        by = batch_y.detach().cpu()
        samples = [(bx[i].clone(), bm[i].clone(), int(by[i].item()))
                    for i in range(bx.size(0))]
        self._buffer.add_batch(samples, [s[2] for s in samples])

    def _sample_replay_emb(self):
        if len(self._buffer) == 0 or self.replay_batch_size <= 0:
            return None
        selected = self._buffer.sample(self.replay_batch_size)
        if not selected:
            return None
        replay_z = torch.stack([s[0] for s in selected], dim=0).to(self.device).float()
        replay_y = torch.tensor(
            [s[1] for s in selected], dtype=torch.long, device=self.device,
        )
        return replay_z, replay_y

    def _sample_replay_raw(self):
        if len(self._buffer) == 0 or self.replay_batch_size <= 0:
            return None
        selected = self._buffer.sample(self.replay_batch_size)
        if not selected:
            return None
        replay_x = torch.stack([s[0] for s in selected], dim=0).to(self.device).float()
        replay_mask = torch.stack([s[1] for s in selected], dim=0).to(self.device)
        replay_y = torch.tensor(
            [s[2] for s in selected], dtype=torch.long, device=self.device,
        )
        return replay_x, replay_mask, replay_y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        self.model.train()
        self.model.encoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Train",
            leave=False,
            disable=True,
        ):
            precomputed = len(batch) == 2

            if precomputed:
                current_emb, batch_y = batch
                current_emb = current_emb.to(self.device).float()
                batch_y = batch_y.to(self.device)
                self._buffer_is_embeddings = True

                replay = self._sample_replay_emb()
                if replay is not None:
                    replay_z, replay_y = replay
                    train_emb = torch.cat([current_emb, replay_z], dim=0)
                    train_y = torch.cat([batch_y, replay_y], dim=0)
                else:
                    train_emb, train_y = current_emb, batch_y

                self.optimizer.zero_grad()
                logits = self.model.head(train_emb)

            else:
                batch_x, batch_mask, batch_y = batch
                batch_x = batch_x.to(self.device).float()
                batch_mask = batch_mask.to(self.device)
                batch_y = batch_y.to(self.device)

                replay = self._sample_replay_raw()
                if replay is not None:
                    rx, rm, ry = replay
                    train_x = torch.cat([batch_x, rx], dim=0)
                    train_mask = torch.cat([batch_mask, rm], dim=0)
                    train_y = torch.cat([batch_y, ry], dim=0)
                else:
                    train_x, train_mask, train_y = batch_x, batch_mask, batch_y

                self.optimizer.zero_grad()
                logits, _ = self.model(train_x, train_mask)

            if self.balanced_loss:
                loss = class_balanced_ce_loss(logits, train_y)
            else:
                loss = F.cross_entropy(logits, train_y)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits, dim=1)
            n = train_y.size(0)
            total_loss += loss.item() * n
            total_correct += (preds == train_y).sum().item()
            total_samples += n

            if precomputed:
                self._add_to_replay_emb(current_emb, batch_y)
            else:
                self._add_to_replay_raw(
                    batch[0].to(self.device).float(),
                    batch[1].to(self.device),
                    batch_y,
                )

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return {"loss": avg_loss, "acc": avg_acc}
