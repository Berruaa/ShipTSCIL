import torch

from methods.base import BaseMethod
from trainers.linear_probe_trainer import train_one_epoch, evaluate


class ReplayDummyMethod(BaseMethod):
    """
    Dummy replay method:
    - currently behaves like linear probing
    - stores placeholders for future replay buffer integration
    """

    def __init__(self, model_name, num_classes, train_dataset, device="cpu", lr=1e-3):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

        self.optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.lr)

        # future replay-related attributes
        self.memory_buffer = []
        self.memory_size = 0
        self.replay_batch_size = 0

    def update_memory(self, batch_x, batch_mask, batch_y):
        # dummy placeholder
        # later you can store exemplars here
        pass

    def train_epoch(self, dataloader):
        metrics = train_one_epoch(
            model=self.model,
            dataloader=dataloader,
            optimizer=self.optimizer,
            device=self.device,
        )

        # future: combine current batch + replay batch
        return metrics

    def evaluate(self, dataloader):
        return evaluate(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
        )