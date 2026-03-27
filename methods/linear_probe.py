import torch

from methods.base import BaseMethod
from trainers.linear_probe_trainer import train_one_epoch, evaluate


class LinearProbeMethod(BaseMethod):
    def __init__(self, model_name, num_classes, train_dataset, device="cpu", lr=1e-3):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            train_dataset=train_dataset,
            device=device,
            lr=lr,
        )

        # only head is trainable
        self.optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.lr)

    def train_epoch(self, dataloader):
        return train_one_epoch(
            model=self.model,
            dataloader=dataloader,
            optimizer=self.optimizer,
            device=self.device,
        )

    def evaluate(self, dataloader):
        return evaluate(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
        )