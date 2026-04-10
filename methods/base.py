import torch

from models.encoder import FrozenMomentEncoder
from models.head import LinearClassifier
from models.model import MomentModel


class BaseMethod:
    def __init__(self, model_name, num_classes, train_dataset, device="cpu", lr=1e-3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.device = device
        self.lr = lr

        self.encoder = FrozenMomentEncoder(model_name=model_name).to(device)
        self.embedding_dim = self._infer_embedding_dim(train_dataset)
        self.head = LinearClassifier(
            in_dim=self.embedding_dim,
            num_classes=num_classes,
        ).to(device)
        self.model = MomentModel(encoder=self.encoder, head=self.head).to(device)

        self.optimizer = None

    @torch.no_grad()
    def _infer_embedding_dim(self, dataset):
        sample_x, sample_mask, _ = dataset[0]
        sample_x = sample_x.unsqueeze(0).to(self.device).float()
        sample_mask = sample_mask.unsqueeze(0).to(self.device)

        embeddings = self.encoder(sample_x, sample_mask)
        return embeddings.shape[-1]

    # ------------------------------------------------------------------
    # Task lifecycle hooks (overridden by methods that need them, e.g. LwF)
    # ------------------------------------------------------------------

    def begin_task(self, task_id, task_classes, old_classes):
        """Called before training on a new task. Default: no-op."""

    def end_task(self, task_id, seen_classes):
        """Called after all epochs for a task are complete. Default: no-op."""

    # ------------------------------------------------------------------

    def train_epoch(self, dataloader):
        raise NotImplementedError

    def evaluate(self, dataloader):
        raise NotImplementedError

    def save(self, checkpoint_path, label_classes, dataset_name, extra_config=None):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "head_state_dict": self.model.head.state_dict(),
            "label_classes": label_classes,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "dataset_name": dataset_name,
            "extra_config": extra_config or {},
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint