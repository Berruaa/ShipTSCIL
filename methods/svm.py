import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from models.encoder import FrozenMomentEncoder


class SVMMethod:
    def __init__(self, model_name, num_classes, train_dataset, device="cpu", lr=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.device = device
        self.lr = lr  # unused, kept for interface consistency

        self.encoder = FrozenMomentEncoder(model_name=model_name).to(device)
        self.encoder.eval()

        self.embedding_dim = self._infer_embedding_dim(train_dataset)
        self.clf = LinearSVC()

    def _infer_embedding_dim(self, dataset):
        x, mask, _ = dataset[0]
        x = x.unsqueeze(0).to(self.device).float()
        mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.encoder(x, mask)

        return emb.shape[-1]

    @torch.no_grad()
    def _extract_features(self, dataloader):
        self.encoder.eval()

        features = []
        labels = []

        for batch_x, batch_mask, batch_y in dataloader:
            batch_x = batch_x.to(self.device).float()
            batch_mask = batch_mask.to(self.device)

            emb = self.encoder(batch_x, batch_mask)

            features.append(emb.cpu().numpy())
            labels.append(batch_y.cpu().numpy())

        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)
        return X, y

    def train_epoch(self, train_loader):
        X_train, y_train = self._extract_features(train_loader)

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_train)
        acc = accuracy_score(y_train, y_pred)

        return {
            "loss": 0.0,
            "acc": acc,
        }

    def evaluate(self, dataloader):
        X, y_true = self._extract_features(dataloader)
        y_pred = self.clf.predict(X)
        acc = accuracy_score(y_true, y_pred)

        return {
            "loss": 0.0,
            "acc": acc,
        }

    def predict(self, dataloader):
        X, y_true = self._extract_features(dataloader)
        y_pred = self.clf.predict(X)
        return y_true, y_pred

    def save(self, checkpoint_path, label_classes, dataset_name, extra_config=None):
        payload = {
            "clf": self.clf,
            "label_classes": label_classes,
            "dataset_name": dataset_name,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "extra_config": extra_config,
        }
        joblib.dump(payload, checkpoint_path)

    def load(self, checkpoint_path):
        payload = joblib.load(checkpoint_path)
        self.clf = payload["clf"]