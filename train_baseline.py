from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from datasets.ts_dataset import TimeSeriesDataset
from models.encoder import FrozenMomentEncoder
from models.head import LinearClassifier
from models.model import MomentModel
from trainers.linear_probe_trainer import train_one_epoch, evaluate
from utils.seed import set_seed


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "ECG5000_TRAIN.ts"
TEST_FILE = DATA_DIR / "ECG5000_TEST.ts"

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
SEED = 42
MODEL_NAME = "AutonLab/MOMENT-1-base"
SAVE_PATH = PROJECT_ROOT / "linear_probe_best.pt"


@torch.no_grad()
def collect_predictions(model, dataloader, device="cpu"):
    model.eval()

    all_preds = []
    all_targets = []

    for batch_x, batch_mask, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)

        logits, _ = model(batch_x, batch_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return y_true, y_pred


def infer_embedding_dim(encoder, dataset, device="cpu"):
    sample_x, sample_mask, _ = dataset[0]

    sample_x = sample_x.unsqueeze(0).to(device).float()
    sample_mask = sample_mask.unsqueeze(0).to(device)

    embeddings = encoder(sample_x, sample_mask)
    return embeddings.shape[-1]


def build_datasets(train_file, test_file):
    train_dataset = TimeSeriesDataset.from_tsfile(
        train_file,
        fit_label_encoder=True,
    )
    label_encoder = train_dataset.label_encoder

    test_dataset = TimeSeriesDataset.from_tsfile(
        test_file,
        label_encoder=label_encoder,
        fit_label_encoder=False,
    )

    return train_dataset, test_dataset, label_encoder


def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test file:  {TEST_FILE}")

    train_dataset, test_dataset, label_encoder = build_datasets(TRAIN_FILE, TEST_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    encoder = FrozenMomentEncoder(model_name=MODEL_NAME).to(DEVICE)
    embedding_dim = infer_embedding_dim(encoder, train_dataset, device=DEVICE)
    num_classes = len(label_encoder.classes_)

    print(f"Embedding dim: {embedding_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {[str(c) for c in label_encoder.classes_]}")

    head = LinearClassifier(in_dim=embedding_dim, num_classes=num_classes)
    model = MomentModel(encoder=encoder, head=head).to(DEVICE)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)

    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=DEVICE)
        test_metrics = evaluate(model, test_loader, device=DEVICE)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f}"
        )

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": model.head.state_dict(),
                    "label_classes": label_encoder.classes_.tolist(),
                    "embedding_dim": embedding_dim,
                    "num_classes": num_classes,
                    "model_name": MODEL_NAME,
                },
                SAVE_PATH,
            )
            # print(f"Saved best model to: {SAVE_PATH}")

    print(f"\nBest test accuracy: {best_test_acc:.4f}")

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = collect_predictions(model, test_loader, device=DEVICE)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()