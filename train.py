from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from datasets.factory import build_dataset_pair
from methods import build_method
from utils.seed import set_seed


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "checkpoints"

CONFIG = {
    "method": "linear_probe",
    "dataset": "ecg5000",
    "train_file": None,  # e.g. DATA_DIR / "MYDATA_TRAIN.ts"
    "test_file": None,   # e.g. DATA_DIR / "MYDATA_TEST.ts"
    "batch_size": 64,
    "epochs": 5,
    "lr": 1e-3,
    "seed": 42,
    "num_workers": 0,
    "model_name": "AutonLab/MOMENT-1-base",
}


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()

    preds_list = []
    targets_list = []

    for batch_x, batch_mask, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)

        logits, _ = model(batch_x, batch_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds_list.append(preds)
        targets_list.append(batch_y.numpy())

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(targets_list)
    return y_true, y_pred


def build_dataloaders(config):
    train_dataset, test_dataset, label_encoder, dataset_info = build_dataset_pair(
        dataset_name=config["dataset"],
        data_dir=DATA_DIR,
        train_file=config["train_file"],
        test_file=config["test_file"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_dataset, test_dataset, label_encoder, dataset_info, train_loader, test_loader


def print_run_info(config, dataset_info, label_encoder, method):
    print(f"Using device: {DEVICE}")
    print(f"Method: {config['method']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Train file: {dataset_info['train_file']}")
    print(f"Test file:  {dataset_info['test_file']}")
    print(f"Embedding dim: {method.embedding_dim}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {[str(c) for c in label_encoder.classes_]}")


def train_and_evaluate(method, train_loader, test_loader, config, label_encoder):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = SAVE_DIR / f"{config['dataset']}_{config['method']}_best.pt"
    best_test_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)

        print(
            f"Epoch [{epoch:02d}/{config['epochs']:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f}"
        )

        # Save the model
        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            method.save(
                checkpoint_path=checkpoint_path,
                label_classes=label_encoder.classes_.tolist(),
                dataset_name=config["dataset"],
                extra_config=config,
            )

    return checkpoint_path, best_test_acc


def print_final_results(method, checkpoint_path, test_loader, label_encoder):
    method.load(checkpoint_path)
    y_true, y_pred = collect_predictions(method.model, test_loader, device=DEVICE)

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


def main():
    set_seed(CONFIG["seed"])

    (
        train_dataset,
        test_dataset,
        label_encoder,
        dataset_info,
        train_loader,
        test_loader,
    ) = build_dataloaders(CONFIG)

    method = build_method(
        method_name=CONFIG["method"],
        model_name=CONFIG["model_name"],
        num_classes=len(label_encoder.classes_),
        train_dataset=train_dataset,
        device=DEVICE,
        lr=CONFIG["lr"],
    )

    print_run_info(CONFIG, dataset_info, label_encoder, method)

    checkpoint_path, best_test_acc = train_and_evaluate(
        method=method,
        train_loader=train_loader,
        test_loader=test_loader,
        config=CONFIG,
        label_encoder=label_encoder,
    )

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    print_final_results(method, checkpoint_path, test_loader, label_encoder)


if __name__ == "__main__":
    main()