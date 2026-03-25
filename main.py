from pathlib import Path
import torch
from torch.utils.data import DataLoader
from momentfm.models.statistical_classifiers import fit_svm

from data_utils import TimeSeriesDataset
from moment_utils import load_moment_model, get_embeddings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def run_tsfile_experiment(train_file, test_file):
    model = load_moment_model(device=DEVICE)

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_embeddings, train_labels = get_embeddings(model, train_loader, device=DEVICE)
    test_embeddings, test_labels = get_embeddings(model, test_loader, device=DEVICE)

    clf = fit_svm(features=train_embeddings, y=train_labels)

    print(f"Train accuracy: {clf.score(train_embeddings, train_labels):.4f}")
    print(f"Test accuracy: {clf.score(test_embeddings, test_labels):.4f}")


def main():
    train_file = DATA_DIR / "ECG5000_TRAIN.ts"
    test_file = DATA_DIR / "ECG5000_TEST.ts"
    run_tsfile_experiment(train_file, test_file)


if __name__ == "__main__":
    main()