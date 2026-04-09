import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.factory import build_dataset_pair


def build_base_datasets_and_info(config, data_dir):
    train_dataset, test_dataset, label_encoder, dataset_info = build_dataset_pair(
        dataset_name=config["dataset"],
        data_dir=data_dir,
        train_file=config["train_file"],
        test_file=config["test_file"],
    )
    return train_dataset, test_dataset, label_encoder, dataset_info


def build_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def extract_targets(dataset):
    targets = []
    for i in range(len(dataset)):
        _, _, y = dataset[i]
        if torch.is_tensor(y):
            y = int(y.item())
        else:
            y = int(y)
        targets.append(y)
    return np.array(targets)


def make_class_subset(dataset, allowed_classes):
    allowed_classes = set(int(c) for c in allowed_classes)
    targets = extract_targets(dataset)
    indices = [i for i, y in enumerate(targets) if int(y) in allowed_classes]
    return Subset(dataset, indices)


def validate_task_splits(task_splits, num_classes):
    flat = [int(c) for task in task_splits for c in task]
    unique = sorted(set(flat))

    if len(flat) != len(unique):
        raise ValueError("task_splits contains duplicate classes.")

    if min(unique) < 0 or max(unique) >= num_classes:
        raise ValueError(
            f"task_splits contains invalid class ids. "
            f"Valid range is [0, {num_classes - 1}]"
        )

    if len(unique) != num_classes:
        raise ValueError(
            f"task_splits must cover all classes exactly once. "
            f"Found {len(unique)} unique classes, expected {num_classes}."
        )
