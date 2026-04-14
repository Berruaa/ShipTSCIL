import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.factory import build_dataset_pair
from datasets.ts_dataset import EmbeddingDataset


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


def _embedding_cache_path(cache_dir, dataset_name, model_name, split):
    """Build a deterministic cache file path for a dataset/model/split combo."""
    model_slug = model_name.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{dataset_name}_{model_slug}_{split}.pt"


@torch.no_grad()
def precompute_embeddings(
    encoder, dataset, device, batch_size=64,
    cache_dir=None, dataset_name=None, model_name=None, split="train",
):
    """Run the frozen encoder once and return an EmbeddingDataset.

    If ``cache_dir`` is provided together with ``dataset_name`` and
    ``model_name``, embeddings are saved to / loaded from disk so
    subsequent runs skip the encoder entirely.
    """
    label_encoder = getattr(dataset, "label_encoder", None)

    # ---- try loading from cache ----
    cache_path = None
    if cache_dir and dataset_name and model_name:
        cache_path = _embedding_cache_path(cache_dir, dataset_name, model_name, split)
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            if cached["num_samples"] == len(dataset):
                print(f"  Loaded cached embeddings from {cache_path}")
                return EmbeddingDataset(
                    embeddings=cached["embeddings"],
                    labels=cached["labels"],
                    label_encoder=label_encoder,
                )
            print(f"  Cache stale (expected {len(dataset)} samples, "
                  f"found {cached['num_samples']}). Recomputing.")

    # ---- compute ----
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()

    all_embs, all_labels = [], []
    for batch in tqdm(loader, desc=f"Precomputing {split} embeddings", leave=False):
        batch_x, batch_mask, batch_y = batch
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)

        emb = encoder(batch_x, batch_mask)
        all_embs.append(emb.cpu())
        all_labels.append(batch_y)

    embeddings = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # ---- save to cache ----
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embeddings": embeddings,
            "labels": labels,
            "num_samples": len(dataset),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "split": split,
        }, cache_path)
        print(f"  Saved embeddings to {cache_path}")

    return EmbeddingDataset(
        embeddings=embeddings,
        labels=labels,
        label_encoder=label_encoder,
    )


def extract_targets(dataset):
    targets = []
    for i in range(len(dataset)):
        item = dataset[i]
        y = item[-1]
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


def validate_task_order(task_order, num_classes):
    flat = [int(c) for task in task_order for c in task]
    unique = sorted(set(flat))

    if len(flat) != len(unique):
        raise ValueError("task_order contains duplicate classes.")

    if min(unique) < 0 or max(unique) >= num_classes:
        raise ValueError(
            f"task_order contains invalid class ids. "
            f"Valid range is [0, {num_classes - 1}]"
        )

    if len(unique) != num_classes:
        raise ValueError(
            f"task_order must cover all classes exactly once. "
            f"Found {len(unique)} unique classes, expected {num_classes}."
        )


def build_task_order(
    num_classes, seed=42, shuffle_class_order=True,
    classes_per_task=2, num_tasks=None,
):
    """Generate class-incremental task order automatically."""
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be >= 1")

    classes = list(range(num_classes))
    if shuffle_class_order:
        rng = random.Random(seed)
        rng.shuffle(classes)

    default_splits = [
        classes[i:i + classes_per_task]
        for i in range(0, num_classes, classes_per_task)
    ]

    if num_tasks is None:
        return default_splits
    if num_tasks <= 0 or num_tasks > num_classes:
        raise ValueError(f"num_tasks must be in [1, {num_classes}], got {num_tasks}")
    if num_tasks == len(default_splits):
        return default_splits

    base = num_classes // num_tasks
    remainder = num_classes % num_tasks
    sizes = [base + (1 if i < remainder else 0) for i in range(num_tasks)]

    task_order, idx = [], 0
    for size in sizes:
        task_order.append(classes[idx:idx + size])
        idx += size
    return task_order
