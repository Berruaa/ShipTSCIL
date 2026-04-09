from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.factory import build_dataset_pair
from methods import build_method
from utils.seed import set_seed
from utils.reporting import (
    print_run_info,
    print_standard_epoch,
    print_sequential_epoch,
    print_final_standard_results,
    print_final_sequential_results,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "checkpoints"

CONFIG = {
    # supported:
    # "svm", "linear_probe", "cil_naive"
    # "cil_replay_raw"        <- placeholder for later
    # "cil_replay_latent"     <- placeholder for later
    "method": "cil_naive",

    "dataset": "electric_devices",
    "train_file": None,
    "test_file": None,

    "batch_size": 64,
    "epochs": 5,
    "lr": 1e-3,
    "seed": 42,
    "num_workers": 0,
    "model_name": "AutonLab/MOMENT-1-base",

    # class-incremental setup
    "task_splits": [[0, 1, 2], [3, 4], [5, 6]],
}


@torch.no_grad()
def collect_predictions_torch_model(model, dataloader, device):
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


def collect_predictions(method, dataloader, device):
    if hasattr(method, "predict"):
        return method.predict(dataloader)

    if hasattr(method, "model"):
        return collect_predictions_torch_model(method.model, dataloader, device=device)

    raise ValueError("Method does not support prediction.")


def build_base_datasets_and_info(config):
    train_dataset, test_dataset, label_encoder, dataset_info = build_dataset_pair(
        dataset_name=config["dataset"],
        data_dir=DATA_DIR,
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


def train_standard(method, train_loader, test_loader, config, label_encoder):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = SAVE_DIR / f"{config['dataset']}_{config['method']}_best.pkl"
    if config["method"] != "svm":
        checkpoint_path = SAVE_DIR / f"{config['dataset']}_{config['method']}_best.pt"

    best_test_acc = 0.0

    if config["method"] == "svm":
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)

        print_standard_epoch(
            epoch=1,
            total_epochs=1,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

        best_test_acc = test_metrics["acc"]
        method.save(
            checkpoint_path=checkpoint_path,
            label_classes=label_encoder.classes_.tolist(),
            dataset_name=config["dataset"],
            extra_config=config,
        )
        return checkpoint_path, best_test_acc

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)

        print_standard_epoch(
            epoch=epoch,
            total_epochs=config["epochs"],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            method.save(
                checkpoint_path=checkpoint_path,
                label_classes=label_encoder.classes_.tolist(),
                dataset_name=config["dataset"],
                extra_config=config,
            )

    return checkpoint_path, best_test_acc


def evaluate_on_seen_classes(method, test_dataset, seen_classes, config):
    eval_subset = make_class_subset(test_dataset, seen_classes)
    eval_loader = build_loader(
        dataset=eval_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    return method.evaluate(eval_loader), eval_loader


def train_sequential(method, train_dataset, test_dataset, config, label_encoder):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = SAVE_DIR / f"{config['dataset']}_{config['method']}_best.pt"
    best_seen_acc = 0.0

    seen_classes = []
    task_results = []

    for task_id, task_classes in enumerate(config["task_splits"], start=1):
        print("\n" + "=" * 80)
        print(f"Starting Task {task_id}/{len(config['task_splits'])}")
        print(f"Current task classes: {task_classes}")

        seen_classes.extend(task_classes)
        seen_classes = sorted(set(seen_classes))

        task_train_subset = make_class_subset(train_dataset, task_classes)
        task_train_loader = build_loader(
            dataset=task_train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )

        print(f"Seen classes so far: {seen_classes}")
        print(f"Task train size: {len(task_train_subset)}")

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = method.train_epoch(task_train_loader)
            seen_test_metrics, _ = evaluate_on_seen_classes(
                method=method,
                test_dataset=test_dataset,
                seen_classes=seen_classes,
                config=config,
            )

            print_sequential_epoch(
                task_id=task_id,
                num_tasks=len(config["task_splits"]),
                epoch=epoch,
                total_epochs=config["epochs"],
                train_metrics=train_metrics,
                seen_test_metrics=seen_test_metrics,
            )

            if seen_test_metrics["acc"] > best_seen_acc:
                best_seen_acc = seen_test_metrics["acc"]
                method.save(
                    checkpoint_path=checkpoint_path,
                    label_classes=label_encoder.classes_.tolist(),
                    dataset_name=config["dataset"],
                    extra_config=config,
                )

        final_seen_metrics, _ = evaluate_on_seen_classes(
            method=method,
            test_dataset=test_dataset,
            seen_classes=seen_classes,
            config=config,
        )
        task_results.append(
            {
                "task_id": task_id,
                "task_classes": deepcopy(task_classes),
                "seen_classes": deepcopy(seen_classes),
                "seen_acc": final_seen_metrics["acc"],
            }
        )

    return checkpoint_path, best_seen_acc, task_results


def main():
    set_seed(CONFIG["seed"])

    train_dataset, test_dataset, label_encoder, dataset_info = build_base_datasets_and_info(CONFIG)

    num_classes = len(label_encoder.classes_)

    if CONFIG["method"] in {
        "cil_naive",
        "cil_replay_raw",
        "cil_replay_latent",
    }:
        validate_task_splits(CONFIG["task_splits"], num_classes)

    if CONFIG["method"] == "svm" and CONFIG["method"] in {
        "cil_naive",
        "cil_replay_raw",
        "cil_replay_latent",
    }:
        raise ValueError("SVM is currently only supported for standard training.")

    method = build_method(
        method_name=CONFIG["method"],
        model_name=CONFIG["model_name"],
        num_classes=num_classes,
        train_dataset=train_dataset,
        device=DEVICE,
        lr=CONFIG["lr"],
    )

    print_run_info(
        config=CONFIG,
        dataset_info=dataset_info,
        label_encoder=label_encoder,
        method=method,
        device=DEVICE,
    )

    if CONFIG["method"] in {"linear_probe", "svm"}:
        train_loader = build_loader(
            dataset=train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
        )
        test_loader = build_loader(
            dataset=test_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

        checkpoint_path, best_test_acc = train_standard(
            method=method,
            train_loader=train_loader,
            test_loader=test_loader,
            config=CONFIG,
            label_encoder=label_encoder,
        )

        method.load(checkpoint_path)
        y_true, y_pred = collect_predictions(method, test_loader, device=DEVICE)

        print_final_standard_results(
            best_test_acc=best_test_acc,
            y_true=y_true,
            y_pred=y_pred,
            label_encoder=label_encoder,
        )
        return

    if CONFIG["method"] in {
        "cil_naive",
        "cil_replay_raw",
        "cil_replay_latent",
    }:
        checkpoint_path, best_seen_acc, task_results = train_sequential(
            method=method,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=CONFIG,
            label_encoder=label_encoder,
        )

        all_seen_classes = sorted(set(c for task in CONFIG["task_splits"] for c in task))
        method.load(checkpoint_path)

        final_subset = make_class_subset(test_dataset, all_seen_classes)
        final_loader = build_loader(
            dataset=final_subset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
        )

        y_true, y_pred = collect_predictions(method, final_loader, device=DEVICE)

        print_final_sequential_results(
            best_seen_acc=best_seen_acc,
            task_results=task_results,
            y_true=y_true,
            y_pred=y_pred,
            label_encoder=label_encoder,
        )
        return

    raise ValueError(f"Unsupported method: {CONFIG['method']}")


if __name__ == "__main__":
    main()