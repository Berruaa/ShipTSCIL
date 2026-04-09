from pathlib import Path

import torch

from methods import build_method
from pipelines import (
    build_base_datasets_and_info,
    build_loader,
    collect_predictions,
    make_class_subset,
    train_sequential,
    train_standard,
    validate_task_splits,
)
from utils.seed import set_seed
from utils.reporting import (
    print_final_sequential_results,
    print_final_standard_results,
    print_run_info,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "checkpoints"

CONFIG = {
    # supported:
    # "svm", "linear_probe", "cil_naive"
    # "cil_replay_raw"
    # "cil_replay_latent"     <- placeholder for later
    "method": "cil_replay_raw",

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
    "replay_buffer_size": 1000,
    "replay_batch_size": 32,
}

STANDARD_METHODS = {"linear_probe", "svm"}
SEQUENTIAL_METHODS = {"cil_naive", "cil_replay_raw", "cil_replay_latent"}


def main():
    set_seed(CONFIG["seed"])

    train_dataset, test_dataset, label_encoder, dataset_info = build_base_datasets_and_info(
        config=CONFIG,
        data_dir=DATA_DIR,
    )

    num_classes = len(label_encoder.classes_)

    if CONFIG["method"] in SEQUENTIAL_METHODS:
        validate_task_splits(CONFIG["task_splits"], num_classes)

    method = build_method(
        method_name=CONFIG["method"],
        model_name=CONFIG["model_name"],
        num_classes=num_classes,
        train_dataset=train_dataset,
        device=DEVICE,
        lr=CONFIG["lr"],
        replay_buffer_size=CONFIG.get("replay_buffer_size", 1000),
        replay_batch_size=CONFIG.get("replay_batch_size", 32),
    )

    print_run_info(
        config=CONFIG,
        dataset_info=dataset_info,
        label_encoder=label_encoder,
        method=method,
        device=DEVICE,
    )

    if CONFIG["method"] in STANDARD_METHODS:
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
            save_dir=SAVE_DIR,
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

    if CONFIG["method"] in SEQUENTIAL_METHODS:
        _, final_checkpoint_path, best_seen_acc, task_results = train_sequential(
            method=method,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=CONFIG,
            label_encoder=label_encoder,
            save_dir=SAVE_DIR,
        )

        all_seen_classes = sorted(set(c for task in CONFIG["task_splits"] for c in task))
        method.load(final_checkpoint_path)

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