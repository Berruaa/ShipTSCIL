from datetime import datetime
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
from utils.plotting import (
    plot_training_curves,
    plot_sequential_training_curves,
    plot_task_accuracy_progression,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_sequential_summary,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CONFIG = {
    # supported:
    # "svm", "linear_probe", "cil_naive"
    # "cil_replay_raw", "cil_replay_latent"
    # "cil_lwf"
    "method": "cil_replay_latent",

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

    # "task_splits": [[0, 1, 2], [3, 4], [5, 6]],
    # "task_splits": [[0], [1], [2], [3], [4], [5], [6]],
    "task_splits": [[0, 1], [2, 3], [4]],
    "replay_buffer_size": 1000,
    "replay_batch_size": 32,
    "balanced_replay": True,        # True = class-balanced buffer, False = plain reservoir
    "balanced_loss": True,          # True = class-weighted CE loss,  False = standard CE

    # LwF distillation (works with cil_replay_latent and cil_lwf)
    "use_distillation": True,       # add KD loss on top of replay
    "distill_temperature": 2.0,     # softmax temperature for knowledge distillation
    "distill_weight": 1.0,          # multiplier for KD loss (additive: CE + λ·KD)
}

STANDARD_METHODS = {"linear_probe", "svm"}
SEQUENTIAL_METHODS = {"cil_naive", "cil_replay_raw", "cil_replay_latent", "cil_lwf"}


def _make_run_dir(config):
    """Create a timestamped folder under results/ for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{config['dataset']}_{config['method']}_{timestamp}"
    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
        balanced_replay=CONFIG.get("balanced_replay", True),
        balanced_loss=CONFIG.get("balanced_loss", True),
        use_distillation=CONFIG.get("use_distillation", False),
        distill_temperature=CONFIG.get("distill_temperature", 2.0),
        distill_weight=CONFIG.get("distill_weight", 1.0),
    )

    print_run_info(
        config=CONFIG,
        dataset_info=dataset_info,
        label_encoder=label_encoder,
        method=method,
        device=DEVICE,
    )

    class_names = [str(c) for c in label_encoder.classes_]
    run_dir = _make_run_dir(CONFIG)
    print(f"\nResults will be saved to: {run_dir}")

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

        checkpoint_path, best_test_acc, history = train_standard(
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

        # ---- plots ----
        method_key = CONFIG["method"]
        dataset_key = CONFIG["dataset"]
        bal_buf = CONFIG.get("balanced_replay")
        bal_loss = CONFIG.get("balanced_loss")
        distill = CONFIG.get("use_distillation")
        plot_training_curves(history, run_dir / "training_curves.png",
                             method_name=method_key, dataset_name=dataset_key,
                             balanced_replay=bal_buf, balanced_loss=bal_loss,
                             use_distillation=distill)
        plot_confusion_matrix(y_true, y_pred, class_names,
                              run_dir / "confusion_matrix.png",
                              method_name=method_key, dataset_name=dataset_key,
                              balanced_replay=bal_buf, balanced_loss=bal_loss,
                              use_distillation=distill)
        plot_per_class_accuracy(y_true, y_pred, class_names,
                                run_dir / "per_class_accuracy.png",
                                method_name=method_key, dataset_name=dataset_key,
                                balanced_replay=bal_buf, balanced_loss=bal_loss,
                                use_distillation=distill)

        print(f"\nPlots saved to: {run_dir}")
        return

    if CONFIG["method"] in SEQUENTIAL_METHODS:
        (
            _,
            final_checkpoint_path,
            best_seen_acc,
            task_results,
            history,
        ) = train_sequential(
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

        # ---- plots ----
        method_key = CONFIG["method"]
        dataset_key = CONFIG["dataset"]
        bal_buf = CONFIG.get("balanced_replay")
        bal_loss = CONFIG.get("balanced_loss")
        distill = CONFIG.get("use_distillation")
        plot_sequential_training_curves(history, run_dir / "training_curves.png",
                                        method_name=method_key, dataset_name=dataset_key,
                                        balanced_replay=bal_buf, balanced_loss=bal_loss,
                                        use_distillation=distill)
        plot_task_accuracy_progression(task_results,
                                       run_dir / "task_accuracy_progression.png",
                                       method_name=method_key, dataset_name=dataset_key,
                                       balanced_replay=bal_buf, balanced_loss=bal_loss,
                                       use_distillation=distill)
        plot_confusion_matrix(y_true, y_pred, class_names,
                              run_dir / "confusion_matrix.png",
                              method_name=method_key, dataset_name=dataset_key,
                              balanced_replay=bal_buf, balanced_loss=bal_loss,
                              use_distillation=distill)
        plot_per_class_accuracy(y_true, y_pred, class_names,
                                run_dir / "per_class_accuracy.png",
                                method_name=method_key, dataset_name=dataset_key,
                                balanced_replay=bal_buf, balanced_loss=bal_loss,
                                use_distillation=distill)
        plot_sequential_summary(history, task_results, y_true, y_pred,
                                class_names, run_dir / "summary.png",
                                method_name=method_key, dataset_name=dataset_key,
                                balanced_replay=bal_buf, balanced_loss=bal_loss,
                                use_distillation=distill)

        print(f"\nPlots saved to: {run_dir}")
        return

    raise ValueError(f"Unsupported method: {CONFIG['method']}")


if __name__ == "__main__":
    main()