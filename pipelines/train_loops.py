from collections import defaultdict
from copy import deepcopy

from torch.utils.data import ConcatDataset

from pipelines.data import build_loader, make_class_subset, stratified_train_val_split
from pipelines.evaluation import evaluate_on_seen_classes
from utils.reporting import print_standard_epoch, print_sequential_epoch


def _early_stopping_enabled(config):
    return bool(config.get("use_early_stopping", True)) and int(config.get("epochs", 1)) > 1


def _metric_improved(current, best, min_delta):
    return current > (best + min_delta)


def _dataset_len(dataset):
    return len(dataset) if dataset is not None else 0


def train_standard(method, train_loader, test_loader, config, label_encoder, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pkl"
    if config["method"] != "svm":
        checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pt"

    best_test_acc = 0.0
    history = defaultdict(list)
    val_loader = None
    patience = int(config.get("early_stopping_patience", 5))
    min_delta = float(config.get("early_stopping_min_delta", 1e-4))

    if _early_stopping_enabled(config) and config["method"] != "svm":
        train_dataset, val_dataset = stratified_train_val_split(
            train_loader.dataset,
            val_fraction=float(config.get("validation_split", 0.1)),
            seed=int(config.get("seed", 42)),
        )
        if val_dataset is not None:
            train_loader = build_loader(
                dataset=train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
            val_loader = build_loader(
                dataset=val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
            )
            print(
                f"Validation split enabled: {_dataset_len(train_dataset)} train / "
                f"{_dataset_len(val_dataset)} val samples"
            )
        else:
            print("Validation split unavailable; early stopping disabled for this run.")

    if config["method"] == "svm":
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)

        print_standard_epoch(
            epoch=1,
            total_epochs=1,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

        best_test_acc = test_metrics["acc"]
        method.save(
            checkpoint_path=checkpoint_path,
            label_classes=label_encoder.classes_.tolist(),
            dataset_name=config["dataset"],
            extra_config=config,
        )
        return checkpoint_path, best_test_acc, dict(history)

    best_val_acc = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)
        val_metrics = method.evaluate(val_loader) if val_loader is not None else None

        print_standard_epoch(
            epoch=epoch,
            total_epochs=config["epochs"],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        if val_metrics is not None:
            print(
                f"                 Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f}"
            )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])
        if val_metrics is not None:
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])

        monitor_acc = val_metrics["acc"] if val_metrics is not None else test_metrics["acc"]
        if _metric_improved(monitor_acc, best_val_acc, min_delta):
            best_val_acc = monitor_acc
            epochs_without_improvement = 0
            method.save(
                checkpoint_path=checkpoint_path,
                label_classes=label_encoder.classes_.tolist(),
                dataset_name=config["dataset"],
                extra_config=config,
            )
        else:
            epochs_without_improvement += 1

        if val_loader is not None and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} after {patience} epochs "
                f"without validation improvement."
            )
            break

    if checkpoint_path.exists():
        method.load(checkpoint_path)
        best_test_acc = method.evaluate(test_loader)["acc"]

    return checkpoint_path, best_test_acc, dict(history)


def train_sequential(method, train_dataset, test_dataset, config, label_encoder, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pt"
    final_checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_final.pt"
    best_seen_acc = 0.0

    seen_classes = []
    task_results = []
    history = {}
    seen_val_datasets = []
    patience = int(config.get("early_stopping_patience", 5))
    min_delta = float(config.get("early_stopping_min_delta", 1e-4))

    for task_id, task_classes in enumerate(config["task_order"], start=1):
        print("\n" + "=" * 80)
        print(f"Starting Task {task_id}/{len(config['task_order'])}")
        print(f"Current task classes: {task_classes}")

        old_classes = list(seen_classes)
        method.begin_task(task_id, task_classes, old_classes)

        seen_classes.extend(task_classes)
        seen_classes = sorted(set(seen_classes))

        task_train_subset = make_class_subset(train_dataset, task_classes)
        task_val_subset = None
        if _early_stopping_enabled(config):
            task_train_subset, task_val_subset = stratified_train_val_split(
                task_train_subset,
                val_fraction=float(config.get("validation_split", 0.1)),
                seed=int(config.get("seed", 42)) + task_id,
            )
            if task_val_subset is not None:
                seen_val_datasets.append(task_val_subset)

        task_train_loader = build_loader(
            dataset=task_train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        seen_val_loader = None
        if seen_val_datasets:
            seen_val_dataset = (
                seen_val_datasets[0]
                if len(seen_val_datasets) == 1
                else ConcatDataset(seen_val_datasets)
            )
            seen_val_loader = build_loader(
                dataset=seen_val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
            )

        print(f"Seen classes so far: {seen_classes}")
        print(f"Task train size: {len(task_train_subset)}")
        if seen_val_loader is not None:
            print(f"Cumulative seen-val size: {len(seen_val_loader.dataset)}")
        elif _early_stopping_enabled(config):
            print("Validation split unavailable for this task; early stopping disabled.")

        task_hist = defaultdict(list)
        best_task_val_acc = float("-inf")
        epochs_without_improvement = 0
        task_checkpoint_path = save_dir / (
            f"{config['dataset']}_{config['method']}_task{task_id:02d}_best.pt"
        )

        for epoch in range(1, config["epochs"] + 1):
            train_metrics = method.train_epoch(task_train_loader)
            seen_test_metrics, _ = evaluate_on_seen_classes(
                method=method,
                test_dataset=test_dataset,
                seen_classes=seen_classes,
                config=config,
            )
            seen_val_metrics = method.evaluate(seen_val_loader) if seen_val_loader is not None else None

            print_sequential_epoch(
                task_id=task_id,
                num_tasks=len(config["task_order"]),
                epoch=epoch,
                total_epochs=config["epochs"],
                train_metrics=train_metrics,
                seen_test_metrics=seen_test_metrics,
            )
            if seen_val_metrics is not None:
                print(
                    f"                 Seen Val Loss: {seen_val_metrics['loss']:.4f} | "
                    f"Seen Val Acc: {seen_val_metrics['acc']:.4f}"
                )

            task_hist["train_loss"].append(train_metrics["loss"])
            task_hist["train_acc"].append(train_metrics["acc"])
            task_hist["seen_test_loss"].append(seen_test_metrics["loss"])
            task_hist["seen_test_acc"].append(seen_test_metrics["acc"])
            if seen_val_metrics is not None:
                task_hist["seen_val_loss"].append(seen_val_metrics["loss"])
                task_hist["seen_val_acc"].append(seen_val_metrics["acc"])

            if seen_test_metrics["acc"] > best_seen_acc:
                best_seen_acc = seen_test_metrics["acc"]
                method.save(
                    checkpoint_path=best_checkpoint_path,
                    label_classes=label_encoder.classes_.tolist(),
                    dataset_name=config["dataset"],
                    extra_config=config,
                )

            monitor_acc = seen_val_metrics["acc"] if seen_val_metrics is not None else None
            if monitor_acc is not None:
                if _metric_improved(monitor_acc, best_task_val_acc, min_delta):
                    best_task_val_acc = monitor_acc
                    epochs_without_improvement = 0
                    method.save(
                        checkpoint_path=task_checkpoint_path,
                        label_classes=label_encoder.classes_.tolist(),
                        dataset_name=config["dataset"],
                        extra_config=config,
                    )
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping task {task_id} at epoch {epoch} after "
                        f"{patience} epochs without validation improvement."
                    )
                    break

        if seen_val_loader is not None and task_checkpoint_path.exists():
            method.load(task_checkpoint_path)

        history[task_id] = dict(task_hist)

        method.end_task(task_id, seen_classes)

        final_seen_metrics, _ = evaluate_on_seen_classes(
            method=method,
            test_dataset=test_dataset,
            seen_classes=seen_classes,
            config=config,
        )

        # Per-task accuracy breakdown (needed for forgetting analysis).
        per_task_acc = {}
        for prev_id, prev_classes in enumerate(config["task_order"][:task_id], start=1):
            prev_metrics, _ = evaluate_on_seen_classes(
                method=method,
                test_dataset=test_dataset,
                seen_classes=prev_classes,
                config=config,
            )
            per_task_acc[prev_id] = prev_metrics["acc"]

        task_results.append(
            {
                "task_id": task_id,
                "task_classes": deepcopy(task_classes),
                "seen_classes": deepcopy(seen_classes),
                "seen_acc": final_seen_metrics["acc"],
                "per_task_acc": per_task_acc,
            }
        )

    method.save(
        checkpoint_path=final_checkpoint_path,
        label_classes=label_encoder.classes_.tolist(),
        dataset_name=config["dataset"],
        extra_config=config,
    )

    return best_checkpoint_path, final_checkpoint_path, best_seen_acc, task_results, history
