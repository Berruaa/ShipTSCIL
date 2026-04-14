from collections import defaultdict
from copy import deepcopy

from pipelines.data import build_loader, make_class_subset
from pipelines.evaluation import evaluate_on_seen_classes
from utils.reporting import print_standard_epoch, print_sequential_epoch


def train_standard(method, train_loader, test_loader, config, label_encoder, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pkl"
    if config["method"] != "svm":
        checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pt"

    best_test_acc = 0.0
    history = defaultdict(list)

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

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = method.train_epoch(train_loader)
        test_metrics = method.evaluate(test_loader)

        print_standard_epoch(
            epoch=epoch,
            total_epochs=config["epochs"],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            method.save(
                checkpoint_path=checkpoint_path,
                label_classes=label_encoder.classes_.tolist(),
                dataset_name=config["dataset"],
                extra_config=config,
            )

    return checkpoint_path, best_test_acc, dict(history)


def train_sequential(method, train_dataset, test_dataset, config, label_encoder, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_best.pt"
    final_checkpoint_path = save_dir / f"{config['dataset']}_{config['method']}_final.pt"
    best_seen_acc = 0.0

    seen_classes = []
    task_results = []
    history = {}

    for task_id, task_classes in enumerate(config["task_order"], start=1):
        print("\n" + "=" * 80)
        print(f"Starting Task {task_id}/{len(config['task_order'])}")
        print(f"Current task classes: {task_classes}")

        old_classes = list(seen_classes)
        method.begin_task(task_id, task_classes, old_classes)

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

        task_hist = defaultdict(list)

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
                num_tasks=len(config["task_order"]),
                epoch=epoch,
                total_epochs=config["epochs"],
                train_metrics=train_metrics,
                seen_test_metrics=seen_test_metrics,
            )

            task_hist["train_loss"].append(train_metrics["loss"])
            task_hist["train_acc"].append(train_metrics["acc"])
            task_hist["seen_test_loss"].append(seen_test_metrics["loss"])
            task_hist["seen_test_acc"].append(seen_test_metrics["acc"])

            if seen_test_metrics["acc"] > best_seen_acc:
                best_seen_acc = seen_test_metrics["acc"]
                method.save(
                    checkpoint_path=best_checkpoint_path,
                    label_classes=label_encoder.classes_.tolist(),
                    dataset_name=config["dataset"],
                    extra_config=config,
                )

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
