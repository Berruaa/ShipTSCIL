from sklearn.metrics import classification_report, confusion_matrix

from methods import SEQUENTIAL_METHODS, REPLAY_METHODS, DISTILLATION_METHODS


def print_run_info(config, dataset_info, label_encoder, method, device):
    print(f"Using device: {device}")
    print(f"Method: {config['method']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Train file: {dataset_info['train_file']}")
    print(f"Test file:  {dataset_info['test_file']}")
    print(f"Embedding dim: {method.embedding_dim}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {[str(c) for c in label_encoder.classes_]}")

    if config["method"] in SEQUENTIAL_METHODS:
        print(f"Task order: {config['task_order']}")

    if config["method"] in REPLAY_METHODS:
        herding = config.get("herding_replay", False)
        balanced = config.get("balanced_replay", True)
        if herding:
            buf_type = "herding (iCaRL exemplar selection)"
        elif balanced:
            buf_type = "class-balanced"
        else:
            buf_type = "reservoir (class-agnostic)"
        print(f"Replay buffer: {buf_type}  "
              f"(size={config.get('replay_buffer_size', 1000)}, "
              f"batch={config.get('replay_batch_size', 32)})")

    bal_loss = config.get("balanced_loss", True)
    loss_type = "class-weighted CE" if bal_loss else "standard CE"
    print(f"Loss function: {loss_type}")

    if config.get("use_lora"):
        modules = config.get("lora_target_modules") or ["q", "v"]
        print(f"LoRA: rank={config.get('lora_rank', 8)}, "
              f"alpha={config.get('lora_alpha', 16)}, "
              f"target={modules}, "
              f"lr={config.get('lora_lr', 'auto')}, "
              f"dropout={config.get('lora_dropout', 0.05)}")

    if config["method"] in DISTILLATION_METHODS:
        use_distill = config.get("use_distillation", False)
        if config["method"] == "cil_lwf" or use_distill:
            print(f"Distillation: T={config.get('distill_temperature', 2.0)}, "
                  f"weight={config.get('distill_weight', 1.0)}")


def print_standard_epoch(epoch, total_epochs, train_metrics, test_metrics):
    print(
        f"Epoch [{epoch:02d}/{total_epochs:02d}] "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['acc']:.4f} | "
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['acc']:.4f}"
    )


def print_sequential_epoch(task_id, num_tasks, epoch, total_epochs, train_metrics, seen_test_metrics):
    print(
        f"Task [{task_id:02d}/{num_tasks:02d}] "
        f"Epoch [{epoch:02d}/{total_epochs:02d}] "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['acc']:.4f} | "
        f"Seen Test Loss: {seen_test_metrics['loss']:.4f} | "
        f"Seen Test Acc: {seen_test_metrics['acc']:.4f}"
    )


def print_sequential_summary(task_results):
    import numpy as np

    print("\n" + "=" * 80)
    print("Sequential Summary")
    for result in task_results:
        print(
            f"Task {result['task_id']:02d} | "
            f"Task classes: {result['task_classes']} | "
            f"Seen classes: {result['seen_classes']} | "
            f"Seen Acc: {result['seen_acc']:.4f}"
        )

    num_tasks = len(task_results)

    # Average Incremental Accuracy
    aia = np.mean([r["seen_acc"] for r in task_results])

    # Per-task forgetting: peak accuracy on task j minus final accuracy
    forgetting = []
    for j in range(num_tasks):
        accs_j = [
            r["per_task_acc"][j + 1]
            for r in task_results
            if (j + 1) in r.get("per_task_acc", {})
        ]
        if len(accs_j) >= 2:
            forgetting.append(max(accs_j) - accs_j[-1])

    avg_forgetting = np.mean(forgetting) if forgetting else 0.0

    # Backward transfer: final_acc(j) - acc_right_after_learning(j)
    bwt_vals = []
    for j in range(num_tasks):
        learned = task_results[j].get("per_task_acc", {}).get(j + 1)
        final = task_results[-1].get("per_task_acc", {}).get(j + 1)
        if learned is not None and final is not None and j < num_tasks - 1:
            bwt_vals.append(final - learned)
    avg_bwt = np.mean(bwt_vals) if bwt_vals else 0.0

    print(f"\nCIL Metrics:")
    print(f"  Average Incremental Accuracy (AIA): {aia:.4f}")
    print(f"  Average Forgetting:                 {avg_forgetting:.4f}")
    print(f"  Backward Transfer (BWT):            {avg_bwt:+.4f}")


def print_classification_report_and_confusion(
    y_true,
    y_pred,
    label_encoder,
    title="Classification Report:",
    confusion_title="Confusion Matrix:",
):
    print(f"\n{title}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            digits=4,
            zero_division=0,
        )
    )
    print(confusion_title)
    print(confusion_matrix(y_true, y_pred))


def print_best_accuracy(best_acc, sequential=False):
    label = "Best seen-class accuracy during training" if sequential else "Best test accuracy"
    print(f"\n{label}: {best_acc:.4f}")


def print_final_standard_results(best_test_acc, y_true, y_pred, label_encoder):
    print_best_accuracy(best_test_acc, sequential=False)
    print_classification_report_and_confusion(
        y_true=y_true,
        y_pred=y_pred,
        label_encoder=label_encoder,
        title="Classification Report:",
        confusion_title="Confusion Matrix:",
    )


def print_final_sequential_results(best_seen_acc, task_results, y_true, y_pred, label_encoder):
    print_sequential_summary(task_results)
    print_classification_report_and_confusion(
        y_true=y_true,
        y_pred=y_pred,
        label_encoder=label_encoder,
        title="Final Classification Report (all classes):",
        confusion_title="Final Confusion Matrix (all classes):",
    )