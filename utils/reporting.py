from sklearn.metrics import classification_report, confusion_matrix


SEQUENTIAL_METHODS = {
    "cil_naive",
    "cil_replay_raw",
    "cil_replay_latent",
    "raw_replay",
    "latent_replay",
}


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
        print(f"Task splits: {config['task_splits']}")


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
    print("\n" + "=" * 80)
    print("Sequential Summary")
    for result in task_results:
        print(
            f"Task {result['task_id']:02d} | "
            f"Task classes: {result['task_classes']} | "
            f"Seen classes: {result['seen_classes']} | "
            f"Seen Acc: {result['seen_acc']:.4f}"
        )


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
    print_best_accuracy(best_seen_acc, sequential=True)
    print_sequential_summary(task_results)
    print_classification_report_and_confusion(
        y_true=y_true,
        y_pred=y_pred,
        label_encoder=label_encoder,
        title="Final Classification Report (all classes):",
        confusion_title="Final Confusion Matrix (all classes):",
    )