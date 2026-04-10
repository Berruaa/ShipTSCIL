"""
Visualization utilities for training results.

All functions accept a ``save_path`` and call ``savefig`` + ``plt.close``
so they can be used headlessly on a server without a display.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, classification_report

# ── Style defaults ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "train_loss": "#1f77b4",
    "test_loss":  "#ff7f0e",
    "train_acc":  "#2ca02c",
    "test_acc":   "#d62728",
}

METHOD_DISPLAY_NAMES = {
    "linear_probe":      "Linear Probe",
    "svm":               "SVM",
    "cil_naive":         "CIL Naive (Fine-tune)",
    "cil_replay_raw":    "CIL Raw Replay",
    "cil_replay_latent": "CIL Latent Replay",
    "cil_lwf":           "CIL LwF (Distill-only)",
    "raw_replay":        "CIL Raw Replay",
    "latent_replay":     "CIL Latent Replay",
    "lwf":               "CIL LwF (Distill-only)",
}

DATASET_DISPLAY_NAMES = {
    "ecg5000":             "ECG5000",
    "electric_devices":    "ElectricDevices",
    "italy_power_demand":  "ItalyPowerDemand",
    "ford_a":              "FordA",
}


REPLAY_METHODS = {"cil_replay_raw", "cil_replay_latent", "raw_replay", "latent_replay"}


def _pretty_method(method_name: str,
                   balanced_replay: bool | None = None,
                   balanced_loss: bool | None = None,
                   use_distillation: bool | None = None) -> str:
    base = METHOD_DISPLAY_NAMES.get(method_name, method_name)
    tags = []
    if method_name in REPLAY_METHODS:
        if balanced_replay is not None:
            tags.append("Bal-Buf" if balanced_replay else "Reservoir")
        if balanced_loss is not None:
            tags.append("Bal-CE" if balanced_loss else "Std-CE")
    if use_distillation:
        tags.append("+Distill")
    if tags:
        return f"{base} ({', '.join(tags)})"
    return base


def _pretty_dataset(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)


def _header(method_name: str, dataset_name: str,
            balanced_replay: bool | None = None,
            balanced_loss: bool | None = None,
            use_distillation: bool | None = None) -> str:
    """Build a consistent 'Method — Dataset' header string."""
    return (f"{_pretty_method(method_name, balanced_replay, balanced_loss, use_distillation)}"
            f"  |  {_pretty_dataset(dataset_name)}")


def _stamp_footer(fig, method_name: str, dataset_name: str,
                  balanced_replay: bool | None = None,
                  balanced_loss: bool | None = None,
                  use_distillation: bool | None = None):
    """Add a small method + dataset footnote at the bottom-right of a figure."""
    fig.text(
        0.99, 0.005,
        (f"Method: {_pretty_method(method_name, balanced_replay, balanced_loss, use_distillation)}"
         f"    Dataset: {_pretty_dataset(dataset_name)}"),
        ha="right", va="bottom", fontsize=7, color="grey", style="italic",
    )


# =====================================================================
#  Training curves
# =====================================================================

def plot_training_curves(history: dict, save_path, *,
                         method_name: str = "", dataset_name: str = "",
                         balanced_replay: bool | None = None,
                         balanced_loss: bool | None = None,
                         use_distillation: bool | None = None):
    """
    Plot loss and accuracy curves for standard (joint) training.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], marker="o", markersize=4,
                 color=COLORS["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["test_loss"], marker="s", markersize=4,
                 color=COLORS["test_loss"], label="Test loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curve")
    ax_loss.legend()
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], marker="o", markersize=4,
                color=COLORS["train_acc"], label="Train acc")
    ax_acc.plot(epochs, history["test_acc"], marker="s", markersize=4,
                color=COLORS["test_acc"], label="Test acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Curve")
    ax_acc.legend()
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(
        f"Training Curves — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}",
        fontsize=14, y=1.02)
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_sequential_training_curves(history: dict, save_path, *,
                                    method_name: str = "", dataset_name: str = "",
                                    balanced_replay: bool | None = None,
                                    balanced_loss: bool | None = None,
                                    use_distillation: bool | None = None):
    """
    Plot loss and accuracy curves for sequential (CIL) training.
    """
    num_tasks = len(history)
    fig, axes = plt.subplots(num_tasks, 2, figsize=(12, 4 * num_tasks),
                             squeeze=False)

    cmap = plt.cm.get_cmap("tab10", num_tasks)

    for i, (task_id, task_hist) in enumerate(sorted(history.items())):
        epochs = range(1, len(task_hist["train_loss"]) + 1)
        color = cmap(i)

        ax_loss = axes[i, 0]
        ax_loss.plot(epochs, task_hist["train_loss"], marker="o", markersize=4,
                     color=color, label="Train loss")
        ax_loss.plot(epochs, task_hist["seen_test_loss"], marker="s", markersize=4,
                     color=color, linestyle="--", label="Seen-class test loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Task {task_id} — Loss")
        ax_loss.legend(fontsize=9)
        ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_loss.grid(alpha=0.3)

        ax_acc = axes[i, 1]
        ax_acc.plot(epochs, task_hist["train_acc"], marker="o", markersize=4,
                    color=color, label="Train acc")
        ax_acc.plot(epochs, task_hist["seen_test_acc"], marker="s", markersize=4,
                    color=color, linestyle="--", label="Seen-class test acc")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"Task {task_id} — Accuracy")
        ax_acc.legend(fontsize=9)
        ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_acc.set_ylim(-0.02, 1.05)
        ax_acc.grid(alpha=0.3)

    fig.suptitle(
        f"Sequential Training Curves — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}",
        fontsize=14, y=1.01,
    )
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Task accuracy progression (sequential only)
# =====================================================================

def plot_task_accuracy_progression(task_results: list[dict], save_path, *,
                                   method_name: str = "", dataset_name: str = "",
                                   balanced_replay: bool | None = None,
                                   balanced_loss: bool | None = None,
                                   use_distillation: bool | None = None):
    """
    Bar + line chart showing seen-class accuracy after each task.

    Parameters
    ----------
    task_results : list of dicts, each with "task_id", "task_classes",
                   "seen_classes", "seen_acc".
    """
    task_ids = [r["task_id"] for r in task_results]
    accs = [r["seen_acc"] for r in task_results]
    labels = [f"Task {r['task_id']}\n+cls {r['task_classes']}" for r in task_results]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(task_ids)), 5))
    bars = ax.bar(task_ids, accs, color=plt.cm.Blues(np.linspace(0.4, 0.85, len(task_ids))),
                  edgecolor="black", linewidth=0.6)
    ax.plot(task_ids, accs, marker="D", color="#d62728", linewidth=2, zorder=5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(task_ids)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Task")
    ax.set_ylabel("Seen-class Test Accuracy")
    ax.set_title(
        f"Accuracy Progression — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}",
        fontsize=13,
    )
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Confusion matrix
# =====================================================================

def plot_confusion_matrix(y_true, y_pred, class_names: list[str], save_path, *,
                          method_name: str = "", dataset_name: str = "",
                          balanced_replay: bool | None = None,
                          balanced_loss: bool | None = None,
                          use_distillation: bool | None = None,
                          title: str | None = None):
    """
    Annotated confusion-matrix heatmap.
    """
    if title is None:
        title = f"Confusion Matrix — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}"

    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    fig_size = max(5, 0.55 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=max(6, 10 - n // 5),
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(class_names, fontsize=max(6, 10 - n // 5))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12)

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Per-class accuracy
# =====================================================================

def plot_per_class_accuracy(y_true, y_pred, class_names: list[str], save_path, *,
                            method_name: str = "", dataset_name: str = "",
                            balanced_replay: bool | None = None,
                            balanced_loss: bool | None = None,
                            use_distillation: bool | None = None,
                            title: str | None = None):
    """
    Horizontal bar chart of per-class accuracy.
    """
    if title is None:
        title = f"Per-Class Accuracy — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}"

    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = cm.diagonal() / cm.sum(axis=1)
        per_class = np.nan_to_num(per_class)

    overall = np.mean(per_class)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(class_names))))

    y_pos = np.arange(len(class_names))
    colors = ["#2ca02c" if a >= overall else "#d62728" for a in per_class]
    bars = ax.barh(y_pos, per_class, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(overall, color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"Mean = {overall:.2%}")

    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2%}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Combined summary figure (sequential only)
# =====================================================================

def plot_sequential_summary(history, task_results, y_true, y_pred,
                            class_names, save_path, *,
                            method_name: str = "", dataset_name: str = "",
                            balanced_replay: bool | None = None,
                            balanced_loss: bool | None = None,
                            use_distillation: bool | None = None):
    """
    All-in-one summary page for a sequential run: a compact accuracy
    progression line at the top plus the final confusion matrix below.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # ---- Accuracy progression (left) ----
    ax_prog = fig.add_subplot(gs[0, 0])
    task_ids = [r["task_id"] for r in task_results]
    accs = [r["seen_acc"] for r in task_results]
    ax_prog.plot(task_ids, accs, marker="D", color="#d62728", linewidth=2)
    ax_prog.fill_between(task_ids, accs, alpha=0.15, color="#d62728")
    for tid, acc in zip(task_ids, accs):
        ax_prog.annotate(f"{acc:.2%}", (tid, acc), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontweight="bold", fontsize=9)
    ax_prog.set_xlabel("Task")
    ax_prog.set_ylabel("Seen-class Accuracy")
    ax_prog.set_title("Accuracy Progression")
    ax_prog.set_ylim(0, 1.12)
    ax_prog.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_prog.grid(alpha=0.3)

    # ---- Confusion matrix (right) ----
    ax_cm = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=max(6, 9 - n // 5),
                       color="white" if cm[i, j] > thresh else "black")
    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right",
                          fontsize=max(6, 9 - n // 5))
    ax_cm.set_yticklabels(class_names, fontsize=max(6, 9 - n // 5))
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Final Confusion Matrix")

    fig.suptitle(
        f"Sequential Summary — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)}",
        fontsize=14, y=1.02,
    )
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
