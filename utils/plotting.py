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
from sklearn.metrics import confusion_matrix

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
    "cil_ncm":           "CIL NCM",   # suffix added dynamically based on herding_replay
    "cil_herding_ncm":   "CIL Herding+NCM (FastICARL-B)",
}

REPLAY_METHODS = {"cil_replay_raw", "cil_replay_latent"}


def _pretty_method(method_name: str,
                   balanced_replay: bool | None = None,
                   balanced_loss: bool | None = None,
                   use_distillation: bool | None = None,
                   herding_replay: bool | None = None) -> str:
    base = METHOD_DISPLAY_NAMES.get(method_name, method_name)
    tags = []

    if method_name == "cil_ncm":
        # cil_ncm routes to Version A or B depending on herding_replay.
        tags.append("FastICARL-B, Herding+NCM" if herding_replay else "FastICARL-A, full mean")
    elif method_name in REPLAY_METHODS:
        if herding_replay:
            tags.append("Herding")
        elif balanced_replay is not None:
            tags.append("Bal-Buf" if balanced_replay else "Reservoir")
        if balanced_loss is not None:
            tags.append("Bal-CE" if balanced_loss else "Std-CE")

    if use_distillation:
        tags.append("+Distill")
    if tags:
        return f"{base} ({', '.join(tags)})"
    return base


def _pretty_dataset(dataset_name: str) -> str:
    """Convert ``snake_case`` key to a display name.

    Each ``_``-separated token is title-cased.  Tokens that contain
    digits are upper-cased entirely (``ecg5000`` → ``ECG5000``).
    """
    parts = dataset_name.split("_")
    return "".join(p.upper() if any(c.isdigit() for c in p) else p.title()
                   for p in parts)


def _header(method_name: str, dataset_name: str,
            balanced_replay: bool | None = None,
            balanced_loss: bool | None = None,
            use_distillation: bool | None = None,
            herding_replay: bool | None = None) -> str:
    """Build a consistent 'Method — Dataset' header string."""
    return (f"{_pretty_method(method_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}"
            f"  |  {_pretty_dataset(dataset_name)}")


def _stamp_footer(fig, method_name: str, dataset_name: str,
                  balanced_replay: bool | None = None,
                  balanced_loss: bool | None = None,
                  use_distillation: bool | None = None,
                  herding_replay: bool | None = None):
    """Add a small method + dataset footnote at the bottom-right of a figure."""
    fig.text(
        0.99, 0.005,
        (f"Method: {_pretty_method(method_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}"
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
                         use_distillation: bool | None = None,
                         herding_replay: bool | None = None):
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
        f"Training Curves — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}",
        fontsize=14, y=1.02)
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_sequential_training_curves(history: dict, save_path, *,
                                    method_name: str = "", dataset_name: str = "",
                                    balanced_replay: bool | None = None,
                                    balanced_loss: bool | None = None,
                                    use_distillation: bool | None = None,
                                    herding_replay: bool | None = None):
    """
    Plot loss and accuracy curves for sequential (CIL) training.
    """
    num_tasks = len(history)
    fig, axes = plt.subplots(num_tasks, 2, figsize=(12, 4 * num_tasks),
                             squeeze=False)

    cmap = plt.colormaps.get_cmap("tab10").resampled(num_tasks)

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
        f"Sequential Training Curves — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}",
        fontsize=14, y=1.01,
    )
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
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
                                   use_distillation: bool | None = None,
                                   herding_replay: bool | None = None):
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
        f"Accuracy Progression — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}",
        fontsize=13,
    )
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
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
                          herding_replay: bool | None = None,
                          title: str | None = None):
    """
    Annotated confusion-matrix heatmap.
    """
    if title is None:
        title = f"Confusion Matrix — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}"

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

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
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
                            herding_replay: bool | None = None,
                            title: str | None = None):
    """
    Horizontal bar chart of per-class accuracy.
    """
    if title is None:
        title = f"Per-Class Accuracy — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}"

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

    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Forgetting analysis (sequential only)
# =====================================================================

def _build_acc_matrix(task_results):
    """Build a (num_tasks x num_tasks) accuracy matrix.

    ``acc_matrix[i, j]`` = accuracy on task *j+1* after training up to
    task *i+1*.  Upper-triangle entries (tasks not yet seen) are NaN.
    """
    num_tasks = len(task_results)
    acc_matrix = np.full((num_tasks, num_tasks), np.nan)
    for row, result in enumerate(task_results):
        per_task = result.get("per_task_acc", {})
        for task_j, acc_val in per_task.items():
            col = int(task_j) - 1
            acc_matrix[row, col] = acc_val
    return acc_matrix


def plot_forgetting_analysis(task_results, save_path, *,
                             method_name: str = "", dataset_name: str = "",
                             balanced_replay: bool | None = None,
                             balanced_loss: bool | None = None,
                             use_distillation: bool | None = None,
                             herding_replay: bool | None = None):
    """
    CIL-specific analysis page:

    * **Left** – accuracy heatmap: rows = "after learning task i",
      columns = "accuracy on task j".  Shows exactly where forgetting
      happens.
    * **Right** – per-task forgetting bars + key scalar metrics (AIA,
      average forgetting, backward transfer).
    """
    acc_matrix = _build_acc_matrix(task_results)
    num_tasks = acc_matrix.shape[0]

    # ── Scalar CIL metrics ──────────────────────────────────────
    # Average Incremental Accuracy: mean of seen-class acc after each task
    aia = np.mean([r["seen_acc"] for r in task_results])

    # Per-task forgetting: max accuracy on task j across all steps up to
    # the current one, minus its final accuracy.
    forgetting = []
    for j in range(num_tasks):
        col = acc_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            forgetting.append(float(np.max(valid) - valid[-1]))
        else:
            forgetting.append(0.0)
    avg_forgetting = np.mean(forgetting) if forgetting else 0.0

    # Backward transfer: for each task j, final_acc(j) - acc_right_after_learning(j)
    bwt_vals = []
    for j in range(num_tasks):
        learned_at = j
        final_at = num_tasks - 1
        if learned_at < final_at and not np.isnan(acc_matrix[learned_at, j]):
            bwt_vals.append(acc_matrix[final_at, j] - acc_matrix[learned_at, j])
    avg_bwt = np.mean(bwt_vals) if bwt_vals else 0.0

    # ── Figure ──────────────────────────────────────────────────
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(14, max(5, 1.2 * num_tasks + 2)))

    # -- Left: accuracy heatmap --
    masked = np.ma.array(acc_matrix, mask=np.isnan(acc_matrix))
    im = ax_heat.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Accuracy")
    for i in range(num_tasks):
        for j in range(num_tasks):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                ax_heat.text(j, i, f"{val:.0%}", ha="center", va="center",
                             fontsize=max(7, 11 - num_tasks // 3),
                             fontweight="bold",
                             color="white" if val < 0.45 else "black")
    ax_heat.set_xticks(range(num_tasks))
    ax_heat.set_yticks(range(num_tasks))
    ax_heat.set_xticklabels([f"Task {i+1}" for i in range(num_tasks)], fontsize=9)
    ax_heat.set_yticklabels([f"After Task {i+1}" for i in range(num_tasks)], fontsize=9)
    ax_heat.set_xlabel("Evaluated on")
    ax_heat.set_ylabel("Model state")
    ax_heat.set_title("Per-Task Accuracy Over Time")

    # -- Right: forgetting bars + metrics --
    task_labels = [f"Task {i+1}" for i in range(num_tasks)]
    y_pos = np.arange(num_tasks)
    colors = ["#d62728" if f > 0.05 else "#2ca02c" for f in forgetting]
    bars = ax_bar.barh(y_pos, forgetting, color=colors, edgecolor="black", linewidth=0.4)
    for bar, f_val in zip(bars, forgetting):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{f_val:.2%}", va="center", fontsize=9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(task_labels, fontsize=9)
    ax_bar.set_xlabel("Forgetting (peak acc − final acc)")
    ax_bar.set_title("Per-Task Forgetting")
    ax_bar.set_xlim(0, max(0.2, max(forgetting) * 1.3 + 0.05) if forgetting else 0.2)
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.3)

    # Metrics text box
    metrics_text = (
        f"Avg. Incremental Acc (AIA): {aia:.2%}\n"
        f"Avg. Forgetting: {avg_forgetting:.2%}\n"
        f"Backward Transfer (BWT): {avg_bwt:+.2%}"
    )
    ax_bar.text(
        0.97, 0.97, metrics_text, transform=ax_bar.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9),
    )

    fig.suptitle(
        f"Forgetting Analysis — {_header(method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)}",
        fontsize=14, y=1.02,
    )
    _stamp_footer(fig, method_name, dataset_name, balanced_replay, balanced_loss, use_distillation, herding_replay)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =====================================================================
#  Convenience wrappers (called from train.py)
# =====================================================================

def _plot_kwargs(config):
    return dict(
        method_name=config["method"],
        dataset_name=config["dataset"],
        balanced_replay=config.get("balanced_replay"),
        balanced_loss=config.get("balanced_loss"),
        use_distillation=config.get("use_distillation"),
        herding_replay=config.get("herding_replay", False),
    )


def save_standard_plots(history, y_true, y_pred, class_names, run_dir, config):
    """Save all plots for a standard (non-sequential) run."""
    pkw = _plot_kwargs(config)
    plot_training_curves(history, run_dir / "training_curves.png", **pkw)
    plot_confusion_matrix(y_true, y_pred, class_names, run_dir / "confusion_matrix.png", **pkw)
    plot_per_class_accuracy(y_true, y_pred, class_names, run_dir / "per_class_accuracy.png", **pkw)
    print(f"\nPlots saved to: {run_dir}")


def save_sequential_plots(history, task_results, y_true, y_pred, class_names, run_dir, config):
    """Save all plots for a sequential (CIL) run."""
    pkw = _plot_kwargs(config)
    plot_sequential_training_curves(history, run_dir / "training_curves.png", **pkw)
    plot_task_accuracy_progression(task_results, run_dir / "task_accuracy_progression.png", **pkw)
    plot_confusion_matrix(y_true, y_pred, class_names, run_dir / "confusion_matrix.png", **pkw)
    plot_per_class_accuracy(y_true, y_pred, class_names, run_dir / "per_class_accuracy.png", **pkw)
    plot_forgetting_analysis(task_results, run_dir / "forgetting_analysis.png", **pkw)
    print(f"\nPlots saved to: {run_dir}")
