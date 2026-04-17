"""
generate_report.py
==================
Generate publication-ready comparison figures for report Section 5.1.

All figures are written to ``results/report_<dataset>/`` (or a path you
specify with ``--output``).  Run this once after finishing all experiments.

Usage examples
--------------
    python generate_report.py
    python generate_report.py --dataset ethanol_level
    python generate_report.py --dataset ethanol_level --output results/my_report
    python generate_report.py --results results/all_results.json --dataset ethanol_level
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix as sk_cm


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Consistent palette for methods (expand as needed)
METHOD_COLORS = {
    "linear_probe":      "#4C72B0",
    "svm":               "#DD8452",
    "cil_naive":         "#55A868",
    "cil_replay_raw":    "#C44E52",
    "cil_replay_latent": "#8172B3",
    "cil_replay_lwf":    "#64B5CD",
    "cil_lwf":           "#937860",
    "cil_ncm":           "#DA8BC3",
    "cil_herding_ncm":   "#8C8C8C",
    "cil_olora":         "#CCB974",
}
DEFAULT_COLOR = "#333333"

METHOD_LABELS = {
    "linear_probe":      "Linear Probe",
    "svm":               "SVM",
    "cil_naive":         "CIL Naive",
    "cil_replay_raw":    "Raw Replay",
    "cil_replay_latent": "Latent Replay",
    "cil_replay_lwf":    "Latent Replay + LwF",
    "cil_lwf":           "LwF",
    "cil_ncm":           "NCM",
    "cil_herding_ncm":   "Herding+NCM",
    "cil_olora":         "O-LoRA",
}

BASELINE_METHODS  = ["linear_probe", "svm", "cil_naive"]
SEQUENTIAL_TYPES  = {"sequential"}

SECTION_511 = "5.1.1"
SECTION_512 = "5.1.2"


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_per_method(runs: list[dict]) -> dict[str, dict]:
    """Return the most-recent run for each method key."""
    best: dict[str, dict] = {}
    for run in runs:
        key = run["method"]
        if key not in best or run["timestamp"] > best[key]["timestamp"]:
            best[key] = run
    return best


def filter_dataset(runs: list[dict], dataset: str) -> list[dict]:
    return [r for r in runs if r["dataset"] == dataset]


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def color(method: str) -> str:
    return METHOD_COLORS.get(method, DEFAULT_COLOR)


def pretty_dataset(name: str) -> str:
    parts = name.split("_")
    return " ".join(p.upper() if any(c.isdigit() for c in p) else p.title()
                    for p in parts)


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _annotate_bars(ax, bars, fmt="{:.1%}", offset=0.005, fontsize=9, horizontal=False):
    for bar in bars:
        if horizontal:
            val = bar.get_width()
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=fontsize)
        else:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    fmt.format(val), ha="center", va="bottom", fontsize=fontsize,
                    fontweight="bold")


# ── Section 5.1.1 — Baseline Results ─────────────────────────────────────────

def plot_511_accuracy(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Grouped bar: final test accuracy for baselines."""
    methods = [m for m in BASELINE_METHODS if m in runs_by_method]
    if not methods:
        print("  [5.1.1] No baseline methods found, skipping accuracy bar chart.")
        return

    accs = [runs_by_method[m]["final_acc"] for m in methods]
    xs   = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(methods)), 5))
    bars = ax.bar(xs, accs,
                  color=[color(m) for m in methods],
                  edgecolor="black", linewidth=0.6, width=0.5)
    _annotate_bars(ax, bars)

    ax.set_xticks(xs)
    ax.set_xticklabels([label(m) for m in methods], fontsize=11)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Baseline Accuracy — {pretty_dataset(dataset_name)}", fontsize=13)
    ax.axhline(max(accs), color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, out_dir / f"{SECTION_511}_baseline_accuracy.png")


def plot_511_confusion_grid(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Side-by-side confusion matrices for each baseline."""
    methods = [m for m in BASELINE_METHODS if m in runs_by_method]
    if not methods:
        return

    n_methods = len(methods)
    # Determine grid size from first run
    sample_run   = runs_by_method[methods[0]]
    class_names  = sample_run["class_names"]
    n_classes    = len(class_names)
    cell_size    = max(0.5, min(0.9, 8 / n_classes))

    fig, axes = plt.subplots(1, n_methods,
                             figsize=(n_methods * n_classes * cell_size + 1,
                                      n_classes * cell_size + 1.5),
                             squeeze=False)

    for col, method in enumerate(methods):
        run = runs_by_method[method]
        cm  = np.array(run["confusion_matrix"])
        ax  = axes[0, col]

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        thresh = cm.max() / 2
        fs = max(6, 10 - n_classes // 4)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=fs,
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fs)
        ax.set_yticklabels(class_names if col == 0 else [], fontsize=fs)
        ax.set_xlabel("Predicted", fontsize=9)
        if col == 0:
            ax.set_ylabel("True", fontsize=9)
        ax.set_title(label(method), fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Confusion Matrices — {pretty_dataset(dataset_name)}", fontsize=13, y=1.02)
    _save(fig, out_dir / f"{SECTION_511}_baseline_confusion_grid.png")


def plot_511_per_class_heatmap(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Heatmap: rows = methods, columns = classes, values = per-class accuracy."""
    methods = [m for m in BASELINE_METHODS if m in runs_by_method]
    if not methods:
        return

    sample_run  = runs_by_method[methods[0]]
    class_names = sample_run["class_names"]
    n_classes   = len(class_names)

    matrix = np.zeros((len(methods), n_classes))
    for i, method in enumerate(methods):
        pca = runs_by_method[method]["per_class_acc"]
        for j, cn in enumerate(class_names):
            matrix[i, j] = pca.get(cn, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n_classes), max(3, 0.9 * len(methods))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)

    fs = max(7, 11 - n_classes // 6)
    for i in range(len(methods)):
        for j in range(n_classes):
            ax.text(j, i, f"{matrix[i, j]:.0%}",
                    ha="center", va="center", fontsize=fs,
                    color="black" if 0.2 < matrix[i, j] < 0.85 else "white")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fs)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([label(m) for m in methods], fontsize=10)
    ax.set_title(f"Per-Class Accuracy — {pretty_dataset(dataset_name)}", fontsize=13)

    _save(fig, out_dir / f"{SECTION_511}_baseline_per_class_heatmap.png")


# ── Section 5.1.2 — All CIL Methods ──────────────────────────────────────────

def _seq_methods(runs_by_method: dict) -> list[str]:
    """Return keys whose stored run is sequential type, sorted."""
    return sorted([k for k, v in runs_by_method.items() if v.get("type") == "sequential"])


def plot_512_metrics_bar(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Grouped bar chart: AIA, Forgetting, BWT for all sequential methods."""
    methods = _seq_methods(runs_by_method)
    if not methods:
        print("  [5.1.2] No sequential results found, skipping CIL metrics bar chart.")
        return

    aias  = [runs_by_method[m]["aia"]           for m in methods]
    fgts  = [runs_by_method[m]["avg_forgetting"] for m in methods]
    bwts  = [runs_by_method[m]["bwt"]            for m in methods]

    xs     = np.arange(len(methods))
    width  = 0.25
    labels = [label(m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(methods)), 5))

    b1 = ax.bar(xs - width, aias, width, label="AIA ↑",
                color="#4C72B0", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(xs,          fgts, width, label="Forgetting ↓",
                color="#C44E52", edgecolor="black", linewidth=0.5)
    b3 = ax.bar(xs + width,  bwts, width, label="BWT ↑",
                color="#55A868", edgecolor="black", linewidth=0.5)

    for bars in (b1, b2, b3):
        _annotate_bars(ax, bars, fmt="{:.2%}", offset=0.004, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(min(-0.15, min(bwts) - 0.1), 1.15)
    ax.set_title(f"CIL Metrics — {pretty_dataset(dataset_name)}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, out_dir / f"{SECTION_512}_cil_metrics_bar.png")


def plot_512_metrics_table(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Render a clean summary table of all CIL metrics as a figure."""
    methods = _seq_methods(runs_by_method)
    if not methods:
        return

    rows    = []
    col_hdr = ["Method", "Final Acc ↑", "AIA ↑", "Forgetting ↓", "BWT"]
    for m in methods:
        r = runs_by_method[m]
        rows.append([
            label(m),
            f"{r['final_acc']:.2%}",
            f"{r['aia']:.2%}",
            f"{r['avg_forgetting']:.2%}",
            f"{r['bwt']:+.2%}",
        ])

    fig_h = max(2.5, 0.55 * (len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_hdr,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    # Style header row
    for col_idx in range(len(col_hdr)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for row_idx in range(1, len(rows) + 1):
        shade = "#F0F4F8" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(col_hdr)):
            tbl[row_idx, col_idx].set_facecolor(shade)

    ax.set_title(f"CIL Summary Table — {pretty_dataset(dataset_name)}",
                 fontsize=13, pad=16, y=1.0)

    _save(fig, out_dir / f"{SECTION_512}_cil_metrics_table.png")


def plot_512_task_progression_overlay(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Seen-class accuracy after each task, all CIL methods on one plot."""
    methods = _seq_methods(runs_by_method)
    if not methods:
        return

    max_tasks = max(
        len(runs_by_method[m]["task_results"]) for m in methods
    )
    fig_size = max(6, 1.4 * max_tasks)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Distinct marker + linestyle per method so overlapping lines stay
    # visually distinguishable (several methods start at identical values).
    style_cycle = [
        ("o", "-"),   ("s", "--"),  ("^", "-."),  ("D", ":"),
        ("v", "-"),   ("P", "--"),  ("X", "-."),  ("*", ":"),
    ]

    # Small horizontal jitter around each integer task id so markers that
    # coincide on the same (task, acc) point don't hide each other.
    n = len(methods)
    jitter_width = 0.28
    offsets = (
        np.linspace(-jitter_width / 2, jitter_width / 2, n)
        if n > 1 else np.array([0.0])
    )

    for idx, method in enumerate(methods):
        run         = runs_by_method[method]
        task_results = run["task_results"]
        task_ids     = [r["task_id"] for r in task_results]
        seen_accs    = [r["seen_acc"] for r in task_results]

        mk, ls = style_cycle[idx % len(style_cycle)]
        x_vals = [t + offsets[idx] for t in task_ids]
        ax.plot(x_vals, seen_accs,
                marker=mk, markersize=9, linewidth=2.2,
                linestyle=ls, alpha=0.9,
                color=color(method), label=label(method),
                markeredgecolor="white", markeredgewidth=0.9)

    ax.set_xlabel("Task")
    ax.set_ylabel("Seen-Class Accuracy")
    ax.set_title(f"Task Accuracy Progression — {pretty_dataset(dataset_name)}", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_box_aspect(1)

    _save(fig, out_dir / f"{SECTION_512}_task_progression_overlay.png")


def plot_512_forgetting_comparison(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Per-task forgetting for every sequential method, shown as grouped bars."""
    methods = _seq_methods(runs_by_method)
    if not methods:
        return

    # Collect per-task forgetting vectors (may differ in length)
    all_forgetting: dict[str, list[float]] = {}
    max_tasks = 0
    for method in methods:
        run          = runs_by_method[method]
        task_results  = run["task_results"]
        n             = len(task_results)
        max_tasks     = max(max_tasks, n)
        fgt = []
        for j in range(n):
            accs_j = [r["per_task_acc"].get(str(j + 1), r["per_task_acc"].get(j + 1))
                      for r in task_results]
            accs_j = [a for a in accs_j if a is not None]
            fgt.append(max(accs_j) - accs_j[-1] if len(accs_j) >= 2 else 0.0)
        all_forgetting[method] = fgt

    xs    = np.arange(max_tasks)
    width = 0.8 / len(methods)
    offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * width

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * max_tasks), 5))

    for i, method in enumerate(methods):
        fgt    = all_forgetting[method]
        x_vals = xs[:len(fgt)] + offsets[i]
        ax.bar(x_vals, fgt, width * 0.9,
               color=color(method), edgecolor="black", linewidth=0.5,
               label=label(method), alpha=0.85)

    ax.set_xticks(xs[:max_tasks])
    ax.set_xticklabels([f"Task {i + 1}" for i in range(max_tasks)], fontsize=10)
    ax.set_ylabel("Forgetting  (peak acc − final acc)")
    ax.set_title(f"Per-Task Forgetting — {pretty_dataset(dataset_name)}", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, out_dir / f"{SECTION_512}_forgetting_comparison.png")


def plot_512_per_class_heatmap(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Heatmap: rows = sequential methods, columns = classes."""
    methods = _seq_methods(runs_by_method)
    if not methods:
        return

    sample_run  = runs_by_method[methods[0]]
    class_names = sample_run["class_names"]
    n_classes   = len(class_names)

    matrix = np.zeros((len(methods), n_classes))
    for i, method in enumerate(methods):
        pca = runs_by_method[method]["per_class_acc"]
        for j, cn in enumerate(class_names):
            matrix[i, j] = pca.get(cn, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n_classes), max(3, 0.9 * len(methods))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)

    fs = max(7, 11 - n_classes // 6)
    for i in range(len(methods)):
        for j in range(n_classes):
            ax.text(j, i, f"{matrix[i, j]:.0%}",
                    ha="center", va="center", fontsize=fs,
                    color="black" if 0.2 < matrix[i, j] < 0.85 else "white")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fs)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([label(m) for m in methods], fontsize=10)
    ax.set_title(f"Per-Class Accuracy (CIL Methods) — {pretty_dataset(dataset_name)}", fontsize=13)

    _save(fig, out_dir / f"{SECTION_512}_cil_per_class_heatmap.png")


def plot_512_all_methods_accuracy(runs_by_method: dict, dataset_name: str, out_dir: Path):
    """Single bar chart: final accuracy for ALL methods (baselines + CIL)."""
    # Preferred display order
    order = [
        "linear_probe", "svm",
        "cil_naive",
        "cil_replay_latent",
        "cil_replay_lwf",
        "cil_lwf",
        "cil_ncm", "cil_herding_ncm",
        "cil_olora",
    ]
    methods = [m for m in order if m in runs_by_method]
    methods += [m for m in runs_by_method if m not in order]
    if not methods:
        return

    accs = [runs_by_method[m]["final_acc"] for m in methods]
    xs   = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(methods)), 5.5))
    bars = ax.bar(xs, accs,
                  color=[color(m) for m in methods],
                  edgecolor="black", linewidth=0.6, width=0.6)
    _annotate_bars(ax, bars)

    # Shade baseline vs CIL regions
    n_base = sum(1 for m in methods if m in BASELINE_METHODS)
    if n_base > 0 and n_base < len(methods):
        ax.axvspan(-0.5, n_base - 0.5, alpha=0.06, color="#4C72B0", label="Baselines")
        ax.axvspan(n_base - 0.5, len(methods) - 0.5, alpha=0.06, color="#C44E52", label="CIL Methods")

    ax.set_xticks(xs)
    ax.set_xticklabels([label(m) for m in methods], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"All Methods — Final Accuracy — {pretty_dataset(dataset_name)}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, out_dir / f"{SECTION_512}_all_methods_accuracy.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Section 5.1 comparison figures from all_results.json"
    )
    parser.add_argument(
        "--results",
        default="results/all_results.json",
        help="Path to all_results.json (default: results/all_results.json)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to report on. If omitted, all datasets are processed.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: results/report_<dataset>_<timestamp>/)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        print("Run at least one experiment first (train.py saves results automatically).")
        sys.exit(1)

    all_runs = load_results(results_path)
    if not all_runs:
        print("No results found in the file yet.")
        sys.exit(1)

    # Determine which datasets to process
    datasets_in_file = sorted({r["dataset"] for r in all_runs})
    if args.dataset:
        if args.dataset not in datasets_in_file:
            print(f"ERROR: dataset '{args.dataset}' not found in results.")
            print(f"Available datasets: {datasets_in_file}")
            sys.exit(1)
        datasets_to_process = [args.dataset]
    else:
        datasets_to_process = datasets_in_file

    print(f"Loaded {len(all_runs)} run(s) from {results_path}")
    print(f"Datasets to process: {datasets_to_process}\n")

    for dataset in datasets_to_process:
        ds_runs       = filter_dataset(all_runs, dataset)
        runs_by_method = latest_per_method(ds_runs)

        methods_found = sorted(runs_by_method.keys())
        print(f"Dataset: {dataset}  ({len(ds_runs)} run(s) total, "
              f"{len(methods_found)} unique method(s): {methods_found})")

        # Determine output directory
        if args.output:
            out_dir = Path(args.output)
        else:
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("results") / f"report_{dataset}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output → {out_dir}\n")

        # ── Section 5.1.1 — Baselines ────────────────────────────────────
        print(f"[{SECTION_511}] Baseline figures …")
        plot_511_accuracy(runs_by_method, dataset, out_dir)
        plot_511_confusion_grid(runs_by_method, dataset, out_dir)
        plot_511_per_class_heatmap(runs_by_method, dataset, out_dir)

        # ── Section 5.1.2 — All CIL methods ─────────────────────────────
        print(f"\n[{SECTION_512}] CIL comparison figures …")
        plot_512_all_methods_accuracy(runs_by_method, dataset, out_dir)
        plot_512_metrics_table(runs_by_method, dataset, out_dir)
        plot_512_metrics_bar(runs_by_method, dataset, out_dir)
        plot_512_task_progression_overlay(runs_by_method, dataset, out_dir)
        plot_512_forgetting_comparison(runs_by_method, dataset, out_dir)
        plot_512_per_class_heatmap(runs_by_method, dataset, out_dir)

        print(f"\nDone — {dataset}\n{'─'*60}\n")

    print("All figures generated.")


if __name__ == "__main__":
    main()
