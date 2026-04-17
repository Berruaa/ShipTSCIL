"""
Persistent results storage.

After every training run, ``append_run`` appends a structured record to
``results/all_results.json``.  The file grows incrementally so results from
different experiments accumulate automatically.

Usage (called internally from train.py):
    from utils.results_logger import build_standard_result, build_sequential_result, append_run
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert numpy / non-JSON types to plain Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _config_subset(config):
    """Return the config dict stripped of any non-serialisable values."""
    skip = {"task_order"}   # stored separately; can contain numpy ints
    safe = {}
    for k, v in config.items():
        if k in skip:
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe


# ── CIL scalar metrics ────────────────────────────────────────────────────────

def _compute_cil_metrics(task_results):
    """Return (AIA, avg_forgetting, avg_bwt) from a list of task result dicts."""
    num_tasks = len(task_results)
    aia = float(np.mean([r["seen_acc"] for r in task_results]))

    def _get(d, key):
        """Lookup by int or string key."""
        return d.get(key, d.get(str(key)))

    forgetting = []
    for j in range(num_tasks):
        accs_j = [_get(r["per_task_acc"], j + 1) for r in task_results]
        accs_j = [a for a in accs_j if a is not None]
        if len(accs_j) >= 2:
            forgetting.append(float(max(accs_j) - accs_j[-1]))
    avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    bwt_vals = []
    for j in range(num_tasks - 1):
        learned = _get(task_results[j]["per_task_acc"], j + 1)
        final   = _get(task_results[-1]["per_task_acc"], j + 1)
        if learned is not None and final is not None:
            bwt_vals.append(float(final - learned))
    avg_bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    return aia, avg_forgetting, avg_bwt


def _per_class_acc(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        pca = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))
    return dict(zip(class_names, pca.tolist()))


# ── Public builders ───────────────────────────────────────────────────────────

def build_standard_result(config, best_test_acc, y_true, y_pred, label_encoder, history):
    """Build a JSON-serialisable record for a standard (non-sequential) run."""
    from sklearn.metrics import confusion_matrix as sk_cm

    y_true_list = [int(v) for v in y_true]
    y_pred_list = [int(v) for v in y_pred]
    class_names  = [str(c) for c in label_encoder.classes_]
    cm           = sk_cm(y_true_list, y_pred_list)
    final_acc    = float(np.mean(np.array(y_true_list) == np.array(y_pred_list)))

    return {
        "run_id":       f"{config['dataset']}_{config['method']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp":    datetime.now().isoformat(),
        "method":       config["method"],
        "dataset":      config["dataset"],
        "seed":         config.get("seed"),
        "type":         "standard",
        "best_test_acc": float(best_test_acc),
        "final_acc":    final_acc,
        "class_names":  class_names,
        "confusion_matrix": _to_serializable(cm),
        "per_class_acc": _per_class_acc(y_true_list, y_pred_list, class_names),
        "y_true":       y_true_list,
        "y_pred":       y_pred_list,
        "history":      _to_serializable(history),
        "config":       _config_subset(config),
    }


def build_sequential_result(config, best_seen_acc, task_results, y_true, y_pred, label_encoder, history):
    """Build a JSON-serialisable record for a sequential (CIL) run."""
    from sklearn.metrics import confusion_matrix as sk_cm

    y_true_list = [int(v) for v in y_true]
    y_pred_list = [int(v) for v in y_pred]
    class_names  = [str(c) for c in label_encoder.classes_]
    cm           = sk_cm(y_true_list, y_pred_list)
    final_acc    = float(np.mean(np.array(y_true_list) == np.array(y_pred_list)))
    aia, avg_forgetting, avg_bwt = _compute_cil_metrics(task_results)

    return {
        "run_id":        f"{config['dataset']}_{config['method']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp":     datetime.now().isoformat(),
        "method":        config["method"],
        "dataset":       config["dataset"],
        "seed":          config.get("seed"),
        "type":          "sequential",
        "best_seen_acc": float(best_seen_acc),
        "final_acc":     final_acc,
        "aia":           aia,
        "avg_forgetting": avg_forgetting,
        "bwt":           avg_bwt,
        "task_results":  _to_serializable(task_results),
        "class_names":   class_names,
        "confusion_matrix": _to_serializable(cm),
        "per_class_acc": _per_class_acc(y_true_list, y_pred_list, class_names),
        "y_true":        y_true_list,
        "y_pred":        y_pred_list,
        "history":       _to_serializable(history),
        "config":        _config_subset(config),
    }


# ── Storage ───────────────────────────────────────────────────────────────────

def append_run(result: dict, results_dir: Path):
    """Append *result* to ``<results_dir>/all_results.json``.

    The file is a JSON array that grows with every call, so no run is ever
    overwritten.  Re-running the same experiment simply adds a newer record.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "all_results.json"

    existing = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(result)

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    print(f"  Results logged → {log_path}  (total runs: {len(existing)})")
