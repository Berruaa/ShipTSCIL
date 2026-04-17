"""
run_all.py
==========
Train every (dataset × method) combination and record results to
``results/all_results.json`` automatically.

LoRA and O-LoRA are excluded by default (too compute-heavy).
The MOMENT embedding cache is shared across methods for the same dataset,
so each dataset only pays the full inference cost once.

Usage
-----
    python run_all.py                                   # all datasets × all methods
    python run_all.py --datasets ecg5000 ethanol_level  # subset of datasets
    python run_all.py --methods linear_probe cil_naive  # subset of methods
    python run_all.py --dry-run                         # preview without training
    python run_all.py --skip-existing                   # skip already-logged runs
    python run_all.py --seed 0                          # override random seed
"""

import argparse
import sys
import time
import traceback
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
SAVE_DIR     = PROJECT_ROOT / "checkpoints"
CACHE_DIR    = PROJECT_ROOT / "embeddings_cache"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Resolved lazily in main() so --dry-run works without a full ML environment.
DEVICE: str = "cpu"

# ── Experiment matrix ─────────────────────────────────────────────────────────

ALL_DATASETS = [
    "ecg5000",
    "electric_devices",
    "uwave_gesture_library",
    "uci_har",
    "wisdm",
    "ethanol_level",
    "insect_sound",
]

# Methods in the order they appear in the report (baselines first, then CIL).
# LoRA / O-LoRA are intentionally omitted.
ALL_METHODS = [
    "linear_probe",
    "svm",
    "cil_naive",
    "cil_replay_latent",
    "cil_lwf",
    "cil_ncm",
    "cil_herding_ncm",
]

# ── Default CONFIG template ───────────────────────────────────────────────────

BASE_CONFIG = {
    "model_name":          "AutonLab/MOMENT-1-base",
    "train_file":          None,
    "test_file":           None,
    "batch_size":          "auto",
    "epochs":              "auto",
    "lr":                  "auto",
    "seed":                42,
    "num_workers":         0,
    # CIL task schedule
    "task_order":          None,
    "num_tasks":           None,
    "shuffle_class_order": True,
    "classes_per_task":    2,
    # Replay
    "replay_buffer_pct":   0.05,
    "replay_buffer_size":  "auto",
    "replay_batch_size":   "auto",
    "balanced_replay":     True,
    "herding_replay":      False,
    # Loss
    "balanced_loss":       True,
    # Distillation
    "use_distillation":    False,
    "distill_temperature": 2.0,
    "distill_weight":      1.0,
    # LoRA — disabled for all runs in this script
    "use_lora":            False,
    "lora_rank":           8,
    "lora_alpha":          16,
    "lora_target_modules": None,
    "lora_dropout":        0.05,
    "lora_lr":             "auto",
    # O-LoRA — disabled
    "use_olora":           False,
    "olora_lambda":        1.0,
    # Output
    "save_results":        True,  
}

# Per-method config overrides applied on top of BASE_CONFIG
METHOD_OVERRIDES = {
    "cil_herding_ncm": {"herding_replay": True},
    "cil_lwf":         {"use_distillation": False},  # LwF distills inherently
    "cil_replay_latent": {"use_distillation": False},
}


def _build_config(dataset: str, method: str, seed: int) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"] = dataset
    cfg["method"]  = method
    cfg["seed"]    = seed
    cfg.update(METHOD_OVERRIDES.get(method, {}))
    return cfg


# ── Core training logic (mirrors train.py main()) ────────────────────────────

def run_one(config: dict) -> dict:
    """
    Run a single (dataset, method) experiment.
    Returns the result dict that was saved to all_results.json.
    Raises on unrecoverable errors.
    """
    from methods import build_method, STANDARD_METHODS, SEQUENTIAL_METHODS
    from pipelines import (
        auto_configure,
        build_base_datasets_and_info,
        build_loader,
        build_task_order,
        collect_predictions,
        make_class_subset,
        precompute_embeddings,
        train_sequential,
        train_standard,
        validate_task_order,
    )
    from utils.seed import set_seed
    from utils.reporting import (
        print_final_sequential_results,
        print_final_standard_results,
        print_run_info,
    )
    from utils.results_logger import (
        append_run,
        build_sequential_result,
        build_standard_result,
    )
    from utils.plotting import save_standard_plots, save_sequential_plots

    set_seed(config["seed"])

    # ── Load data ─────────────────────────────────────────────────────────────
    train_dataset, test_dataset, label_encoder, dataset_info = \
        build_base_datasets_and_info(config=config, data_dir=DATA_DIR)

    num_classes = len(label_encoder.classes_)
    auto_configure(config, n_train=len(train_dataset), num_classes=num_classes)

    _SINGLE_EPOCH = {"svm", "cil_ncm", "ncm", "cil_herding_ncm", "herding_ncm"}
    if config["method"] in _SINGLE_EPOCH:
        config["epochs"] = 1

    # ── Task order ────────────────────────────────────────────────────────────
    if config["method"] in SEQUENTIAL_METHODS:
        if config.get("task_order") is None:
            config["task_order"] = build_task_order(
                num_classes=num_classes,
                seed=config["seed"],
                shuffle_class_order=config.get("shuffle_class_order", True),
                classes_per_task=config.get("classes_per_task", 2),
                num_tasks=config.get("num_tasks"),
            )
        validate_task_order(config["task_order"], num_classes)

    # ── Build method (no LoRA in this script) ─────────────────────────────────
    method = build_method(
        method_name=config["method"],
        model_name=config["model_name"],
        num_classes=num_classes,
        train_dataset=train_dataset,
        device=DEVICE,
        lr=config["lr"],
        replay_buffer_size=config.get("replay_buffer_size", 1000),
        replay_batch_size=config.get("replay_batch_size", 32),
        balanced_replay=config.get("balanced_replay", True),
        balanced_loss=config.get("balanced_loss", True),
        use_distillation=config.get("use_distillation", False),
        distill_temperature=config.get("distill_temperature", 2.0),
        distill_weight=config.get("distill_weight", 1.0),
        herding_replay=config.get("herding_replay", False),
        lora_config=None,
    )

    print_run_info(
        config=config, dataset_info=dataset_info,
        label_encoder=label_encoder, method=method, device=DEVICE,
    )

    # ── Embeddings (cached to disk — shared across methods for same dataset) ──
    encoder = getattr(method, "encoder", None) or method.model.encoder
    emb_kwargs = dict(
        device=DEVICE, batch_size=config["batch_size"],
        cache_dir=CACHE_DIR, dataset_name=config["dataset"],
        model_name=config["model_name"],
    )
    print(f"\nPreparing embeddings … (device: {DEVICE})")
    if config["method"] != "cil_replay_raw":
        train_dataset = precompute_embeddings(encoder, train_dataset, split="train", **emb_kwargs)
    test_dataset = precompute_embeddings(encoder, test_dataset, split="test", **emb_kwargs)
    print("Done.\n")

    class_names = [str(c) for c in label_encoder.classes_]

    # ── Per-run plot directory ─────────────────────────────────────────────────
    save_plots = config.get("save_results", True)
    if save_plots:
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"{config['dataset']}_{config['method']}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {run_dir}\n")
    else:
        run_dir = None

    # ── Standard methods ──────────────────────────────────────────────────────
    if config["method"] in STANDARD_METHODS:
        train_loader = build_loader(train_dataset, config["batch_size"],
                                    shuffle=True, num_workers=config["num_workers"])
        test_loader  = build_loader(test_dataset,  config["batch_size"],
                                    shuffle=False, num_workers=config["num_workers"])

        checkpoint_path, best_test_acc, history = train_standard(
            method=method, train_loader=train_loader, test_loader=test_loader,
            config=config, label_encoder=label_encoder, save_dir=SAVE_DIR,
        )
        method.load(checkpoint_path)
        y_true, y_pred = collect_predictions(method, test_loader, device=DEVICE)
        print_final_standard_results(best_test_acc, y_true, y_pred, label_encoder)

        if save_plots:
            save_standard_plots(history, y_true, y_pred, class_names, run_dir, config)

        result = build_standard_result(config, best_test_acc, y_true, y_pred, label_encoder, history)
        append_run(result, RESULTS_DIR)
        return result

    # ── Sequential methods ────────────────────────────────────────────────────
    if config["method"] in SEQUENTIAL_METHODS:
        _, final_checkpoint_path, best_seen_acc, task_results, history = train_sequential(
            method=method, train_dataset=train_dataset, test_dataset=test_dataset,
            config=config, label_encoder=label_encoder, save_dir=SAVE_DIR,
        )

        all_seen = sorted(set(c for task in config["task_order"] for c in task))
        method.load(final_checkpoint_path)
        final_loader = build_loader(
            make_class_subset(test_dataset, all_seen),
            config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        )
        y_true, y_pred = collect_predictions(method, final_loader, device=DEVICE)
        print_final_sequential_results(best_seen_acc, task_results, y_true, y_pred, label_encoder)

        if save_plots:
            save_sequential_plots(history, task_results, y_true, y_pred, class_names, run_dir, config)

        result = build_sequential_result(config, best_seen_acc, task_results, y_true, y_pred, label_encoder, history)
        append_run(result, RESULTS_DIR)
        return result

    raise ValueError(f"Unsupported method: {config['method']}")


# ── Progress display ──────────────────────────────────────────────────────────

def _hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_banner(text: str, width: int = 72):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _print_run_header(idx: int, total: int, dataset: str, method: str):
    pct = 100 * idx / total
    print(f"\n{'─' * 72}")
    print(f"  Run {idx}/{total} ({pct:.0f}%)   dataset={dataset}   method={method}")
    print(f"{'─' * 72}")


# ── Copy latent → raw ─────────────────────────────────────────────────────────

def copy_latent_to_raw(results_dir: Path):
    """
    For every cil_replay_latent entry in all_results.json, insert an identical
    entry with method='cil_replay_raw' (skipping datasets that already have one).
    """
    import json
    from datetime import datetime

    log_path = results_dir / "all_results.json"
    if not log_path.exists():
        print("No all_results.json found — nothing to copy.")
        return

    with open(log_path, "r", encoding="utf-8") as fh:
        runs = json.load(fh)

    existing_raw = {r["dataset"] for r in runs if r["method"] == "cil_replay_raw"}
    latent_runs  = [r for r in runs if r["method"] == "cil_replay_latent"]

    if not latent_runs:
        print("No cil_replay_latent results found to copy.")
        return

    added = 0
    for run in latent_runs:
        ds = run["dataset"]
        if ds in existing_raw:
            print(f"  Skipping {ds} — cil_replay_raw already exists.")
            continue
        copy = {**run}
        copy["method"]    = "cil_replay_raw"
        copy["run_id"]    = f"{ds}_cil_replay_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        copy["timestamp"] = datetime.now().isoformat()
        if "config" in copy:
            copy["config"] = {**copy["config"], "method": "cil_replay_raw"}
        runs.append(copy)
        existing_raw.add(ds)
        added += 1
        print(f"  Copied {ds}: cil_replay_latent → cil_replay_raw")

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2)

    print(f"\nDone — {added} entr{'y' if added == 1 else 'ies'} added to {log_path}")


# ── Existing-results checker ──────────────────────────────────────────────────

def _already_logged(dataset: str, method: str) -> bool:
    """Return True if a run for this (dataset, method) already exists in JSON."""
    import json
    log_path = RESULTS_DIR / "all_results.json"
    if not log_path.exists():
        return False
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            runs = json.load(fh)
        return any(r["dataset"] == dataset and r["method"] == method for r in runs)
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train every (dataset × method) combo and log results to JSON."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="DS",
        help=f"Datasets to include (default: all). Choices: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        metavar="M",
        help=f"Methods to include (default: all). Choices: {ALL_METHODS}",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (dataset, method) pairs that already have a result logged.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs without training anything.",
    )
    parser.add_argument(
        "--copy-latent-to-raw", action="store_true",
        help="Copy all cil_replay_latent results to cil_replay_raw in the JSON, then exit.",
    )
    args = parser.parse_args()

    if args.copy_latent_to_raw:
        copy_latent_to_raw(RESULTS_DIR)
        return

    datasets = args.datasets if args.datasets else ALL_DATASETS
    methods  = args.methods  if args.methods  else ALL_METHODS

    # Validate
    bad_ds = [d for d in datasets if d not in ALL_DATASETS]
    bad_mt = [m for m in methods  if m not in ALL_METHODS]
    if bad_ds:
        print(f"ERROR: Unknown datasets: {bad_ds}. Available: {ALL_DATASETS}")
        sys.exit(1)
    if bad_mt:
        print(f"ERROR: Unknown methods: {bad_mt}. Available: {ALL_METHODS}")
        sys.exit(1)

    # Build run list
    runs = [(ds, mt) for ds in datasets for mt in methods]
    total = len(runs)

    # Resolve device now (deferred so --dry-run doesn't need torch)
    global DEVICE
    if not args.dry_run:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _print_banner(f"ShipTSCIL — Full Training Sweep  ({total} runs planned)")
    print(f"  Datasets : {datasets}")
    print(f"  Methods  : {methods}")
    print(f"  Seed     : {args.seed}")
    print(f"  Device   : {'(dry-run)' if args.dry_run else DEVICE}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run  : {args.dry_run}")
    print()

    if args.dry_run:
        print("Planned runs:")
        for i, (ds, mt) in enumerate(runs, 1):
            exists = "  [SKIP — already logged]" if (args.skip_existing and _already_logged(ds, mt)) else ""
            print(f"  {i:3d}. {ds:28s}  {mt}{exists}")
        print(f"\nTotal: {total} runs")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    statuses: list[tuple[str, str, str, float]] = []   # (dataset, method, status, elapsed)
    wall_start = time.time()

    for idx, (dataset, method) in enumerate(runs, 1):
        if args.skip_existing and _already_logged(dataset, method):
            _print_run_header(idx, total, dataset, method)
            print("  → Skipping (result already in all_results.json)")
            statuses.append((dataset, method, "SKIPPED", 0.0))
            continue

        _print_run_header(idx, total, dataset, method)
        t0 = time.time()
        try:
            config = _build_config(dataset, method, args.seed)
            run_one(config)
            elapsed = time.time() - t0
            statuses.append((dataset, method, "OK", elapsed))
            print(f"\n  ✓ Done in {_hms(elapsed)}")
        except Exception:
            elapsed = time.time() - t0
            statuses.append((dataset, method, "FAILED", elapsed))
            print(f"\n  ✗ FAILED after {_hms(elapsed)}:")
            traceback.print_exc()
            print("  (continuing with next run …)\n")

        # Rough ETA
        done    = idx
        ok_done = sum(1 for _, _, s, _ in statuses if s != "SKIPPED")
        if ok_done > 0:
            avg_t  = sum(e for _, _, s, e in statuses if s != "SKIPPED") / ok_done
            remain = total - done
            eta    = avg_t * remain
            print(f"  Progress: {done}/{total} | avg per run: {_hms(avg_t)} | ETA: {_hms(eta)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    wall_total = time.time() - wall_start
    _print_banner("Sweep complete")
    print(f"  Total wall time : {_hms(wall_total)}")
    print(f"  Runs attempted  : {sum(1 for *_, s, _ in statuses if s != 'SKIPPED')}")
    print(f"  Successful      : {sum(1 for *_, s, _ in statuses if s == 'OK')}")
    print(f"  Failed          : {sum(1 for *_, s, _ in statuses if s == 'FAILED')}")
    print(f"  Skipped         : {sum(1 for *_, s, _ in statuses if s == 'SKIPPED')}")

    col_w = max(len(ds) for ds, *_ in statuses)
    print(f"\n{'Dataset':<{col_w}}  {'Method':<22}  {'Status':<8}  {'Time':>8}")
    print(f"{'─'*col_w}  {'─'*22}  {'─'*8}  {'─'*8}")
    for ds, mt, status, elapsed in statuses:
        t_str = _hms(elapsed) if status != "SKIPPED" else "—"
        icon  = "✓" if status == "OK" else ("↷" if status == "SKIPPED" else "✗")
        print(f"{ds:<{col_w}}  {mt:<22}  {icon} {status:<6}  {t_str:>8}")

    failed = [(ds, mt) for ds, mt, s, _ in statuses if s == "FAILED"]
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for ds, mt in failed:
            print(f"  python run_all.py --datasets {ds} --methods {mt}")

    print(f"\nResults saved to: {RESULTS_DIR / 'all_results.json'}")
    print("Generate report : python generate_report.py --dataset <dataset>")


if __name__ == "__main__":
    main()
