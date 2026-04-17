"""
run_loras.py
============
Train every (dataset × method) combination **with LoRA / O-LoRA enabled**
and record results to ``results/lora/all_results.json``.

This script is the LoRA-family counterpart to ``run_all.py``.  Results
are written to a *separate* results directory so the frozen-encoder
comparisons in the main report are not affected.

Why these specific methods?
---------------------------
LoRA adapters learn *only if the encoder receives a gradient signal*.
The combinations below are the ones where that signal actually exists
(CE + optional replay / distillation on a trainable linear head):

* ``linear_probe``        — standard LoRA fine-tuning baseline
* ``cil_naive``           — CIL baseline: just fine-tune with LoRA, no anti-forgetting
* ``cil_replay_raw``      — CIL + raw replay (herding + balanced loss + balanced replay)
* ``cil_replay_raw_lwf``  — CIL + raw replay + LwF distillation (the combo variant)
* ``cil_lwf``             — CIL + LwF knowledge distillation
* ``cil_olora``           — O-LoRA with CE loss + orthogonality constraint

NCM / Herding-NCM are *excluded* on purpose: they perform **no gradient
updates**, so LoRA adapters would stay at their identity init and the
run would be numerically identical to plain NCM.  Latent-replay variants
are also excluded — they pre-cache embeddings from a frozen encoder,
which is incompatible with a trainable LoRA encoder.

Usage
-----
    python run_loras.py                                   # every dataset × every method
    python run_loras.py --datasets ecg5000 uci_har        # subset of datasets
    python run_loras.py --methods cil_lwf cil_olora       # subset of methods
    python run_loras.py --olora-only                      # only run cil_olora
    python run_loras.py --lora-only                       # only run the LoRA methods
    python run_loras.py --dry-run                         # preview without training
    python run_loras.py --skip-existing                   # skip already-logged runs
    python run_loras.py --seed 0                          # override random seed
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
SAVE_DIR     = PROJECT_ROOT / "checkpoints" / "lora"
CACHE_DIR    = PROJECT_ROOT / "embeddings_cache"   # unused by LoRA runs, kept for signature
RESULTS_DIR  = PROJECT_ROOT / "results" / "lora"

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

# LoRA fine-tuning methods (use_lora=True).
LORA_METHODS = [
    "linear_probe",
    "cil_naive",
    "cil_replay_raw",
    "cil_replay_raw_lwf",
    "cil_lwf",
]

# O-LoRA native method (use_olora=True).
OLORA_METHODS = [
    "cil_olora",
]

ALL_METHODS = LORA_METHODS + OLORA_METHODS

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
    "herding_replay":      True,
    # Loss
    "balanced_loss":       True,
    # Distillation
    "distill_temperature": 2.0,
    "distill_weight":      1.0,
    # LoRA — enabled by default for this sweep; can be overridden per method
    "use_lora":            True,
    "lora_rank":           8,
    "lora_alpha":          16,
    "lora_target_modules": None,
    "lora_dropout":        0.05,
    "lora_lr":             "auto",
    # O-LoRA — enabled only for cil_olora via METHOD_OVERRIDES below
    "use_olora":           False,
    "olora_lambda":        1.0,
    # Output
    "save_results":        True,
}

# Per-method config overrides applied on top of BASE_CONFIG.
METHOD_OVERRIDES = {
    "linear_probe": {
        # Standard fine-tuning baseline — no replay/distill config needed.
        "use_lora":       True,
        "use_olora":      False,
    },
    "cil_naive": {
        "use_lora":       True,
        "use_olora":      False,
        "balanced_replay": False,
        "herding_replay":  False,
        "balanced_loss":   False,
    },
    "cil_replay_raw": {
        "use_lora":       True,
        "use_olora":      False,
        "balanced_replay": True,
        "herding_replay":  True,
        "balanced_loss":   True,
    },
    "cil_replay_raw_lwf": {
        "use_lora":       True,
        "use_olora":      False,
        "balanced_replay": True,
        "herding_replay":  True,
        "balanced_loss":   True,
    },
    "cil_lwf": {
        "use_lora":       True,
        "use_olora":      False,
        "balanced_replay": True,
        "herding_replay":  False,
        "balanced_loss":   True,
    },
    "cil_olora": {
        # O-LoRA owns the LoRA adapters itself; use_olora flips it on.
        "use_lora":       True,
        "use_olora":      True,
        "balanced_replay": False,
        "herding_replay":  False,
        "balanced_loss":   True,
    },
}


def _build_config(dataset: str, method: str, seed: int) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"] = dataset
    cfg["method"]  = method
    cfg["seed"]    = seed
    cfg.update(METHOD_OVERRIDES.get(method, {}))
    return cfg


def _build_lora_config(cfg: dict) -> dict | None:
    """Assemble the ``lora_config`` dict passed to ``build_method``."""
    if cfg.get("use_olora", False):
        return {
            "enabled":        True,
            "olora":          True,
            "rank":           cfg.get("lora_rank", 8),
            "alpha":          cfg.get("lora_alpha", 16),
            "target_modules": cfg.get("lora_target_modules"),
            "dropout":        cfg.get("lora_dropout", 0.05),
            "lr":             cfg.get("lora_lr", cfg["lr"] * 0.2 if isinstance(cfg.get("lr"), (int, float)) else cfg.get("lr")),
            "olora_lambda":   cfg.get("olora_lambda", 1.0),
        }
    if cfg.get("use_lora", False):
        return {
            "enabled":        True,
            "rank":           cfg.get("lora_rank", 8),
            "alpha":          cfg.get("lora_alpha", 16),
            "target_modules": cfg.get("lora_target_modules"),
            "dropout":        cfg.get("lora_dropout", 0.05),
            "lr":             cfg.get("lora_lr", cfg["lr"] * 0.2 if isinstance(cfg.get("lr"), (int, float)) else cfg.get("lr")),
        }
    return None


# ── Core training logic (LoRA-aware; mirrors train.py main()) ────────────────

def run_one(config: dict) -> dict:
    """
    Run a single (dataset, method) LoRA experiment.
    Returns the result dict that was saved to lora/all_results.json.
    """
    from methods import build_method, STANDARD_METHODS, SEQUENTIAL_METHODS
    from pipelines import (
        auto_configure,
        build_base_datasets_and_info,
        build_loader,
        build_task_order,
        collect_predictions,
        make_class_subset,
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

    # ── LoRA / O-LoRA config ──────────────────────────────────────────────────
    lora_config = _build_lora_config(config)

    # ── Build method ──────────────────────────────────────────────────────────
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
        distill_temperature=config.get("distill_temperature", 2.0),
        distill_weight=config.get("distill_weight", 1.0),
        herding_replay=config.get("herding_replay", False),
        lora_config=lora_config,
    )

    print_run_info(
        config=config, dataset_info=dataset_info,
        label_encoder=label_encoder, method=method, device=DEVICE,
    )

    # ── Embeddings: SKIPPED for LoRA (encoder is trainable) ───────────────────
    print("\nLoRA enabled — skipping embedding precomputation "
          "(encoder is trainable).\n")

    class_names = [str(c) for c in label_encoder.classes_]

    # ── Per-run plot directory ─────────────────────────────────────────────────
    save_plots = config.get("save_results", True)
    if save_plots:
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        tag = "olora" if config.get("use_olora") else "lora"
        run_dir = RESULTS_DIR / f"{config['dataset']}_{config['method']}_{tag}_{timestamp}"
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


def _print_run_header(idx: int, total: int, dataset: str, method: str, tag: str):
    pct = 100 * idx / total
    print(f"\n{'─' * 72}")
    print(f"  Run {idx}/{total} ({pct:.0f}%)   dataset={dataset}   method={method}   [{tag}]")
    print(f"{'─' * 72}")


# ── Existing-results checker ──────────────────────────────────────────────────

def _already_logged(dataset: str, method: str) -> bool:
    """Return True if a LoRA run for this (dataset, method) already exists."""
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
        description="Train every (dataset × method) combo with LoRA/O-LoRA "
                    "and log results to results/lora/all_results.json.",
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
        "--lora-only", action="store_true",
        help="Run only the plain-LoRA methods (skip cil_olora).",
    )
    parser.add_argument(
        "--olora-only", action="store_true",
        help="Run only cil_olora.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (dataset, method) pairs that already have a LoRA result logged.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs without training anything.",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_DATASETS

    if args.lora_only and args.olora_only:
        print("ERROR: --lora-only and --olora-only are mutually exclusive.")
        sys.exit(1)

    if args.methods:
        methods = args.methods
    elif args.lora_only:
        methods = list(LORA_METHODS)
    elif args.olora_only:
        methods = list(OLORA_METHODS)
    else:
        methods = list(ALL_METHODS)

    # Validate
    bad_ds = [d for d in datasets if d not in ALL_DATASETS]
    bad_mt = [m for m in methods  if m not in ALL_METHODS]
    if bad_ds:
        print(f"ERROR: Unknown datasets: {bad_ds}. Available: {ALL_DATASETS}")
        sys.exit(1)
    if bad_mt:
        print(f"ERROR: Unknown methods: {bad_mt}. Available: {ALL_METHODS}")
        sys.exit(1)

    runs = [(ds, mt) for ds in datasets for mt in methods]
    total = len(runs)

    # Resolve device now (deferred so --dry-run doesn't need torch)
    global DEVICE
    if not args.dry_run:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _print_banner(f"ShipTSCIL — LoRA / O-LoRA Sweep  ({total} runs planned)")
    print(f"  Datasets     : {datasets}")
    print(f"  Methods      : {methods}")
    print(f"  Seed         : {args.seed}")
    print(f"  Device       : {'(dry-run)' if args.dry_run else DEVICE}")
    print(f"  Results dir  : {RESULTS_DIR}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    if args.dry_run:
        print("Planned runs:")
        for i, (ds, mt) in enumerate(runs, 1):
            tag = "O-LoRA" if mt in OLORA_METHODS else "LoRA"
            exists = "  [SKIP — already logged]" if (args.skip_existing and _already_logged(ds, mt)) else ""
            print(f"  {i:3d}. {ds:28s}  {mt:<22s}  [{tag}]{exists}")
        print(f"\nTotal: {total} runs")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    statuses: list[tuple[str, str, str, float]] = []
    wall_start = time.time()

    for idx, (dataset, method) in enumerate(runs, 1):
        tag = "O-LoRA" if method in OLORA_METHODS else "LoRA"

        if args.skip_existing and _already_logged(dataset, method):
            _print_run_header(idx, total, dataset, method, tag)
            print("  → Skipping (result already in lora/all_results.json)")
            statuses.append((dataset, method, "SKIPPED", 0.0))
            continue

        _print_run_header(idx, total, dataset, method, tag)
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

        done    = idx
        ok_done = sum(1 for _, _, s, _ in statuses if s != "SKIPPED")
        if ok_done > 0:
            avg_t  = sum(e for _, _, s, e in statuses if s != "SKIPPED") / ok_done
            remain = total - done
            eta    = avg_t * remain
            print(f"  Progress: {done}/{total} | avg per run: {_hms(avg_t)} | ETA: {_hms(eta)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    wall_total = time.time() - wall_start
    _print_banner("LoRA sweep complete")
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
            print(f"  python run_loras.py --datasets {ds} --methods {mt}")

    print(f"\nResults saved to: {RESULTS_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
