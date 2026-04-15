from datetime import datetime
from pathlib import Path

import torch

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
from utils.reporting import print_final_sequential_results, print_final_standard_results, print_run_info
from utils.plotting import save_standard_plots, save_sequential_plots


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "checkpoints"
CACHE_DIR = PROJECT_ROOT / "embeddings_cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# ======================================================================
#  Run configuration
# ======================================================================
CONFIG = {
    # ── What to run ───────────────────────────────────────────────
    # Methods: svm, linear_probe,
    # cil_naive,
    # cil_replay_raw, cil_replay_latent,
    # cil_lwf (Distillation),
    # cil_ncm  (FastICARL — NCM (default); Herding+NCM when herding_replay=True)
    "method":     "cil_replay_raw",
    "dataset":    "ethanol_level",
    "model_name": "AutonLab/MOMENT-1-base",

    # Set to file paths to override the dataset registry lookup.
    "train_file": None,
    "test_file":  None,

    # ── Training ("auto" = derived from dataset size) ─────────────
    "batch_size": "auto",
    "epochs":     50,
    "lr":         "auto",
    "seed":       42,
    "num_workers": 0,

    # ── Class-incremental setup ───────────────────────────────────
    "task_order":          None,   # auto-generated if None
    "num_tasks":           None,   # optional override for number of tasks
    "shuffle_class_order": True,
    "classes_per_task":    2,

    # ── Replay buffer (replay methods only) ───────────────────────
    "replay_buffer_pct":  0.05,     # buffer = this fraction of training set
    "replay_buffer_size": "auto",  # override: set an int to bypass pct
    "replay_batch_size":  "auto",  # override: set an int to bypass auto
    "balanced_replay":    True,    # class-balanced vs reservoir
    "herding_replay":     True,   # iCaRL herding exemplar selection (overrides balanced_replay)

    # ── Loss ─────────────────────────────────────────────────────
    "balanced_loss":       True,   # class-weighted CE vs standard CE

    # ── Distillation (cil_lwf / cil_replay_latent only) ──────────
    "use_distillation":    False,
    "distill_temperature": 2.0,
    "distill_weight":      1.0,

    # ── LoRA (parameter-efficient fine-tuning of backbone) ─────────
    # When enabled the MOMENT encoder receives low-rank adapters on
    # selected attention projections.  All original weights stay frozen.
    # Compatible methods: linear_probe, cil_naive, cil_replay_raw, cil_lwf.
    #
    # When use_lora is True, "auto" values above are adjusted:
    #   lr:       2e-4 (vs 5e-4 frozen) — more conservative head LR
    #   lora_lr:  auto = 0.2× head LR; 0.1× on small tasks (<200 samples)
    #   epochs:   ceiling 60 (vs 100), target 150 grad-steps (vs 200)
    #   batch_size: floor 16 (vs 8) — stabilises encoder gradients
    #   replay_buffer_size: 1.5× larger (cap 8000 vs 5000)
    #   replay_batch_size:  75% of batch (vs 50%)
    "use_lora":            False,
    "lora_rank":           8,
    "lora_alpha":          16,
    "lora_target_modules": None,   # default: ["q", "v"] (T5 attention)
    "lora_dropout":        0.05,
    "lora_lr":             "auto", # see above for auto logic

    # ── O-LoRA (orthogonal LoRA for continual learning) ──────────
    # Wang et al. (2023) "Orthogonal Subspace Learning for Language
    # Model Continual Learning".  Each task gets its own LoRA adapters;
    # previous adapters are frozen and an orthogonality loss prevents
    # interference.  No replay or distillation needed.
    # Set method to "cil_olora" and use_olora to True.
    # Inherits lora_rank, lora_alpha, lora_target_modules, lora_dropout.
    "use_olora":           True,
    "olora_lambda":        1.0,    # weight of orthogonality loss

    # ── Output controls ────────────────────────────────────────────
    "save_results":        False,   # save plots/figures under results/
}


# ======================================================================
#  Main
# ======================================================================

def _make_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{config['dataset']}_{config['method']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    set_seed(CONFIG["seed"])

    # ── Load data ─────────────────────────────────────────────────
    train_dataset, test_dataset, label_encoder, dataset_info = \
        build_base_datasets_and_info(config=CONFIG, data_dir=DATA_DIR)

    num_classes = len(label_encoder.classes_)
    auto_configure(CONFIG, n_train=len(train_dataset), num_classes=num_classes)

    # Methods that do no per-epoch learning only need a single pass.
    _SINGLE_EPOCH_METHODS = {"svm", "cil_ncm", "ncm", "cil_herding_ncm", "herding_ncm"}
    if CONFIG["method"] in _SINGLE_EPOCH_METHODS and CONFIG.get("epochs") != 1:
        print(f"Auto-setting epochs=1 for '{CONFIG['method']}' (no per-epoch learning).")
        CONFIG["epochs"] = 1

    # ── Task order (sequential methods only) ──────────────────────
    if CONFIG["method"] in SEQUENTIAL_METHODS:
        if CONFIG.get("task_order") is None:
            CONFIG["task_order"] = build_task_order(
                num_classes=num_classes,
                seed=CONFIG["seed"],
                shuffle_class_order=CONFIG.get("shuffle_class_order", True),
                classes_per_task=CONFIG.get("classes_per_task", 2),
                num_tasks=CONFIG.get("num_tasks"),
            )
            print(f"Auto-generated task_order: {CONFIG['task_order']}")
        validate_task_order(CONFIG["task_order"], num_classes)

    # ── LoRA / O-LoRA config ─────────────────────────────────────
    lora_config = None
    use_olora = CONFIG.get("use_olora", False)
    if use_olora:
        CONFIG["use_lora"] = True
        lora_config = {
            "enabled":        True,
            "olora":          True,
            "rank":           CONFIG.get("lora_rank", 8),
            "alpha":          CONFIG.get("lora_alpha", 16),
            "target_modules": CONFIG.get("lora_target_modules"),
            "dropout":        CONFIG.get("lora_dropout", 0.05),
            "lr":             CONFIG.get("lora_lr", CONFIG["lr"] * 0.2),
            "olora_lambda":   CONFIG.get("olora_lambda", 1.0),
        }
    elif CONFIG.get("use_lora", False):
        lora_config = {
            "enabled":        True,
            "rank":           CONFIG.get("lora_rank", 8),
            "alpha":          CONFIG.get("lora_alpha", 16),
            "target_modules": CONFIG.get("lora_target_modules"),
            "dropout":        CONFIG.get("lora_dropout", 0.05),
            "lr":             CONFIG.get("lora_lr", CONFIG["lr"] * 0.2),
        }

    # ── Build method ──────────────────────────────────────────────
    method = build_method(
        method_name=CONFIG["method"],
        model_name=CONFIG["model_name"],
        num_classes=num_classes,
        train_dataset=train_dataset,
        device=DEVICE,
        lr=CONFIG["lr"],
        replay_buffer_size=CONFIG.get("replay_buffer_size", 1000),
        replay_batch_size=CONFIG.get("replay_batch_size", 32),
        balanced_replay=CONFIG.get("balanced_replay", True),
        balanced_loss=CONFIG.get("balanced_loss", True),
        use_distillation=CONFIG.get("use_distillation", False),
        distill_temperature=CONFIG.get("distill_temperature", 2.0),
        distill_weight=CONFIG.get("distill_weight", 1.0),
        herding_replay=CONFIG.get("herding_replay", False),
        lora_config=lora_config,
    )

    print_run_info(
        config=CONFIG, dataset_info=dataset_info,
        label_encoder=label_encoder, method=method, device=DEVICE,
    )

    # ── Precompute embeddings (cached to disk) ────────────────────
    # When LoRA is active the encoder is trainable — embeddings change
    # every step, so precomputation (and caching) must be skipped entirely.
    # Without LoRA, cil_replay_raw still needs raw training series for
    # its replay buffer; all other frozen-encoder methods precompute.
    use_lora = CONFIG.get("use_lora", False)
    if use_lora:
        print("\nLoRA enabled — skipping embedding precomputation "
              "(encoder is trainable).\n")
    else:
        encoder = getattr(method, "encoder", None) or method.model.encoder
        emb_kwargs = dict(
            device=DEVICE, batch_size=CONFIG["batch_size"],
            cache_dir=CACHE_DIR, dataset_name=CONFIG["dataset"],
            model_name=CONFIG["model_name"],
        )
        print(f"\nPreparing embeddings … (device: {DEVICE})")
        if CONFIG["method"] != "cil_replay_raw":
            train_dataset = precompute_embeddings(encoder, train_dataset, split="train", **emb_kwargs)
        test_dataset = precompute_embeddings(encoder, test_dataset, split="test", **emb_kwargs)
        print("Done.\n")

    class_names = [str(c) for c in label_encoder.classes_]
    save_results = CONFIG.get("save_results", True)
    run_dir = _make_run_dir(CONFIG) if save_results else None
    if save_results:
        print(f"Results will be saved to: {run_dir}\n")
    else:
        print("Result saving disabled (CONFIG['save_results']=False).\n")

    # ── Standard methods (svm, linear_probe) ──────────────────────
    if CONFIG["method"] in STANDARD_METHODS:
        train_loader = build_loader(train_dataset, CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
        test_loader = build_loader(test_dataset, CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

        checkpoint_path, best_test_acc, history = train_standard(
            method=method, train_loader=train_loader, test_loader=test_loader,
            config=CONFIG, label_encoder=label_encoder, save_dir=SAVE_DIR,
        )

        method.load(checkpoint_path)
        y_true, y_pred = collect_predictions(method, test_loader, device=DEVICE)
        print_final_standard_results(best_test_acc, y_true, y_pred, label_encoder)
        if save_results:
            save_standard_plots(history, y_true, y_pred, class_names, run_dir, CONFIG)
        return

    # ── Sequential methods (cil_*) ────────────────────────────────
    if CONFIG["method"] in SEQUENTIAL_METHODS:
        _, final_checkpoint_path, best_seen_acc, task_results, history = \
            train_sequential(
                method=method, train_dataset=train_dataset, test_dataset=test_dataset,
                config=CONFIG, label_encoder=label_encoder, save_dir=SAVE_DIR,
            )

        all_seen = sorted(set(c for task in CONFIG["task_order"] for c in task))
        method.load(final_checkpoint_path)
        final_loader = build_loader(
            make_class_subset(test_dataset, all_seen),
            CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"],
        )

        y_true, y_pred = collect_predictions(method, final_loader, device=DEVICE)
        print_final_sequential_results(best_seen_acc, task_results, y_true, y_pred, label_encoder)
        if save_results:
            save_sequential_plots(history, task_results, y_true, y_pred, class_names, run_dir, CONFIG)
        return

    raise ValueError(f"Unsupported method: {CONFIG['method']}")


if __name__ == "__main__":
    main()
