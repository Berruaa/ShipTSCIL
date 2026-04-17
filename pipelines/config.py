"""Resolve ``"auto"`` hyper-parameters based on dataset statistics.

Three regimes are maintained:

* **Frozen encoder** (default) — only the linear head is trained.
  Cheap forward passes allow large batches and somewhat longer training.
* **LoRA** — encoder adapters are trainable so batches must fit encoder
  activations + gradients in VRAM. Tiny tasks can overfit quickly.
* **O-LoRA** — per-task orthogonal adapters; no replay needed. It still
  trains encoder-side parameters, so tiny-task overfitting remains a risk.

Design principles
-----------------
* **Batch size** targets a fixed number of batches per epoch, clamped to
  a hardware-aware range. Frozen encoders can safely use larger batches.
* **Epochs** are controlled by two constraints:
  1. a target total gradient-step budget, and
  2. a size-aware epoch ceiling.
  The ceiling is intentionally much stricter for trainable-encoder
  regimes on tiny tasks (e.g. ECG5000) to avoid memorisation.
* **Replay buffer** is sized per-class rather than as a flat percentage,
  ensuring that even small datasets retain enough exemplars per class
  for meaningful replay.
"""

from methods import SEQUENTIAL_METHODS

_AUTO = "auto"
_UWAVE_DATASET = "uwave_gesture_library"
_UWAVE_EPOCH_OVERRIDE = 30

# =====================================================================
#  Regime constants
# =====================================================================

# ---- Frozen-encoder ----
_FROZEN_LR = 5e-4
_FROZEN_TARGET_STEPS = 250             # total gradient-step budget per task
_FROZEN_TARGET_BATCHES = 10            # desired batches per epoch
_FROZEN_EPOCH_RANGE = (5, 100)
_FROZEN_BS_RANGE = (8, 128)            # frozen = no encoder gradients → big batches ok
_FROZEN_REPLAY_PER_CLASS = 30          # baseline exemplars per class
_FROZEN_REPLAY_MIN_PER_CLASS = 10
_FROZEN_REPLAY_CAP = 5_000
_FROZEN_REPLAY_BS_RATIO = 0.5
_FROZEN_EPOCH_CAPS = (
    (250, 18),
    (500, 20),
    (1_000, 18),
    (2_500, 12),
    (float("inf"), 8),
)

# ---- LoRA ----
# The classifier head is still trained from scratch, so keep its default LR
# close to the frozen-encoder regime even when the encoder uses adapters.
_LORA_HEAD_LR = 5e-4
_LORA_TARGET_STEPS = 200
_LORA_TARGET_BATCHES = 10
_LORA_EPOCH_RANGE = (4, 60)
_LORA_BS_RANGE = (16, 64)             # VRAM constrained
_LORA_REPLAY_PER_CLASS = 50           # bigger buffer — encoder also forgets
_LORA_REPLAY_MIN_PER_CLASS = 15
_LORA_REPLAY_CAP = 8_000
_LORA_REPLAY_BS_RATIO = 0.75
_LORA_LR_RATIO_DEFAULT = 0.2
_LORA_LR_RATIO_SMALL = 0.1
_LORA_SMALL_DATASET_THRESHOLD = 200
_LORA_EPOCH_CAPS = (
    (250, 6),
    (500, 8),
    (1_000, 10),
    (2_500, 8),
    (float("inf"), 5),
)

# ---- O-LoRA ----
# O-LoRA also uses a freshly trained classifier head; too-small head learning
# rates on tiny tasks often stall at chance while the loss drifts downward.
_OLORA_HEAD_LR = 5e-4
_OLORA_TARGET_STEPS = 250
_OLORA_TARGET_BATCHES = 10
_OLORA_EPOCH_RANGE = (4, 60)
_OLORA_BS_RANGE = (16, 64)
_OLORA_LR_RATIO_DEFAULT = 0.2
_OLORA_LR_RATIO_SMALL = 0.1
_OLORA_SMALL_DATASET_THRESHOLD = 200
_OLORA_EPOCH_CAPS = (
    (250, 8),
    (500, 10),
    (1_000, 12),
    (2_500, 9),
    (float("inf"), 6),
)


# =====================================================================
#  Helpers
# =====================================================================

def _epoch_ceiling_from_size(effective_size, caps):
    """Return a regime-specific max epoch count from task size.

    ``caps`` is an ordered sequence of ``(max_effective_size, max_epochs)``.
    This keeps tiny trainable-encoder tasks from receiving dozens of epochs
    just because each epoch has only a few batches.
    """
    for max_size, max_epochs in caps:
        if effective_size <= max_size:
            return max_epochs
    return caps[-1][1]


def _replay_buffer_size(num_classes, samples_per_class, *,
                        target_per_class, min_per_class, cap):
    """Class-aware replay buffer sizing.

    Each class gets *target_per_class* exemplars, but no more than the
    number of training samples available for that class.  A sqrt taper
    keeps large-class datasets from blowing up the buffer.
    """
    import math

    tapered = int(target_per_class * math.sqrt(
        min(samples_per_class, target_per_class * 4) / (target_per_class * 4)
    ) * 2)
    per_class = max(min_per_class, min(target_per_class, tapered))
    per_class = min(per_class, int(samples_per_class))
    total = num_classes * per_class
    return max(num_classes * min_per_class, min(cap, total))


# =====================================================================
#  Main entry point
# =====================================================================

def auto_configure(config, n_train, num_classes):
    """Fill in ``"auto"`` values in *config* in-place.

    Explicit (non-``"auto"``) values are never overwritten.
    """
    config.setdefault("use_early_stopping", True)
    config.setdefault("validation_split", 0.1)
    config.setdefault("early_stopping_min_delta", 1e-4)
    config.setdefault("early_stopping_patience", _AUTO)

    use_lora = config.get("use_lora", False)
    use_olora = config.get("use_olora", False)
    is_sequential = config["method"] in SEQUENTIAL_METHODS
    classes_per_task = config.get("classes_per_task", 2)
    samples_per_class = n_train / max(1, num_classes)

    if is_sequential:
        effective_size = int(samples_per_class * classes_per_task)
    else:
        effective_size = n_train

    # ── Pick regime constants ─────────────────────────────────────
    if use_olora:
        default_lr      = _OLORA_HEAD_LR
        target_steps     = _OLORA_TARGET_STEPS
        target_batches   = _OLORA_TARGET_BATCHES
        epoch_lo, epoch_hi = _OLORA_EPOCH_RANGE
        epoch_caps       = _OLORA_EPOCH_CAPS
        bs_lo, bs_hi     = _OLORA_BS_RANGE
        rp_per_class     = 0
        rp_min_per_class = 0
        rp_cap           = 0
        rp_bs_ratio      = _FROZEN_REPLAY_BS_RATIO
    elif use_lora:
        default_lr      = _LORA_HEAD_LR
        target_steps     = _LORA_TARGET_STEPS
        target_batches   = _LORA_TARGET_BATCHES
        epoch_lo, epoch_hi = _LORA_EPOCH_RANGE
        epoch_caps       = _LORA_EPOCH_CAPS
        bs_lo, bs_hi     = _LORA_BS_RANGE
        rp_per_class     = _LORA_REPLAY_PER_CLASS
        rp_min_per_class = _LORA_REPLAY_MIN_PER_CLASS
        rp_cap           = _LORA_REPLAY_CAP
        rp_bs_ratio      = _LORA_REPLAY_BS_RATIO
    else:
        default_lr      = _FROZEN_LR
        target_steps     = _FROZEN_TARGET_STEPS
        target_batches   = _FROZEN_TARGET_BATCHES
        epoch_lo, epoch_hi = _FROZEN_EPOCH_RANGE
        epoch_caps       = _FROZEN_EPOCH_CAPS
        bs_lo, bs_hi     = _FROZEN_BS_RANGE
        rp_per_class     = _FROZEN_REPLAY_PER_CLASS
        rp_min_per_class = _FROZEN_REPLAY_MIN_PER_CLASS
        rp_cap           = _FROZEN_REPLAY_CAP
        rp_bs_ratio      = _FROZEN_REPLAY_BS_RATIO

    resolved = {}

    # ── batch_size ────────────────────────────────────────────────
    # Target a fixed number of batches per epoch so training isn't
    # dominated by optimizer overhead on tiny batches or starved of
    # gradient updates with one huge batch.
    if config.get("batch_size") == _AUTO:
        import math

        ideal = math.ceil(effective_size / max(1, target_batches))
        bs = max(bs_lo, min(bs_hi, ideal))
        # Guarantee at least 4 batches per epoch so each epoch is
        # meaningful. On tiny tasks we relax the normal regime floor;
        # otherwise LoRA/O-LoRA can end up with only 1-2 updates/epoch.
        if effective_size > 0:
            max_bs_for_four_batches = max(1, math.ceil(effective_size / 4))
            if max_bs_for_four_batches < bs_lo:
                relaxed_floor = 4 if (use_lora or use_olora) else bs_lo
                bs = min(bs, max(relaxed_floor, max_bs_for_four_batches))
            else:
                bs = min(bs, max_bs_for_four_batches)
        config["batch_size"] = bs
        resolved["batch_size"] = bs

    bs = config["batch_size"]
    import math

    batches_per_epoch = max(1, math.ceil(effective_size / max(1, bs)))

    # ── epochs ────────────────────────────────────────────────────
    # Two constraints: (1) target gradient-step budget, and (2) a
    # regime-specific size ceiling. The stricter of the two wins.
    if config.get("epochs") == _AUTO:
        step_based = max(1, target_steps // batches_per_epoch)
        pass_ceiling = _epoch_ceiling_from_size(effective_size, epoch_caps)
        epochs = max(epoch_lo, min(epoch_hi, step_based, pass_ceiling))
        config["epochs"] = epochs
        resolved["epochs"] = epochs
        if step_based > pass_ceiling:
            resolved["epochs_note"] = (
                f"capped by task-size ceiling: {step_based} → {pass_ceiling} "
                f"(~{effective_size} samples/task)"
            )

    if config.get("early_stopping_patience") == _AUTO:
        patience = max(2, min(8, config["epochs"] // 3))
        config["early_stopping_patience"] = patience
        resolved["early_stopping_patience"] = patience

    # ── Dataset-specific override: UWave Gesture Library ───────────
    # This dataset often stays near chance in the first few epochs.
    # Give it a longer fixed horizon and disable early stopping so it can recover.
    if config.get("dataset") == _UWAVE_DATASET:
        if config.get("use_early_stopping", True):
            config["use_early_stopping"] = False
            resolved["use_early_stopping"] = False

        if isinstance(config.get("epochs"), int) and config["epochs"] < _UWAVE_EPOCH_OVERRIDE:
            config["epochs"] = _UWAVE_EPOCH_OVERRIDE
            resolved["epochs"] = _UWAVE_EPOCH_OVERRIDE

    # ── lr (head) ─────────────────────────────────────────────────
    if config.get("lr") == _AUTO:
        config["lr"] = default_lr
        resolved["lr"] = config["lr"]

    # ── replay_buffer_size (class-aware) ──────────────────────────
    if config.get("replay_buffer_size") == _AUTO:
        if rp_per_class > 0:
            buf = _replay_buffer_size(
                num_classes, samples_per_class,
                target_per_class=rp_per_class,
                min_per_class=rp_min_per_class,
                cap=rp_cap,
            )
        else:
            buf = 10
        config["replay_buffer_size"] = buf
        resolved["replay_buffer_size"] = buf
        per_cls = buf // max(1, num_classes)
        resolved["replay_per_class"] = f"~{per_cls}"

    # ── replay_batch_size ─────────────────────────────────────────
    # Scale with the main batch so replay doesn't overwhelm or
    # get drowned out by current-task data.
    if config.get("replay_batch_size") == _AUTO:
        rbs = max(8, int(config["batch_size"] * rp_bs_ratio))
        config["replay_batch_size"] = rbs
        resolved["replay_batch_size"] = rbs

    # ── lora_lr (shared by LoRA and O-LoRA) ───────────────────────
    if (use_lora or use_olora) and config.get("lora_lr") == _AUTO:
        if use_olora:
            threshold = _OLORA_SMALL_DATASET_THRESHOLD
            ratio_default = _OLORA_LR_RATIO_DEFAULT
            ratio_small = _OLORA_LR_RATIO_SMALL
        else:
            threshold = _LORA_SMALL_DATASET_THRESHOLD
            ratio_default = _LORA_LR_RATIO_DEFAULT
            ratio_small = _LORA_LR_RATIO_SMALL

        if effective_size < threshold:
            ratio = ratio_small
        else:
            ratio = ratio_default
        config["lora_lr"] = config["lr"] * ratio
        resolved["lora_lr"] = config["lora_lr"]
        if effective_size < threshold:
            resolved["lora_lr_note"] = (
                f"reduced (small task: ~{effective_size} samples < "
                f"{threshold} threshold)"
            )

    # ── Summary ───────────────────────────────────────────────────
    if resolved:
        regime = "O-LoRA" if use_olora else ("LoRA" if use_lora else "frozen-encoder")
        parts = [f"{k}={v}" for k, v in resolved.items()]
        print(f"Auto-configured ({regime}): {', '.join(parts)}")
        total_steps = config["epochs"] * batches_per_epoch
        if is_sequential:
            print(f"  {n_train} train samples, {num_classes} classes, "
                  f"~{effective_size} samples/task, "
                  f"~{batches_per_epoch} batches/epoch, "
                  f"~{total_steps} total grad steps/task")
        else:
            print(f"  {n_train} train samples, {num_classes} classes, "
                  f"~{batches_per_epoch} batches/epoch, "
                  f"~{total_steps} total grad steps")
