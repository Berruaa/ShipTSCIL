"""Resolve ``"auto"`` hyper-parameters based on dataset statistics.

Two regimes are maintained:

* **Frozen encoder** (default) — only the linear head is trained.
  Conservative defaults (lr=5e-4, up to 200 grad-steps, small replay).
* **LoRA** — the encoder receives low-rank adapters that are trained
  alongside the head.  The encoder can both overfit small CIL tasks
  and underfit if training is too short, so all heuristics are adjusted:
  lower head LR, adaptive LoRA LR, larger replay, and a tighter epoch
  ceiling.
"""

from methods import SEQUENTIAL_METHODS

_AUTO = "auto"

# ---------- Frozen-encoder defaults ----------
_FROZEN_LR = 5e-4
_FROZEN_MIN_GRAD_STEPS = 200
_FROZEN_EPOCH_RANGE = (10, 100)
_FROZEN_BS_RANGE = (8, 64)
_FROZEN_REPLAY_PCT_DEFAULT = 0.2
_FROZEN_REPLAY_CAP = 5_000
_FROZEN_REPLAY_BS_RATIO = 0.5          # replay_batch = main_batch × ratio

# ---------- LoRA-specific defaults -----------
_LORA_HEAD_LR = 2e-4
_LORA_MIN_GRAD_STEPS = 150
_LORA_EPOCH_RANGE = (10, 60)
_LORA_BS_RANGE = (16, 64)
_LORA_REPLAY_PCT_BOOST = 1.5           # multiply user's pct by this
_LORA_REPLAY_CAP = 8_000
_LORA_REPLAY_BS_RATIO = 0.75
_LORA_LR_RATIO_DEFAULT = 0.2           # lora_lr = head_lr × ratio
_LORA_LR_RATIO_SMALL = 0.1            # used when effective_size < threshold
_LORA_SMALL_DATASET_THRESHOLD = 200    # samples-per-task below this = "small"


def auto_configure(config, n_train, num_classes):
    """Fill in ``"auto"`` values in *config* in-place.

    Heuristics (frozen encoder)
    ---------------------------
    * **batch_size** – sized so each epoch has several batches, even for
      the smallest CIL task.  Clamped to [8, 64].
    * **epochs** – targets ~200 gradient steps per task so the linear
      head converges, clamped to [10, 100].
    * **lr** – 5e-4 (safe default for Adam on a linear head).
    * **replay_buffer_size** – scales with the training set; capped at 5 000.
    * **replay_batch_size** – half of the main batch size.

    LoRA adjustments
    ----------------
    When ``use_lora`` is True the following changes apply automatically:

    * **lr** – lowered to 2e-4 so the head doesn't outpace the adapters.
    * **lora_lr** – 0.2× head LR by default, reduced to 0.1× on small
      tasks (< 200 samples/task) to limit overfitting.
    * **epochs** – ceiling reduced from 100 → 60; target gradient steps
      lowered from 200 → 150.
    * **batch_size** – floor raised from 8 → 16 to stabilise gradients
      flowing through the encoder.
    * **replay_buffer_size** – 1.5× larger (capped at 8 000).  With a
      trainable encoder the replay buffer is the primary defence against
      forgetting in the representation space, not just the head.
    * **replay_batch_size** – 75 % of main batch (was 50 %) to strengthen
      the old-class signal each step.

    Explicit (non-``"auto"``) values are never overwritten.
    """
    use_lora = config.get("use_lora", False)
    is_sequential = config["method"] in SEQUENTIAL_METHODS
    classes_per_task = config.get("classes_per_task", 2)

    if is_sequential:
        samples_per_class = n_train / max(1, num_classes)
        effective_size = int(samples_per_class * classes_per_task)
    else:
        effective_size = n_train

    # Pick regime-specific constants.
    if use_lora:
        default_lr = _LORA_HEAD_LR
        min_grad_steps = _LORA_MIN_GRAD_STEPS
        epoch_lo, epoch_hi = _LORA_EPOCH_RANGE
        bs_lo, bs_hi = _LORA_BS_RANGE
        replay_pct_mult = _LORA_REPLAY_PCT_BOOST
        replay_cap = _LORA_REPLAY_CAP
        replay_bs_ratio = _LORA_REPLAY_BS_RATIO
    else:
        default_lr = _FROZEN_LR
        min_grad_steps = _FROZEN_MIN_GRAD_STEPS
        epoch_lo, epoch_hi = _FROZEN_EPOCH_RANGE
        bs_lo, bs_hi = _FROZEN_BS_RANGE
        replay_pct_mult = 1.0
        replay_cap = _FROZEN_REPLAY_CAP
        replay_bs_ratio = _FROZEN_REPLAY_BS_RATIO

    resolved = {}

    # ── batch_size ────────────────────────────────────────────────
    if config.get("batch_size") == _AUTO:
        config["batch_size"] = max(bs_lo, min(bs_hi, effective_size // 4))
        resolved["batch_size"] = config["batch_size"]

    bs = config["batch_size"]
    batches_per_epoch = max(1, effective_size // bs)

    # ── epochs ────────────────────────────────────────────────────
    if config.get("epochs") == _AUTO:
        config["epochs"] = max(epoch_lo, min(epoch_hi, min_grad_steps // batches_per_epoch))
        resolved["epochs"] = config["epochs"]

    # ── lr (head) ─────────────────────────────────────────────────
    if config.get("lr") == _AUTO:
        config["lr"] = default_lr
        resolved["lr"] = config["lr"]

    # ── replay_buffer_size ────────────────────────────────────────
    if config.get("replay_buffer_size") == _AUTO:
        base_pct = config.get("replay_buffer_pct", _FROZEN_REPLAY_PCT_DEFAULT)
        effective_pct = base_pct * replay_pct_mult
        config["replay_buffer_size"] = max(10, min(replay_cap, int(n_train * effective_pct)))
        resolved["replay_buffer_size"] = config["replay_buffer_size"]
        resolved["replay_buffer_pct"] = f"{effective_pct:.0%}"

    # ── replay_batch_size ─────────────────────────────────────────
    if config.get("replay_batch_size") == _AUTO:
        config["replay_batch_size"] = max(8, int(config["batch_size"] * replay_bs_ratio))
        resolved["replay_batch_size"] = config["replay_batch_size"]

    # ── lora_lr ───────────────────────────────────────────────────
    if use_lora and config.get("lora_lr") == _AUTO:
        if effective_size < _LORA_SMALL_DATASET_THRESHOLD:
            ratio = _LORA_LR_RATIO_SMALL
        else:
            ratio = _LORA_LR_RATIO_DEFAULT
        config["lora_lr"] = config["lr"] * ratio
        resolved["lora_lr"] = config["lora_lr"]
        if effective_size < _LORA_SMALL_DATASET_THRESHOLD:
            resolved["lora_lr_note"] = (
                f"reduced (small task: ~{effective_size} samples < "
                f"{_LORA_SMALL_DATASET_THRESHOLD} threshold)"
            )

    # ── Summary ───────────────────────────────────────────────────
    if resolved:
        regime = "LoRA" if use_lora else "frozen-encoder"
        parts = [f"{k}={v}" for k, v in resolved.items()]
        print(f"Auto-configured ({regime}): {', '.join(parts)}")
        if is_sequential:
            print(f"  (based on {n_train} train samples, "
                  f"{num_classes} classes, ~{effective_size} samples/task)")
        else:
            print(f"  (based on {n_train} train samples, {num_classes} classes)")
