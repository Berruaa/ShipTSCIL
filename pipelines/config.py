"""Resolve ``"auto"`` hyper-parameters based on dataset statistics."""

from methods import SEQUENTIAL_METHODS

_AUTO = "auto"
_MIN_GRAD_STEPS = 200


def auto_configure(config, n_train, num_classes):
    """Fill in ``"auto"`` values in *config* in-place.

    Heuristics
    ----------
    * **batch_size** – sized so each epoch has several batches, even for
      the smallest CIL task.  Clamped to [8, 64].
    * **epochs** – targets ~200 gradient steps per task so the linear
      head converges, clamped to [10, 100].
    * **lr** – 5e-4 (safe default for Adam on a linear head).
    * **replay_buffer_size** – scales with the training set; capped at 5 000.
    * **replay_batch_size** – half of the main batch size.

    Explicit (non-``"auto"``) values are never overwritten.
    """
    is_sequential = config["method"] in SEQUENTIAL_METHODS
    classes_per_task = config.get("classes_per_task", 2)

    if is_sequential:
        samples_per_class = n_train / max(1, num_classes)
        effective_size = int(samples_per_class * classes_per_task)
    else:
        effective_size = n_train

    resolved = {}

    if config.get("batch_size") == _AUTO:
        config["batch_size"] = max(8, min(64, effective_size // 4))
        resolved["batch_size"] = config["batch_size"]

    bs = config["batch_size"]
    batches_per_epoch = max(1, effective_size // bs)

    if config.get("epochs") == _AUTO:
        config["epochs"] = max(10, min(100, _MIN_GRAD_STEPS // batches_per_epoch))
        resolved["epochs"] = config["epochs"]

    if config.get("lr") == _AUTO:
        config["lr"] = 5e-4
        resolved["lr"] = config["lr"]

    if config.get("replay_buffer_size") == _AUTO:
        pct = config.get("replay_buffer_pct", 0.2)
        config["replay_buffer_size"] = max(10, min(5000, int(n_train * pct)))
        resolved["replay_buffer_size"] = config["replay_buffer_size"]
        resolved["replay_buffer_pct"] = f"{pct:.0%}"

    if config.get("replay_batch_size") == _AUTO:
        config["replay_batch_size"] = max(8, config["batch_size"] // 2)
        resolved["replay_batch_size"] = config["replay_batch_size"]

    if resolved:
        parts = [f"{k}={v}" for k, v in resolved.items()]
        print(f"Auto-configured: {', '.join(parts)}")
        if is_sequential:
            print(f"  (based on {n_train} train samples, "
                  f"{num_classes} classes, ~{effective_size} samples/task)")
        else:
            print(f"  (based on {n_train} train samples, {num_classes} classes)")
