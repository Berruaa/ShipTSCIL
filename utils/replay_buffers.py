"""
Replay buffers for continual learning.

Three strategies with compatible APIs so they can be swapped via a single flag:

* ``ReservoirBuffer``      – class-agnostic reservoir sampling.
* ``ClassBalancedBuffer``  – per-class partitioned buffer with stratified
                             sampling (prevents majority-class domination).
* ``HerdingBuffer``        – iCaRL-style herding: picks exemplars whose
                             running mean best approximates the class mean
                             embedding.  Requires a ``rebuild()`` call at
                             the end of each task (see method end_task hooks).
"""

import random
from collections import defaultdict

import torch


# =====================================================================
#  ReservoirBuffer  (class-agnostic, original behaviour)
# =====================================================================

class ReservoirBuffer:
    """
    Fixed-capacity buffer using standard reservoir sampling.

    Every incoming sample has an equal probability of being retained,
    regardless of its class label.  Simple and unbiased w.r.t. the data
    stream, but can lead to class-imbalanced replay when the training
    distribution is skewed.
    """

    def __init__(self, total_size: int):
        self.total_size = total_size
        self._buffer: list = []
        self._seen: int = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, sample, label: int):          # noqa: ARG002 – label unused
        """Add a single sample via reservoir sampling (label stored inside sample)."""
        self._seen += 1
        if len(self._buffer) < self.total_size:
            self._buffer.append(sample)
        else:
            idx = random.randint(0, self._seen - 1)
            if idx < self.total_size:
                self._buffer[idx] = sample

    def add_batch(self, samples: list, labels: list[int]):
        for sample, label in zip(samples, labels):
            self.add(sample, label)

    def sample(self, batch_size: int):
        if not self._buffer or batch_size <= 0:
            return []
        k = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, k=k)

    def class_distribution(self) -> dict[int, int]:
        return {}


# =====================================================================
#  ClassBalancedBuffer
# =====================================================================

class ClassBalancedBuffer:
    """
    Fixed-capacity buffer that keeps an equal number of exemplars per class.

    Storage
    -------
    Internally a ``dict[int, list]`` mapping each class label to its exemplar
    list.  Each class is allowed at most ``total_size // num_seen_classes``
    slots.  When a previously unseen class is encountered the budget is
    recomputed and all existing classes are trimmed to make room.

    Within each class, samples are managed with reservoir sampling so early
    and late samples are retained with equal probability.

    Sampling
    --------
    ``sample(batch_size)`` returns a **stratified** batch — it draws an equal
    number of samples from every class (remainder distributed round-robin) so
    that no class is over-represented regardless of the buffer's fill level.
    """

    def __init__(self, total_size: int):
        self.total_size = total_size
        self._buffers: dict[int, list] = defaultdict(list)
        self._counts: dict[int, int] = defaultdict(int)
        self._max_per_class: int = total_size

    @property
    def num_classes(self) -> int:
        return len(self._buffers)

    def __len__(self) -> int:
        return sum(len(v) for v in self._buffers.values())

    # ------------------------------------------------------------------
    #  Add
    # ------------------------------------------------------------------

    def add(self, sample, label: int):
        """Add a single sample with reservoir sampling within its class."""
        if label not in self._buffers:
            self._rebalance_for_new_class(label)

        self._counts[label] += 1
        buf = self._buffers[label]

        if len(buf) < self._max_per_class:
            buf.append(sample)
        else:
            idx = random.randint(0, self._counts[label] - 1)
            if idx < self._max_per_class:
                buf[idx] = sample

    def add_batch(self, samples: list, labels: list[int]):
        """Add multiple samples at once."""
        for sample, label in zip(samples, labels):
            self.add(sample, label)

    # ------------------------------------------------------------------
    #  Rebalance
    # ------------------------------------------------------------------

    def _rebalance_for_new_class(self, new_label: int):
        """Recompute per-class budget and trim existing classes."""
        self._buffers[new_label] = []
        self._counts[new_label] = 0

        new_max = max(1, self.total_size // len(self._buffers))
        self._max_per_class = new_max

        for cls, buf in self._buffers.items():
            if len(buf) > new_max:
                random.shuffle(buf)
                self._buffers[cls] = buf[:new_max]

    # ------------------------------------------------------------------
    #  Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int):
        """
        Return a class-balanced list of samples.

        Distributes ``batch_size`` as evenly as possible across all classes.
        If a class has fewer stored samples than its share, all its samples
        are included and the shortfall is redistributed to other classes.
        """
        if len(self) == 0 or batch_size <= 0:
            return []

        classes = list(self._buffers.keys())
        random.shuffle(classes)

        base = batch_size // len(classes)
        remainder = batch_size % len(classes)

        per_class_n = {}
        for i, cls in enumerate(classes):
            per_class_n[cls] = base + (1 if i < remainder else 0)

        selected = []
        for cls in classes:
            buf = self._buffers[cls]
            n = min(per_class_n[cls], len(buf))
            selected.extend(random.sample(buf, k=n))

        return selected

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def class_distribution(self) -> dict[int, int]:
        """Return {class_label: num_stored} for inspection."""
        return {cls: len(buf) for cls, buf in sorted(self._buffers.items())}


# =====================================================================
#  HerdingBuffer
# =====================================================================

def _herding_select(embs: torch.Tensor, k: int) -> list[int]:
    """
    Greedy herding selection (iCaRL-style).

    Iteratively picks the sample whose addition makes the running mean of
    selected exemplars closest to the true class mean.

    Parameters
    ----------
    embs : (N, D) float tensor of class embeddings.
    k    : number of exemplars to select (clamped to N).

    Returns
    -------
    List of selected indices into *embs* (length min(k, N)).
    """
    k = min(k, embs.size(0))
    mean = embs.mean(dim=0)           # (D,)
    selected: list[int] = []
    running_sum = torch.zeros_like(mean)
    remaining = list(range(embs.size(0)))

    for step in range(k):
        # Vectorised: compute candidate means for all remaining samples at once.
        cand_embs = embs[remaining]                                    # (R, D)
        cand_means = (running_sum.unsqueeze(0) + cand_embs) / (step + 1)  # (R, D)
        dists = torch.norm(cand_means - mean.unsqueeze(0), dim=1)     # (R,)
        best_local = int(dists.argmin().item())
        best_global = remaining[best_local]

        selected.append(best_global)
        running_sum = running_sum + embs[best_global]
        remaining.pop(best_local)

    return selected


class HerdingBuffer:
    """
    Fixed-capacity buffer that selects exemplars via iCaRL-style herding.

    Workflow
    --------
    1. Call ``begin_task()`` at the start of each new task to reset the
       staging area.
    2. Call ``add_batch(samples, labels)`` during training — samples are
       accumulated in a per-class staging list, not yet committed.
    3. Call ``rebuild(max_per_class)`` at the end of the task.  This
       deduplicates staged samples (safe with a frozen encoder), runs greedy
       herding to select the best exemplars for new classes, and trims old
       classes proportionally.

    Each sample must be a tuple ``(embedding_tensor, label)`` so that
    embedding distances can be computed during ``rebuild``.
    Sampling returns the same tuple format, matching ``ClassBalancedBuffer``.
    """

    def __init__(self, total_size: int):
        self.total_size = total_size
        # Committed exemplars: {label: [(emb, label), ...]}
        self._exemplars: dict[int, list] = {}
        # Staging area for the current task: {label: [(emb, label), ...]}
        self._staging: dict[int, list] = defaultdict(list)

    @property
    def num_classes(self) -> int:
        return len(self._exemplars)

    def __len__(self) -> int:
        return sum(len(v) for v in self._exemplars.values())

    # ------------------------------------------------------------------
    #  Task lifecycle
    # ------------------------------------------------------------------

    def begin_task(self):
        """Clear the staging area at the start of a new task."""
        self._staging.clear()

    def rebuild(self, max_per_class: int):
        """
        Commit exemplars for all staged (new) classes and trim old ones.

        Parameters
        ----------
        max_per_class : target number of exemplars per class after rebuild
                        (typically ``total_size // num_seen_classes``).
        """
        for label, candidates in self._staging.items():
            # Deduplicate: frozen encoder → same input = identical embedding.
            seen_keys: set[bytes] = set()
            unique: list = []
            for c in candidates:
                key = c[0].numpy().tobytes()
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique.append(c)

            embs = torch.stack([c[0] for c in unique])   # (N, D)
            indices = _herding_select(embs, max_per_class)
            self._exemplars[label] = [unique[i] for i in indices]

        # Trim existing classes that were seen in earlier tasks.
        for label in list(self._exemplars.keys()):
            if label not in self._staging:
                self._exemplars[label] = self._exemplars[label][:max_per_class]

        self._staging.clear()

    # ------------------------------------------------------------------
    #  Add / Sample (same API as the other buffers)
    # ------------------------------------------------------------------

    def add_batch(self, samples: list, labels: list[int]):
        """Accumulate samples into the staging area (not exemplars yet)."""
        for sample, label in zip(samples, labels):
            self._staging[label].append(sample)

    def sample(self, batch_size: int) -> list:
        """Stratified sample from committed exemplars (same as ClassBalancedBuffer)."""
        if len(self) == 0 or batch_size <= 0:
            return []

        classes = list(self._exemplars.keys())
        random.shuffle(classes)

        base = batch_size // len(classes)
        remainder = batch_size % len(classes)

        selected = []
        for i, cls in enumerate(classes):
            buf = self._exemplars[cls]
            n = min(base + (1 if i < remainder else 0), len(buf))
            selected.extend(random.sample(buf, k=n))
        return selected

    def class_distribution(self) -> dict[int, int]:
        """Return {class_label: num_exemplars} for inspection."""
        return {cls: len(buf) for cls, buf in sorted(self._exemplars.items())}


# =====================================================================
#  Factory
# =====================================================================

def build_replay_buffer(total_size: int, balanced: bool = True, herding: bool = False):
    """
    Return the appropriate replay buffer.

    Priority: herding > balanced > reservoir.
    """
    if herding:
        return HerdingBuffer(total_size)
    if balanced:
        return ClassBalancedBuffer(total_size)
    return ReservoirBuffer(total_size)
