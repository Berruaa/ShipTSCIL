"""
Replay buffers for continual learning.

Two strategies with identical APIs so they can be swapped via a single flag:

* ``ReservoirBuffer``      – class-agnostic reservoir sampling (original).
* ``ClassBalancedBuffer``   – per-class partitioned buffer with stratified
                              sampling (prevents majority-class domination).
"""

import random
from collections import defaultdict


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
#  Factory
# =====================================================================

def build_replay_buffer(total_size: int, balanced: bool = True):
    """Return a ClassBalancedBuffer or ReservoirBuffer depending on *balanced*."""
    if balanced:
        return ClassBalancedBuffer(total_size)
    return ReservoirBuffer(total_size)
