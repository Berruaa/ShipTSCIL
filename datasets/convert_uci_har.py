"""
Convert the UCI HAR (Human Activity Recognition) dataset to UCR/UEA .ts format.

Usage:
    python datasets/convert_uci_har.py

Input  (expected layout under data/UCI HAR Dataset/):
    train/Inertial Signals/body_acc_x_train.txt   (7352 × 128)
    train/Inertial Signals/body_acc_y_train.txt
    train/Inertial Signals/body_acc_z_train.txt
    train/Inertial Signals/body_gyro_x_train.txt
    train/Inertial Signals/body_gyro_y_train.txt
    train/Inertial Signals/body_gyro_z_train.txt
    train/Inertial Signals/total_acc_x_train.txt
    train/Inertial Signals/total_acc_y_train.txt
    train/Inertial Signals/total_acc_z_train.txt
    train/y_train.txt                              (7352 integer labels 1–6)
    test/Inertial Signals/  (same structure, suffix _test)
    test/y_test.txt
    activity_labels.txt

Output (written to data/):
    UCIHAR_TRAIN.ts   — 7352 samples, 9 channels, 128 time-steps
    UCIHAR_TEST.ts    — 2947 samples, 9 channels, 128 time-steps

The .ts format (UCR/UEA multivariate):
    Header lines starting with @, then @data, then one row per sample.
    Each row: chan1_t1,chan1_t2,...,chan1_t128:chan2_t1,...:...:label_string
"""

from pathlib import Path

import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UCI_HAR_DIR  = PROJECT_ROOT / "data" / "UCI HAR Dataset"
OUTPUT_DIR   = PROJECT_ROOT / "data"

CHANNEL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

PROBLEM_NAME = "UCIHAR"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_signal(path: Path) -> np.ndarray:
    """Load a single inertial signal file into an (N, 128) float32 array."""
    return np.loadtxt(path, dtype=np.float32)


def _load_labels(path: Path, label_map: dict) -> list:
    """Load integer labels (1-indexed) and map to activity name strings."""
    raw = np.loadtxt(path, dtype=int)
    return [label_map[i] for i in raw]


def _load_activity_map(path: Path) -> dict:
    """Return {int_id: 'ACTIVITY_NAME'} from activity_labels.txt."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            mapping[int(parts[0])] = parts[1]
    return mapping


def _load_split(split: str, label_map: dict):
    """
    Load all 9 channel arrays and labels for a given split ('train' or 'test').

    Returns
    -------
    signals : np.ndarray  shape (N, 9, 128)
    labels  : list[str]   length N
    """
    sig_dir = UCI_HAR_DIR / split / "Inertial Signals"
    channels = []
    for ch in CHANNEL_NAMES:
        fname = f"{ch}_{split}.txt"
        arr = _load_signal(sig_dir / fname)       # (N, 128)
        channels.append(arr)

    signals = np.stack(channels, axis=1)           # (N, 9, 128)

    labels_path = UCI_HAR_DIR / split / f"y_{split}.txt"
    labels = _load_labels(labels_path, label_map)

    return signals, labels


# ── Writer ────────────────────────────────────────────────────────────────────

def _write_ts(signals: np.ndarray, labels: list, out_path: Path, class_labels: list):
    """
    Write (N, C, T) signals + string labels to UCR/UEA .ts format.

    File layout
    -----------
    @problemName  <name>
    @timeStamps   false
    @missing      false
    @univariate   false
    @dimensions   <C>
    @equalLength  true
    @seriesLength <T>
    @classLabel   true <label1> <label2> ...
    @data
    t0,t1,...,tT-1:t0,...:...:label
    """
    n_samples, n_channels, series_len = signals.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"@problemName {PROBLEM_NAME}\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate false\n")
        f.write(f"@dimensions {n_channels}\n")
        f.write("@equalLength true\n")
        f.write(f"@seriesLength {series_len}\n")
        class_str = " ".join(class_labels)
        f.write(f"@classLabel true {class_str}\n")
        f.write("@data\n")

        for i in range(n_samples):
            channel_strs = []
            for c in range(n_channels):
                # Use enough decimal places to preserve float32 precision.
                vals = ",".join(f"{v:.6f}" for v in signals[i, c])
                channel_strs.append(vals)
            row = ":".join(channel_strs) + ":" + labels[i]
            f.write(row + "\n")

    print(f"  Written {n_samples} samples → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def convert():
    labels_file = UCI_HAR_DIR / "activity_labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(
            f"Expected UCI HAR root at:\n  {UCI_HAR_DIR}\n"
            "Make sure the dataset is extracted there."
        )

    label_map = _load_activity_map(labels_file)
    # Sorted alphabetically so the class list is deterministic.
    class_labels = sorted(label_map.values())

    print(f"UCI HAR activities: {class_labels}")
    print(f"Loading training split …")
    train_signals, train_labels = _load_split("train", label_map)

    print(f"Loading test split …")
    test_signals, test_labels = _load_split("test", label_map)

    print(f"\nWriting .ts files to {OUTPUT_DIR} …")
    _write_ts(train_signals, train_labels, OUTPUT_DIR / "UCIHAR_TRAIN.ts", class_labels)
    _write_ts(test_signals,  test_labels,  OUTPUT_DIR / "UCIHAR_TEST.ts",  class_labels)

    print("\nDone.  Add 'uci_har' to train.py CONFIG to use this dataset:")
    print('  "dataset": "uci_har"')


if __name__ == "__main__":
    convert()
