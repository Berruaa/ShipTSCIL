"""
Convert the WISDM dataset (raw phone accelerometer) to UCR/UEA .ts format.

Usage:
    python datasets/convert_wisdm.py

Source:  data/WISDM/raw/phone/accel/data_<id>_accel_phone.txt
Format per line:  subject_id,activity_code,timestamp,x,y,z;

Windowing:
    Window size : 200 samples  (10 sec @ 20 Hz)
    Stride      : 200 samples  (no overlap)
    Each window becomes one sample with 3 channels (acc_x, acc_y, acc_z).

Activities (18 classes, letter codes):
    A=walking   B=jogging   C=stairs    D=sitting   E=standing
    F=typing    G=teeth     H=soup      I=chips     J=pasta
    K=drinking  L=sandwich  M=kicking   O=catch     P=dribbling
    Q=writing   R=clapping  S=folding

Train / test split (subject-independent):
    Subjects are sorted by ID; first 80% → train, last 20% → test.
    51 subjects (1600–1650) → train: 41 subjects, test: 10 subjects.

Output (written to data/):
    WISDM_TRAIN.ts
    WISDM_TEST.ts
"""

from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
WISDM_RAW_DIR = PROJECT_ROOT / "data" / "WISDM" / "raw" / "phone" / "accel"
OUTPUT_DIR    = PROJECT_ROOT / "data"

# ── Activity map ──────────────────────────────────────────────────────────────

ACTIVITY_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}

WINDOW_SIZE = 200   # 10 sec × 20 Hz
PROBLEM_NAME = "WISDM"


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_file(path: Path) -> list:
    """
    Parse one subject's accelerometer file.

    Returns a list of (activity_code, x, y, z) tuples in order.
    Lines ending with ';' and any trailing empty/malformed lines are handled.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            try:
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except ValueError:
                continue
            if activity not in ACTIVITY_MAP:
                continue
            records.append((activity, x, y, z))
    return records


def _load_all_subjects() -> dict:
    """
    Load every subject file.

    Returns
    -------
    subject_data : dict  {subject_id: [(activity, x, y, z), ...]}
    """
    files = sorted(WISDM_RAW_DIR.glob("data_*_accel_phone.txt"))
    if not files:
        raise FileNotFoundError(f"No raw files found in {WISDM_RAW_DIR}")

    subject_data = {}
    for path in files:
        subject_id = int(path.stem.split("_")[1])
        records = _parse_file(path)
        if records:
            subject_data[subject_id] = records
    return subject_data


# ── Windowing ─────────────────────────────────────────────────────────────────

def _segment_subject(records: list) -> tuple:
    """
    Slice a subject's records into non-overlapping windows.

    Only windows where every sample shares the same activity label are kept
    (no cross-activity contamination).

    Returns
    -------
    windows : np.ndarray  shape (N, 3, WINDOW_SIZE)
    labels  : list[str]   length N  — activity name strings
    """
    # Group consecutive same-activity runs.
    # Build per-activity buffers without splitting mid-window.
    activity_buffers: dict = defaultdict(list)
    for act, x, y, z in records:
        activity_buffers[act].append((x, y, z))

    windows, labels = [], []
    for act_code, buf in activity_buffers.items():
        arr = np.array(buf, dtype=np.float32)   # (M, 3)
        n_windows = len(arr) // WINDOW_SIZE
        for i in range(n_windows):
            chunk = arr[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE]   # (200, 3)
            windows.append(chunk.T)                                 # (3, 200)
            labels.append(ACTIVITY_MAP[act_code])

    return windows, labels


def _build_split(subject_ids: list, subject_data: dict) -> tuple:
    """Concatenate windowed data for a list of subject IDs."""
    all_windows, all_labels = [], []
    for sid in subject_ids:
        if sid not in subject_data:
            continue
        w, l = _segment_subject(subject_data[sid])
        all_windows.extend(w)
        all_labels.extend(l)

    signals = np.stack(all_windows, axis=0)   # (N, 3, 200)
    return signals, all_labels


# ── Writer ────────────────────────────────────────────────────────────────────

def _write_ts(
    signals: np.ndarray,
    labels: list,
    out_path: Path,
    class_labels: list,
):
    """Write (N, C, T) array + string labels to UCR/UEA .ts format."""
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
        f.write(f"@classLabel true {' '.join(class_labels)}\n")
        f.write("@data\n")

        for i in range(n_samples):
            channel_strs = [
                ",".join(f"{v:.6f}" for v in signals[i, c])
                for c in range(n_channels)
            ]
            f.write(":".join(channel_strs) + ":" + labels[i] + "\n")

    print(f"  Written {n_samples} samples  ({n_channels} ch × {series_len} steps)  →  {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def convert():
    if not WISDM_RAW_DIR.exists():
        raise FileNotFoundError(
            f"WISDM raw phone accel directory not found at:\n  {WISDM_RAW_DIR}"
        )

    print("Loading raw files …")
    subject_data = _load_all_subjects()

    subject_ids = sorted(subject_data.keys())
    n_train = int(len(subject_ids) * 0.8)
    train_ids = subject_ids[:n_train]
    test_ids  = subject_ids[n_train:]

    print(f"Subjects  : {len(subject_ids)}  (IDs {subject_ids[0]}–{subject_ids[-1]})")
    print(f"Train     : {len(train_ids)} subjects  ({train_ids[0]}–{train_ids[-1]})")
    print(f"Test      : {len(test_ids)} subjects  ({test_ids[0]}–{test_ids[-1]})")

    class_labels = sorted(ACTIVITY_MAP.values())

    print(f"\nBuilding training split …")
    train_signals, train_labels = _build_split(train_ids, subject_data)

    print(f"Building test split …")
    test_signals, test_labels = _build_split(test_ids, subject_data)

    print(f"\nWriting .ts files to {OUTPUT_DIR} …")
    _write_ts(train_signals, train_labels, OUTPUT_DIR / "WISDM_TRAIN.ts", class_labels)
    _write_ts(test_signals,  test_labels,  OUTPUT_DIR / "WISDM_TEST.ts",  class_labels)

    print("\nDone.  Add 'wisdm' to train.py CONFIG to use this dataset:")
    print('  "dataset": "wisdm"')


if __name__ == "__main__":
    convert()
