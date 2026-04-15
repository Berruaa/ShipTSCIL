# ShipTSCIL — Class-Incremental Learning on Time-Series Foundation Models

A framework for benchmarking **Class-Incremental Learning (CIL)** strategies on top of the [MOMENT](https://huggingface.co/AutonLab/MOMENT-1-base) time-series foundation model. The frozen (or LoRA-adapted) MOMENT encoder produces fixed-length embeddings from variable-length, potentially multivariate time series, while a lightweight classification head is trained incrementally as new classes arrive.

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Datasets](#datasets)
- [Methods](#methods)
- [Evaluation Metrics](#evaluation-metrics)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New CIL Method](#adding-a-new-cil-method)

## Key Features

- **Foundation-model backbone** — uses MOMENT (a T5-based time-series model) as a frozen feature extractor. Sequences longer than 512 steps are automatically chunked and mean-pooled.
- **Six CIL strategies** — naive fine-tuning, raw/latent experience replay, Learning without Forgetting (LwF), and two Nearest-Class-Mean variants (standard and herding-based).
- **Two standard baselines** — joint linear probe and SVM on the full training set (upper-bound references).
- **LoRA support** — optional low-rank adapters on the MOMENT encoder for methods that benefit from a trainable backbone.
- **Auto-configuration** — batch size, learning rate, epochs, and replay buffer sizes are derived from dataset statistics when set to `"auto"`.
- **Embedding caching** — precomputed embeddings are saved to disk, so subsequent runs skip the encoder forward pass entirely.
- **Seven built-in datasets** spanning ECG, energy, gesture, HAR, spectroscopy, and entomology domains.
- **Automated reporting** — per-task accuracy tables, Average Incremental Accuracy (AIA), forgetting, Backward Transfer (BWT), confusion matrices, and publication-ready plots.

## Project Structure

```
ShipTSCIL/
├── train.py                       # Entry point and run configuration
│
├── datasets/
│   ├── factory.py                 # Dataset registry and loader builder
│   ├── ts_dataset.py              # TimeSeriesDataset / EmbeddingDataset
│   ├── convert_uci_har.py         # UCI HAR → .ts converter
│   └── convert_wisdm.py           # WISDM → .ts converter
│
├── methods/
│   ├── __init__.py                # Method registry and factory
│   ├── base.py                    # BaseMethod (encoder + head + save/load)
│   ├── linear_probe.py            # Joint linear probe (standard baseline)
│   ├── svm.py                     # SVM baseline
│   ├── cil_naive.py               # Naive sequential fine-tuning
│   ├── cil_replay_raw.py          # Raw experience replay
│   ├── cil_replay_latent.py       # Latent (embedding) replay + optional KD
│   ├── cil_lwf.py                 # Learning without Forgetting
│   ├── cil_ncm.py                 # Nearest-Class-Mean (FastICARL-style)
│   └── cil_herding_ncm.py         # Herding exemplar selection + NCM
│
├── models/
│   ├── encoder.py                 # FrozenMomentEncoder (chunked inference)
│   ├── head.py                    # LinearClassifier
│   ├── lora.py                    # LoRA injection for MOMENT
│   └── model.py                   # MomentModel (encoder + head composite)
│
├── pipelines/
│   ├── config.py                  # Auto-configuration heuristics
│   ├── data.py                    # Data loading, embedding cache, task splits
│   ├── evaluation.py              # Prediction collection and seen-class eval
│   └── train_loops.py             # Standard and sequential training loops
│
├── trainers/
│   └── linear_probe_trainer.py    # Single-epoch train/eval for linear heads
│
├── utils/
│   ├── seed.py                    # Reproducibility (Python/NumPy/PyTorch)
│   ├── metrics.py                 # Accuracy, confusion matrix, per-class acc
│   ├── losses.py                  # Distillation loss, class-balanced CE
│   ├── replay_buffers.py          # Reservoir, class-balanced, herding buffers
│   ├── reporting.py               # Console output (run info, AIA, BWT, etc.)
│   └── plotting.py                # Matplotlib plots (curves, heatmaps, CM)
│
├── data/                          # .ts dataset files (see Datasets section)
├── checkpoints/                   # Best/final model checkpoints (auto-created)
├── embeddings_cache/              # Cached encoder outputs (auto-created)
└── results/                       # Plots and figures (auto-created)
```

## Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended but not required (CPU fallback is automatic)

### Installation

```bash
git clone <repo-url> ShipTSCIL
cd ShipTSCIL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu
pip install momentfm scikit-learn numpy tqdm matplotlib joblib
```

### Dataset preparation

Most datasets ship as `.ts` files in `data/` and work out of the box. Two datasets require a conversion step from their original formats:

```bash
# UCI HAR — download and extract to data/UCI HAR Dataset/, then:
python datasets/convert_uci_har.py

# WISDM — download and extract to data/WISDM/raw/phone/accel/, then:
python datasets/convert_wisdm.py
```

## Quick Start

All configuration lives in the `CONFIG` dictionary at the top of `train.py`. Edit the values you need, then run:

```bash
python train.py
```

**Example: Joint linear probe on ECG5000**

```python
CONFIG = {
    "method":  "linear_probe",
    "dataset": "ecg5000",
    ...
}
```

**Example: Latent replay CIL on WISDM with 3 classes per task**

```python
CONFIG = {
    "method":          "cil_replay_latent",
    "dataset":         "wisdm",
    "classes_per_task": 3,
    "save_results":     True,
    ...
}
```

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | str | `"cil_replay_latent"` | Learning method (see [Methods](#methods)) |
| `dataset` | str | `"insect_sound"` | Dataset registry key (see [Datasets](#datasets)) |
| `model_name` | str | `"AutonLab/MOMENT-1-base"` | Hugging Face model ID for MOMENT |
| `train_file` / `test_file` | str \| None | `None` | Override registry with explicit file paths |
| `batch_size` | int \| `"auto"` | `"auto"` | Training batch size |
| `epochs` | int \| `"auto"` | `"auto"` | Training epochs per task |
| `lr` | float \| `"auto"` | `"auto"` | Head learning rate (Adam) |
| `seed` | int | `42` | Global random seed |
| `num_workers` | int | `0` | DataLoader worker processes |

### Class-incremental settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `task_order` | list \| None | `None` | Manual task schedule, e.g. `[[0,1],[2,3]]`; auto-generated if `None` |
| `num_tasks` | int \| None | `None` | Override number of tasks (alternative to `classes_per_task`) |
| `shuffle_class_order` | bool | `True` | Randomise class order before splitting into tasks |
| `classes_per_task` | int | `2` | Number of new classes introduced per task |

### Replay buffer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `replay_buffer_pct` | float | `0.05` | Buffer = this fraction of training set (when size is auto) |
| `replay_buffer_size` | int \| `"auto"` | `"auto"` | Explicit buffer capacity (overrides pct) |
| `replay_batch_size` | int \| `"auto"` | `"auto"` | Replay samples per training batch |
| `balanced_replay` | bool | `True` | Class-balanced buffer vs reservoir sampling |
| `herding_replay` | bool | `True` | iCaRL herding exemplar selection |

### Loss and distillation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `balanced_loss` | bool | `True` | Class-weighted cross-entropy |
| `use_distillation` | bool | `False` | Knowledge distillation (LwF / latent replay) |
| `distill_temperature` | float | `2.0` | Softmax temperature for KD |
| `distill_weight` | float | `1.0` | KD loss weight relative to CE |

### LoRA

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_lora` | bool | `False` | Enable low-rank adapters on the MOMENT encoder |
| `lora_rank` | int | `8` | LoRA rank |
| `lora_alpha` | int | `16` | LoRA scaling factor |
| `lora_target_modules` | list \| None | `None` | Attention projections to adapt (default: `["q", "v"]`) |
| `lora_dropout` | float | `0.05` | Dropout on LoRA layers |
| `lora_lr` | float \| `"auto"` | `"auto"` | Adapter learning rate (default: 0.2x head LR) |

### Output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_results` | bool | `False` | Save plots and figures to `results/` |

### Auto-configuration

When a value is set to `"auto"`, `pipelines.config.auto_configure` resolves it from the dataset size and number of classes. Two regimes exist:

| Parameter | Frozen encoder | LoRA |
|-----------|---------------|------|
| Learning rate | 5e-4 | 2e-4 |
| Epoch range | 10 -- 100 | 10 -- 60 |
| Batch size range | 8 -- 64 | 16 -- 64 |
| Replay buffer cap | 5,000 | 8,000 (1.5x pct) |
| Replay batch ratio | 50% of batch | 75% of batch |
| Grad-step target | 200 | 150 |

## Datasets

Seven datasets are registered in `datasets/factory.py`:

| Key | Domain | Train | Test | Length | Ch. | Classes |
|-----|--------|------:|-----:|-------:|----:|--------:|
| `ecg5000` | Medical / cardiology | 500 | 4,500 | 140 | 1 | 5 |
| `electric_devices` | Energy / smart metering | 8,926 | 7,711 | 96 | 1 | 7 |
| `uwave_gesture_library` | Gesture recognition | 2,238 | 2,241 | 315 | 3 | 8 |
| `uci_har` | Wearable HAR | 7,352 | 2,947 | 128 | 9 | 6 |
| `wisdm` | Wearable HAR | 16,071 | 7,326 | 200 | 3 | 18 |
| `ethanol_level` | Spectroscopy | 504 | 251 | 1,751 | 1 | 4 |
| `insect_sound` | Entomology / bioacoustics | 25,000 | 25,000 | 600 | 1 | 10 |

All datasets use the UCR/UEA `.ts` format. UCI HAR and WISDM require running their respective converter scripts first (see [Setup](#setup)).

## Methods

### Standard baselines (joint training on all classes)

| Config value | Strategy |
|-------------|----------|
| `linear_probe` | Frozen MOMENT encoder + trainable linear head (Adam). Upper-bound reference. |
| `svm` | Frozen encoder + scikit-learn `LinearSVC` on extracted embeddings. |

### Class-incremental methods (sequential task arrival)

| Config value | Strategy | Replay | Distillation |
|-------------|----------|:------:|:------------:|
| `cil_naive` | Fine-tune head on current-task data only. No anti-forgetting mechanism. | -- | -- |
| `cil_replay_raw` | Store raw time-series (or embeddings) in a replay buffer; concatenate with current-task batch during training. | Yes | -- |
| `cil_replay_latent` | Replay stored embeddings. Optionally add LwF-style knowledge distillation on old-class logits. | Yes | Optional |
| `cil_lwf` | Learning without Forgetting: snapshot the head at task start; distill old-class logits alongside new-class CE. No replay buffer. | -- | Yes |
| `cil_ncm` | Nearest-Class-Mean: maintain per-class prototype vectors (mean embeddings); classify by nearest prototype at inference. | -- | -- |
| `cil_herding_ncm` | iCaRL-style herding exemplar selection + NCM inference. Prototypes are computed from herding-selected exemplars. | Yes (herding) | -- |

**Aliases**: `cil_ncm` with `herding_replay=True` automatically uses the herding variant. Additional shorthand aliases: `raw_replay`, `latent_replay`, `lwf`, `ncm`, `herding_ncm`.

### Replay buffer strategies

Three buffer implementations are available in `utils/replay_buffers.py`:

- **Reservoir** (`balanced_replay=False`) — classic reservoir sampling; uniform random replacement.
- **Class-balanced** (`balanced_replay=True`) — maintains equal capacity per class; oldest samples evicted when a class slot is full.
- **Herding** (`herding_replay=True`) — iCaRL-style: selects exemplars closest to the running class mean in embedding space. Overrides the balanced flag.

## Evaluation Metrics

### During sequential training

After each task, the framework evaluates on the test subset restricted to **all classes seen so far** and records:

- **Seen-class test accuracy** — accuracy on the growing set of known classes.
- **Per-task accuracy** — accuracy on each previous task's classes (used to compute forgetting).

### Final summary

| Metric | Definition |
|--------|-----------|
| **AIA** (Average Incremental Accuracy) | Mean of the seen-class accuracy recorded after each task. |
| **Average Forgetting** | For each task, the difference between its peak accuracy and its accuracy after the final task, averaged across tasks. |
| **BWT** (Backward Transfer) | Mean change in per-task accuracy between when a task was last trained and after all tasks complete. Negative values indicate forgetting. |

The final report also includes a full **classification report** (precision, recall, F1 per class) and a **confusion matrix**, both from scikit-learn.

### Saved plots (`save_results=True`)

When enabled, a timestamped directory under `results/` contains:

- Training loss/accuracy curves (per task for CIL methods)
- Task accuracy progression chart
- Confusion matrix heatmap
- Per-class accuracy bar chart
- Forgetting heatmap (CIL methods)

## LoRA Fine-Tuning

By default the MOMENT encoder is completely frozen and only the linear head is trained. Enabling LoRA injects trainable low-rank adapters into the encoder's attention layers, allowing the representations to adapt to the target domain.

```python
CONFIG = {
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16,
    ...
}
```

When LoRA is active:

- Embedding precomputation and caching are **skipped** (representations change every step).
- Auto-configured hyperparameters shift to a more conservative regime (see table above).
- Compatible with: `linear_probe`, `cil_naive`, `cil_replay_raw`, `cil_lwf`.
- **Not** compatible with: `svm`, `cil_replay_latent`, `cil_ncm`, `cil_herding_ncm` (these rely on fixed embeddings).

## Adding a New Dataset

1. Place `<Name>_TRAIN.ts` and `<Name>_TEST.ts` in `data/`.

2. Register in `datasets/factory.py`:

```python
"my_dataset": {
    "train": "MyDataset_TRAIN.ts",
    "test": "MyDataset_TEST.ts",
},
```

3. Set `CONFIG["dataset"] = "my_dataset"` in `train.py`.

For non-`.ts` source data, write a converter script under `datasets/` following the pattern of `convert_uci_har.py` or `convert_wisdm.py`.

## Adding a New CIL Method

1. Create `methods/cil_<name>.py` with a class inheriting from `BaseMethod` (or `LinearProbeMethod`).

2. Implement at minimum:
   - `begin_task(task_classes)` — called before each task's training.
   - `train_epoch(loader)` — one epoch of training.
   - `evaluate(loader)` — returns `(loss, accuracy)`.
   - `end_task(task_classes, train_dataset)` — called after each task (update buffers, prototypes, etc.).

3. Register in `methods/__init__.py` by adding to `SEQUENTIAL_METHODS` and the `build_method` factory.
