from pathlib import Path

from datasets.ts_dataset import TimeSeriesDataset


# ── Dataset Registry ──────────────────────────────────────────────────────────
# ┌─────────────────────────────┬───────┬────────┬───────┬───────┬────────────┐
# │ Key                         │ Train │  Test  │  Len  │  Ch.  │  Classes   │
# ├─────────────────────────────┼───────┼────────┼───────┼───────┼────────────┤
# │ ecg5000                     │   500 │  4 500 │   140 │   1   │     5      │
# │ electric_devices            │ 8 926 │  7 711 │    96 │   1   │     7      │
# │ uwave_gesture_library       │ 2 238 │  2 241 │   315 │   3   │     8      │
# │ uci_har            *        │ 7 352 │  2 947 │   128 │   9   │     6      │
# │ wisdm              *        │16 071 │  7 326 │   200 │   3   │    18      │
# │ ethanol_level               │   504 │    251 │ 1 751 │   1   │     4      │
# │ insect_sound                │25 000 │ 25 000 │   600 │   1   │    10      │
# └─────────────────────────────┴───────┴────────┴───────┴───────┴────────────┘
# * converted from raw data

DATASET_REGISTRY = {
    # ── ECG5000 ───────────────────────────────────────────────────────────────
    # Source    : BIDMC Congestive Heart Failure Database (PhysioNet), record
    #             "chf07" — 5 000 heartbeats randomly sampled from a 20-hour ECG.
    # Task      : ECG heartbeat classification (normal vs. 4 abnormal types).
    # Domain    : Medical / cardiology.
    "ecg5000": {
        "train": "ECG5000_TRAIN.ts",
        "test": "ECG5000_TEST.ts",
    },

    # ── ElectricDevices ───────────────────────────────────────────────────────
    # Source    : UK "Powering the Nation" study — 251 households, electricity
    #             consumption recorded at 2-minute intervals over one month.
    # Task      : Classify the type of household electrical device in use.
    # Domain    : Energy / smart metering.
    "electric_devices": {
        "train": "ElectricDevices_TRAIN.ts",
        "test": "ElectricDevices_TEST.ts",
    },

    # ── UWaveGestureLibrary ───────────────────────────────────────────────────
    # Source    : uWave system — accelerometer recordings (X, Y, Z axes) of
    #             8 predefined hand gestures performed with a Wiimote-style device.
    # Task      : Hand gesture recognition.
    # Domain    : Human-computer interaction / gesture recognition.
    "uwave_gesture_library": {
        "train": "UWaveGestureLibrary_TRAIN.ts",
        "test": "UWaveGestureLibrary_TEST.ts",
    },

    # ── UCI HAR ───────────────────────────────────────────────────────────────
    # Source    : 30 volunteers (age 19–48) wearing a Samsung Galaxy S II on
    #             the waist; triaxial accelerometer + gyroscope at 50 Hz.
    # Task      : Human activity recognition (6 activities).
    # Domain    : Wearable sensing / HAR.
    "uci_har": {
        "train": "UCIHAR_TRAIN.ts",
        "test": "UCIHAR_TEST.ts",
    },

    # ── WISDM ─────────────────────────────────────────────────────────────────
    # Source    : 51 participants (IDs 1600–1650) performing 18 activities for
    #             3 minutes each; phone accelerometer recorded at 20 Hz.
    # Task      : Fine-grained activity and gesture recognition.
    # Domain    : Wearable sensing / HAR.
    "wisdm": {
        "train": "WISDM_TRAIN.ts",
        "test": "WISDM_TEST.ts",
    },

    # ── EthanolLevel ──────────────────────────────────────────────────────────
    # Source    : Scotch Whisky Research Institute — vibrational spectrographs
    #             of 20 bottle types at four alcohol concentrations (35 %, 38 %,
    #             40 %, 45 %).  Train/test split is bottle-disjoint (no bottle
    #             type appears in both splits), making resampling inappropriate.
    # Task      : Classify alcohol concentration from the spectrograph.
    #             Class 1 → E35, Class 2 → E38, Class 3 → E40, Class 4 → E45.
    # Domain    : Food & beverage authentication / vibrational spectroscopy.
    "ethanol_level": {
        "train": "EthanolLevel_TRAIN.ts",
        "test": "EthanolLevel_TEST.ts",
    },

    # ── InsectSound ───────────────────────────────────────────────────────────
    # Source    : UCR computational entomology group — amplitude-modulated
    #             intervals from infrared beam occlusion in single-species fly
    #             enclosures; 10 ms segments sampled at 6 000 Hz (length 600).
    # Task      : Classify fly species / sex from wingbeat sound.
    #             Classes: Aedes_female, Aedes_male, Fruit_flies, House_flies,
    #             Quinx_female, Quinx_male, Stigma_female, Stigma_male,
    #             Tarsalis_female, Tarsalis_male — 5 000 instances each.
    # Domain    : Computational entomology / bioacoustics.
    "insect_sound": {
        "train": "InsectSound_TRAIN.ts",
        "test": "InsectSound_TEST.ts",
    },


}


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_").replace(" ", "_")


def build_dataset_pair(dataset_name, data_dir, train_file=None, test_file=None):
    if train_file is not None and test_file is not None:
        train_path = Path(train_file)
        test_path = Path(test_file)
    else:
        dataset_key = _normalize_dataset_name(dataset_name)
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )
        spec = DATASET_REGISTRY[dataset_key]
        train_path = Path(data_dir) / spec["train"]
        test_path = Path(data_dir) / spec["test"]

    train_dataset = TimeSeriesDataset.from_tsfile(
        train_path,
        fit_label_encoder=True,
    )
    label_encoder = train_dataset.label_encoder

    test_dataset = TimeSeriesDataset.from_tsfile(
        test_path,
        label_encoder=label_encoder,
        fit_label_encoder=False,
    )

    dataset_info = {
        "train_file": str(train_path),
        "test_file": str(test_path),
    }

    return train_dataset, test_dataset, label_encoder, dataset_info
