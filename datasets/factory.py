from pathlib import Path

from datasets.ts_dataset import TimeSeriesDataset


DATASET_REGISTRY = {
    "ecg5000": {
        "train": "ECG5000_TRAIN.ts",
        "test": "ECG5000_TEST.ts",
    },
    "italy_power_demand": {
        "train": "ItalyPowerDemand_TRAIN.ts",
        "test": "ItalyPowerDemand_TEST.ts",
    },

    "electric_devices": {
        "train": "ElectricDevices_TRAIN.ts",
        "test": "ElectricDevices_TEST.ts",
    }
    # Add more datasets here later
    # "ucr_dataset_name": {
    #     "train": "SomeDataset_TRAIN.ts",
    #     "test": "SomeDataset_TEST.ts",
    # },
}


def build_dataset_pair(dataset_name, data_dir, train_file=None, test_file=None):
    if train_file is not None and test_file is not None:
        train_path = Path(train_file)
        test_path = Path(test_file)
    else:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )
        spec = DATASET_REGISTRY[dataset_name]
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