import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from momentfm.utils.data import load_from_tsfile


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, label_encoder=None):
        data = np.asarray(data)
        labels = np.asarray(labels)

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_encoder = label_encoder

    @classmethod
    def from_tsfile(cls, file_path, label_encoder=None, fit_label_encoder=False):
        data, labels = load_from_tsfile(str(file_path))
        return cls.from_arrays(
            data=data,
            labels=labels,
            label_encoder=label_encoder,
            fit_label_encoder=fit_label_encoder,
        )

    @classmethod
    def from_arrays(cls, data, labels, label_encoder=None, fit_label_encoder=False):
        labels = np.asarray(labels)

        if label_encoder is None:
            label_encoder = LabelEncoder()

        if fit_label_encoder:
            encoded_labels = label_encoder.fit_transform(labels)
        else:
            encoded_labels = label_encoder.transform(labels)

        return cls(data=data, labels=encoded_labels, label_encoder=label_encoder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        # Accept [T] or [C, T]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"Expected sample shape [T] or [C, T], got {tuple(x.shape)}")

        seq_len = x.shape[-1]
        input_mask = torch.ones(seq_len, dtype=torch.long)

        return x, input_mask, self.labels[idx]