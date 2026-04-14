import os
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from momentfm.utils.data import load_from_tsfile


def _load_tsfile_safe(file_path):
    """Wrapper around ``load_from_tsfile`` that tolerates malformed headers.

    Some .ts files from the UEA/UCR archives have extra whitespace in header
    tags (e.g. ``@missing  false`` instead of ``@missing false``).  The
    momentfm parser rejects these.  This function normalises the header lines
    (everything before ``@data``) into single-space-separated tokens, writes
    a cleaned temp file, and delegates to the upstream loader.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = raw.split("\n")
    cleaned = []
    in_header = True
    for line in lines:
        if in_header:
            stripped = line.strip()
            if stripped.lower() == "@data":
                in_header = False
                cleaned.append("@data")
            elif stripped.startswith("@") or stripped.startswith("#"):
                cleaned.append(" ".join(stripped.split()))
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ts", delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write("\n".join(cleaned))
        tmp_path = tmp.name

    try:
        return load_from_tsfile(tmp_path)
    finally:
        os.unlink(tmp_path)


class EmbeddingDataset(Dataset):
    """Stores precomputed (embedding, label) pairs for head-only training."""

    def __init__(self, embeddings, labels, label_encoder=None):
        self.embeddings = embeddings
        self.labels = labels
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, label_encoder=None):
        data = np.asarray(data)
        labels = np.asarray(labels)

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_encoder = label_encoder

    @classmethod
    def from_tsfile(cls, file_path, label_encoder=None, fit_label_encoder=False):
        data, labels = _load_tsfile_safe(str(file_path))
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

        # Accept [T] or 2D multivariate sample.
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            # Normalize to channel-first [C, T].
            #
            # UCR/UEA loaders may return multivariate samples as either [C, T]
            # or [T, C]. MOMENT expects channel-first with time on the last dim.
            # In practice C is usually much smaller than T, so when the first
            # axis is larger than the second, treat input as [T, C] and transpose.
            if x.shape[0] > x.shape[1]:
                x = x.transpose(0, 1).contiguous()
        else:
            raise ValueError(f"Expected sample shape [T] or [C, T], got {tuple(x.shape)}")

        seq_len = x.shape[-1]
        input_mask = torch.ones(seq_len, dtype=torch.long)

        y = self.labels[idx]
        return x, input_mask, y