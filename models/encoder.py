import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline


class FrozenMomentEncoder(nn.Module):
    """Frozen MOMENT encoder that transparently handles long sequences.

    MOMENT's T5 backbone is trained with ``seq_len=512``.  Longer series
    are automatically split into non-overlapping 512-step chunks, each
    chunk is embedded independently, and the per-chunk embeddings are
    averaged.  Because the encoder is frozen the result is deterministic
    and lossless.
    """

    def __init__(self, model_name="AutonLab/MOMENT-1-base"):
        super().__init__()

        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.model.init()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        self.max_seq_len = getattr(self.model.config, "seq_len", 512)

    def _embed_chunk(self, x_chunk, mask_chunk):
        """Embed a single chunk, padding to ``max_seq_len`` if needed."""
        chunk_len = x_chunk.shape[-1]
        if chunk_len < self.max_seq_len:
            pad = self.max_seq_len - chunk_len
            x_chunk = F.pad(x_chunk, (0, pad))
            mask_chunk = F.pad(mask_chunk, (0, pad))
        output = self.model(x_enc=x_chunk, input_mask=mask_chunk)
        return output.embeddings

    def forward(self, x, input_mask):
        seq_len = x.shape[-1]

        if seq_len <= self.max_seq_len:
            with torch.no_grad():
                return self._embed_chunk(x, input_mask)

        chunk_embeddings = []
        for start in range(0, seq_len, self.max_seq_len):
            end = min(start + self.max_seq_len, seq_len)
            x_chunk = x[..., start:end]
            mask_chunk = input_mask[:, start:end]

            with torch.no_grad():
                chunk_embeddings.append(self._embed_chunk(x_chunk, mask_chunk))

        return torch.stack(chunk_embeddings, dim=0).mean(dim=0)