"""
Low-Rank Adaptation (LoRA) for the MOMENT encoder.

Provides a manual LoRA implementation (no ``peft`` dependency) and a
LoRA-enabled MOMENT encoder that can be swapped in for FrozenMomentEncoder.

Key differences from FrozenMomentEncoder:
* Selected attention projections receive trainable low-rank adapters.
* ``forward()`` does NOT wrap computation in ``torch.no_grad()`` so that
  gradients flow through the LoRA parameters during training.
* All original MOMENT weights remain frozen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline


# ------------------------------------------------------------------
#  LoRA building blocks
# ------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a low-rank adapter.

    Output = original(x) + (α / r) · x A^T B^T

    A is Kaiming-initialised; B is zero-initialised so the adapter
    contributes nothing at the start of training.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora


def inject_lora(model, target_modules, rank, alpha, dropout=0.0):
    """Replace matching ``nn.Linear`` layers in *model* with ``LoRALinear``.

    Parameters
    ----------
    model : nn.Module
    target_modules : list[str]
        Leaf module-name suffixes to match (e.g. ``["q", "v"]`` for T5).
    rank, alpha, dropout : LoRA hyper-parameters.

    Returns
    -------
    list[nn.Parameter] — all newly created LoRA parameters.
    """
    target_set = set(target_modules)
    lora_params: list[nn.Parameter] = []

    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if name.split(".")[-1] in target_set and isinstance(module, nn.Linear)
    ]

    for full_name, module in replacements:
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    return lora_params


# ------------------------------------------------------------------
#  LoRA-enabled MOMENT encoder
# ------------------------------------------------------------------

class LoRAMomentEncoder(nn.Module):
    """MOMENT encoder with LoRA adapters on selected attention projections.

    The base MOMENT weights stay frozen; only the low-rank adapter
    matrices (A, B per layer) are trainable.  Forward passes are
    executed **with** gradient tracking so that back-propagation can
    reach the LoRA parameters.
    """

    def __init__(
        self,
        model_name="AutonLab/MOMENT-1-base",
        lora_rank=8,
        lora_alpha=16,
        lora_target_modules=None,
        lora_dropout=0.05,
    ):
        super().__init__()

        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.model.init()

        for param in self.model.parameters():
            param.requires_grad = False

        self.max_seq_len = getattr(self.model.config, "seq_len", 512)

        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]

        self._lora_params = inject_lora(
            self.model,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        n_lora = sum(p.numel() for p in self._lora_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[LoRA] Injected {len(self._lora_params)} adapter matrices "
            f"({n_lora:,} trainable / {n_total:,} total params, "
            f"{n_lora / n_total:.2%})"
        )

    # -- parameter access ---------------------------------------------------

    def lora_parameters(self):
        """Iterator over the trainable LoRA parameters only."""
        return iter(self._lora_params)

    def lora_param_count(self):
        return sum(p.numel() for p in self._lora_params)

    # -- forward (NO torch.no_grad — gradients flow through LoRA) ----------

    def _embed_chunk(self, x_chunk, mask_chunk):
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
            return self._embed_chunk(x, input_mask)

        chunk_embeddings = []
        for start in range(0, seq_len, self.max_seq_len):
            end = min(start + self.max_seq_len, seq_len)
            chunk_embeddings.append(
                self._embed_chunk(x[..., start:end], input_mask[:, start:end])
            )

        return torch.stack(chunk_embeddings, dim=0).mean(dim=0)
