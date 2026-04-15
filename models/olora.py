"""
Orthogonal Low-Rank Adaptation (O-LoRA) for continual learning.

Implements the approach from Wang et al. (2023) "Orthogonal Subspace Learning
for Language Model Continual Learning" (EMNLP 2023 Findings).

Each sequential task receives its own LoRA adapter pair (A_t, B_t).  Previous
adapters are frozen and an orthogonality regulariser is applied so that the
current task's update subspace stays orthogonal to all past subspaces:

    L_orth = sum_{i < t} || A_i^T @ A_t ||_F^2

This prevents catastrophic forgetting without replay or distillation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline


# ------------------------------------------------------------------
#  O-LoRA building blocks
# ------------------------------------------------------------------

class OLoRALinear(nn.Module):
    """``nn.Linear`` wrapper that supports multiple task-specific LoRA adapters.

    For task *t* the output is:

        y = original(x) + sum_{i=1}^{t} (alpha / r) * x @ A_i^T @ B_i^T

    Adapters for tasks < t are frozen; only the latest (A_t, B_t) is trainable.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.original.parameters():
            param.requires_grad = False

        self._task_A: nn.ParameterList = nn.ParameterList()
        self._task_B: nn.ParameterList = nn.ParameterList()

    # -- task management ------------------------------------------------

    def add_task(self):
        """Create a new (A, B) adapter pair for the next task.

        Freezes all existing adapters before adding the new one.
        """
        for p in self._task_A:
            p.requires_grad = False
        for p in self._task_B:
            p.requires_grad = False

        device = self.original.weight.device
        dtype = self.original.weight.dtype
        A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device, dtype=dtype))
        B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        self._task_A.append(A)
        self._task_B.append(B)

    @property
    def num_tasks(self) -> int:
        return len(self._task_A)

    def current_lora_params(self) -> list[nn.Parameter]:
        """Return the (A, B) of the latest task (the only trainable pair)."""
        if self.num_tasks == 0:
            return []
        return [self._task_A[-1], self._task_B[-1]]

    def all_lora_params(self) -> list[nn.Parameter]:
        """Return every (A, B) across all tasks."""
        return list(self._task_A) + list(self._task_B)

    # -- orthogonality loss ---------------------------------------------

    def orthogonality_loss(self) -> torch.Tensor:
        """Compute  sum_{i < t} || A_i^T @ A_t ||_F^2  for this layer."""
        if self.num_tasks < 2:
            return torch.tensor(0.0, device=self._task_A[-1].device)

        A_current = self._task_A[-1]                        # (r, d)
        loss = torch.tensor(0.0, device=A_current.device)
        for A_prev in self._task_A[:-1]:
            overlap = A_prev @ A_current.T                  # (r, r)
            loss = loss + overlap.pow(2).sum()
        return loss

    # -- forward --------------------------------------------------------

    def forward(self, x):
        base = self.original(x)
        if self.num_tasks == 0:
            return base

        lora_out = torch.zeros_like(base)
        x_drop = self.lora_dropout(x)
        for A, B in zip(self._task_A, self._task_B):
            lora_out = lora_out + x_drop @ A.T @ B.T * self.scaling
        return base + lora_out


# ------------------------------------------------------------------
#  Injection helper
# ------------------------------------------------------------------

def inject_olora(model, target_modules, rank, alpha, dropout=0.0):
    """Replace matching ``nn.Linear`` layers with ``OLoRALinear``.

    Returns the list of newly created ``OLoRALinear`` modules (not parameters,
    since adapters are added per-task later).
    """
    target_set = set(target_modules)
    olora_layers: list[OLoRALinear] = []

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

        olora_layer = OLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], olora_layer)
        olora_layers.append(olora_layer)

    return olora_layers


# ------------------------------------------------------------------
#  O-LoRA-enabled MOMENT encoder
# ------------------------------------------------------------------

class OLoRAMomentEncoder(nn.Module):
    """MOMENT encoder with per-task orthogonal LoRA adapters.

    Usage pattern (driven by the CIL method):
        encoder = OLoRAMomentEncoder(...)
        for task_id in tasks:
            encoder.add_task()       # new adapters, old ones frozen
            ... train ...
            orth_loss = encoder.orthogonality_loss()
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

        self._olora_layers = inject_olora(
            self.model,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        print(
            f"[O-LoRA] Injected {len(self._olora_layers)} OLoRA layers "
            f"(rank={lora_rank}, alpha={lora_alpha}, targets={lora_target_modules})"
        )

    # -- task management ------------------------------------------------

    def add_task(self):
        """Add a new adapter set for the next task across all OLoRA layers."""
        for layer in self._olora_layers:
            layer.add_task()

        n_trainable = sum(p.numel() for p in self.current_lora_parameters())
        n_total_lora = sum(p.numel() for layer in self._olora_layers for p in layer.all_lora_params())
        task_id = self._olora_layers[0].num_tasks if self._olora_layers else 0
        print(
            f"[O-LoRA] Task {task_id}: added {n_trainable:,} trainable params "
            f"({n_total_lora:,} total LoRA params across {task_id} task(s))"
        )

    # -- parameter access ------------------------------------------------

    def current_lora_parameters(self):
        """Trainable LoRA parameters for the current task only."""
        params = []
        for layer in self._olora_layers:
            params.extend(layer.current_lora_params())
        return params

    def lora_parameters(self):
        """Alias for current_lora_parameters (API compat with LoRAMomentEncoder)."""
        return iter(self.current_lora_parameters())

    def lora_param_count(self):
        return sum(p.numel() for p in self.current_lora_parameters())

    # -- orthogonality loss ----------------------------------------------

    def orthogonality_loss(self) -> torch.Tensor:
        """Aggregate orthogonality loss across all OLoRA layers."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for layer in self._olora_layers:
            loss = loss + layer.orthogonality_loss()
        return loss

    # -- forward (NO torch.no_grad — gradients flow through adapters) ----

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
