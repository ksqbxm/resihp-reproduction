"""Self-built decoder-only Transformer with stable global layer identity.

The reference model is assembled from primitive layers -- token/position
embedding, LayerNorm, multi-head causal self-attention, MLP, residual, LM head
-- rather than a pre-built Transformer, so that TP sharding and PP owner changes
in later tasks can address individual parameters.

Two identity guarantees from plan section 3.1 are why this lives as more than a
bare ``nn.Module``:

* Every Transformer block carries a **stable global layer id** ``0..num_layers-1``.
* Every parameter has a **stable logical name** (e.g. ``layers.2.attn.q_proj.weight``)
  keyed by that global id, so a layer keeps its name after it migrates to a
  different PP stage or is resharded across a different TP degree.

Attention and MLP projections are bias-free, so one layer holds exactly
``12 * model_dim**2 + 4 * model_dim`` parameters -- the same per-layer figure the
single memory calculator in :mod:`resihp.memory` budgets: the four attention
projections give ``4 * model_dim**2``, the two MLP matrices (``model_dim`` up to
``4 * model_dim`` and back) give ``8 * model_dim**2``, and the two LayerNorms'
weight and bias give ``4 * model_dim``.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainConfig


# Plan section 一 fixes the run at FP32, and principle A compares the distributed run
# against the reference at the level of a few float32 ulps. On Ampere and later, CUDA
# answers ``matmul`` with TF32 -- ten mantissa bits -- whenever the precision setting
# allows it, which moves a single GEMM by ~1e-3 relative: the sharded and unsharded
# forms of the same math then disagree by reduced precision rather than by
# reassociation, and the comparison stops measuring anything. The setting is a
# process-wide default that has changed across torch versions, so the run pins it here
# rather than inheriting it. This module is imported by every process that builds a
# model -- reference or sharded -- which makes it the one place that covers them all.
torch.set_float32_matmul_precision("highest")

#: MLP hidden width as a multiple of ``model_dim``.
MLP_RATIO = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        batch, seq, dim = x.shape
        shape = (batch, seq, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(context)


class MLP(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, MLP_RATIO * model_dim, bias=False)
        self.fc2 = nn.Linear(MLP_RATIO * model_dim, model_dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``."""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ReferenceTransformer(nn.Module):
    """Decoder-only Transformer keyed by stable global layer ids."""

    def __init__(self, config: TrainConfig, *, vocab_size: int, sequence_length: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(sequence_length, config.model_dim)
        # ModuleDict keyed by the global layer id (as a string): the key is the
        # stable identity that survives PP/TP replanning, not a positional index.
        self.layers = nn.ModuleDict(
            {str(gid): TransformerBlock(config.model_dim, config.num_heads) for gid in range(config.num_layers)}
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, vocab_size, bias=False)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        # The ModuleDict keys are the authoritative global ids, sorted into
        # execution order; forward and every summary read them, never a range.
        return tuple(sorted(int(gid) for gid in self.layers))

    def forward(self, tokens):
        _, seq = tokens.shape
        if seq != self.sequence_length:
            raise ValueError("token sequence length does not match the model")
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for gid in self.layer_ids:
            x = self.layers[str(gid)](x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def logical_state_dict(self) -> dict[str, "torch.Tensor"]:
        """Parameters keyed by stable logical name (``layers.<gid>.<...>``)."""
        return {name: param for name, param in self.named_parameters()}
