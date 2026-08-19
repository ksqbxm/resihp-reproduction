"""Real sharded tensor parallelism with genuine TP all-reduce (T10).

Each process is one TP rank and holds only its shard of every weight -- Q/K/V and
MLP ``fc1`` sharded over the output (head) dimension, ``out_proj`` and ``fc2`` over
the input dimension, the token embedding and LM head over vocabulary -- while the
two LayerNorms and the position embedding are replicated. This is the shard
layout the memory model budgets (:mod:`resihp.memory`); each shard is a real leaf
parameter with its own gradient and AdamW moments, the state fail-stop recovery
gathers and reshards in T11+.

Execution is genuine tensor parallelism over a TP process group (NCCL on GPU,
Gloo on CPU), using the two Megatron collectives:

* ``f`` -- identity forward, **all-reduce backward** -- wraps the input of every
  column-parallel region (Q/K/V, ``fc1``) so the replicated input gradient is
  summed across the group.
* ``g`` -- **all-reduce forward**, identity backward -- reduces the partial output
  of every row-parallel region (``out_proj``, ``fc2``) and of the vocab-parallel
  token embedding into the full, replicated result.

The LM head is column-parallel over vocab; its shard logits are all-gathered into
the full logits (split on backward) so the loss is the ordinary cross-entropy.
Because the hidden state stays identically replicated across ranks (every ``g``
all-reduce and residual add produces the same bytes on each rank) the replicated
LayerNorm / position-embedding gradients are identical across ranks and stay in
sync without extra communication.

Numerical accuracy (plan principle A, relaxed to ``allclose``). Summing per-shard
partials reorders FP32 accumulation relative to the reference's single matmul, so
results match the single-process reference within floating-point tolerance rather
than bit-for-bit; at TP degree 1 every collective is a no-op and the match is
exact. No resharding lives here -- static TP only; degree/member changes land in
T11+.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..config import TrainConfig
from ..model import ReferenceTransformer


def _require_divisible(config: TrainConfig, vocab_size: int, tp_size: int) -> None:
    if config.model_dim % tp_size or config.num_heads % tp_size:
        raise ValueError("tp_size must divide model_dim and num_heads")
    if vocab_size % tp_size:
        raise ValueError("tp_size must divide vocab_size")


def _local_shard(weight: torch.Tensor, tp_rank: int, tp_size: int, dim: int) -> nn.Parameter:
    """This rank's contiguous shard of ``weight`` along ``dim``."""
    return nn.Parameter(torch.chunk(weight, tp_size, dim=dim)[tp_rank].detach().clone())


def _replicated(weight: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(weight.detach().clone())


class _CopyToRegion(torch.autograd.Function):
    """``f``: identity forward, all-reduce the gradient backward."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromRegion(torch.autograd.Function):
    """``g``: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _GatherLastDim(torch.autograd.Function):
    """All-gather shards along the last dim forward; take the local slice backward."""

    @staticmethod
    def forward(ctx, x, group, tp_rank, tp_size):
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        if tp_size == 1:
            return x
        gathered = [torch.empty_like(x) for _ in range(tp_size)]
        dist.all_gather(gathered, x.contiguous(), group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        if ctx.tp_size == 1:
            return grad, None, None, None
        local = grad.chunk(ctx.tp_size, dim=-1)[ctx.tp_rank]
        return local.contiguous(), None, None, None


class _TPBlock(nn.Module):
    """One Transformer block's local shards and replicated LayerNorms."""

    def __init__(self, state: dict, gid: int, tp_rank: int, tp_size: int):
        super().__init__()
        prefix = f"layers.{gid}"
        self.attn_norm_weight = _replicated(state[f"{prefix}.attn_norm.weight"])
        self.attn_norm_bias = _replicated(state[f"{prefix}.attn_norm.bias"])
        self.q_proj = _local_shard(state[f"{prefix}.attn.q_proj.weight"], tp_rank, tp_size, 0)
        self.k_proj = _local_shard(state[f"{prefix}.attn.k_proj.weight"], tp_rank, tp_size, 0)
        self.v_proj = _local_shard(state[f"{prefix}.attn.v_proj.weight"], tp_rank, tp_size, 0)
        self.out_proj = _local_shard(state[f"{prefix}.attn.out_proj.weight"], tp_rank, tp_size, 1)
        self.mlp_norm_weight = _replicated(state[f"{prefix}.mlp_norm.weight"])
        self.mlp_norm_bias = _replicated(state[f"{prefix}.mlp_norm.bias"])
        self.fc1 = _local_shard(state[f"{prefix}.mlp.fc1.weight"], tp_rank, tp_size, 0)
        self.fc2 = _local_shard(state[f"{prefix}.mlp.fc2.weight"], tp_rank, tp_size, 1)

    def local_shards(self, gid: int) -> dict[str, tuple[nn.Parameter, int | None]]:
        """Map logical name -> (local parameter, shard dim); ``None`` = replicated."""
        prefix = f"layers.{gid}"
        return {
            f"{prefix}.attn_norm.weight": (self.attn_norm_weight, None),
            f"{prefix}.attn_norm.bias": (self.attn_norm_bias, None),
            f"{prefix}.attn.q_proj.weight": (self.q_proj, 0),
            f"{prefix}.attn.k_proj.weight": (self.k_proj, 0),
            f"{prefix}.attn.v_proj.weight": (self.v_proj, 0),
            f"{prefix}.attn.out_proj.weight": (self.out_proj, 1),
            f"{prefix}.mlp_norm.weight": (self.mlp_norm_weight, None),
            f"{prefix}.mlp_norm.bias": (self.mlp_norm_bias, None),
            f"{prefix}.mlp.fc1.weight": (self.fc1, 0),
            f"{prefix}.mlp.fc2.weight": (self.fc2, 1),
        }


class TensorParallelTransformer(nn.Module):
    """Decoder-only Transformer executed as real tensor parallelism over a TP group.

    ``group`` is the TP process group (``None`` -> the default world group); the
    degree and this rank's index come from it, so the module works for a whole-world
    TP run or a TP subgroup of a larger 3D layout. Pass ``source_state`` (a
    reference's ``logical_state_dict``) to shard a known init; with none, the same
    fixed init as the reference is rebuilt from ``config.seed`` and this rank keeps
    only its shard.
    """

    def __init__(
        self,
        config: TrainConfig,
        *,
        vocab_size: int,
        sequence_length: int,
        group=None,
        source_state: dict | None = None,
    ):
        super().__init__()
        self.group = group
        self.tp_rank = dist.get_rank(group)
        self.tp_size = dist.get_world_size(group)
        _require_divisible(config, vocab_size, self.tp_size)
        if source_state is None:
            torch.manual_seed(config.seed)
            source_state = ReferenceTransformer(
                config, vocab_size=vocab_size, sequence_length=sequence_length
            ).logical_state_dict()

        self.config = config
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_heads = config.num_heads
        self.dim = config.model_dim
        self.vocab_per_rank = vocab_size // self.tp_size
        self.vocab_start = self.tp_rank * self.vocab_per_rank

        self.token_embedding = _local_shard(source_state["token_embedding.weight"], self.tp_rank, self.tp_size, 0)
        self.position_embedding = _replicated(source_state["position_embedding.weight"])
        self.blocks = nn.ModuleDict(
            {str(gid): _TPBlock(source_state, gid, self.tp_rank, self.tp_size) for gid in range(config.num_layers)}
        )
        self.final_norm_weight = _replicated(source_state["final_norm.weight"])
        self.final_norm_bias = _replicated(source_state["final_norm.bias"])
        self.lm_head = _local_shard(source_state["lm_head.weight"], self.tp_rank, self.tp_size, 0)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(gid) for gid in self.blocks))

    def local_shards(self) -> dict[str, tuple[nn.Parameter, int | None]]:
        """Every local parameter keyed by logical name -> (param, shard dim | None)."""
        shards = {
            "token_embedding.weight": (self.token_embedding, 0),
            "position_embedding.weight": (self.position_embedding, None),
            "final_norm.weight": (self.final_norm_weight, None),
            "final_norm.bias": (self.final_norm_bias, None),
            "lm_head.weight": (self.lm_head, 0),
        }
        for gid in self.layer_ids:
            shards.update(self.blocks[str(gid)].local_shards(gid))
        return shards

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # Vocab-parallel embedding: look up only owned rows, zero the rest, then
        # all-reduce (``g``) into the full replicated embedding.
        outside = (tokens < self.vocab_start) | (tokens >= self.vocab_start + self.vocab_per_rank)
        local_ids = (tokens - self.vocab_start).clamp(0, self.vocab_per_rank - 1)
        local = F.embedding(local_ids, self.token_embedding).masked_fill(outside.unsqueeze(-1), 0.0)
        return _ReduceFromRegion.apply(local, self.group)

    def _attention(self, block: _TPBlock, x: torch.Tensor) -> torch.Tensor:
        local_heads = self.num_heads // self.tp_size
        head_dim = self.dim // self.num_heads
        batch, seq, _ = x.shape
        shape = (batch, seq, local_heads, head_dim)
        q = F.linear(x, block.q_proj).view(shape).transpose(1, 2)
        k = F.linear(x, block.k_proj).view(shape).transpose(1, 2)
        v = F.linear(x, block.v_proj).view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, seq, local_heads * head_dim)
        partial = F.linear(context, block.out_proj)  # row-parallel partial
        return _ReduceFromRegion.apply(partial, self.group)

    def _mlp(self, block: _TPBlock, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(F.linear(x, block.fc1))
        partial = F.linear(hidden, block.fc2)  # row-parallel partial
        return _ReduceFromRegion.apply(partial, self.group)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq = tokens.shape
        if seq != self.sequence_length:
            raise ValueError("token sequence length does not match the model")
        positions = torch.arange(seq, device=tokens.device)
        x = self._embed(tokens) + F.embedding(positions, self.position_embedding)
        for gid in self.layer_ids:
            block = self.blocks[str(gid)]
            normed = F.layer_norm(x, (self.dim,), block.attn_norm_weight, block.attn_norm_bias)
            x = x + self._attention(block, _CopyToRegion.apply(normed, self.group))
            normed = F.layer_norm(x, (self.dim,), block.mlp_norm_weight, block.mlp_norm_bias)
            x = x + self._mlp(block, _CopyToRegion.apply(normed, self.group))
        x = F.layer_norm(x, (self.dim,), self.final_norm_weight, self.final_norm_bias)
        # LM head is column-parallel over vocab, so like every column-parallel
        # region its input passes through ``f``: the per-rank partials of the input
        # gradient are all-reduced back into the full gradient of the norm output.
        local_logits = F.linear(_CopyToRegion.apply(x, self.group), self.lm_head)
        return _GatherLastDim.apply(local_logits, self.group, self.tp_rank, self.tp_size)
