"""Pipeline-parallel runtime (1F1B) and layer placement / migration planning (T12).

Plan section 3.4. The model is cut into contiguous PP stages, one process per
stage; a stage holds only its own layers (the first stage also the embeddings, the
last stage the final LayerNorm and LM head), so no stage stores the whole model.

:class:`PipelineRuntime` executes one training iteration as a genuine **1F1B**
schedule over a PP process group, emitting the plan's Forward / Backward / Send /
Recv / WeightUpdate primitives: warmup forwards, then one-forward-one-backward,
then cooldown backwards, micro-batch gradients accumulating into a single AdamW
update. Activations travel forward and gradients backward as real point-to-point
transfers (NCCL on GPU, Gloo on CPU).

The two places where a send and a receive must overlap -- ``send_forward +
recv_backward`` and ``send_backward + recv_forward`` in the steady state -- are
issued as one fused :func:`torch.distributed.batch_isend_irecv`. That is not an
optimization: on NCCL each stage's transfers are ordered on its stream, so issuing
those two separately deadlocks (a stage blocks on a send its peer cannot match
until the peer's own blocked send is matched). Fusing them lets both directions
progress together.

Splitting the batch into micro-batches reorders the FP32 loss reduction relative to
the reference's single full-batch pass, so results match the single-process
reference within ``allclose`` -- the same tolerance the TP path adopted in T10 --
and exactly at one micro-batch.

Layer migration reuses the single recovery path rather than adding a second
mechanism. :func:`plan_migration` composes the T4 repartition (which layer lands on
which stage) with the TP degree on each side, giving every layer its old and new
``(owner, degree)``; the tensors themselves move through the T11 reshard
(:func:`resihp.parallel.reshard.reshard_tp_state`) -- collect the full logical
tensor from healthy peers, verify against the checkpoint, re-chunk to the new
layout. A layer needs work when it changes owner (it moves) **or** when its degree
changes (it reshards) -- the second case includes a layer that stays put on a stage
that lost a TP rank, which a moved-layers-only view would silently leave in the old
layout. A receiving stage of a different degree is handed the target layout
directly; no old-layout compatibility is kept.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..config import TrainConfig
from ..model import TransformerBlock
from ..planner.pp import repartition_pp
from ..reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY
from .reshard import shard_dims


def balanced_layers(num_layers: int, num_stages: int) -> tuple[tuple[int, ...], ...]:
    """Contiguous global layer ids per stage, the remainder going to earlier stages.

    The base partition a run starts from, before any fail-stop repartition.
    """
    if num_stages < 1 or num_layers < num_stages:
        raise ValueError("need at least one layer per stage")
    base, remainder = divmod(num_layers, num_stages)
    stages = []
    start = 0
    for stage in range(num_stages):
        count = base + (1 if stage < remainder else 0)
        stages.append(tuple(range(start, start + count)))
        start += count
    return tuple(stages)


class PipelineStage(nn.Module):
    """One PP stage: a contiguous subset of layers plus the boundary modules.

    Submodules are named exactly as the reference, so ``named_parameters`` yields
    the stable logical names (``layers.<gid>.attn.q_proj.weight``) and the stage is
    loaded from a reference ``logical_state_dict`` by those names -- a layer keeps
    its name after moving here from another stage. The first stage owns the
    token/position embeddings and the last stage the final norm and LM head, which
    is plan 3.4's "embedding / LM head belong to the first / last executable stage".
    """

    def __init__(
        self,
        config: TrainConfig,
        *,
        vocab_size: int,
        sequence_length: int,
        layer_ids,
        is_first: bool,
        is_last: bool,
        source_state: dict,
    ):
        super().__init__()
        self.layer_ids = tuple(sorted(int(gid) for gid in layer_ids))
        self.is_first = is_first
        self.is_last = is_last
        self.dim = config.model_dim
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        if is_first:
            self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
            self.position_embedding = nn.Embedding(sequence_length, config.model_dim)
        self.layers = nn.ModuleDict(
            {str(gid): TransformerBlock(config.model_dim, config.num_heads) for gid in self.layer_ids}
        )
        if is_last:
            self.final_norm = nn.LayerNorm(config.model_dim)
            self.lm_head = nn.Linear(config.model_dim, vocab_size, bias=False)
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(source_state[name])

    def forward(self, *, tokens=None, hidden=None):
        if self.is_first:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = self.token_embedding(tokens) + self.position_embedding(positions)
        else:
            x = hidden
        for gid in self.layer_ids:
            x = self.layers[str(gid)](x)
        if self.is_last:
            x = self.lm_head(self.final_norm(x))
        return x

    def logical_state_dict(self) -> dict:
        """Owned parameters keyed by their stable logical name."""
        return {name: param for name, param in self.named_parameters()}


class PipelineRuntime:
    """1F1B execution of one training iteration over a linear pipeline of stages.

    ``stage_ranks`` are the **global** ranks of the stages in pipeline order, so
    this rank's neighbours are its adjacent entries; ``group`` is the process group
    the transfers ride on. :meth:`train_step` runs the schedule and applies one
    AdamW WeightUpdate; :attr:`schedule` records the forward/backward primitives it
    issued, in order.
    """

    def __init__(self, stage: PipelineStage, *, stage_ranks, num_micro_batches: int, group=None):
        self.stage = stage
        self.group = group
        self.stage_ranks = tuple(int(rank) for rank in stage_ranks)
        self.num_stages = len(self.stage_ranks)
        self.index = self.stage_ranks.index(dist.get_rank())
        self.is_first = self.index == 0
        self.is_last = self.index == self.num_stages - 1
        self.prev_rank = None if self.is_first else self.stage_ranks[self.index - 1]
        self.next_rank = None if self.is_last else self.stage_ranks[self.index + 1]
        self.micro = num_micro_batches
        self.schedule: list[str] = []
        self.optimizer = torch.optim.AdamW(
            stage.parameters(),
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    def train_step(self, tokens):
        """Run one 1F1B iteration over ``tokens``; return the loss on the last stage."""
        if tokens.shape[0] % self.micro:
            raise ValueError("batch size must be divisible by the micro-batch count")
        device = next(self.stage.parameters()).device
        chunks = tokens.chunk(self.micro, dim=0)
        shape = (chunks[0].shape[0], self.stage.sequence_length, self.stage.dim)
        self.optimizer.zero_grad(set_to_none=True)
        self.schedule = []

        # --- Send / Recv primitives -------------------------------------------------
        def exchange(*ops):
            # ``ops`` stays referenced for the whole call, keeping every send buffer
            # alive until its transfer has completed.
            for work in dist.batch_isend_irecv(list(ops)):
                work.wait()

        def send_op(tensor, peer):
            return dist.P2POp(dist.isend, tensor.contiguous(), peer, self.group)

        def recv_op(buffer, peer):
            return dist.P2POp(dist.irecv, buffer, peer, self.group)

        def buffer():
            return torch.empty(shape, device=device)

        def recv_forward():
            if self.is_first:
                return None
            received = buffer()
            exchange(recv_op(received, self.prev_rank))
            return received.requires_grad_(True)

        def send_forward(output):
            if not self.is_last:
                exchange(send_op(output.detach(), self.next_rank))

        def send_forward_recv_backward(output):
            """Push this activation downstream while pulling the oldest gradient back."""
            if self.is_last:
                return None
            grad = buffer()
            exchange(send_op(output.detach(), self.next_rank), recv_op(grad, self.next_rank))
            return grad

        def recv_backward():
            if self.is_last:
                return None
            grad = buffer()
            exchange(recv_op(grad, self.next_rank))
            return grad

        def send_backward(grad):
            if not self.is_first:
                exchange(send_op(grad, self.prev_rank))

        def send_backward_recv_forward(grad):
            """Return the oldest gradient upstream while pulling the next activation."""
            if self.is_first:
                return None
            received = buffer()
            exchange(send_op(grad, self.prev_rank), recv_op(received, self.prev_rank))
            return received.requires_grad_(True)

        # --- Forward / Backward primitives ------------------------------------------
        inputs: list = []   # received activation leaf per in-flight micro-batch
        outputs: list = []  # its forward output (the scaled loss on the last stage)
        total_loss = torch.zeros((), device=device)
        forward_count = 0

        def forward(received):
            nonlocal forward_count
            index = forward_count
            forward_count += 1
            self.schedule.append(f"F{index}")
            output = self.stage(tokens=chunks[index]) if self.is_first else self.stage(hidden=received)
            if self.is_last:
                # Scale by the micro-batch count so the accumulated gradient is the
                # full-batch mean, matching the reference's single-pass loss.
                output = F.cross_entropy(
                    output[:, :-1].reshape(-1, self.stage.vocab_size),
                    chunks[index][:, 1:].reshape(-1),
                ) / self.micro
                total_loss.add_(output.detach())
            inputs.append(received)
            outputs.append(output)
            return output

        def backward(grad_output):
            # The oldest in-flight micro-batch is the one being retired.
            self.schedule.append(f"B{forward_count - len(outputs)}")
            received = inputs.pop(0)
            output = outputs.pop(0)
            if self.is_last:
                output.backward()
            else:
                output.backward(grad_output)
            return None if self.is_first else received.grad

        # --- 1F1B schedule ----------------------------------------------------------
        warmup = min(self.num_stages - 1 - self.index, self.micro)
        steady = self.micro - warmup

        for _ in range(warmup):
            send_forward(forward(recv_forward()))

        received = recv_forward() if steady else None
        for iteration in range(steady):
            grad_output = send_forward_recv_backward(forward(received))
            grad_input = backward(grad_output)
            if iteration == steady - 1:
                send_backward(grad_input)
            else:
                received = send_backward_recv_forward(grad_input)

        for _ in range(warmup):
            send_backward(backward(recv_backward()))

        self.schedule.append("W")
        self.optimizer.step()
        return float(total_loss) if self.is_last else None


@dataclass(frozen=True)
class LayerPlacement:
    """Where one global layer lives before and after a repartition, and at what degree."""

    layer: int
    old_owner: int
    new_owner: int
    old_degree: int
    new_degree: int

    @property
    def moved(self) -> bool:
        """True when the layer changes PP stage, so its state must be transferred."""
        return self.new_owner != self.old_owner

    @property
    def resharded(self) -> bool:
        """True when the TP degree changes, so its state must be re-chunked.

        Independent of :attr:`moved`: a layer that stays on a stage which lost a TP
        rank still has to be resharded to the stage's new degree.
        """
        return self.new_degree != self.old_degree


def _owners(stage_layers) -> list[int]:
    """Global layer id -> owning stage, from contiguous per-stage layer counts."""
    return [stage for stage, count in enumerate(stage_layers) for _ in range(count)]


def plan_migration(old_stage_layers, old_tp_degrees, new_tp_degrees):
    """Repartition the pipeline and describe where every layer's state must end up.

    Returns ``(plan, placements)``: ``plan`` is the T4 :class:`~resihp.planner.pp.PPPlan`
    (new layer ranges and the embedding / LM-head owners); ``placements`` covers
    **every** global layer with its old and new owner and TP degree, so the caller
    can hand each layer needing work straight to
    :func:`resihp.parallel.reshard.reshard_tp_state` with the right old and new
    degree. Layers that neither moved nor changed degree stay where they are.
    """
    plan = repartition_pp(old_stage_layers, old_tp_degrees, new_tp_degrees)
    old_owner = _owners(old_stage_layers)
    new_owner = _owners(plan.stage_layers)
    placements = tuple(
        LayerPlacement(
            layer=layer,
            old_owner=old_owner[layer],
            new_owner=new_owner[layer],
            old_degree=old_tp_degrees[old_owner[layer]],
            new_degree=new_tp_degrees[new_owner[layer]],
        )
        for layer in range(len(old_owner))
    )
    return plan, placements


def reshard_layout(placements, *, new_owner: int) -> dict[str, int | None]:
    """Shard layout of the layers ``new_owner`` must (re)acquire state for.

    The layout to pass to :func:`resihp.parallel.reshard.reshard_tp_state`. It
    covers every layer the stage ends up owning that changed stage **or** changed
    degree -- not just the arriving ones: a layer that stays put on a stage which
    lost a TP rank still has to be re-chunked, and selecting only moved layers would
    leave that stage's state in the old layout. Layers that changed neither already
    hold their state in the right shape and are omitted.

    One call covers one ``old_degree -> new_degree`` pair, since ``reshard_tp_state``
    takes a single ``old_size``; group the placements by :attr:`LayerPlacement.old_degree`
    when a stage receives layers from sources of differing degree.

    ``shard_dims`` also carries the embedding / LM-head entries, which belong to the
    first / last stage rather than to any layer, so only per-layer names are kept.
    """
    layers = sorted(
        place.layer
        for place in placements
        if place.new_owner == new_owner and (place.moved or place.resharded)
    )
    return {name: dim for name, dim in shard_dims(layers).items() if name.startswith("layers.")}
