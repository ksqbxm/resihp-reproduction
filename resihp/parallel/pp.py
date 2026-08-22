"""The single pipeline runtime: assignment-driven 1F1B execution (T12 + T13).

Plan sections 3.4 and 3.5. The model is cut into contiguous PP stages; a stage holds
only its own layers (the replica's first stage also the embeddings, its last stage the
final LayerNorm and LM head), so no stage stores the whole model. The stage itself is
:class:`resihp.parallel.tp.TensorParallelStage` -- there is one stage class for the
whole project, so PP ownership and TP layout are always expressed together and a
TP-degree-1 pipeline is just that class over a one-rank group.

:class:`PipelineRuntime` is the **only** runtime in the project: the control plane
drives it and the tests exercise the same class. It executes one training iteration as
a genuine **1F1B** schedule -- warmup forwards, then one-forward-one-backward, then
cooldown backwards -- emitting the plan's Forward / Backward / Send / Recv /
WeightUpdate primitives, with micro-batch gradients accumulating into a single AdamW
update. The order comes from :func:`resihp.planner.pp.pipeline_phases`, the same
definition the analytical memory model sizes peak activations from.

Everything topological is read from the current :class:`~resihp.plan.ExecutionPlan`,
never inferred from an older layout: which micro-batches this rank runs, which stage it
is, how many stages its pipeline has, and which ranks its neighbours are all come from
the plan's :class:`~resihp.planner.dp.DPAssignment`.

**Stage boundaries.** A stage's activation (and the gradient of its input) is
replicated across its TP group, so a boundary moves exactly one authoritative copy
between the two stages' leaders and the receiving group replicates it -- never a
per-rank sum, which would double-count, and never a single rank holding it, which would
starve the others. That is what makes a boundary between stages of *different* TP degree
work, and it is the project's only heterogeneous-boundary implementation.

The two places where a send and a receive must overlap -- ``send_forward +
recv_backward`` and ``send_backward + recv_forward`` in the steady state -- are issued
as one fused :func:`torch.distributed.batch_isend_irecv`. That is not an optimization:
each stage's transfers are ordered on its own stream, so issuing those two separately
deadlocks (a stage blocks on a send its peer cannot match until the peer's own blocked
send is matched). Fusing them lets both directions progress together. Because NCCL runs
*batched* P2P on the group's own collective communicator -- every rank of that group
would then have to issue it, in the same order -- each hop rides a process group holding
**exactly its two leaders** (:func:`resihp.plan.boundary_pairs` names them, the control
plane builds them). The un-fused hops use the same group for the same reason.

Splitting the batch into micro-batches reorders the FP32 loss reduction relative to the
reference's single full-batch pass, so results match the single-process reference within
``allclose`` -- the same tolerance the TP path adopted in T10 -- and exactly at one
micro-batch.
"""

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ..planner.pp import pipeline_phases
from ..reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY
from .dp import (
    ActivationLog,
    dp_combine_gradients,
    executor_route,
    global_micro_count,
    stage_pipeline,
)


class PipelineRuntime:
    """One rank's 1F1B execution of one training iteration, driven by the assignment.

    ``stage`` is this rank's :class:`resihp.parallel.tp.TensorParallelStage`;
    ``assignment`` is the plan's :class:`~resihp.planner.dp.DPAssignment`, which names
    the executor ranks of every ``(micro_batch, stage)``. ``boundary_groups`` maps a
    sorted leader pair to the two-rank process group its hop rides on, and ``group`` is
    the group spanning every rank the plan places -- the DP gradient combine rides
    there. Ranks are **global** throughout, because that is what the assignment names.

    :meth:`train_step` runs the schedule, combines gradients across DP replicas, and
    applies one AdamW WeightUpdate; :attr:`schedule` records the primitives it issued,
    in order, tagged with the micro-batch each acted on.
    """

    def __init__(self, stage, *, replica_id, assignment, boundary_groups=None, group=None):
        self.stage = stage
        self.replica_id = int(replica_id)
        self.assignment = assignment
        self.group = group
        self.boundary_groups = dict(boundary_groups or {})
        self.rank = dist.get_rank()
        self.routes = executor_route(assignment, self.rank)
        self.micro_batches = [route["micro_batch"] for route in self.routes]
        self.micro_total = global_micro_count(assignment)
        self.schedule: list[str] = []
        self.activation_log = ActivationLog()
        self.stage_index, self.num_stages = self._position()
        self.executors, self.upstream, self.downstream = self._neighbours()
        # The stage's ends come from the plan and its neighbours from the assignment;
        # the plan invariants make them agree. A mismatch would not deadlock -- it would
        # silently feed the first stage a received activation instead of its tokens, or
        # drop one -- so it is rejected here rather than left to show up as numerics.
        if self.routes and (
            (self.upstream is None) != stage.is_first
            or (self.downstream is None) != stage.is_last
        ):
            raise ValueError("the stage's pipeline ends disagree with the assignment")
        self.optimizer = torch.optim.AdamW(
            stage.parameters(),
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    # --- reading the assignment (the only authority on who talks to whom) ----------

    def _position(self) -> tuple[int, int]:
        """This rank's index in its pipeline, and that pipeline's stage count.

        Both are read from the assignment. Every micro-batch this rank runs must place
        it at the same index of an equally long pipeline -- the plan invariant that a
        micro-batch runs its replica's stages exactly once already guarantees it -- so
        one 1F1B schedule covers the whole iteration.
        """
        positions = set()
        for micro_batch in self.micro_batches:
            pipeline = stage_pipeline(self.assignment, micro_batch)
            index = next(
                order
                for order, placement in enumerate(pipeline)
                if self.rank in placement.executor_ranks
            )
            positions.add((index, len(pipeline)))
        if len(positions) > 1:
            raise ValueError("assignment puts this rank at differing pipeline positions")
        return positions.pop() if positions else (0, 1)

    def _neighbours(self):
        """This rank's stage executors, and its upstream / downstream executors.

        The steady state fuses a forward send with a backward receive into one transfer
        on one hop, so a rank's neighbours must be the same for every micro-batch it
        runs. The plan invariant that a micro-batch never splits across replicas is
        exactly that guarantee; a hand-built assignment that breaks it is rejected here
        rather than deadlocking later.
        """
        seen = {
            (route["executors"], route["upstream"], route["downstream"])
            for route in self.routes
        }
        if len(seen) > 1:
            raise ValueError("assignment gives this rank differing neighbours across micro-batches")
        return seen.pop() if seen else ((self.rank,), None, None)

    def _hop(self, peer_leader: int):
        """The two-rank process group this rank's hop to ``peer_leader`` rides on."""
        return self.boundary_groups[(min(self.rank, peer_leader), max(self.rank, peer_leader))]

    # --- Send / Recv primitives ----------------------------------------------------

    def _exchange(self, ops) -> None:
        # ``ops`` stays referenced for the whole call, keeping every send buffer alive
        # until its transfer has completed.
        for work in dist.batch_isend_irecv(list(ops)):
            work.wait()

    def _send_op(self, tensor, peer):
        return dist.P2POp(dist.isend, tensor.contiguous(), peer, self._hop(peer))

    def _recv_op(self, buffer, peer):
        return dist.P2POp(dist.irecv, buffer, peer, self._hop(peer))

    def _replicate(self, buffer):
        """Give every TP rank of this stage the copy its leader moved across the hop."""
        if self.stage.tp_size > 1:
            dist.broadcast(buffer, src=self.executors[0], group=self.stage.group)
        return buffer

    @property
    def _is_leader(self) -> bool:
        return self.rank == self.executors[0]

    def _recv_forward(self, shape, device):
        if self.upstream is None:
            return None
        received = torch.empty(shape, device=device)
        if self._is_leader:
            self._exchange([self._recv_op(received, self.upstream[0])])
        return self._replicate(received).requires_grad_(True)

    def _send_forward(self, output) -> None:
        if self.downstream is not None and self._is_leader:
            self._exchange([self._send_op(output.detach(), self.downstream[0])])

    def _send_forward_recv_backward(self, output, shape, device):
        """Push this activation downstream while pulling the oldest gradient back."""
        if self.downstream is None:
            return None
        grad = torch.empty(shape, device=device)
        if self._is_leader:
            peer = self.downstream[0]
            self._exchange([self._send_op(output.detach(), peer), self._recv_op(grad, peer)])
        return self._replicate(grad)

    def _recv_backward(self, shape, device):
        if self.downstream is None:
            return None
        grad = torch.empty(shape, device=device)
        if self._is_leader:
            self._exchange([self._recv_op(grad, self.downstream[0])])
        return self._replicate(grad)

    def _send_backward(self, grad) -> None:
        if self.upstream is not None and self._is_leader:
            self._exchange([self._send_op(grad, self.upstream[0])])

    def _send_backward_recv_forward(self, grad, shape, device):
        """Return the oldest gradient upstream while pulling the next activation."""
        if self.upstream is None:
            return None
        received = torch.empty(shape, device=device)
        if self._is_leader:
            peer = self.upstream[0]
            self._exchange([self._send_op(grad, peer), self._recv_op(received, peer)])
        return self._replicate(received).requires_grad_(True)

    # --- the iteration -------------------------------------------------------------

    def train_step(self, tokens):
        """Run this rank's 1F1B schedule over ``tokens``; the loss on the last stage.

        ``tokens`` is the full global batch, and a micro-batch's slice is indexed by its
        **global** micro-batch id, so which executor runs it cannot change which data it
        is. Every micro-batch's loss is scaled by the global micro-batch count, so the
        accumulated gradient is the full-batch mean however the executors are
        distributed -- rerouting cannot change the normalization.
        """
        if tokens.shape[0] % self.micro_total:
            raise ValueError("batch size must be divisible by the micro-batch count")
        device = next(self.stage.parameters()).device
        chunks = tokens.chunk(self.micro_total, dim=0)
        shape = (chunks[0].shape[0], self.stage.sequence_length, self.stage.dim)
        self.optimizer.zero_grad(set_to_none=True)
        self.schedule = []
        self.activation_log = ActivationLog()

        inputs: list = []     # received activation leaf per in-flight micro-batch
        outputs: list = []    # its forward output (the scaled loss on the last stage)
        held: list[int] = []  # the micro-batch id each of those belongs to
        total_loss = torch.zeros((), device=device)
        forward_count = 0

        def forward(received):
            nonlocal forward_count
            index = self.micro_batches[forward_count]
            forward_count += 1
            self.schedule.append(f"F{index}")
            output = (
                self.stage(tokens=chunks[index].to(device))
                if self.stage.is_first
                else self.stage(hidden=received)
            )
            if self.stage.is_last:
                # Scale by the global micro-batch count so the accumulated gradient is
                # the full-batch mean, matching the reference's single-pass loss.
                output = F.cross_entropy(
                    output[:, :-1].reshape(-1, self.stage.vocab_size),
                    chunks[index][:, 1:].reshape(-1).to(device),
                ) / self.micro_total
                total_loss.add_(output.detach())
            # An activation is counted in memory from here until its backward retires it
            # (plan 3.5), which is what makes the peak the 1F1B in-flight count.
            self.activation_log.retain(index)
            inputs.append(received)
            outputs.append(output)
            held.append(index)
            return output

        def backward(grad_output):
            # The oldest in-flight micro-batch is the one being retired.
            index = held.pop(0)
            self.schedule.append(f"B{index}")
            received = inputs.pop(0)
            output = outputs.pop(0)
            if self.stage.is_last:
                output.backward()
            else:
                output.backward(grad_output)
            self.activation_log.release(index)
            return None if self.stage.is_first else received.grad

        warmup, steady = pipeline_phases(
            len(self.micro_batches), stage_index=self.stage_index, num_stages=self.num_stages
        )
        for _ in range(warmup):
            self._send_forward(forward(self._recv_forward(shape, device)))

        received = self._recv_forward(shape, device) if steady else None
        for iteration in range(steady):
            grad_output = self._send_forward_recv_backward(forward(received), shape, device)
            grad_input = backward(grad_output)
            if iteration == steady - 1:
                self._send_backward(grad_input)
            else:
                received = self._send_backward_recv_forward(grad_input, shape, device)

        for _ in range(warmup):
            self._send_backward(backward(self._recv_backward(shape, device)))

        self.schedule.append("W")
        self._combine_and_step(device)
        return float(total_loss) if self.stage.is_last else None

    def _combine_and_step(self, device) -> None:
        """Sum each logical gradient across replicas, then apply one AdamW update.

        The shard metadata is the stage's own, so a replica running a different TP degree
        (or a different PP layering) still combines through the one
        reconstruct-sum-rechunk path. Every rank of ``group`` joins, including one the
        assignment left with no micro-batch to run, because the combine is collective.
        """
        shards = self.stage.local_shards()
        local = {
            name: {
                "grad": param.grad if param.grad is not None else torch.zeros_like(param),
                "shard_dim": shard_dim,
                "shard_index": self.stage.tp_rank,
                "old_size": self.stage.tp_size,
                "replica": self.replica_id,
            }
            for name, (param, shard_dim) in shards.items()
        }
        combined = dp_combine_gradients(local, group=self.group)
        with torch.no_grad():
            for name, (param, _dim) in shards.items():
                param.grad = combined[name].to(device)
        self.optimizer.step()
