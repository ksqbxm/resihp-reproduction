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

**Stage boundaries: scatter/gather.** A stage's activation (and the gradient of its
input) is replicated across its TP group, so exactly one authoritative copy has to cross
the boundary -- never a per-rank sum, which would double-count, and never a single rank
holding it, which would starve the others. Following ResiHP's P2P communication
optimization (paper Fig 7), that copy is *not* moved leader to leader: it is cut into
``N = max(TP_send, TP_recv)`` equal contiguous chunks, each chunk is P2P-sent on its own
``(sender, receiver)`` rank pair (:func:`scatter_routing`), and the receiving stage
rebuilds the whole tensor with a fast intra-node **all-gather** over its own TP group.
Every pair is distinct and carries ``1/N`` of the tensor, so the slow inter-stage link
still moves one copy in total -- now spread across ``N`` parallel links instead of one
leader link, and reassembled by a gather rather than a broadcast. That is also what
makes a boundary between stages of *different* TP degree work, and it is the project's
only boundary implementation.

The two places where a send and a receive must overlap -- ``send_forward +
recv_backward`` and ``send_backward + recv_forward`` in the steady state -- are issued
as one fused :func:`torch.distributed.batch_isend_irecv`. That is not an optimization:
each stage's transfers are ordered on its own stream, so issuing those two separately
deadlocks (a stage blocks on a send its peer cannot match until the peer's own blocked
send is matched). Fusing them lets both directions progress together -- a chunk's
forward activation and its gradient ride the same rank pair, so the fused batch pairs up
chunk for chunk. Because NCCL runs *batched* P2P on the group's own collective
communicator -- every rank of that group would then have to issue it, in the same order
-- every chunk of a hop rides one process group holding **the union of both stages'
ranks** (:func:`resihp.plan.boundary_hops` names them, the control plane builds them),
so one communicator serves the whole hop. The un-fused hops use the same group for the
same reason.

Splitting the batch into micro-batches reorders the FP32 loss reduction relative to the
reference's single full-batch pass, so results match the single-process reference within
``allclose`` -- the same tolerance the TP path adopted in T10 -- and exactly at one
micro-batch.
"""

import math

import torch
import torch.distributed as dist

from ..planner.pp import pipeline_phases
from ..reference import adamw, next_token_loss
from .dp import (
    ActivationLog,
    dp_combine_gradients,
    executor_route,
    global_micro_count,
    stage_pipeline,
)


def scatter_routing(up_members, down_members):
    """The ``(sender, receiver)`` rank pair carrying each chunk of one pipeline hop.

    ResiHP's P2P communication optimization. The boundary tensor is replicated inside
    each stage's TP group, so only one copy has to cross the link between the stages;
    it is cut into ``N = max(U, D)`` equal contiguous chunks and chunk ``k`` travels
    from ``up_members[k*U//N]`` to ``down_members[k*D//N]``.

    TP degrees are powers of two, so ``N`` is a multiple of both ``U`` and ``D``: every
    sender owns a contiguous run of ``N/U`` chunks, every receiver a contiguous run of
    ``N/D``, every member of both stages appears, and the ``N`` pairs are distinct. The
    cross-link volume is therefore still exactly one copy, now spread over ``N`` parallel
    links. The backward pass reuses this table with the roles reversed, so a chunk's
    activation and its gradient ride the same rank pair and the steady state's fused
    ``batch_isend_irecv`` still pairs up chunk for chunk.

    Pure: no tensor, no device, no process group.
    """
    up, down = tuple(up_members), tuple(down_members)
    chunks = max(len(up), len(down))
    return tuple(
        (up[index * len(up) // chunks], down[index * len(down) // chunks])
        for index in range(chunks)
    )


class PipelineRuntime:
    """One rank's 1F1B execution of one training iteration, driven by the assignment.

    ``stage`` is this rank's :class:`resihp.parallel.tp.TensorParallelStage`;
    ``assignment`` is the plan's :class:`~resihp.planner.dp.DPAssignment`, which names
    the executor ranks of every ``(micro_batch, stage)``. ``boundary_groups`` maps the
    sorted union of two adjacent stages' ranks to the process group every chunk of that
    hop rides on, and ``group`` is the group spanning every rank the plan places -- the
    DP gradient combine rides there. Ranks are **plan ranks** throughout, because that
    is what the assignment names: the stable identity a rank keeps for the whole run,
    not its index in the current world, which changes every time a fail-stop re-forms
    it. :meth:`_peer_rank` is the one place the two meet.

    :meth:`train_step` runs the schedule, combines gradients across DP replicas, and
    applies one AdamW WeightUpdate; :attr:`schedule` records the primitives it issued,
    in order, tagged with the micro-batch each acted on.
    """

    def __init__(self, stage, *, rank, replica_id, assignment, boundary_groups=None, group=None):
        self.stage = stage
        self.replica_id = int(replica_id)
        self.assignment = assignment
        self.group = group
        self.boundary_groups = dict(boundary_groups or {})
        self.rank = int(rank)
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
        # The scatter routing indexes the stage's executors in ascending order and the
        # gather reassembles the chunks in TP-rank order; the two are the same ordering
        # only because the TP group is built from those very executors. A mismatch would
        # reassemble the tensor wrongly or hang in the gather, so it is rejected here.
        if self.routes and len(self.executors) != stage.tp_size:
            raise ValueError("the stage's TP group disagrees with the assignment's executors")
        self.optimizer = adamw(stage.parameters())

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

    # --- Send / Recv primitives: the scatter/gather boundary ------------------------
    #
    # Every hop is named by the two stages it joins, upstream first, whichever side this
    # rank is on: :func:`scatter_routing` is then the one table both sides read, so the
    # forward chunk and its gradient always agree on which rank pair they belong to.

    def _hop(self, up_members, down_members):
        """This hop's members, and the union process group every one of its chunks rides.

        The member tuple is sorted, which is the order the group was built in, so a
        member's position in it is its rank *inside* that group.
        """
        key = tuple(sorted(set(up_members) | set(down_members)))
        return key, self.boundary_groups[key]

    def _peer_rank(self, hop, group, peer):
        """The torch rank a P2P op must name for the plan rank ``peer``.

        A fail-stop kills a process and the survivors re-form the world, so a rank's
        torch rank is only its index in the current membership while its plan rank is
        fixed. Going through the hop group -- position in the hop, then that group's
        own translation -- keeps this correct after any number of re-formations without
        the runtime having to know the membership at all.
        """
        return dist.get_global_rank(group, hop.index(peer))

    def _my_chunks(self, up_members, down_members):
        """``(chunk index, peer rank)`` for every chunk of this hop this rank carries.

        Adjacent stages own disjoint ranks, so this rank is the hop's sender or its
        receiver, never both, and the peer is whichever end it is not. Because the
        degrees are powers of two the indices are a contiguous run -- exactly this rank's
        ``1/tp_size`` share of the flat tensor, which is what :meth:`_reconstruct` needs.
        """
        sending = self.rank in up_members
        return tuple(
            (index, receiver if sending else sender)
            for index, (sender, receiver) in enumerate(
                scatter_routing(up_members, down_members)
            )
            if (sender if sending else receiver) == self.rank
        )

    def _exchange(self, ops) -> None:
        """Issue one hop's chunks as a single batch and wait for all of them.

        A chunk is a slice of the flattened activation, so ``ops`` holds the only
        reference keeping those buffers alive; it stays referenced for the whole call and
        every transfer is waited on before returning, so no chunk is dropped in flight.
        """
        ops = list(ops)
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def _scatter_send_ops(self, tensor, up_members, down_members):
        """Sends of this rank's chunks of ``tensor``, one per pair it takes part in.

        The chunk index is global to the hop, because this rank holds the whole
        replicated tensor and must put the receiver's own slice on the wire.
        """
        hop, group = self._hop(up_members, down_members)
        flat = tensor.detach().reshape(-1)
        width = flat.numel() // max(len(up_members), len(down_members))
        return [
            dist.P2POp(
                dist.isend,
                flat[index * width : (index + 1) * width].contiguous(),
                self._peer_rank(hop, group, peer),
                group,
            )
            for index, peer in self._my_chunks(up_members, down_members)
        ]

    def _gather_recv(self, shape, device, up_members, down_members):
        """A slab for this rank's chunks of the hop, and the receives that fill it.

        The chunks are a contiguous run, so filling the slab in order makes it precisely
        this rank's contiguous share of the flat tensor -- the shape
        :meth:`_reconstruct` all-gathers.
        """
        hop, group = self._hop(up_members, down_members)
        mine = self._my_chunks(up_members, down_members)
        width = math.prod(shape) // max(len(up_members), len(down_members))
        slab = torch.empty(len(mine) * width, device=device)
        ops = [
            dist.P2POp(
                dist.irecv,
                slab[slot * width : (slot + 1) * width],
                self._peer_rank(hop, group, peer),
                group,
            )
            for slot, (_index, peer) in enumerate(mine)
        ]
        return slab, ops

    def _reconstruct(self, slab, shape):
        """Rebuild the whole boundary tensor from this stage's slabs (intra-node gather).

        Each TP rank holds its own contiguous share and the group is built from the
        stage's executors -- the same ascending order the routing indexed them by -- so
        concatenating the gathered shares in TP-rank order restores the flat tensor
        exactly. At degree 1 the rank already holds all of it and no collective is issued.
        """
        if self.stage.tp_size == 1:
            return slab.view(shape)
        shares = [torch.empty_like(slab) for _ in range(self.stage.tp_size)]
        dist.all_gather(shares, slab, group=self.stage.group)
        return torch.cat(shares).view(shape)

    def _recv_forward(self, shape, device):
        if self.upstream is None:
            return None
        slab, ops = self._gather_recv(shape, device, self.upstream, self.executors)
        self._exchange(ops)
        return self._reconstruct(slab, shape).requires_grad_(True)

    def _send_forward(self, output) -> None:
        if self.downstream is not None:
            self._exchange(self._scatter_send_ops(output, self.executors, self.downstream))

    def _send_forward_recv_backward(self, output, shape, device):
        """Push this activation downstream while pulling the oldest gradient back."""
        if self.downstream is None:
            return None
        slab, receives = self._gather_recv(shape, device, self.executors, self.downstream)
        self._exchange(
            self._scatter_send_ops(output, self.executors, self.downstream) + receives
        )
        return self._reconstruct(slab, shape)

    def _recv_backward(self, shape, device):
        if self.downstream is None:
            return None
        slab, ops = self._gather_recv(shape, device, self.executors, self.downstream)
        self._exchange(ops)
        return self._reconstruct(slab, shape)

    def _send_backward(self, grad) -> None:
        if self.upstream is not None:
            self._exchange(self._scatter_send_ops(grad, self.upstream, self.executors))

    def _send_backward_recv_forward(self, grad, shape, device):
        """Return the oldest gradient upstream while pulling the next activation."""
        if self.upstream is None:
            return None
        slab, receives = self._gather_recv(shape, device, self.upstream, self.executors)
        self._exchange(
            self._scatter_send_ops(grad, self.upstream, self.executors) + receives
        )
        return self._reconstruct(slab, shape).requires_grad_(True)

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
                output = (
                    next_token_loss(
                        output, chunks[index].to(device), self.stage.vocab_size
                    )
                    / self.micro_total
                )
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
