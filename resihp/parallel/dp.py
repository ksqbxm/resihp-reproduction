"""Data-parallel cross-replica execution driven by the T5 assignment (T13).

Plan section 3.5's runtime guarantees. The T5 planner (:mod:`resihp.planner.dp`)
produces a :class:`~resihp.planner.dp.DPAssignment`: every ``(micro_batch, stage)``
names the executor ranks that must run it. T13 turns that assignment into real
execution -- and does so **by reading the assignment**, never by inferring peers
from a fixed topology (plan 3.2):

* a micro-batch's forward activation is sent to the *actual* executor of its next
  stage, and its backward gradient returned to the *actual* executor of its previous
  stage -- so a micro-batch whose stages were rerouted onto different replicas
  crosses replicas transparently;
* an activation stays alive (counted in memory) from its forward until the matching
  backward retires it -- exposed as :class:`ActivationLog` for the lifecycle gate;
* a micro-batch belongs to exactly one executor per stage, so no workload is owned
  by source and target at once;
* every micro-batch's loss is scaled by the **global** micro-batch count, so the
  accumulated gradient is the full-batch mean no matter how the executors are
  distributed -- rerouting cannot change the normalization;
* replicas may differ in PP layering and TP degree: the DP gradient combine works on
  *full logical* tensors (reconstruct each replica's tensor, sum across replicas,
  re-chunk to each replica's own layout), reusing the single T11 reshard path rather
  than a second all-reduce, so heterogeneous replicas combine uniformly.

Only the DP dimension lives here. Sharded TP forward/backward is T10
(:class:`resihp.parallel.tp.TensorParallelStage`) and the 1F1B schedule is T12
(:class:`resihp.parallel.pp.PipelineRuntime`); :func:`dp_combine_gradients` is written
generally so a TP-degree-heterogeneous DP combine reuses it directly. Micro-batching
reorders the FP32 reduction, so results match the single-process reference within the
same ``allclose`` tolerance T10 fixed, and exactly at one micro-batch.
"""

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ..reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY
from .reshard import local_slice, reconstruct_full


def stage_pipeline(assignment, micro_batch):
    """The placements of one micro-batch in stage order (its executor pipeline)."""
    placements = assignment.by_micro_batch[micro_batch]
    return tuple(sorted(placements, key=lambda placement: placement.stage_id))


def executor_route(assignment, rank):
    """For ``rank``, the micro-batches it runs at each stage and their neighbours.

    Pure lookup used by the runtime and unit-tested on its own: for every
    ``(micro_batch, stage)`` this rank executes, it gives the upstream and downstream
    executor ranks taken straight from the assignment -- the neighbour may sit in a
    different replica, which is exactly how a rerouted micro-batch crosses replicas.
    ``executors`` are the ranks running this ``(micro_batch, stage)`` -- the whole TP
    group of the executing stage; ``upstream``/``downstream`` are ``None`` at the
    pipeline ends.
    """
    routes = []
    for micro_batch in sorted(assignment.by_micro_batch):
        pipeline = stage_pipeline(assignment, micro_batch)
        for index, placement in enumerate(pipeline):
            if rank not in placement.executor_ranks:
                continue
            upstream = pipeline[index - 1].executor_ranks if index > 0 else None
            downstream = pipeline[index + 1].executor_ranks if index < len(pipeline) - 1 else None
            routes.append(
                {
                    "micro_batch": micro_batch,
                    "stage_id": placement.stage_id,
                    "executors": placement.executor_ranks,
                    "upstream": upstream,
                    "downstream": downstream,
                }
            )
    return routes


def global_micro_count(assignment):
    """Total micro-batches in the assignment (the global batch's normalizer)."""
    return len(assignment.by_micro_batch)


class ActivationLog:
    """Which activations are live (forwarded, not yet backwarded) over time.

    An activation is retained at its forward and released at its matching backward,
    so :attr:`live` never holds one whose backward has completed and :attr:`peak` is
    the high-water mark of concurrently-held activations. The lifecycle gate asserts
    on this directly (plan 3.5: an activation is counted in memory until its backward
    completes).
    """

    def __init__(self):
        self.live: set = set()
        self.peak = 0

    def retain(self, key):
        self.live.add(key)
        self.peak = max(self.peak, len(self.live))

    def release(self, key):
        if key not in self.live:
            raise KeyError(f"release of activation {key} that is not live")
        self.live.discard(key)


def dp_combine_gradients(local, *, group=None):
    """Sum each logical gradient across the DP replica copies that hold it.

    ``local`` is ``{name: {"grad": T, "shard_dim": int|None, "shard_index": int,
    "old_size": int, "replica": int}}`` for the shards this rank owns. Every rank in
    ``group`` joins one ``all_gather_object``; then, per logical name, each replica's
    full tensor is reconstructed from its shards (:func:`reconstruct_full`), the full
    tensors are summed across replicas, and this rank slices its own shard back out
    (:func:`local_slice`). Because each replica already holds the partial sum of its
    own micro-batches' gradients (scaled by the global micro count), summing the
    replicas yields the full-batch mean -- the executor split cannot change it.

    Working on full logical tensors is the single path that also handles replicas of
    different PP layering or TP degree: a name a replica lays out differently is still
    reconstructed and re-chunked to that replica's own layout.
    """
    payload = {
        name: {
            "grad": entry["grad"].detach().cpu(),
            "shard_dim": entry["shard_dim"],
            "shard_index": entry["shard_index"],
            "old_size": entry["old_size"],
            "replica": entry["replica"],
        }
        for name, entry in local.items()
    }
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)

    # name -> replica -> {"old_size": deg, "shards": {shard_index: grad}}. The shard
    # dim is a layout constant per logical name, but the degree (old_size) is
    # per-replica, so each replica is reconstructed under its own degree.
    by_name: dict[str, dict[int, dict]] = {}
    dims: dict[str, int | None] = {}
    for contribution in gathered:
        for name, entry in contribution.items():
            dims[name] = entry["shard_dim"]
            replica = by_name.setdefault(name, {}).setdefault(
                entry["replica"], {"old_size": entry["old_size"], "shards": {}}
            )
            replica["shards"][entry["shard_index"]] = entry["grad"]

    full_by_name: dict[str, torch.Tensor] = {}
    for name, per_replica in by_name.items():
        total = None
        for replica in per_replica.values():
            full, _ = reconstruct_full(name, dims[name], replica["old_size"], replica["shards"])
            total = full.clone() if total is None else total + full
        full_by_name[name] = total

    return {
        name: local_slice(full_by_name[name], entry["shard_dim"], entry["shard_index"], entry["old_size"])
        for name, entry in local.items()
    }


class DataParallelRuntime:
    """Cross-replica execution of one iteration for one stage.

    ``stage`` is this rank's :class:`resihp.parallel.tp.TensorParallelStage` (it owns
    a contiguous layer slice plus, on the ends, the embeddings / LM head, each
    sharded over its TP group). The runtime runs only the micro-batches the
    assignment routes to this rank, exchanging activations and gradients with the
    *actual* neighbour executors, then combines gradients across DP replicas and
    applies one AdamW update.

    ``group`` is the process group the transfers and the combine ride on -- it must
    contain every rank the assignment places. Ranks are **global** throughout, since
    that is what the assignment names.
    """

    def __init__(self, stage, *, replica_id, assignment, group=None):
        self.stage = stage
        self.replica_id = int(replica_id)
        self.assignment = assignment
        self.group = group
        self.rank = dist.get_rank()
        self.routes = executor_route(assignment, self.rank)
        self.micro_total = global_micro_count(assignment)
        self.activation_log = ActivationLog()
        self.optimizer = torch.optim.AdamW(
            stage.parameters(),
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

    def _recv(self, route, key, shape, device):
        """Pull one boundary tensor in and give every TP rank of this stage a copy.

        A stage's activation (and the gradient of its input) is replicated across its
        TP group, so the boundary moves exactly one authoritative copy between the two
        stages' leaders and the receiving group replicates it -- never a per-rank sum,
        which would double-count, and never a single rank holding it, which would
        starve the others.
        """
        buffer = torch.empty(shape, device=device)
        leader = route["executors"][0]
        if self.rank == leader:
            op = dist.P2POp(dist.irecv, buffer, route[key][0], self.group)
            for work in dist.batch_isend_irecv([op]):
                work.wait()
        if self.stage.tp_size > 1:
            dist.broadcast(buffer, src=leader, group=self.stage.group)
        return buffer

    def _send(self, route, key, tensor):
        """Push one authoritative copy from this stage's leader to the peer's."""
        if self.rank != route["executors"][0]:
            return
        op = dist.P2POp(dist.isend, tensor.contiguous(), route[key][0], self.group)
        for work in dist.batch_isend_irecv([op]):
            work.wait()

    def train_step(self, tokens):
        """Run every routed micro-batch's forward then backward; return the last loss.

        ``tokens`` is the full global batch; a micro-batch's ``tokens[micro]`` slice is
        this rank's chunk when it is the first stage. Returns the summed loss on the
        last stage (``None`` otherwise), and combines gradients across replicas before
        the AdamW step so the update matches the reference full-batch update.
        """
        device = next(self.stage.parameters()).device
        chunks = tokens.chunk(self.micro_total, dim=0)
        shape = (chunks[0].shape[0], self.stage.sequence_length, self.stage.dim)
        self.optimizer.zero_grad(set_to_none=True)
        self.activation_log = ActivationLog()

        held: dict[int, dict] = {}  # micro_batch -> forward context kept until backward
        total_loss = torch.zeros((), device=device)

        # --- forward: pull from the actual upstream, push to the actual downstream ---
        for route in self.routes:
            index = route["micro_batch"]
            if self.stage.is_first:
                received = None
                output = self.stage(tokens=chunks[index].to(device))
            else:
                received = self._recv(route, "upstream", shape, device).requires_grad_(True)
                output = self.stage(hidden=received)
            self.activation_log.retain(index)
            if self.stage.is_last:
                loss = F.cross_entropy(
                    output[:, :-1].reshape(-1, self.stage.vocab_size),
                    chunks[index][:, 1:].reshape(-1).to(device),
                ) / self.micro_total
                total_loss = total_loss + loss.detach()
                held[index] = {"received": received, "output": loss}
            else:
                self._send(route, "downstream", output.detach())
                held[index] = {"received": received, "output": output}

        # --- backward: return each gradient to the actual upstream executor ----------
        for route in reversed(self.routes):
            index = route["micro_batch"]
            context = held.pop(index)
            if self.stage.is_last:
                context["output"].backward()
            else:
                grad = self._recv(route, "downstream", shape, device)
                context["output"].backward(grad)
            if not self.stage.is_first:
                self._send(route, "upstream", context["received"].grad)
            self.activation_log.release(index)

        self._combine_and_step(device)
        return float(total_loss) if self.stage.is_last else None

    def _combine_and_step(self, device):
        """Sum each logical gradient across replicas, then apply one AdamW update.

        The shard metadata is the stage's own, so a replica running a different TP
        degree (or a different PP layering) still combines through the one
        reconstruct-sum-rechunk path.
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
