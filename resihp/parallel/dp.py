"""Data-parallel routing and cross-replica gradient combination (T13).

Plan section 3.5. The T5 planner (:mod:`resihp.planner.dp`) produces a
:class:`~resihp.planner.dp.DPAssignment`: every ``(micro_batch, stage)`` names the
executor ranks that must run it. This module turns that assignment into the two things
the DP dimension contributes to execution, and it does so **by reading the assignment**,
never by inferring peers from a fixed topology (plan 3.2):

* :func:`executor_route` -- for every ``(micro_batch, stage)`` a rank executes, the
  *actual* upstream and downstream executor ranks, so a micro-batch whose stages were
  rerouted reaches the executors the current plan named and no others;
* :func:`dp_combine_gradients` -- each logical gradient summed across the replicas that
  hold it, on *full logical* tensors (reconstruct each replica's tensor, sum across
  replicas, re-chunk to each replica's own layout), reusing the single T11 reshard path
  rather than a second all-reduce, so replicas that differ in PP layering or TP degree
  combine uniformly.

:class:`ActivationLog` records the lifecycle the plan requires: an activation stays
alive (counted in memory) from its forward until the matching backward retires it.

The execution itself -- the 1F1B schedule, the boundary transfers, the AdamW update --
is :class:`resihp.parallel.pp.PipelineRuntime`, the project's one runtime; sharded
forward/backward within a stage is :class:`resihp.parallel.tp.TensorParallelStage`.
"""

import torch
import torch.distributed as dist

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
