"""TP resharding on degree/member change, and the heterogeneous TP boundary (T11).

When a fail-stop drops a TP rank the surviving group must move to a new degree (or
swap in a replacement rank), which means every logical tensor has to be *resharded*
under the new layout. Plan section 3.3 fixes the single recovery path:

1. collect the full logical tensor from the shards healthy ranks still hold
   (shards of the same logical layer live on peer DP replicas too);
2. only when a shard index is missing on *every* healthy rank, fall back to the
   pre-failure checkpoint;
3. re-chunk ``param``/``grad``/``exp_avg``/``exp_avg_sq`` by the new degree;
4. hand each new-group rank its new shard;
5. verify the reconstructed full logical state matches the checkpoint tensor by
   tensor before anyone resumes.

Reconstruction is a pure function (:func:`reconstruct_full`) so the donor-recovery
and checkpoint-fallback branches are unit-testable without any process group; the
distributed driver (:func:`reshard_tp_state`) is a thin ``all_gather_object`` on top.

The heterogeneous TP boundary (:func:`cross_tp_boundary`) bridges two PP stages
whose TP degrees differ. The activation is replicated within each stage's TP group,
so the boundary moves one authoritative copy forward and returns one authoritative
gradient copy backward -- never the per-rank sum -- so the boundary gradient equals
the single-process reference element for element (functional correctness only, no
P2P performance optimization).
"""

import torch
import torch.distributed as dist


class ReshardError(RuntimeError):
    """Raised when a logical tensor is available from neither peers nor checkpoint."""


#: Shard dim of each non-layer logical tensor; ``None`` means replicated.
_GLOBAL_SHARD_DIMS = {
    "token_embedding.weight": 0,
    "position_embedding.weight": None,
    "final_norm.weight": None,
    "final_norm.bias": None,
    "lm_head.weight": 0,
}
#: Shard dim of each per-layer tensor, keyed by the suffix after ``layers.<gid>.``.
_LAYER_SHARD_DIMS = {
    "attn_norm.weight": None,
    "attn_norm.bias": None,
    "attn.q_proj.weight": 0,
    "attn.k_proj.weight": 0,
    "attn.v_proj.weight": 0,
    "attn.out_proj.weight": 1,
    "mlp_norm.weight": None,
    "mlp_norm.bias": None,
    "mlp.fc1.weight": 0,
    "mlp.fc2.weight": 1,
}

#: Per-parameter tensors that are resharded like the parameter itself.
_SHARDED_FIELDS = ("param", "grad", "exp_avg", "exp_avg_sq")
#: The AdamW step count is a replicated scalar, not a shard.
_REPLICATED_FIELDS = ("step",)
_ALL_FIELDS = _SHARDED_FIELDS + _REPLICATED_FIELDS


def shard_dims(layer_ids) -> dict[str, int | None]:
    """Logical name -> shard dim (``None`` = replicated) for the whole model.

    The single source of truth for the TP shard layout, matching
    :meth:`resihp.parallel.tp.TensorParallelTransformer.local_shards`; a torch-gated
    test asserts the two agree so they cannot drift apart.
    """
    dims = dict(_GLOBAL_SHARD_DIMS)
    for gid in layer_ids:
        for suffix, dim in _LAYER_SHARD_DIMS.items():
            dims[f"layers.{gid}.{suffix}"] = dim
    return dims


def reconstruct_full(name, shard_dim, old_size, contributions, checkpoint=None):
    """Rebuild one full logical tensor from the shards healthy ranks contributed.

    ``contributions`` maps a shard index to the tensor a healthy rank still holds.
    A replicated tensor (``shard_dim is None``) needs any one copy; a sharded tensor
    needs every index ``0..old_size-1``. When a required shard is absent everywhere
    fall back to ``checkpoint[name]``; if that is missing too the layer is
    unrecoverable and we raise -- the consistent-stop condition wired up in T14.

    Returns ``(full_tensor, source)`` with ``source`` in ``{"peer", "checkpoint"}``.
    """
    if shard_dim is None:
        if contributions:
            first = min(contributions)
            return contributions[first], "peer"
    elif all(index in contributions for index in range(old_size)):
        parts = [contributions[index] for index in range(old_size)]
        return torch.cat(parts, dim=shard_dim), "peer"
    if checkpoint is not None and name in checkpoint:
        return checkpoint[name], "checkpoint"
    raise ReshardError(f"logical tensor {name} unavailable from peers and checkpoint")


def local_slice(full, shard_dim, new_rank, new_size):
    """This rank's contiguous shard of ``full`` under the new degree."""
    if shard_dim is None:
        return full.detach().clone()
    return torch.chunk(full, new_size, dim=shard_dim)[new_rank].contiguous().clone()


def _to_cpu(state):
    """Canonicalize a rank's shards to CPU before the object gather.

    A reconstructed logical tensor is a device-agnostic canonical form (the
    checkpoint anchor it is verified against is CPU too). Gathering live CUDA shards
    would tag each tensor with its owner's device ordinal, so ``reconstruct_full``'s
    ``torch.cat`` would try to concatenate across devices and fail; canonicalizing to
    CPU first keeps reconstruction device-safe on NCCL. The caller places the
    returned shards back on the compute device.
    """
    return {
        name: {
            key: (value.detach().cpu() if torch.is_tensor(value) else value)
            for key, value in entry.items()
        }
        for name, entry in state.items()
    }


def _merge(gathered):
    """Fold per-rank contributions into ``{name: {field: {shard_index: tensor}}}``."""
    merged: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for contribution in gathered:
        for name, entry in contribution.items():
            index = entry["shard_index"]
            fields = merged.setdefault(name, {})
            for field in _ALL_FIELDS:
                if field in entry:
                    fields.setdefault(field, {})[index] = entry[field]
    return merged


def reshard_tp_state(
    local_state,
    *,
    layout,
    old_size,
    new_size,
    new_rank,
    group=None,
    checkpoint=None,
    verify=True,
):
    """Reshard param/grad/AdamW state across a TP degree or membership change.

    ``local_state`` is ``{name: {"shard_index": int, "param": T, "grad": T,
    "exp_avg": T, "exp_avg_sq": T, "step": T}}`` for the shards this rank holds; a
    brand-new replacement rank passes ``{}``. Every rank in ``group`` (all healthy
    ranks, across DP replicas) joins the ``all_gather_object`` so each reconstructs
    the identical full logical state. ``new_rank`` is this rank's index in the new
    TP group, or ``None`` if it is being dropped -- dropped ranks still gather (so
    their shards feed peers) but receive nothing back.

    ``checkpoint`` (when given) is ``{name: {field: full_tensor}}``; it supplies any
    shard missing from all peers and, with ``verify``, is checked tensor by tensor
    against every peer reconstruction (plan step 5). Returns this rank's new shards
    as ``{name: {field: tensor}}`` (empty when ``new_rank is None``).
    """
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, _to_cpu(local_state), group=group)
    merged = _merge(gathered)

    new_local: dict[str, dict[str, torch.Tensor]] = {}
    for name, shard_dim in layout.items():
        rebuilt = {}
        ckpt_entry = None if checkpoint is None else checkpoint.get(name)
        for field in _ALL_FIELDS:
            contributions = merged.get(name, {}).get(field, {})
            # Consult the checkpoint for this field only if it carries it: a real
            # checkpoint has no ``grad``, so a peer-supplied grad must not KeyError.
            ckpt_full = None if ckpt_entry is None else ckpt_entry.get(field)
            if not contributions and ckpt_full is None:
                continue  # field absent from the run (e.g. no grad) and from checkpoint
            dim = None if field in _REPLICATED_FIELDS else shard_dim
            ckpt = None if ckpt_full is None else {name: ckpt_full}
            full, source = reconstruct_full(name, dim, old_size, contributions, ckpt)
            if verify and source == "peer" and ckpt_full is not None:
                if not torch.equal(full, ckpt_full):
                    raise ReshardError(f"reshard verify failed for {name}.{field}")
            rebuilt[field] = full
        if new_rank is None:
            continue
        new_local[name] = {
            field: local_slice(
                full, None if field in _REPLICATED_FIELDS else shard_dim, new_rank, new_size
            )
            for field, full in rebuilt.items()
        }
    return new_local


class _ReplicatedBridge(torch.autograd.Function):
    """Move a replicated activation across a TP-degree (PP stage) boundary.

    Both stages hold the full activation replicated within their own TP group, so
    forward broadcasts the authoritative copy from ``fwd_src`` (an upstream rank) to
    every rank in ``group`` -- the downstream group thereby receives it. Backward
    broadcasts the authoritative gradient from ``bwd_src`` (a downstream rank) back
    to every rank, so each upstream rank gets exactly one copy of the reference
    gradient: the per-rank duplicates a naive all-reduce would sum are never added,
    and nothing is dropped.
    """

    @staticmethod
    def forward(ctx, x, group, fwd_src, bwd_src):
        ctx.group = group
        ctx.bwd_src = bwd_src
        y = x.contiguous().clone()
        dist.broadcast(y, src=fwd_src, group=group)
        return y

    @staticmethod
    def backward(ctx, grad):
        g = grad.contiguous().clone()
        dist.broadcast(g, src=ctx.bwd_src, group=ctx.group)
        return g, None, None, None


def cross_tp_boundary(x, *, upstream_leader, downstream_leader, group=None):
    """Bridge a replicated activation from the upstream TP group to the downstream one.

    ``upstream_leader``/``downstream_leader`` are the authoritative global ranks for
    the forward value and the backward gradient respectively.
    """
    return _ReplicatedBridge.apply(x, group, upstream_leader, downstream_leader)
