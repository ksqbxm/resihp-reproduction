"""TP reshard and heterogeneous TP boundary (T11).

Pure-function gates (single process, torch only) cover the two reconstruction
branches -- donor recovery from peer shards and checkpoint fallback -- plus the
layout that keeps :func:`shard_dims` in step with the TP module.

The distributed gates run the identical logic on two backends, exactly like the
T10 TP gates: CPU/**Gloo** (always available where torch is installed) and
GPU/**NCCL** (real device shards and collectives, skipped when there are fewer GPUs
than the case needs). They cover the real ``all_gather_object`` reshard for a
``TP2->TP1`` degree drop and a ``{0,1}->{0,2}`` member swap with no shard lost, and
the heterogeneous TP boundary -- a TP1 stage feeding a TP2 stage -- executed by the
production runtime, whose backward gradient must match the single-process reference
(no double counting, nothing dropped).

There is exactly one heterogeneous-boundary implementation in the project, the
scatter/gather hop inside :class:`resihp.parallel.pp.PipelineRuntime`, and this gate
drives that one. Nothing here reshards a ``grad``: gradients are not persistent state,
so they are in no checkpoint and no transfer (plan principle A).
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resihp.parallel.reshard import (
    ReshardError,
    local_slice,
    reconstruct_full,
    shard_dims,
)


requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- pure reconstruction branches -------------------------------------------------


@requires_torch
def test_reconstruct_from_peer_shards():
    """Donor recovery: every shard index present -> reconstruct from peers, no ckpt."""
    a = torch.arange(6.0).reshape(2, 3)
    b = torch.arange(6.0, 12.0).reshape(2, 3)
    full, source = reconstruct_full("w", shard_dim=1, old_size=2, contributions={0: a, 1: b})
    assert source == "peer"
    assert torch.equal(full, torch.cat([a, b], dim=1))


@requires_torch
def test_reconstruct_replicated_needs_one_copy():
    a = torch.ones(3)
    full, source = reconstruct_full("n", shard_dim=None, old_size=2, contributions={1: a})
    assert source == "peer"
    assert torch.equal(full, a)


@requires_torch
def test_reconstruct_falls_back_to_checkpoint_when_shard_missing():
    """A shard absent from every peer -> checkpoint fallback, not a peer stitch."""
    a = torch.arange(6.0).reshape(2, 3)
    ckpt_full = torch.zeros(2, 6)
    full, source = reconstruct_full(
        "w", shard_dim=1, old_size=2, contributions={0: a}, checkpoint={"w": ckpt_full}
    )
    assert source == "checkpoint"
    assert torch.equal(full, ckpt_full)


@requires_torch
def test_reconstruct_raises_when_unavailable_everywhere():
    a = torch.arange(6.0).reshape(2, 3)
    with pytest.raises(ReshardError):
        reconstruct_full("w", shard_dim=1, old_size=2, contributions={0: a}, checkpoint=None)


@requires_torch
def test_local_slice_matches_chunk():
    full = torch.arange(8.0)
    assert torch.equal(local_slice(full, 0, 1, 2), full.chunk(2, dim=0)[1])
    assert torch.equal(local_slice(full, None, 0, 2), full)


@requires_torch
def test_shard_dims_matches_tp_module():
    """Layout source of truth must equal the TP module's own shard dims."""
    import torch.multiprocessing as mp

    mp.spawn(_layout_worker, args=(_free_port(),), nprocs=1, join=True)


def _layout_worker(rank, port):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1")
    import torch.distributed as dist

    from resihp.config import TrainConfig
    from resihp.model import ReferenceTransformer
    from resihp.parallel.reshard import shard_logical_state
    from resihp.parallel.tp import TensorParallelStage

    dist.init_process_group(backend="gloo")
    config = TrainConfig(
        model_dim=16, num_layers=3, num_heads=4, batch_size=4,
        micro_batch_size=2, seed=7, tp=1, pp=1, dp=1, iterations=1,
    )
    layer_ids = range(config.num_layers)
    layout = shard_dims(layer_ids)
    reference = ReferenceTransformer(config, vocab_size=32, sequence_length=8)
    stage = TensorParallelStage(
        config,
        vocab_size=32,
        sequence_length=8,
        layer_ids=layer_ids,
        is_first=True,
        is_last=True,
        local_state=shard_logical_state(
            reference.logical_state_dict(), layout=layout, tp_rank=0, tp_size=1
        ),
    )
    from_module = {name: dim for name, (_, dim) in stage.local_shards().items()}
    assert from_module == layout, (from_module, layout)
    dist.destroy_process_group()


# --- distributed reshard: logic shared by Gloo and NCCL ---------------------------

_LAYOUT = shard_dims([0, 1])
#: Persistent per-parameter state only -- see the module docstring on ``grad``.
_FIELDS = ("param", "exp_avg", "exp_avg_sq")


def _full_state(seed):
    """A deterministic full logical state: every tensor with its moments and step."""
    import torch

    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    dim = 4
    shapes = {name: (dim, dim) for name in _LAYOUT}
    shapes["token_embedding.weight"] = (8, dim)  # sharded dim 0
    shapes["lm_head.weight"] = (8, dim)
    shapes["position_embedding.weight"] = (8, dim)  # replicated
    full = {}
    for name in _LAYOUT:
        base = rand(*shapes[name])
        full[name] = {
            "param": base,
            "exp_avg": base * 0.5,
            "exp_avg_sq": base.abs() + 1.0,
            "step": torch.tensor(3.0),
        }
    return full


def _shard_of(full_field, dim, index, size):
    import torch

    if dim is None:
        return full_field.clone()
    return torch.chunk(full_field, size, dim=dim)[index].contiguous().clone()


def _local_state_for(full, shard_index, old_size, device):
    return {
        name: {
            "shard_index": shard_index,
            **{f: _shard_of(full[name][f], _LAYOUT[name], shard_index, old_size).to(device) for f in _FIELDS},
            "step": full[name]["step"].to(device),
        }
        for name in full
    }


def _run_reshard(rank, world_size, kind, device):
    """Reshard on ``device`` and return a per-rank match summary (device-agnostic)."""
    import torch

    from resihp.parallel.reshard import reshard_tp_state

    full = _full_state(seed=123)  # CPU logical anchor, like a real checkpoint
    checkpoint = {name: dict(fields) for name, fields in full.items()}

    if kind == "degrade":  # TP2 -> TP1: shard 0 on rank 0, shard 1 on rank 1
        old_size, new_size = 2, 1
        local_state = _local_state_for(full, rank, old_size, device)
        new_rank = 0 if rank == 0 else None
    else:  # replace: old group {0,1}, new group {0,2}; rank 2 is the newcomer
        old_size, new_size = 2, 2
        local_state = _local_state_for(full, rank, old_size, device) if rank in (0, 1) else {}
        new_rank = {0: 0, 1: None, 2: 1}[rank]

    local_was_device = bool(local_state) and next(iter(local_state.values()))["param"].device == device
    new_local = reshard_tp_state(
        local_state, layout=_LAYOUT, old_size=old_size, new_size=new_size,
        new_rank=new_rank, checkpoint=checkpoint,
    )

    if new_rank is None:
        return {"dropped": True, "empty": new_local == {}, "local_was_device": local_was_device}
    ok = True
    max_diff = 0.0
    for name, dim in _LAYOUT.items():
        for field in _FIELDS + ("step",):
            dim_for = None if field == "step" else dim
            want = _shard_of(full[name][field], dim_for, new_rank, new_size)  # CPU anchor slice
            got = new_local[name][field]
            ok &= torch.equal(got, want)
            max_diff = max(max_diff, (got - want).abs().max().item())
    return {"dropped": False, "match": bool(ok), "max_diff": max_diff, "local_was_device": local_was_device}


BOUNDARY_CONFIG_KWARGS = dict(
    model_dim=16,
    num_layers=4,
    num_heads=4,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=1,
    iterations=1,
)
BOUNDARY_VOCAB = 32
BOUNDARY_SEQLEN = 8
BOUNDARY_MICRO = 4
#: Reassociation band: real TP all-reduce plus micro-batch splitting (T10/T12).
RTOL = 1e-4
ATOL = 1e-5


def _run_boundary(rank, device):
    """Upstream TP1 (rank 0) -> downstream TP2 (ranks 1, 2), through the real runtime.

    The two stages run at *different* TP degrees, which is the case plan 3.3 singles
    out. The activation is replicated inside each stage's TP group, so the boundary
    must move exactly one authoritative copy: scatter/gather cuts it into
    ``N = max(1, 2) = 2`` chunks on the distinct pairs ``0 -> 1`` and ``0 -> 2``, and
    the downstream group all-gathers them back. A per-rank sum would double the gradient
    the upstream stage sees (by the downstream degree), and a single receiving rank
    would starve its peer. Both failures show up as a gradient that no longer matches
    the single-process reference, which is what this asserts.

    This drives :class:`resihp.parallel.pp.PipelineRuntime` -- the project's only
    heterogeneous-boundary implementation and the one the control plane runs. Three
    ranks, so it runs on a 3-GPU box.
    """
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F

    from resihp.config import TrainConfig
    from resihp.model import ReferenceTransformer
    from resihp.parallel.pp import PipelineRuntime
    from resihp.parallel.reshard import shard_logical_state
    from resihp.parallel.tp import TensorParallelStage
    from resihp.planner.dp import DPAssignment, DPPlacement
    from resihp.planner.pp import balanced_layers
    from resihp.reference import adamw

    config = TrainConfig(**BOUNDARY_CONFIG_KWARGS)
    torch.manual_seed(config.seed)
    reference = ReferenceTransformer(
        config, vocab_size=BOUNDARY_VOCAB, sequence_length=BOUNDARY_SEQLEN
    )
    source = {name: p.detach().clone() for name, p in reference.logical_state_dict().items()}

    # Stage 0 is TP1 on rank 0; stage 1 is TP2 on ranks 1 and 2 -- a heterogeneous
    # boundary. Every rank creates every group in the same order.
    tp_groups = [dist.new_group([0]), dist.new_group([1, 2])]
    stage_id = 0 if rank == 0 else 1
    tp_group = tp_groups[stage_id]
    executors = ((0,), (1, 2))
    hop = dist.new_group([0, 1, 2])  # the union of both stages: every rank carries a chunk

    layers = balanced_layers(config.num_layers, 2)[stage_id]
    stage = TensorParallelStage(
        config,
        vocab_size=BOUNDARY_VOCAB,
        sequence_length=BOUNDARY_SEQLEN,
        layer_ids=layers,
        is_first=stage_id == 0,
        is_last=stage_id == 1,
        local_state=shard_logical_state(
            source,
            layout=shard_dims(layers),
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
        ),
        group=tp_group,
    ).to(device)
    stage.train()

    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(
        0, BOUNDARY_VOCAB, (config.batch_size, BOUNDARY_SEQLEN), generator=generator
    ).to(device)

    reference = reference.to(device).train()
    ref_opt = adamw(reference.parameters())
    logits = reference(tokens)
    ref_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, BOUNDARY_VOCAB), tokens[:, 1:].reshape(-1)
    )
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in reference.logical_state_dict().items()}

    assignment = DPAssignment(
        step=0,
        failure_signature=(),
        placements=tuple(
            DPPlacement(micro_batch=micro, stage_id=sid, replica_id=0, executor_ranks=executors[sid])
            for micro in range(BOUNDARY_MICRO)
            for sid in (0, 1)
        ),
    )
    runtime = PipelineRuntime(
        stage, rank=rank, replica_id=0, assignment=assignment, boundary_groups={(0, 1, 2): hop}
    )
    loss = runtime.train_step(tokens)

    close = True
    max_grad_diff = 0.0
    for name, (param, dim) in stage.local_shards().items():
        want = local_slice(ref_grads[name], dim, stage.tp_rank, stage.tp_size)
        close &= torch.allclose(param.grad, want, rtol=RTOL, atol=ATOL)
        max_grad_diff = max(max_grad_diff, (param.grad - want).abs().max().item())
    return {
        "stage_id": stage_id,
        "degree": stage.tp_size,
        "grad_matches_reference": bool(close),
        "max_grad_diff": max_grad_diff,
        "loss": loss,
        "reference_loss": float(ref_loss.detach()),
        "schedule": list(runtime.schedule),
        "is_cuda": bool(next(stage.parameters()).is_cuda),
    }


# --- backend wrappers -------------------------------------------------------------


def _gloo_reshard_worker(rank, world_size, kind, result_dir, port):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _run_reshard(rank, world_size, kind, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_reshard_worker(rank, world_size, kind, result_dir, port):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    result = _run_reshard(rank, world_size, kind, torch.device(f"cuda:{rank}"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _gloo_boundary_worker(rank, world_size, result_dir, port):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _run_boundary(rank, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_boundary_worker(rank, world_size, result_dir, port):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    result = _run_boundary(rank, torch.device(f"cuda:{rank}"))
    assert result["is_cuda"]
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _results(result_dir, count):
    return [json.loads(Path(result_dir, f"result_{rank}.json").read_text()) for rank in range(count)]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


def _assert_degrade(results):
    r0, r1 = results
    assert r0["match"], r0  # surviving rank now holds the full (degree-1) tensors
    assert r1["dropped"] and r1["empty"], r1  # removed rank keeps nothing


def _assert_replace(results):
    r0, r1, r2 = results
    assert r0["match"], r0  # kept rank's shard 0 unchanged
    assert r1["dropped"] and r1["empty"], r1  # replaced rank drops out
    assert r2["match"], r2  # newcomer receives shard 1 intact -- nothing lost


def _assert_boundary(results):
    upstream, *downstream = results
    assert [r["degree"] for r in results] == [1, 2, 2], results  # genuinely heterogeneous
    for result in results:
        # A doubled boundary gradient (the naive all-reduce) is ~2x off, far outside
        # the reassociation band; a dropped one leaves the upstream stage at zero.
        assert result["grad_matches_reference"], result
    assert upstream["loss"] is None, upstream  # only the last stage produces the loss
    for result in downstream:
        assert abs(result["loss"] - result["reference_loss"]) < 1e-4, result


# --- Gloo gates (always run where torch is installed) -----------------------------


@requires_torch
def test_reshard_tp2_to_tp1_gloo(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_reshard_worker, args=(2, "degrade", str(tmp_path), _free_port()), nprocs=2, join=True)
    _assert_degrade(_results(tmp_path, 2))


@requires_torch
def test_reshard_member_replacement_gloo(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_reshard_worker, args=(3, "replace", str(tmp_path), _free_port()), nprocs=3, join=True)
    _assert_replace(_results(tmp_path, 3))


@requires_torch
def test_heterogeneous_boundary_gloo(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_boundary_worker, args=(3, str(tmp_path), _free_port()), nprocs=3, join=True)
    _assert_boundary(_results(tmp_path, 3))


# --- NCCL gates (real GPU device shards + collectives; skip without enough GPUs) --


@requires_torch
def test_reshard_tp2_to_tp1_cuda_nccl(tmp_path):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(2)
    mp.spawn(_nccl_reshard_worker, args=(2, "degrade", str(tmp_path), _free_port()), nprocs=2, join=True)
    results = _results(tmp_path, 2)
    assert results[0]["local_was_device"], results  # shards were genuinely on GPU
    _assert_degrade(results)


@requires_torch
def test_reshard_member_replacement_cuda_nccl(tmp_path):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(3)
    mp.spawn(_nccl_reshard_worker, args=(3, "replace", str(tmp_path), _free_port()), nprocs=3, join=True)
    results = _results(tmp_path, 3)
    assert results[0]["local_was_device"], results
    _assert_replace(results)


@requires_torch
def test_heterogeneous_boundary_cuda_nccl(tmp_path):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(3)
    mp.spawn(_nccl_boundary_worker, args=(3, str(tmp_path), _free_port()), nprocs=3, join=True)
    results = _results(tmp_path, 3)
    for result in results:
        assert result["is_cuda"], result
    _assert_boundary(results)
