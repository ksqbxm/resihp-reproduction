"""TP reshard and heterogeneous TP boundary (T11).

Pure-function gates (single process, torch only) cover the two reconstruction
branches -- donor recovery from peer shards and checkpoint fallback -- plus the
layout that keeps :func:`shard_dims` in step with the TP module.

The distributed gates run the identical logic on two backends, exactly like the
T10 TP gates: CPU/**Gloo** (always available where torch is installed) and
GPU/**NCCL** (real device shards and collectives, skipped when there are fewer GPUs
than the case needs). They cover the real ``all_gather_object`` reshard for a
``TP2->TP1`` degree drop and a ``{0,1}->{0,2}`` member swap with no shard lost, and
the heterogeneous boundary whose backward gradient must equal the single-process
reference element for element (no double counting, nothing dropped).
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
_FIELDS = ("param", "grad", "exp_avg", "exp_avg_sq")


def _full_state(seed):
    """A deterministic full logical state: every tensor with all four fields + step."""
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
            "grad": base * 2.0,
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


def _run_boundary(rank, device):
    """Upstream TP1 (rank 0) -> downstream TP2 (ranks 0,1): bridge one activation.

    Downstream degree (2) exceeds upstream (1), so the downstream grad is replicated
    on both ranks; the bridge must return a *single* copy to the upstream rank, not
    the sum -- rank 0's activation gradient must equal the single-process reference (a
    naive all-reduce would double it). rank 1 is downstream-only and must receive the
    real activation on the forward, so its loss equals the reference. Two ranks, so it
    runs on a 2-GPU box.
    """
    import torch

    from resihp.parallel.reshard import cross_tp_boundary

    torch.manual_seed(0)
    inp = torch.rand(2, 4, device=device)
    w1 = torch.rand(4, 4, device=device)
    w2 = torch.rand(4, 4, device=device)

    # Reference (identical on every rank): full pipeline, single copy of the grad.
    x_ref = (inp @ w1).requires_grad_(True)
    (x_ref @ w2).sum().backward()
    grad_ref = x_ref.grad.detach().clone()
    reference_loss = float(((inp @ w1) @ w2).sum())

    if rank == 0:  # upstream: authoritative real activation for the forward
        x = (inp @ w1).detach().clone().requires_grad_(True)
    else:  # downstream only: placeholder, receives the real activation via the bridge
        x = torch.zeros(2, 4, device=device, requires_grad=True)
    x.retain_grad()

    bridged = cross_tp_boundary(x, upstream_leader=0, downstream_leader=0)
    loss = (bridged @ w2).sum()  # both ranks are downstream
    loss.backward()

    return {
        "grad_matches_reference": bool(torch.equal(x.grad, grad_ref)) if rank == 0 else None,
        "loss": float(loss.detach()),
        "reference_loss": reference_loss,
        "is_cuda": bool(x.is_cuda),
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
    r0, r1 = results
    assert r0["grad_matches_reference"], r0  # one grad copy returned, not doubled
    assert abs(r0["loss"] - r0["reference_loss"]) < 1e-6, r0
    assert abs(r1["loss"] - r1["reference_loss"]) < 1e-6, r1  # got real activation forward


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

    mp.spawn(_gloo_boundary_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    _assert_boundary(_results(tmp_path, 2))


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

    _skip_if_few_gpus(2)
    mp.spawn(_nccl_boundary_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    results = _results(tmp_path, 2)
    for result in results:
        assert result["is_cuda"], result
    _assert_boundary(results)
