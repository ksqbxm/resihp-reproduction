"""Data-parallel cross-replica execution against the reference (T13).

Pure gates (single process) cover the assignment-driven routing and the activation
lifecycle. The distributed gates run identical logic on two backends, exactly like
T10/T11/T12: CPU/**Gloo** (always available where torch is installed) and GPU/**NCCL**
(real device tensors and real point-to-point transfers, skipped when there are fewer
GPUs than the case needs), so a 2-GPU box runs the two-rank gates for real.

* ``*_dp_normalization`` -- two full-model replicas, an imbalanced micro-batch split
  (3 vs 1) standing in for a post-reroute imbalance: each replica scales its losses by
  the *global* micro count and the DP combine sums the replicas, so the AdamW update
  matches the single-process reference no matter how lopsided the split is.
* ``*_cross_replica`` -- PP2xDP2 with one micro-batch rerouted so its stage-0 executor
  is in replica A and its stage-1 executor in replica B: the forward activation crosses
  the replica boundary to the *actual* downstream executor and the gradient returns to
  the *actual* upstream executor, and every stage's combined gradient matches the
  reference.
* ``*_pp_heterogeneous`` -- replica A runs two PP stages while replica B runs one stage
  owning every layer; both reach the reference gradient after the DP combine.
* ``*_tp_heterogeneous`` -- replica A executes real TP2 sharded forward/backward while
  replica B runs TP1; the DP combine reconstructs each replica's full gradient, sums
  them, and re-chunks to each replica's own degree, matching the reference shard for
  shard.
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resihp.parallel.dp import (
    ActivationLog,
    executor_route,
    global_micro_count,
    stage_pipeline,
)
from resihp.planner.dp import DPAssignment, DPPlacement


CONFIG_KWARGS = dict(
    model_dim=16,
    num_layers=4,
    num_heads=4,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=3,
)
VOCAB = 32
SEQLEN = 8
MICRO = 4  # batch_size // micro_batch_size
TOKEN_SEED = 99
# Micro-batching reorders the FP32 reduction relative to the reference's single pass,
# so the same tolerance the TP/PP paths fixed in T10/T12 applies here.
RTOL = 1e-4
ATOL = 1e-5

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _config():
    from resihp.config import TrainConfig

    return TrainConfig(**CONFIG_KWARGS)


def _assignment(spec):
    """Build a DPAssignment from ``(micro, stage, replica, executor_ranks)`` tuples."""
    placements = tuple(
        DPPlacement(micro_batch=m, stage_id=s, replica_id=r, executor_ranks=e) for (m, s, r, e) in spec
    )
    return DPAssignment(step=0, failure_signature=(), placements=placements)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- pure: assignment-driven routing and activation lifecycle ----------------------

# A 4-rank cross-replica assignment: replica A (ranks 0,1) and replica B (ranks 2,3)
# each run a two-stage pipeline, and micro-batch 3 is rerouted so its stage-0 executor
# is rank 0 (replica A) while its stage-1 executor is rank 3 (replica B).
_CROSS_SPEC = [
    (0, 0, 0, (0,)), (0, 1, 0, (1,)),
    (1, 0, 0, (0,)), (1, 1, 0, (1,)),
    (2, 0, 1, (2,)), (2, 1, 1, (3,)),
    (3, 0, 0, (0,)), (3, 1, 1, (3,)),
]


@requires_torch
def test_executor_route_reads_actual_downstream_and_upstream():
    assignment = _assignment(_CROSS_SPEC)
    assert global_micro_count(assignment) == 4

    rank0 = executor_route(assignment, 0)
    # rank 0's downstream executor varies per micro-batch (rank 1, rank 1, then rank 3):
    # it is read from the assignment, never inferred from a fixed neighbour.
    assert [(r["micro_batch"], r["downstream"]) for r in rank0] == [(0, (1,)), (1, (1,)), (3, (3,))]

    rank3 = executor_route(assignment, 3)
    # rank 3 pulls micro-batch 3's activation from rank 0 -- across the replica boundary.
    assert [(r["micro_batch"], r["upstream"]) for r in rank3] == [(2, (2,)), (3, (0,))]

    rank1 = executor_route(assignment, 1)
    assert [r["micro_batch"] for r in rank1] == [0, 1]
    assert all(r["downstream"] is None for r in rank1)  # stage 1 is the pipeline end


@requires_torch
def test_stage_pipeline_orders_placements_by_stage():
    assignment = _assignment(_CROSS_SPEC)
    assert [p.stage_id for p in stage_pipeline(assignment, 3)] == [0, 1]
    assert stage_pipeline(assignment, 3)[0].executor_ranks == (0,)  # stage 0 in replica A
    assert stage_pipeline(assignment, 3)[1].executor_ranks == (3,)  # stage 1 in replica B


@requires_torch
def test_activation_log_holds_each_activation_until_its_backward():
    log = ActivationLog()
    for micro in (0, 1, 2):
        log.retain(micro)
    assert log.live == {0, 1, 2} and log.peak == 3
    log.release(1)
    assert log.live == {0, 2}
    log.release(0)
    log.release(2)
    assert log.live == set()
    assert log.peak == 3  # high-water mark survives the releases
    with pytest.raises(KeyError):
        log.release(9)  # releasing an activation whose backward already ran is a bug


# --- distributed helpers ----------------------------------------------------------


def _adamw(params):
    from resihp.reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY

    return torch.optim.AdamW(
        params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )


def _reference(device):
    """The single-process reference: initial weights, one full-batch step, its gradients."""
    from torch.nn import functional as F

    from resihp.model import ReferenceTransformer

    config = _config()
    torch.manual_seed(config.seed)
    reference = ReferenceTransformer(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    source = {name: p.detach().clone() for name, p in reference.logical_state_dict().items()}

    generator = torch.Generator().manual_seed(TOKEN_SEED)
    tokens = torch.randint(0, VOCAB, (config.batch_size, SEQLEN), generator=generator).to(device)

    reference = reference.to(device).train()
    opt = _adamw(reference.parameters())
    logits = reference(tokens)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in reference.logical_state_dict().items()}
    opt.step()
    updated = {name: p.detach().clone() for name, p in reference.logical_state_dict().items()}
    return source, tokens, grads, updated, float(loss.detach())


def _stage(source, *, layer_ids, is_first, is_last, device, world_size):
    """A TP-degree-1 stage: one TP rank per stage, so every rank gets a solo group."""
    import torch.distributed as dist

    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage

    solo = [dist.new_group([peer]) for peer in range(world_size)][dist.get_rank()]
    return TensorParallelStage(
        _config(),
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        layer_ids=layer_ids,
        is_first=is_first,
        is_last=is_last,
        local_state=shard_logical_state(
            source, layout=shard_dims(layer_ids), tp_rank=0, tp_size=1
        ),
        group=solo,
    ).to(device).train()


def _runtime_result(stage, *, replica_id, assignment, tokens, ref_grads, ref_updated):
    from resihp.parallel.dp import DataParallelRuntime

    runtime = DataParallelRuntime(stage, replica_id=replica_id, assignment=assignment)
    loss = runtime.train_step(tokens)

    owned = stage.logical_state_dict()
    grad_close = step_close = True
    max_grad_diff = max_param_diff = 0.0
    for name, param in owned.items():
        grad_close &= torch.allclose(param.grad, ref_grads[name], rtol=RTOL, atol=ATOL)
        step_close &= torch.allclose(param.detach(), ref_updated[name], rtol=RTOL, atol=ATOL)
        max_grad_diff = max(max_grad_diff, (param.grad - ref_grads[name]).abs().max().item())
        max_param_diff = max(max_param_diff, (param.detach() - ref_updated[name]).abs().max().item())

    return {
        "owned": sorted(owned),
        "loss": loss,
        "processed": [(r["micro_batch"], r["stage_id"]) for r in runtime.routes],
        "activation_peak": runtime.activation_log.peak,
        "activation_drained": runtime.activation_log.live == set(),
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "is_cuda": bool(next(stage.parameters()).is_cuda),
    }


def _assert_runtime(results, ref_loss, *, label):
    for rank, result in enumerate(results):
        print(f"{label} rank {rank}: grad={result['max_grad_diff']:.2e} param={result['max_param_diff']:.2e}")
    for result in results:
        assert result["grad_close"], result
        assert result["step_close"], result
        assert result["activation_drained"], result  # every activation released by its backward
        assert result["activation_peak"] == len(result["processed"]), result

    # Every (micro, stage) executed exactly once across all ranks (no shared workload).
    # JSON turns the tuples into lists on the round-trip through the result files.
    pairs = [tuple(pair) for result in results for pair in result["processed"]]
    assert len(pairs) == len(set(pairs)), pairs

    losses = [result["loss"] for result in results if result["loss"] is not None]
    assert abs(sum(losses) - ref_loss) < 1e-4, (losses, ref_loss)


# --- scenario runners -------------------------------------------------------------


def _run_dp_normalization(rank, world_size, device):
    """Two full-model replicas, an imbalanced 3-vs-1 micro-batch split."""
    source, tokens, ref_grads, ref_updated, ref_loss = _reference(device)
    stage = _stage(
        source,
        layer_ids=range(_config().num_layers),
        is_first=True,
        is_last=True,
        device=device,
        world_size=world_size,
    )
    assignment = _assignment([(0, 0, 0, (0,)), (1, 0, 0, (0,)), (2, 0, 0, (0,)), (3, 0, 1, (1,))])
    result = _runtime_result(
        stage, replica_id=rank, assignment=assignment, tokens=tokens, ref_grads=ref_grads, ref_updated=ref_updated
    )
    result["reference_loss"] = ref_loss
    return result


def _run_cross_replica(rank, world_size, device):
    """PP2xDP2 where micro-batch 3's stages sit in different replicas."""
    from resihp.parallel.pp import balanced_layers

    source, tokens, ref_grads, ref_updated, ref_loss = _reference(device)
    groups = balanced_layers(_config().num_layers, 2)
    stage_id = rank % 2
    replica = rank // 2
    stage = _stage(
        source,
        layer_ids=groups[stage_id],
        is_first=stage_id == 0,
        is_last=stage_id == 1,
        device=device,
        world_size=world_size,
    )
    assignment = _assignment(_CROSS_SPEC)
    result = _runtime_result(
        stage, replica_id=replica, assignment=assignment, tokens=tokens, ref_grads=ref_grads, ref_updated=ref_updated
    )
    result["reference_loss"] = ref_loss
    # rank 0's downstream for the rerouted micro-batch 3 is rank 3, in the other replica.
    routes = executor_route(assignment, rank)
    result["crossed"] = rank == 0 and any(r["micro_batch"] == 3 and r["downstream"] == (3,) for r in routes)
    return result


def _run_pp_heterogeneous(rank, world_size, device):
    """Replica A runs two PP stages; replica B runs one stage owning every layer."""
    source, tokens, ref_grads, ref_updated, ref_loss = _reference(device)
    if rank == 0:
        replica, layer_ids, is_first, is_last = 0, (0, 1), True, False
    elif rank == 1:
        replica, layer_ids, is_first, is_last = 0, (2, 3), False, True
    else:
        replica, layer_ids, is_first, is_last = 1, range(_config().num_layers), True, True
    stage = _stage(
        source,
        layer_ids=layer_ids,
        is_first=is_first,
        is_last=is_last,
        device=device,
        world_size=world_size,
    )
    assignment = _assignment([
        (0, 0, 0, (0,)), (0, 1, 0, (1,)),
        (1, 0, 0, (0,)), (1, 1, 0, (1,)),
        (2, 0, 1, (2,)),
        (3, 0, 1, (2,)),
    ])
    result = _runtime_result(
        stage, replica_id=replica, assignment=assignment, tokens=tokens, ref_grads=ref_grads, ref_updated=ref_updated
    )
    result["reference_loss"] = ref_loss
    result["owns_whole_model"] = rank == 2 and len(result["owned"]) == len(ref_grads)
    return result


def _run_tp_heterogeneous(rank, world_size, device):
    """Replica A executes real TP2 sharded forward/backward; replica B runs TP1.

    Only the DP gradient combine is under test here; the sharded TP forward/backward is
    T10's. Each replica scales its micro-batch losses by the global count, and the
    combine must reconstruct each replica's full gradient, sum them, and re-chunk to
    each rank's own degree -- matching the reference shard for shard.
    """
    import torch.distributed as dist
    from torch.nn import functional as F

    from resihp.parallel.dp import dp_combine_gradients
    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage

    source, tokens, ref_grads, _ref_updated, _ref_loss = _reference(device)
    group_a = dist.new_group([0, 1])
    group_b = dist.new_group([2])
    chunks = tokens.chunk(MICRO, dim=0)

    if rank in (0, 1):
        replica, tp_group, micros = 0, group_a, [0, 1]
    else:
        replica, tp_group, micros = 1, group_b, [2, 3]

    layer_ids = range(_config().num_layers)
    model = TensorParallelStage(
        _config(),
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        layer_ids=layer_ids,
        is_first=True,
        is_last=True,
        local_state=shard_logical_state(
            source,
            layout=shard_dims(layer_ids),
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
        ),
        group=tp_group,
    ).to(device).train()
    for index in micros:
        logits = model(tokens=chunks[index])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), chunks[index][:, 1:].reshape(-1)
        ) / MICRO
        loss.backward()

    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    shards = model.local_shards()
    local = {
        name: {
            "grad": param.grad if param.grad is not None else torch.zeros_like(param),
            "shard_dim": shard_dim,
            "shard_index": tp_rank,
            "old_size": tp_size,
            "replica": replica,
        }
        for name, (param, shard_dim) in shards.items()
    }
    combined = dp_combine_gradients(local, group=None)

    close = True
    max_diff = 0.0
    for name, (param, shard_dim) in shards.items():
        reference = ref_grads[name].detach().cpu()
        if shard_dim is not None:
            reference = torch.chunk(reference, tp_size, dim=shard_dim)[tp_rank].contiguous()
        close &= torch.allclose(combined[name], reference, rtol=RTOL, atol=ATOL)
        max_diff = max(max_diff, (combined[name] - reference).abs().max().item())
    return {"close": bool(close), "max_diff": max_diff, "degree": tp_size, "is_cuda": bool(next(model.parameters()).is_cuda)}


# --- backend wrappers -------------------------------------------------------------


def _init_env(rank, world_size, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )


def _gloo_worker(runner, rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = runner(rank, world_size, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_worker(runner, rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    assert torch.cuda.current_device() == rank
    result = runner(rank, world_size, torch.device(f"cuda:{rank}"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


# ``mp.spawn`` needs module-level targets, so name one wrapper per runner per backend.
def _gloo_dp_normalization(rank, world_size, result_dir, port):
    _gloo_worker(_run_dp_normalization, rank, world_size, result_dir, port)


def _gloo_cross_replica(rank, world_size, result_dir, port):
    _gloo_worker(_run_cross_replica, rank, world_size, result_dir, port)


def _gloo_pp_heterogeneous(rank, world_size, result_dir, port):
    _gloo_worker(_run_pp_heterogeneous, rank, world_size, result_dir, port)


def _gloo_tp_heterogeneous(rank, world_size, result_dir, port):
    _gloo_worker(_run_tp_heterogeneous, rank, world_size, result_dir, port)


def _nccl_dp_normalization(rank, world_size, result_dir, port):
    _nccl_worker(_run_dp_normalization, rank, world_size, result_dir, port)


def _nccl_cross_replica(rank, world_size, result_dir, port):
    _nccl_worker(_run_cross_replica, rank, world_size, result_dir, port)


def _nccl_pp_heterogeneous(rank, world_size, result_dir, port):
    _nccl_worker(_run_pp_heterogeneous, rank, world_size, result_dir, port)


def _nccl_tp_heterogeneous(rank, world_size, result_dir, port):
    _nccl_worker(_run_tp_heterogeneous, rank, world_size, result_dir, port)


def _results(result_dir, count):
    return [json.loads(Path(result_dir, f"result_{rank}.json").read_text()) for rank in range(count)]


def _spawn(target, world_size, tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(target, args=(world_size, str(tmp_path), _free_port()), nprocs=world_size, join=True)
    return _results(tmp_path, world_size)


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- Gloo gates (always run where torch is installed) -----------------------------


@requires_torch
def test_dp_normalization_matches_reference_gloo(tmp_path):
    results = _spawn(_gloo_dp_normalization, 2, tmp_path)
    assert [len(r["processed"]) for r in results] == [3, 1]  # imbalanced split
    _assert_runtime(results, results[0]["reference_loss"], label="DP Gloo")


@requires_torch
def test_cross_replica_matches_reference_gloo(tmp_path):
    results = _spawn(_gloo_cross_replica, 4, tmp_path)
    assert results[0]["crossed"], results[0]  # micro-batch 3's activation crossed replicas
    _assert_runtime(results, results[0]["reference_loss"], label="cross-replica Gloo")


@requires_torch
def test_pp_heterogeneous_matches_reference_gloo(tmp_path):
    results = _spawn(_gloo_pp_heterogeneous, 3, tmp_path)
    assert results[2]["owns_whole_model"], results[2]  # replica B is a single PP stage
    assert not (set(results[0]["owned"]) & set(results[1]["owned"]))  # replica A really split
    _assert_runtime(results, results[0]["reference_loss"], label="PP-hetero Gloo")


@requires_torch
def test_tp_heterogeneous_combine_matches_reference_gloo(tmp_path):
    results = _spawn(_gloo_tp_heterogeneous, 3, tmp_path)
    assert [r["degree"] for r in results] == [2, 2, 1]  # replica A TP2, replica B TP1
    for result in results:
        assert result["close"], result


# --- NCCL gates (real GPU tensors + real transfers; skip without enough GPUs) -----


@requires_torch
def test_dp_normalization_matches_reference_cuda_nccl(tmp_path):
    _skip_if_few_gpus(2)
    results = _spawn(_nccl_dp_normalization, 2, tmp_path)
    for result in results:
        assert result["is_cuda"], result
    _assert_runtime(results, results[0]["reference_loss"], label="DP NCCL")


@requires_torch
def test_cross_replica_matches_reference_cuda_nccl(tmp_path):
    _skip_if_few_gpus(4)
    results = _spawn(_nccl_cross_replica, 4, tmp_path)
    assert results[0]["crossed"], results[0]
    for result in results:
        assert result["is_cuda"], result
    _assert_runtime(results, results[0]["reference_loss"], label="cross-replica NCCL")


@requires_torch
def test_pp_heterogeneous_matches_reference_cuda_nccl(tmp_path):
    _skip_if_few_gpus(3)
    results = _spawn(_nccl_pp_heterogeneous, 3, tmp_path)
    for result in results:
        assert result["is_cuda"], result
    _assert_runtime(results, results[0]["reference_loss"], label="PP-hetero NCCL")


@requires_torch
def test_tp_heterogeneous_combine_matches_reference_cuda_nccl(tmp_path):
    _skip_if_few_gpus(3)
    results = _spawn(_nccl_tp_heterogeneous, 3, tmp_path)
    assert [r["degree"] for r in results] == [2, 2, 1]
    for result in results:
        assert result["is_cuda"], result
        assert result["close"], result
