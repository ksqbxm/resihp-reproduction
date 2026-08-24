"""The single 1F1B pipeline runtime and layer-state migration (T12).

``PipelineRuntime`` is the runtime the control plane drives, so these gates exercise
the production class directly -- there is no separate test-only scheduler to compare
against. The pure schedule itself (warmup / steady / cooldown, in-flight peaks) is
locked in ``test_planner_pp.py``; what is checked here is that the runtime issues
*that* order and that the transfers it issues are correct.

The distributed gates run identical logic on two backends, exactly like T10/T11:
CPU/**Gloo** (always available where torch is installed) and GPU/**NCCL** (real device
tensors and real point-to-point transfers, skipped when there are fewer GPUs than the
case needs), so a 2-GPU box actually runs them.

* ``*_matches_reference`` -- two stages, four micro-batches: each rank owns only its
  slice of the model, activations and gradients cross the stage boundary as real
  transfers, and every stage's gradients and post-AdamW weights match the
  single-process reference for the names it owns. The emitted primitive order is
  asserted equal to :func:`resihp.planner.pp.pipeline_schedule`, so it is genuine 1F1B
  and not all-forwards-then-all-backwards.
* ``*_layer_migration_is_lossless`` -- a stage that must reacquire two layers at once
  (one arriving from another stage, one staying put but re-chunked because its stage
  lost a TP rank) carries ``param``/``exp_avg``/``exp_avg_sq``/``step`` through a real
  collective and lands tensor-for-tensor equal to the checkpoint anchor. There is no
  ``grad``: gradients are not persistent state and never cross a safe point.
* ``*_heterogeneous_boundary_matches_reference`` -- TP2 -> TP1 and TP1 -> TP2 pipelines
  (world 3), where the scatter/gather boundary is the only thing that can make the two
  differently sharded stages agree. The chunk routing itself is proved separately and
  purely by ``test_scatter_routing_*``.
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resihp.config import TrainConfig
from resihp.parallel.pp import scatter_routing
from resihp.planner.pp import balanced_layers, pipeline_schedule


CONFIG = TrainConfig(
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
MICRO = CONFIG.batch_size // CONFIG.micro_batch_size  # 4 micro-batches
# Micro-batching reorders the FP32 loss reduction relative to the reference's single
# full-batch pass, so the same tolerance the TP path fixed in T10 applies here.
RTOL = 1e-4
ATOL = 1e-5

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)

#: The 1F1B primitive order each stage must emit at 2 stages / 4 micro-batches:
#: stage 0 warms up one forward then alternates, stage 1 alternates throughout.
EXPECTED_SCHEDULE = {
    0: ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3", "W"],
    1: ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3", "W"],
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@requires_torch
def test_expected_schedule_is_the_planner_definition():
    """The literal above is only a readable spelling of the shared schedule function."""
    for stage_index, expected in EXPECTED_SCHEDULE.items():
        order = pipeline_schedule(MICRO, stage_index=stage_index, num_stages=2)
        assert [step[0] for step in expected[:-1]] == list(order)


# --- pure: the chunk routing (no process group, no tensor, no device) --------------

#: ``(up members, down members)``: equal degrees, both directions of a halving boundary,
#: and a 4 -> 2 hop where several senders feed one receiver.
_ROUTING_CASES = [
    ((0,), (1,)),
    ((0, 1), (2, 3)),
    ((0, 1), (2,)),
    ((0,), (1, 2)),
    ((0, 1, 2, 3), (4, 5)),
]


@requires_torch
@pytest.mark.parametrize("up,down", _ROUTING_CASES)
def test_scatter_routing_spreads_one_copy_over_n_links(up, down):
    """The traffic-reduction proof: N distinct pairs, each carrying 1/N of the tensor.

    A hop moves ``N = max(U, D)`` chunks. If every chunk sits on a pair of its own and no
    pair repeats, the whole tensor crosses the link exactly once -- the same volume the
    old leader-to-leader hop moved -- but over ``N`` links in parallel instead of one.
    """
    routing = scatter_routing(up, down)
    chunks = max(len(up), len(down))
    assert len(routing) == chunks  # exactly N chunks
    assert len(set(routing)) == chunks  # each pair distinct, so each is used once
    # One copy in total, not one per receiver: the chunks partition the tensor rather
    # than duplicating it, so N chunks of ``numel // N`` add back up to exactly ``numel``.
    numel = 256
    assert len(routing) * (numel // chunks) == numel


@requires_torch
@pytest.mark.parametrize("up,down", _ROUTING_CASES)
def test_scatter_routing_gives_every_rank_a_contiguous_share(up, down):
    """Every member takes part, and its chunks are the contiguous share it gathers.

    The receiver rebuilds the tensor with a plain all-gather over its TP group, which is
    only correct if each member's chunks are its own contiguous ``1/degree`` slice, in
    ascending member order. Same on the way back for the senders and the gradient.
    """
    routing = scatter_routing(up, down)
    chunks = max(len(up), len(down))
    for members, end in ((up, 0), (down, 1)):
        runs = [[k for k, pair in enumerate(routing) if pair[end] == member] for member in members]
        assert all(runs), (members, runs)  # no member is left idle
        for run in runs:
            assert run == list(range(run[0], run[0] + len(run)))  # contiguous
            assert len(run) == chunks // len(members)  # equal shares
        # Ascending member order is ascending chunk order, so ``cat`` in TP-rank order
        # restores the flat tensor.
        assert [k for run in runs for k in run] == list(range(chunks))


# --- distributed: 1F1B execution against the reference ----------------------------


def _pipeline(rank, num_stages):
    """A one-rank-per-stage assignment plus the union group of every hop.

    Exactly the shape the plan produces for a single DP replica: every micro-batch runs
    every stage, executed by that stage's own rank. At degree 1 the union of two adjacent
    stages *is* the two-rank pair, and the routing has a single chunk on it, so this
    keeps working unchanged. ``new_group`` is collective over the world, so every rank
    creates every hop group in the same order and keeps the ones it joins.
    """
    import torch.distributed as dist

    from resihp.planner.dp import DPAssignment, DPPlacement

    assignment = DPAssignment(
        step=0,
        failure_signature=(),
        placements=tuple(
            DPPlacement(micro_batch=micro, stage_id=stage, replica_id=0, executor_ranks=(stage,))
            for micro in range(MICRO)
            for stage in range(num_stages)
        ),
    )
    boundary = {}
    for pair in zip(range(num_stages), range(1, num_stages)):
        group = dist.new_group(list(pair))
        if rank in pair:
            boundary[pair] = group
    return assignment, boundary


def _compare_pp(rank, num_stages, device):
    """Run the reference and the pipeline on ``device``; return a per-rank summary."""
    import torch
    from torch.nn import functional as F

    import torch.distributed as dist

    from resihp.model import ReferenceTransformer
    from resihp.parallel.pp import PipelineRuntime
    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage
    from resihp.reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY

    def adamw(params):
        return torch.optim.AdamW(
            params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
        )

    torch.manual_seed(CONFIG.seed)
    reference = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    # Clone the init before the reference's own step mutates it: the stage must start
    # from the same weights the reference starts from.
    source = {name: param.detach().clone() for name, param in reference.logical_state_dict().items()}

    groups = balanced_layers(CONFIG.num_layers, num_stages)
    # One TP rank per stage: every rank builds every solo group, in the same order,
    # because ``new_group`` is collective over the world.
    solo = [dist.new_group([peer]) for peer in range(num_stages)][rank]
    assignment, boundary = _pipeline(rank, num_stages)
    stage = TensorParallelStage(
        CONFIG,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        layer_ids=groups[rank],
        is_first=rank == 0,
        is_last=rank == num_stages - 1,
        local_state=shard_logical_state(
            source, layout=shard_dims(groups[rank]), tp_rank=0, tp_size=1
        ),
        group=solo,
    ).to(device)
    stage.train()

    reference = reference.to(device).train()
    ref_opt = adamw(reference.parameters())

    generator = torch.Generator().manual_seed(99)
    tokens = torch.randint(0, VOCAB, (CONFIG.batch_size, SEQLEN), generator=generator).to(device)

    ref_logits = reference(tokens)
    ref_loss = F.cross_entropy(ref_logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_grads = {name: p.grad.detach().clone() for name, p in reference.logical_state_dict().items()}
    ref_opt.step()
    ref_updated = {name: p.detach().clone() for name, p in reference.logical_state_dict().items()}

    runtime = PipelineRuntime(
        stage, replica_id=0, assignment=assignment, boundary_groups=boundary
    )
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
        "all_names": sorted(source),
        "schedule": runtime.schedule,
        # Derived on the rank that ran it, so the comparison below is against the
        # planner's definition and not against a literal that could drift from it.
        "planned": list(
            pipeline_schedule(MICRO, stage_index=runtime.stage_index, num_stages=num_stages)
        ),
        "activation_peak": runtime.activation_log.peak,
        "activation_drained": runtime.activation_log.live == set(),
        "loss": loss,
        "reference_loss": float(ref_loss.detach()),
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "is_cuda": bool(next(stage.parameters()).is_cuda),
    }


def _diffs(result):
    """Compact diff view for the failure printout."""
    return {key: result[key] for key in ("loss", "reference_loss", "max_grad_diff", "max_param_diff")}


def _assert_matches_reference(results, num_stages, label):
    for rank, result in enumerate(results):
        # Surfaced so a failure shows the magnitude (reassociation vs real bug).
        print(f"{label} stage {rank}: {_diffs(result)}")
    for result in results:
        assert result["grad_close"], result
        assert result["step_close"], result

    last = results[-1]
    assert abs(last["loss"] - last["reference_loss"]) < 1e-4, last
    for result in results[:-1]:
        assert result["loss"] is None, result  # only the last stage produces the loss

    owned = [set(result["owned"]) for result in results]
    union = set().union(*owned)
    assert union == set(results[0]["all_names"])  # every parameter owned exactly once
    for i, left in enumerate(owned):
        for right in owned[i + 1 :]:
            assert not (left & right), (left, right)

    for rank, result in enumerate(results):
        # The runtime issues exactly the shared schedule -- the tie that keeps the
        # memory model's in-flight peaks describing the real thing.
        assert [step[0] for step in result["schedule"][:-1]] == result["planned"], result
        assert result["schedule"][-1] == "W", result  # one WeightUpdate, at the end
        assert result["activation_drained"], result  # every activation released
        # 1F1B, not GPipe: a stage holds at most one activation per stage below it,
        # never one per micro-batch.
        assert result["activation_peak"] == min(num_stages - rank, MICRO), result

    if num_stages > 1:
        for result in results:
            # Real pipelining: no stage holds the whole model.
            assert set(result["owned"]) < set(result["all_names"]), result
        assert any(n.startswith("token_embedding") for n in results[0]["owned"])
        assert any(n.startswith("lm_head") for n in results[-1]["owned"])
        for rank, result in enumerate(results):
            assert result["schedule"] == EXPECTED_SCHEDULE[rank], result["schedule"]
        # The whole point: the first stage really does hold two activations at once,
        # which a GPipe schedule would have made four.
        assert results[0]["activation_peak"] == 2 < MICRO, results[0]


# --- distributed: the heterogeneous scatter/gather boundary ------------------------

#: ``case -> (stage 0 members, stage 1 members)`` over a world of 3. Both directions of
#: an unequal boundary, which is the case only scatter/gather handles: the sending and
#: receiving TP groups have different sizes, so N = max(U, D) chunks are routed onto
#: N distinct rank pairs and the receiver reassembles by all-gather.
_HETERO = {"tp2_tp1": ((0, 1), (2,)), "tp1_tp2": ((0,), (1, 2))}


def _source(config):
    """The fixed initial full logical state -- the init the reference starts from."""
    import torch

    from resihp.model import ReferenceTransformer

    torch.manual_seed(config.seed)
    model = ReferenceTransformer(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    return {name: param.detach().clone() for name, param in model.logical_state_dict().items()}


def _compare_hetero(rank, case, device):
    """Run one side of an unequal-degree boundary and measure it against the reference."""
    import torch.distributed as dist

    from resihp import verify
    from resihp.parallel.pp import PipelineRuntime
    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage
    from resihp.planner.dp import DPAssignment, DPPlacement

    members = _HETERO[case]
    union = tuple(sorted(set(members[0]) | set(members[1])))
    # ``new_group`` is collective over the world, so every rank makes every call in the
    # same order and keeps only the handles it joined.
    tp_groups = [dist.new_group(list(stage_members)) for stage_members in members]
    hop = dist.new_group(list(union))
    executors = dist.new_group(list(union))

    stage_id = 0 if rank in members[0] else 1
    # The TP group is built from the stage's executors, so a rank's index in the members
    # tuple *is* its TP rank -- the correspondence the gather reassembly relies on.
    tp_index, tp_size = members[stage_id].index(rank), len(members[stage_id])

    reference = verify.reference_steps(
        CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN, count=1, device=device
    )[0]
    layer_ids = balanced_layers(CONFIG.num_layers, len(members))[stage_id]
    stage = (
        TensorParallelStage(
            CONFIG,
            vocab_size=VOCAB,
            sequence_length=SEQLEN,
            layer_ids=layer_ids,
            is_first=stage_id == 0,
            is_last=stage_id == len(members) - 1,
            local_state=shard_logical_state(
                _source(CONFIG), layout=shard_dims(layer_ids), tp_rank=tp_index, tp_size=tp_size
            ),
            group=tp_groups[stage_id],
        )
        .to(device)
        .train()
    )

    # Every micro-batch runs both stages; each stage's executors are its whole TP group.
    assignment = DPAssignment(
        step=0,
        failure_signature=(),
        placements=tuple(
            DPPlacement(
                micro_batch=micro, stage_id=sid, replica_id=0, executor_ranks=members[sid]
            )
            for micro in range(MICRO)
            for sid in (0, 1)
        ),
    )
    runtime = PipelineRuntime(
        stage,
        replica_id=0,
        assignment=assignment,
        # One union group for the whole hop -- every rank of both stages joins it,
        # because every one of them carries a chunk.
        boundary_groups={union: hop},
        group=executors,
    )
    loss = runtime.train_step(
        verify.batch(CONFIG, 0, vocab_size=VOCAB, sequence_length=SEQLEN, device=device)
    )

    result = verify.compare_shards(stage, reference)
    result.update(
        loss=loss,
        stage_id=stage_id,
        tp_index=tp_index,
        schedule=list(runtime.schedule),
        all_names=sorted(reference["params"]),
        routing=[list(pair) for pair in scatter_routing(*members)],
        is_cuda=bool(next(stage.parameters()).is_cuda),
    )
    return result


def _assert_hetero(results, case, label):
    from resihp import verify

    members = _HETERO[case]
    for rank, result in enumerate(results):
        print(f"{label} rank {rank}: {_diffs(result)}")  # magnitude, for a failure
    for rank, result in enumerate(results):
        # Sliced by this rank's own TP layout, so a TP2 shard is held to its half of the
        # reference and a TP1 stage to the whole of it.
        verify.assert_matches_reference(result, label=f"{label} rank {rank}")

    last = [result for result in results if result["stage_id"] == 1]
    for result in last:
        assert abs(result["loss"] - result["reference_loss"]) < 1e-4, result
    for result in results:
        if result["stage_id"] == 0:
            assert result["loss"] is None, result  # only the last stage produces the loss

    # Real pipelining across an unequal boundary: the two stages own disjoint names and
    # together own the whole model, whatever their degrees are.
    by_stage = [
        set().union(*[set(r["owned"]) for r in results if r["stage_id"] == sid]) for sid in (0, 1)
    ]
    assert not (by_stage[0] & by_stage[1]), by_stage
    assert by_stage[0] | by_stage[1] == set(results[0]["all_names"])

    # The boundary really was scattered: N = max(U, D) distinct pairs, spanning ranks of
    # both stages -- not one leader link.
    chunks = max(len(members[0]), len(members[1]))
    routing = [tuple(pair) for pair in results[0]["routing"]]
    assert len(routing) == len(set(routing)) == chunks, routing
    assert {pair[0] for pair in routing} == set(members[0]), routing
    assert {pair[1] for pair in routing} == set(members[1]), routing

    for result in results:
        assert result["schedule"] == EXPECTED_SCHEDULE[result["stage_id"]], result
        assert result["tp_size"] == len(members[result["stage_id"]]), result


# --- distributed: layer state migration -------------------------------------------

_MIGRATION_DIM = 4
#: Persistent per-parameter state only. ``grad`` is absent by contract: a safe point is
#: reached after the iteration's AdamW step, so the next iteration recomputes it.
_MIGRATION_FIELDS = ("param", "exp_avg", "exp_avg_sq")
#: Layers 3 and 4 of a six-layer model: layer 4 arrives from another stage while layer 3
#: stays put, and both drop from TP2 to TP1 -- one reshard covers the pair. Which layers
#: these are is the plan's decision (``ExecutionPlan.state_routes``); what is under test
#: here is that moving them loses nothing.
_MIGRATION_LAYERS = (3, 4)
_MIGRATION_OLD_DEGREE = 2
_MIGRATION_NEW_DEGREE = 1


def _layer_shape(name, dim):
    suffix = name.split(".", 2)[2]
    if suffix.endswith(("norm.weight", "norm.bias")):
        return (dim,)
    if suffix == "mlp.fc1.weight":
        return (4 * dim, dim)
    if suffix == "mlp.fc2.weight":
        return (dim, 4 * dim)
    return (dim, dim)


def _anchor_state(layout, seed):
    """The layer's full logical state, as the pre-failure checkpoint holds it (CPU)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    anchor = {}
    for name in layout:
        base = torch.rand(_layer_shape(name, _MIGRATION_DIM), generator=generator)
        anchor[name] = {
            "param": base,
            "exp_avg": base * 0.5,
            "exp_avg_sq": base.abs() + 1.0,
            "step": torch.tensor(5.0),
        }
    return anchor


def _shard_of(full, dim, index, size):
    import torch

    if dim is None:
        return full.clone()
    return torch.chunk(full, size, dim=dim)[index].contiguous().clone()


def _migration_layout():
    """Shard dims of the two layers the receiving stage must reacquire."""
    from resihp.parallel.reshard import shard_dims

    prefixes = tuple(f"layers.{layer}." for layer in _MIGRATION_LAYERS)
    return {
        name: dim
        for name, dim in shard_dims(_MIGRATION_LAYERS).items()
        if name.startswith(prefixes)
    }


def _run_migration(rank, device):
    """A stage's whole state workload at TP2 -> TP1, over a real collective.

    The two ranks are the current TP2 shards; afterwards rank 0 must hold the full
    logical tensors byte for byte equal to the checkpoint anchor, AdamW moments and step
    included, and the dropped rank must keep nothing.
    """
    import torch

    from resihp.parallel.reshard import reshard_tp_state

    layout = _migration_layout()
    anchor = _anchor_state(layout, seed=2024)

    local_state = {
        name: {
            "shard_index": rank,
            **{
                field: _shard_of(
                    anchor[name][field], layout[name], rank, _MIGRATION_OLD_DEGREE
                ).to(device)
                for field in _MIGRATION_FIELDS
            },
            "step": anchor[name]["step"].to(device),
        }
        for name in layout
    }
    local_was_device = next(iter(local_state.values()))["param"].device == device

    new_local = reshard_tp_state(
        local_state,
        layout=layout,
        old_size=_MIGRATION_OLD_DEGREE,
        new_size=_MIGRATION_NEW_DEGREE,
        new_rank=0 if rank == 0 else None,
        checkpoint=anchor,
    )

    if rank != 0:
        return {"dropped": True, "empty": new_local == {}, "local_was_device": local_was_device}

    matched = True
    for name in layout:
        for field in _MIGRATION_FIELDS + ("step",):
            matched &= torch.equal(new_local[name][field], anchor[name][field])
    return {
        "dropped": False,
        "match": bool(matched),
        "layers": sorted({name.split(".")[1] for name in new_local}),
        "names": sorted(new_local),
        "fields": sorted(next(iter(new_local.values()))),
        "local_was_device": local_was_device,
    }


def _assert_migration(results):
    receiver, donor = results
    assert receiver["match"], receiver  # every field equals the checkpoint anchor
    assert receiver["layers"] == ["3", "4"], receiver
    assert len(receiver["names"]) == 20, receiver  # both layers whole, 10 tensors each
    # Nothing carries a gradient across the boundary -- it is not persistent state.
    assert receiver["fields"] == ["exp_avg", "exp_avg_sq", "param", "step"], receiver
    assert donor["dropped"] and donor["empty"], donor


# --- backend wrappers -------------------------------------------------------------


def _init_env(rank, world_size, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )


def _gloo_pp_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _compare_pp(rank, world_size, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_pp_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    assert torch.cuda.current_device() == rank
    result = _compare_pp(rank, world_size, torch.device(f"cuda:{rank}"))
    assert result["is_cuda"]
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _gloo_hetero_worker(rank, world_size, result_dir, port, case):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _compare_hetero(rank, case, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_hetero_worker(rank, world_size, result_dir, port, case):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    result = _compare_hetero(rank, case, torch.device(f"cuda:{rank}"))
    assert result["is_cuda"]
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _gloo_migration_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _run_migration(rank, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_migration_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    result = _run_migration(rank, torch.device(f"cuda:{rank}"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _results(result_dir, count):
    return [json.loads(Path(result_dir, f"result_{rank}.json").read_text()) for rank in range(count)]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- Gloo gates (always run where torch is installed) -----------------------------


@requires_torch
@pytest.mark.parametrize("num_stages", [1, 2])
def test_pp_matches_reference_gloo(tmp_path, num_stages):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_pp_worker, args=(num_stages, str(tmp_path), _free_port()), nprocs=num_stages, join=True)
    _assert_matches_reference(_results(tmp_path, num_stages), num_stages, f"PP{num_stages} Gloo")


@requires_torch
@pytest.mark.parametrize("case", sorted(_HETERO))
def test_pp_heterogeneous_boundary_matches_reference_gloo(tmp_path, case):
    """TP2 -> TP1 and TP1 -> TP2 through the real runtime, over a world of 3."""
    import torch.multiprocessing as mp

    mp.spawn(
        _gloo_hetero_worker, args=(3, str(tmp_path), _free_port(), case), nprocs=3, join=True
    )
    _assert_hetero(_results(tmp_path, 3), case, f"{case} Gloo")


@requires_torch
def test_pp_layer_migration_is_lossless_gloo(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_migration_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    _assert_migration(_results(tmp_path, 2))


# --- NCCL gates (real GPU tensors + real transfers; skip without enough GPUs) -----


@requires_torch
@pytest.mark.parametrize("num_stages", [1, 2])
def test_pp_cuda_nccl_matches_reference(tmp_path, num_stages):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(num_stages)
    mp.spawn(_nccl_pp_worker, args=(num_stages, str(tmp_path), _free_port()), nprocs=num_stages, join=True)
    results = _results(tmp_path, num_stages)
    for result in results:
        assert result["is_cuda"], result
    _assert_matches_reference(results, num_stages, f"PP{num_stages} NCCL")


@requires_torch
@pytest.mark.parametrize("case", sorted(_HETERO))
def test_pp_heterogeneous_boundary_cuda_nccl(tmp_path, case):
    """The same unequal boundary on real devices: NCCL coalesced P2P over the union group."""
    import torch.multiprocessing as mp

    _skip_if_few_gpus(3)
    mp.spawn(
        _nccl_hetero_worker, args=(3, str(tmp_path), _free_port(), case), nprocs=3, join=True
    )
    results = _results(tmp_path, 3)
    for result in results:
        assert result["is_cuda"], result
    _assert_hetero(results, case, f"{case} NCCL")


@requires_torch
def test_pp_layer_migration_is_lossless_cuda_nccl(tmp_path):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(2)
    mp.spawn(_nccl_migration_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    results = _results(tmp_path, 2)
    assert results[0]["local_was_device"], results  # shards genuinely started on GPU
    _assert_migration(results)
