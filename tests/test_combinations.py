"""Pairwise combination gates (T15).

Plan section 四.D's first bullet: the seven two-component combinations must each
**execute a real forward/backward**, not merely compare plans. Every gate here runs
genuine sharded / pipelined / replicated training and checks the result against the
single-process reference (:mod:`resihp.reference`) -- the only anchor principle A
recognizes. No feature is added; these are tests over the T10-T14 code as it stands.

Each gate is the *seam* between two components, chosen so it is not already locked
elsewhere:

* ``tp_pp`` -- TP2 inside PP2 through the 1F1B :class:`~resihp.parallel.pp.PipelineRuntime`.
  T10 runs TP with a single stage and T12 runs the pipeline at TP degree 1; nothing
  ran real TP all-reduces *inside* a 1F1B schedule. The boundary activation is
  replicated within a TP group, so the pipeline hop is per TP index: two columns,
  ``(0,2)`` and ``(1,3)``, which is how TP and PP compose.
* ``tp_dp`` -- TP2 x DP2 through :class:`~resihp.parallel.dp.DataParallelRuntime`.
  T13 exercises :func:`~resihp.parallel.dp.dp_combine_gradients` on hand-built shards;
  here the whole runtime drives it, so the combine sees shards produced by its own
  sharded forward/backward.
* ``pp_dp`` -- PP2 x DP2 with an imbalanced 3-vs-1 micro-batch split. T13 covers a
  balanced split with one rerouted micro-batch, and an imbalanced split with no
  pipeline; the two together are what a post-reroute topology actually looks like.
* ``scheduler_groups`` -- the real scheduler (:func:`~resihp.plan.build_plan`) feeding
  :meth:`~resihp.control.ControlPlane.build_training_groups`, then training on exactly
  those groups. T14 drives this path but never checks its numbers; this gate does.
* ``scheduler_migration`` -- a fail-stop replan plus the state migration it implies,
  then **principle A's after-resume half**: the migrated run's next iteration equals a
  reference restarted from the same checkpoint under the new topology. T14 asserts
  only the before-resume half. The two replicas end up at *different* TP degrees.
* ``migration_checkpoint`` -- the sharded state gathered into the one atomic checkpoint
  and back out again: the committed anchor is the whole logical model at the values the
  reference reached, each rank then reloads exactly its slice of it (through the
  checkpoint-fallback branch, since with no peer replica the dead rank's shard exists
  nowhere else), and training continues correctly from there.
* ``dynamic_groups_pipeline`` -- process groups destroyed and rebuilt *between*
  iterations, with :class:`~resihp.parallel.pp.PipelineRuntime` running on the fresh
  communicators and the AdamW state carried across, so iteration 2 still matches the
  reference's iteration 2. No failure is involved, so what is under test is only the
  group churn itself.

Every gate runs on CPU/**Gloo** and on GPU/**NCCL** with real device tensors and real
NCCL training groups; the world group is Gloo on both, because it is the control
plane's always-alive group (plan 3.2). NCCL gates skip only when there are fewer GPUs
than the case needs.

Tolerance: TP all-reduce and micro-batch splitting each reorder FP32 accumulation
relative to the reference's single full-batch pass, so the comparison is the
``allclose`` band T10 fixed, not bit-for-bit.
"""

import faulthandler
import importlib.util
import json
import os
import socket
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
from torch.nn import functional as F

from resihp.checkpoint import load_anchor
from resihp.config import TrainConfig
from resihp.control import ControlPlane
from resihp.model import ReferenceTransformer
from resihp.parallel.reshard import local_slice, shard_dims, shard_logical_state
from resihp.parallel.tp import TensorParallelStage
from resihp.plan import build_plan
from resihp.planner.dp import DPAssignment, DPPlacement
from resihp.recovery import initial_run, stage_layout, stage_of
from resihp.reference import (
    ADAM_BETAS,
    ADAM_EPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _OPTIM_STATES,
    _token_stream,
)


VOCAB = 32
SEQLEN = 8
#: Reassociation band: real TP all-reduce plus micro-batch splitting (T10/T12/T13).
RTOL = 1e-4
ATOL = 1e-5

_BASE = dict(
    model_dim=16,
    num_layers=4,
    num_heads=4,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    iterations=4,
)
#: One layout per gate; the world size a gate needs is ``tp * pp * dp``.
LAYOUTS = {
    "tp_pp": dict(_BASE, tp=2, pp=2, dp=1),
    "tp_dp": dict(_BASE, tp=2, pp=1, dp=2),
    "pp_dp": dict(_BASE, tp=1, pp=2, dp=2),
    # Six layers so that halving stage 0's degree also pushes a layer across the stage
    # boundary: one event, a reshard and a real layer migration (as in T14).
    "pipeline": dict(_BASE, num_layers=6, tp=2, pp=2, dp=1),
    "solo_pp": dict(_BASE, tp=1, pp=2, dp=1),
}
MICRO = _BASE["batch_size"] // _BASE["micro_batch_size"]  # 4 micro-batches

#: The 1F1B primitive order at 2 stages / 4 micro-batches, by pipeline index (T12).
EXPECTED_SCHEDULE = {
    0: ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3", "W"],
    1: ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3", "W"],
}

#: A rank stuck in a collective would hang the suite forever; fail the gate instead.
JOIN_TIMEOUT = 300.0
#: A blocked rank is invisible from outside, so each dumps its own stack after this long.
STACK_DUMP_AFTER = 60.0

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _config(name) -> TrainConfig:
    return TrainConfig(**LAYOUTS[name])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _batch(config, index, device):
    """Iteration ``index``'s fixed token batch -- the same stream the run consumes."""
    stream = _token_stream(VOCAB, SEQLEN, config.batch_size, config.iterations, config.seed)
    return stream[index].to(device)


def _adamw(params):
    return torch.optim.AdamW(
        params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )


# --- reference anchors ------------------------------------------------------------


def _source(config):
    """The fixed initial full logical state -- the init the reference starts from."""
    torch.manual_seed(config.seed)
    model = ReferenceTransformer(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    return {name: param.detach().clone() for name, param in model.logical_state_dict().items()}


def _step_record(model, optimizer, tokens):
    """One full-batch reference iteration: its gradients, new weights, new moments."""
    logits = model(tokens)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    named = model.logical_state_dict()
    grads = {name: param.grad.detach().clone() for name, param in named.items()}
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grads": grads,
        "params": {name: param.detach().clone() for name, param in named.items()},
        "moments": {
            name: {key: optimizer.state[param][key].detach().clone() for key in _OPTIM_STATES}
            for name, param in named.items()
            if param in optimizer.state
        },
    }


def _reference_steps(config, device, steps):
    """The single-process reference over ``steps`` iterations of the fixed stream."""
    torch.manual_seed(config.seed)
    model = ReferenceTransformer(config, vocab_size=VOCAB, sequence_length=SEQLEN).to(device)
    model.train()
    optimizer = _adamw(model.parameters())
    return [_step_record(model, optimizer, _batch(config, index, device)) for index in range(steps)]


def _reference_from_anchor(config, anchor, tokens, device):
    """Principle A's after-resume reference: one step from the checkpoint anchor.

    The baseline is *not* an uninterrupted run from iteration 0 but "the same
    checkpoint + the new topology + the new configuration's actual batch + the same
    seed" -- which is exactly this: the checkpoint's full logical parameters and AdamW
    moments, stepped once on the batch the resumed run consumes.
    """
    model = ReferenceTransformer(config, vocab_size=VOCAB, sequence_length=SEQLEN).to(device)
    model.train()
    named = model.logical_state_dict()
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(anchor[name]["param"].to(device))
    optimizer = _adamw(model.parameters())
    optimizer.state.clear()
    for name, fields in anchor.items():
        if "exp_avg" not in fields:
            continue
        optimizer.state[named[name]] = {
            "exp_avg": fields["exp_avg"].to(device).clone(),
            "exp_avg_sq": fields["exp_avg_sq"].to(device).clone(),
            "step": fields["step"].clone(),  # AdamW keeps its step count on the CPU
        }
    return _step_record(model, optimizer, tokens)


# --- comparisons ------------------------------------------------------------------


def _compare_shards(stage, record):
    """Compare this rank's shards to the reference, sliced by its own TP layout."""
    grad_close = step_close = True
    max_grad_diff = max_param_diff = 0.0
    for name, (param, dim) in stage.local_shards().items():
        want_param = local_slice(record["params"][name], dim, stage.tp_rank, stage.tp_size)
        step_close &= torch.allclose(param.detach(), want_param, rtol=RTOL, atol=ATOL)
        max_param_diff = max(max_param_diff, (param.detach() - want_param).abs().max().item())
        if param.grad is None:
            grad_close = False  # every owned parameter must have taken a gradient
            continue
        want_grad = local_slice(record["grads"][name], dim, stage.tp_rank, stage.tp_size)
        grad_close &= torch.allclose(param.grad, want_grad, rtol=RTOL, atol=ATOL)
        max_grad_diff = max(max_grad_diff, (param.grad - want_grad).abs().max().item())
    return {
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "owned": sorted(stage.logical_state_dict()),
        "tp_size": stage.tp_size,
        "reference_loss": record["loss"],
    }


def _matches_anchor(run, plan, rank, checkpoint):
    """Principle A's before-resume half: exactly the checkpoint, re-sharded by the plan."""
    stage = stage_of(plan, rank)
    if stage is None:
        return run is None  # a dropped rank must hold no state at all
    layout = stage_layout(plan, stage)
    shards = run.stage.local_shards()
    if set(shards) != set(layout):
        return False  # the stage holds its plan's names, no more and no fewer
    anchor, _completed = load_anchor(checkpoint)
    moments = run.runtime.optimizer.state
    index = stage.tp_members.index(rank)
    ok = True
    for name, dim in layout.items():
        param = shards[name][0]
        ok &= torch.equal(
            param.detach().cpu(), local_slice(anchor[name]["param"], dim, index, stage.tp_degree)
        )
        held = moments[param]
        for field in ("exp_avg", "exp_avg_sq"):
            ok &= torch.equal(
                held[field].detach().cpu(),
                local_slice(anchor[name][field], dim, index, stage.tp_degree),
            )
        ok &= torch.equal(held["step"].detach().cpu(), anchor[name]["step"])
    return bool(ok)


# --- distributed building blocks --------------------------------------------------


def _groups(specs, backend, rank):
    """Create every group on every process in one order; keep the ones this rank joins.

    ``new_group`` is collective over the world, so a non-member must still make the
    call -- it just gets a sentinel back, which is dropped here.
    """
    mine = {}
    for label, ranks in specs:
        group = dist.new_group(ranks=list(ranks), backend=backend)
        if rank in ranks:
            mine[label] = group
    return mine


def _stage(config, source, *, layer_ids, is_first, is_last, group, tp_index, tp_size, device):
    """This rank's stage: the given layers, sharded to its index in ``group``.

    ``source`` is the full logical value of every name it carries. ``shard_dims`` always
    names the boundary tensors (embeddings, final norm, LM head) because they belong to
    *some* stage, so the layout is narrowed to what ``source`` actually holds: a full
    logical anchor carries all of them, while a stage rebuilt from its own shards
    carries only the ones it owns -- and at degree 1, where the only rebuild here
    happens, those shards *are* the full logical value. ``TensorParallelStage`` then
    takes exactly what ``is_first``/``is_last`` say it needs.
    """
    layer_ids = tuple(layer_ids)
    layout = {name: dim for name, dim in shard_dims(layer_ids).items() if name in source}
    return (
        TensorParallelStage(
            config,
            vocab_size=VOCAB,
            sequence_length=SEQLEN,
            layer_ids=layer_ids,
            is_first=is_first,
            is_last=is_last,
            local_state=shard_logical_state(
                source, layout=layout, tp_rank=tp_index, tp_size=tp_size
            ),
            group=group,
        )
        .to(device)
        .train()
    )


def _assignment(spec):
    """A DPAssignment from ``(micro, stage, replica, executor_ranks)`` tuples."""
    return DPAssignment(
        step=0,
        failure_signature=(),
        placements=tuple(
            DPPlacement(micro_batch=m, stage_id=s, replica_id=r, executor_ranks=e)
            for m, s, r, e in spec
        ),
    )


def _attach(control, plan, rank, device, checkpoint):
    """Give the control plane this rank's real training state under ``plan``."""
    run = initial_run(
        plan,
        rank=rank,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        tp_group=control.tp_group,
        executor_group=control.executor_group,
        device=device,
    )
    control.attach_run(run, checkpoint_path=checkpoint, device=device)
    return run


def _training_backend(control):
    """The backend the training groups really ride on; ``None`` for an idle rank."""
    return None if control.tp_group is None else dist.get_backend(control.tp_group)


def _micro_batch_map(plan):
    """``(replica, micro_batch)`` pairs the plan assigns -- the scheduler's own split."""
    return sorted({(place.replica_id, place.micro_batch) for place in plan.placements})


# --- gate runners -----------------------------------------------------------------


def _run_tp_pp(rank, world_size, device, backend, result_dir):
    """TP2 inside PP2: two pipeline columns, each carrying the replicated activation."""
    from resihp.parallel.pp import PipelineRuntime, balanced_layers

    config = _config("tp_pp")
    stage_id, tp_index = rank // config.tp, rank % config.tp
    mine = _groups(
        [("tp0", (0, 1)), ("tp1", (2, 3)), ("pp0", (0, 2)), ("pp1", (1, 3))], backend, rank
    )
    tp_group, pp_group = mine[f"tp{stage_id}"], mine[f"pp{tp_index}"]

    reference = _reference_steps(config, device, 1)[0]
    stage = _stage(
        config,
        _source(config),
        layer_ids=balanced_layers(config.num_layers, config.pp)[stage_id],
        is_first=stage_id == 0,
        is_last=stage_id == config.pp - 1,
        group=tp_group,
        tp_index=tp_index,
        tp_size=config.tp,
        device=device,
    )
    runtime = PipelineRuntime(
        stage,
        stage_ranks=(tp_index, tp_index + config.tp),
        num_micro_batches=MICRO,
        group=pp_group,
    )
    loss = runtime.train_step(_batch(config, 0, device))

    result = _compare_shards(stage, reference)
    result.update(
        loss=loss,
        schedule=list(runtime.schedule),
        pipeline_index=stage_id,
        all_names=sorted(reference["params"]),
        backend=dist.get_backend(tp_group),
        is_cuda=bool(next(stage.parameters()).is_cuda),
    )
    return result


def _run_tp_dp(rank, world_size, device, backend, result_dir):
    """TP2 x DP2 through the assignment-driven runtime: sharded execution, then combine."""
    from resihp.parallel.dp import DataParallelRuntime

    config = _config("tp_dp")
    replica, tp_index = rank // config.tp, rank % config.tp
    mine = _groups(
        [("tp0", (0, 1)), ("tp1", (2, 3)), ("exec", tuple(range(world_size)))], backend, rank
    )
    tp_group, executors = mine[f"tp{replica}"], mine["exec"]

    reference = _reference_steps(config, device, 1)[0]
    stage = _stage(
        config,
        _source(config),
        layer_ids=range(config.num_layers),
        is_first=True,
        is_last=True,
        group=tp_group,
        tp_index=tp_index,
        tp_size=config.tp,
        device=device,
    )
    # Micro-batches 0,1 run on replica 0 (ranks 0,1); 2,3 on replica 1 (ranks 2,3).
    assignment = _assignment(
        [(micro, 0, micro // 2, (0, 1) if micro < 2 else (2, 3)) for micro in range(MICRO)]
    )
    runtime = DataParallelRuntime(stage, replica_id=replica, assignment=assignment, group=executors)
    loss = runtime.train_step(_batch(config, 0, device))

    result = _compare_shards(stage, reference)
    result.update(
        loss=loss,
        replica=replica,
        processed=[(route["micro_batch"], route["stage_id"]) for route in runtime.routes],
        activation_peak=runtime.activation_log.peak,
        activation_drained=runtime.activation_log.live == set(),
        backend=dist.get_backend(tp_group),
        is_cuda=bool(next(stage.parameters()).is_cuda),
    )
    return result


def _run_pp_dp(rank, world_size, device, backend, result_dir):
    """PP2 x DP2 with an imbalanced 3-vs-1 split across two pipelined replicas."""
    from resihp.parallel.dp import DataParallelRuntime
    from resihp.parallel.pp import balanced_layers

    config = _config("pp_dp")
    replica, stage_id = rank // config.pp, rank % config.pp
    mine = _groups(
        [(f"solo{peer}", (peer,)) for peer in range(world_size)]
        + [("exec", tuple(range(world_size)))],
        backend,
        rank,
    )
    tp_group, executors = mine[f"solo{rank}"], mine["exec"]

    reference = _reference_steps(config, device, 1)[0]
    stage = _stage(
        config,
        _source(config),
        layer_ids=balanced_layers(config.num_layers, config.pp)[stage_id],
        is_first=stage_id == 0,
        is_last=stage_id == config.pp - 1,
        group=tp_group,
        tp_index=0,
        tp_size=1,
        device=device,
    )
    # Replica 0 (ranks 0,1) takes three micro-batches, replica 1 (ranks 2,3) takes one.
    spec = [(micro, index, 0, (index,)) for micro in range(3) for index in (0, 1)]
    spec += [(3, index, 1, (2 + index,)) for index in (0, 1)]
    runtime = DataParallelRuntime(
        stage, replica_id=replica, assignment=_assignment(spec), group=executors
    )
    loss = runtime.train_step(_batch(config, 0, device))

    result = _compare_shards(stage, reference)
    result.update(
        loss=loss,
        replica=replica,
        processed=[(route["micro_batch"], route["stage_id"]) for route in runtime.routes],
        activation_peak=runtime.activation_log.peak,
        activation_drained=runtime.activation_log.live == set(),
        all_names=sorted(reference["params"]),
        backend=dist.get_backend(executors),
        is_cuda=bool(next(stage.parameters()).is_cuda),
    )
    return result


def _run_scheduler_groups(rank, world_size, device, backend, result_dir):
    """The scheduler's plan builds the process groups, and training rides on them."""
    config = _config("tp_dp")
    control = ControlPlane(
        rank, world_size, dist.group.WORLD, backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
    run = _attach(control, plan, rank, device, result_dir / "ckpt.pt")

    reference = _reference_steps(config, device, 1)[0]
    loss = control.training_step()

    stage = stage_of(plan, rank)
    result = _compare_shards(run.stage, reference)
    result.update(
        loss=loss,
        digest=plan.digest,
        version=plan.version,
        replica=stage.replica_id,
        tp_members=list(stage.tp_members),
        micro_batches=_micro_batch_map(plan),
        backend=_training_backend(control),
        is_cuda=bool(next(run.stage.parameters()).is_cuda),
    )
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    return result


def _run_scheduler_migration(rank, world_size, device, backend, result_dir):
    """A replan plus the migration it implies, then principle A's after-resume half.

    Rank 1 fails, so replica 0 drops to TP1 while replica 1 keeps TP2 -- the replicas
    run *different* degrees from here on. The scheduler reassigns the micro-batches,
    the single recovery path moves the state, and the next iteration must equal a
    reference restarted from the same checkpoint under the new topology.
    """
    config = _config("tp_dp")
    checkpoint = result_dir / "ckpt.pt"
    control = ControlPlane(
        rank, world_size, dist.group.WORLD, backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
    run = _attach(control, plan, rank, device, checkpoint)
    is_cuda = bool(next(run.stage.parameters()).is_cuda)
    control.training_step()  # iteration 1 on the pristine topology

    plan, failed = control.safe_point(config, plan, (), 1, next_step=1)
    anchor, completed = load_anchor(checkpoint)
    stage = stage_of(plan, rank)
    result = {
        "version": plan.version,
        "failed": list(failed),
        "completed": completed,
        "degree": None if stage is None else stage.tp_degree,
        "micro_batches": _micro_batch_map(plan),
        "matches_checkpoint": _matches_anchor(control.training_run, plan, rank, checkpoint),
        "backend": _training_backend(control),
        "is_cuda": is_cuda,
    }

    reference = _reference_from_anchor(config, anchor, _batch(config, completed, device), device)
    result["loss"] = control.training_step()  # iteration 2 on the new topology
    if control.training_run is not None:
        result.update(_compare_shards(control.training_run.stage, reference))
        result["replica"] = stage.replica_id
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    return result


def _run_migration_checkpoint(rank, world_size, device, backend, result_dir):
    """State migration and the one atomic checkpoint, round-tripped.

    TP2 x PP2 over six layers: dropping a rank of stage 0 halves its degree *and*
    pushes a layer across the stage boundary. With no peer replica the dead rank's
    shard exists nowhere else, so this is the checkpoint-fallback branch.
    """
    config = _config("pipeline")
    checkpoint = result_dir / "ckpt.pt"
    control = ControlPlane(
        rank, world_size, dist.group.WORLD, backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
    run = _attach(control, plan, rank, device, checkpoint)
    is_cuda = bool(next(run.stage.parameters()).is_cuda)

    reference = _reference_steps(config, device, 1)[0]
    first_loss = control.training_step()  # iteration 1: real TP2 x PP2 forward/backward

    plan, failed = control.safe_point(config, plan, (), 1, next_step=1)
    anchor, completed = load_anchor(checkpoint)

    # The gather-and-commit is lossless: the anchor holds the whole logical model at
    # the values the reference reached, with the AdamW moments intact.
    anchor_close = True
    max_anchor_diff = 0.0
    wanted = [(name, "param", tensor) for name, tensor in reference["params"].items()]
    wanted += [
        (name, field, fields[field])
        for name, fields in reference["moments"].items()
        for field in ("exp_avg", "exp_avg_sq")
    ]
    for name, field, want in wanted:
        got, target = anchor[name][field], want.detach().cpu()
        anchor_close &= torch.allclose(got, target, rtol=RTOL, atol=ATOL)
        max_anchor_diff = max(max_anchor_diff, (got - target).abs().max().item())

    stage = stage_of(plan, rank)
    result = {
        "first_loss": first_loss,
        "first_reference_loss": reference["loss"],
        "version": plan.version,
        "failed": list(failed),
        "completed": completed,
        "anchor_close": bool(anchor_close),
        "max_anchor_diff": max_anchor_diff,
        "anchor_names": sorted(anchor),
        "anchor_steps": sorted({int(fields["step"]) for fields in anchor.values()}),
        "all_names": sorted(reference["params"]),
        "degree": None if stage is None else stage.tp_degree,
        "layers": None if stage is None else list(range(*stage.layer_range)),
        "matches_checkpoint": _matches_anchor(control.training_run, plan, rank, checkpoint),
        "backend": _training_backend(control),
        "is_cuda": is_cuda,
    }

    resumed = _reference_from_anchor(config, anchor, _batch(config, completed, device), device)
    result["loss"] = control.training_step()  # iteration 2, from the migrated state
    if control.training_run is not None:
        result.update(_compare_shards(control.training_run.stage, resumed))
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    return result


def _run_dynamic_groups_pipeline(rank, world_size, device, backend, result_dir):
    """Groups torn down and rebuilt between iterations, with 1F1B on the new ones.

    The safe point destroys and rebuilds every training group in unison; this gate
    isolates that from any failure, so what is under test is only that
    :class:`PipelineRuntime` runs correctly on communicators created after the run
    began. The stage is re-instantiated on the new TP group from its own current
    shards and the AdamW moments are carried across, so iteration 2 must still match
    the reference's iteration 2.
    """
    from resihp.parallel.pp import PipelineRuntime

    config = _config("solo_pp")
    control = ControlPlane(
        rank, world_size, dist.group.WORLD, backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)

    ordered = sorted(plan.stages, key=lambda entry: entry.stage_id)
    stage_ranks = tuple(entry.tp_members[0] for entry in ordered)
    mine = stage_of(plan, rank)
    layer_ids = tuple(range(*mine.layer_range))
    is_first = mine.stage_id == ordered[0].stage_id
    is_last = mine.stage_id == ordered[-1].stage_id

    reference = _reference_steps(config, device, 2)

    def pipeline(stage):
        return PipelineRuntime(
            stage,
            stage_ranks=stage_ranks,
            num_micro_batches=MICRO,
            group=control.executor_group,
        )

    def build(source):
        return _stage(
            config,
            source,
            layer_ids=layer_ids,
            is_first=is_first,
            is_last=is_last,
            group=control.tp_group,
            tp_index=0,
            tp_size=1,
            device=device,
        )

    stage = build(_source(config))
    runtime = pipeline(stage)
    loss = runtime.train_step(_batch(config, 0, device))
    rounds = [dict(_compare_shards(stage, reference[0]), loss=loss, schedule=list(runtime.schedule))]

    before = (control.tp_group, control.executor_group)
    control.destroy_training_groups()
    control.build_training_groups(plan)
    rebuilt = (control.tp_group, control.executor_group) != before

    # The old groups are gone, so the stage is rebuilt on the new ones from its own
    # current shards, and the optimizer moments are carried over rather than reset.
    moments = runtime.optimizer.state_dict()
    stage = build({name: param.detach().clone() for name, param in stage.logical_state_dict().items()})
    runtime = pipeline(stage)
    runtime.optimizer.load_state_dict(moments)
    loss = runtime.train_step(_batch(config, 1, device))
    rounds.append(dict(_compare_shards(stage, reference[1]), loss=loss, schedule=list(runtime.schedule)))

    result = {
        "rounds": rounds,
        "rebuilt": bool(rebuilt),
        "pipeline_index": stage_ranks.index(rank),
        "all_names": sorted(reference[0]["params"]),
        "backend": _training_backend(control),
        "is_cuda": bool(next(stage.parameters()).is_cuda),
    }
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    return result


# --- shared assertions ------------------------------------------------------------


def _assert_numerics(results, label, *, ranks=None):
    """Every listed rank's gradients and post-AdamW weights match the reference."""
    for rank, result in enumerate(results):
        if ranks is not None and rank not in ranks:
            continue
        # Surfaced so a failure shows the magnitude (reassociation vs a real bug).
        print(
            f"{label} rank {rank}: grad={result['max_grad_diff']:.2e} "
            f"param={result['max_param_diff']:.2e}"
        )
        assert result["grad_close"], (label, rank, result)
        assert result["step_close"], (label, rank, result)


def _assert_loss(results, label):
    """The distinct replicas' losses sum to the reference's full-batch loss.

    Keyed by replica because every TP rank of a stage computes the identical value,
    so only one copy per replica may be counted.
    """
    seen = {}
    reference_loss = None
    for result in results:
        if result.get("loss") is None:
            continue
        seen[result["replica"]] = result["loss"]
        reference_loss = result["reference_loss"]
    assert reference_loss is not None, (label, results)
    assert abs(sum(seen.values()) - reference_loss) < 1e-4, (label, seen, reference_loss)


def _assert_tiles_the_model(results, indices, all_names):
    """The listed ranks' stages tile the whole model, and none of them holds all of it."""
    owned = [set(results[index]["owned"]) for index in indices]
    union = set().union(*owned)
    assert union == set(all_names), (union ^ set(all_names))
    for position, left in enumerate(owned):
        assert left < set(all_names), left  # real pipelining: no stage holds everything
        for right in owned[position + 1 :]:
            assert not left & right, (left, right)


# --- per-gate assertions ----------------------------------------------------------


def _assert_tp_pp(results, label):
    _assert_numerics(results, label)
    for result in results:
        assert result["tp_size"] == 2, result  # real TP inside the pipeline
        assert result["schedule"] == EXPECTED_SCHEDULE[result["pipeline_index"]], result
    # TP peers of a stage own the same names; the two stages tile the model.
    assert results[1]["owned"] == results[0]["owned"], results
    assert results[3]["owned"] == results[2]["owned"], results
    _assert_tiles_the_model(results, (0, 2), results[0]["all_names"])
    # Only the last stage produces the loss, and both of its TP ranks agree on it.
    assert results[0]["loss"] is None and results[1]["loss"] is None, results
    assert results[2]["loss"] == pytest.approx(results[3]["loss"]), results
    assert abs(results[2]["loss"] - results[2]["reference_loss"]) < 1e-4, results[2]


def _assert_tp_dp(results, label):
    _assert_numerics(results, label)
    for result in results:
        assert result["tp_size"] == 2, result
        assert result["activation_drained"], result
        assert result["activation_peak"] == len(result["processed"]), result
    assert [result["replica"] for result in results] == [0, 0, 1, 1], results
    # Each replica ran its own two micro-batches; together, every one exactly once.
    micro = {result["replica"]: sorted(pair[0] for pair in result["processed"]) for result in results}
    assert micro == {0: [0, 1], 1: [2, 3]}, micro
    _assert_loss(results, label)


def _assert_pp_dp(results, label):
    _assert_numerics(results, label)
    # The imbalanced split: replica 0 runs three micro-batches, replica 1 runs one.
    assert [len(result["processed"]) for result in results] == [3, 3, 1, 1], results
    for result in results:
        assert result["activation_drained"], result
        assert result["activation_peak"] == len(result["processed"]), result
    for indices in ((0, 1), (2, 3)):  # each replica really is a two-stage pipeline
        _assert_tiles_the_model(results, indices, results[0]["all_names"])
    # Every (micro, stage) executed exactly once across all ranks.
    pairs = [tuple(pair) for result in results for pair in result["processed"]]
    assert len(pairs) == len(set(pairs)) == 2 * MICRO, pairs
    _assert_loss(results, label)


def _assert_scheduler_groups(results, label):
    _assert_numerics(results, label)
    assert len({result["digest"] for result in results}) == 1, results  # one agreed plan
    for result in results:
        assert result["version"] == 0, result
        assert result["tp_size"] == 2, result  # groups built at the plan's TP degree
        assert result["initialized"] is False, result  # no residual process group
    assert [result["tp_members"] for result in results] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    # The scheduler's own micro-batch split, executed exactly as it was planned.
    assert results[0]["micro_batches"] == [[0, 0], [0, 1], [1, 2], [1, 3]], results[0]
    _assert_loss(results, label)


def _assert_scheduler_migration(results, label):
    for result in results:
        assert result["version"] == 1 and result["failed"] == [1], result
        assert result["completed"] == 1, result  # one iteration ran before the event
        assert result["matches_checkpoint"], result  # principle A, before resume
        assert result["initialized"] is False, result
    # Replica 0 halved to TP1 while replica 1 kept TP2: different degrees, one plan.
    assert [result["degree"] for result in results] == [1, None, 2, 2], results
    assert results[1]["loss"] is None, results[1]  # the dead rank does no training work
    assert results[0]["micro_batches"] == [[0, 0], [0, 1], [1, 2], [1, 3]], results[0]
    # Principle A, after resume: the migrated run matches the checkpoint-anchored
    # reference for the new topology.
    _assert_numerics(results, label, ranks={0, 2, 3})
    _assert_loss([results[0], results[2]], label)


def _assert_migration_checkpoint(results, label):
    for result in results:
        print(f"{label}: anchor diff {result['max_anchor_diff']:.2e}")
        assert result["version"] == 1 and result["failed"] == [1], result
        assert result["anchor_close"], result  # the commit gathered a lossless anchor
        assert result["anchor_names"] == result["all_names"], result  # the whole model
        assert result["anchor_steps"] == [1], result  # the AdamW step count survived
        assert result["matches_checkpoint"], result  # each rank reloaded its own slice
        assert result["initialized"] is False, result
    # Iteration 1 itself was a correct TP2 x PP2 run, not just a correctly saved one.
    assert results[0]["first_loss"] is None, results[0]
    assert abs(results[2]["first_loss"] - results[2]["first_reference_loss"]) < 1e-4, results[2]
    # The event both halved stage 0's degree and moved a layer across the boundary.
    assert [result["degree"] for result in results] == [1, None, 2, 2], results
    assert results[0]["layers"] == [0, 1], results[0]
    assert results[2]["layers"] == [2, 3, 4, 5], results[2]
    # And training continues correctly from the migrated state.
    _assert_numerics(results, label, ranks={0, 2, 3})
    assert results[0]["loss"] is None and results[2]["loss"] is not None, results
    assert abs(results[2]["loss"] - results[2]["reference_loss"]) < 1e-4, results[2]


def _assert_dynamic_groups_pipeline(results, label):
    every = set(results[0]["all_names"])
    for rank, result in enumerate(results):
        assert result["rebuilt"], result  # the groups really were replaced
        assert result["initialized"] is False, result
        for index, round_result in enumerate(result["rounds"]):
            print(
                f"{label} rank {rank} round {index}: grad={round_result['max_grad_diff']:.2e} "
                f"param={round_result['max_param_diff']:.2e}"
            )
            assert round_result["grad_close"], (rank, index, round_result)
            assert round_result["step_close"], (rank, index, round_result)
            assert round_result["schedule"] == EXPECTED_SCHEDULE[result["pipeline_index"]]
            assert set(round_result["owned"]) < every, round_result
    # Both rounds still produce exactly one loss, on the last stage.
    for index in (0, 1):
        first_stage, last_stage = (result["rounds"][index] for result in results)
        assert first_stage["loss"] is None, (index, first_stage)
        assert abs(last_stage["loss"] - last_stage["reference_loss"]) < 1e-4, (index, last_stage)


#: ``gate -> (runner, world size, assertion)``.
GATES = {
    "tp_pp": (_run_tp_pp, 4, _assert_tp_pp),
    "tp_dp": (_run_tp_dp, 4, _assert_tp_dp),
    "pp_dp": (_run_pp_dp, 4, _assert_pp_dp),
    "scheduler_groups": (_run_scheduler_groups, 4, _assert_scheduler_groups),
    "scheduler_migration": (_run_scheduler_migration, 4, _assert_scheduler_migration),
    "migration_checkpoint": (_run_migration_checkpoint, 4, _assert_migration_checkpoint),
    "dynamic_groups_pipeline": (_run_dynamic_groups_pipeline, 2, _assert_dynamic_groups_pipeline),
}


# --- process entry point ----------------------------------------------------------


def _entry(rank, world_size, gate, backend, result_dir, port):
    """One spawned rank: Gloo world, training groups on ``backend``, then the gate."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    # Kept open for the process's lifetime: faulthandler writes into it from a timer.
    stack_file = Path(result_dir, f"stack_{rank}.txt").open("w")
    faulthandler.dump_traceback_later(STACK_DUMP_AFTER, repeat=True, file=stack_file)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        assert torch.cuda.current_device() == rank
    # The world group is Gloo on both backends -- it is the control plane's
    # always-alive group (plan 3.2); only the training groups switch to NCCL.
    dist.init_process_group(backend="gloo")
    result = GATES[gate][0](rank, world_size, device, backend, Path(result_dir))
    faulthandler.cancel_dump_traceback_later()
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(gate, backend, tmp_path):
    """Spawn the ranks and require *all* of them to exit before the timeout."""
    import torch.multiprocessing as mp

    world_size = GATES[gate][1]
    context = mp.spawn(
        _entry,
        args=(world_size, gate, backend, str(tmp_path), _free_port()),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + JOIN_TIMEOUT
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            stacks = "\n".join(
                f"--- rank {peer} ---\n{Path(tmp_path, f'stack_{peer}.txt').read_text()}"
                for peer in range(world_size)
                if Path(tmp_path, f"stack_{peer}.txt").exists()
            )
            pytest.fail(f"{gate}/{backend}: a rank is blocked and did not exit\n{stacks}")
    return [
        json.loads(Path(tmp_path, f"result_{rank}.json").read_text()) for rank in range(world_size)
    ]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- Gloo gates (always run where torch is installed) -----------------------------


@requires_torch
@pytest.mark.parametrize("gate", sorted(GATES))
def test_combination_gloo(tmp_path, gate):
    results = _spawn(gate, "gloo", tmp_path)
    GATES[gate][2](results, f"{gate} Gloo")


# --- NCCL gates (real GPU tensors and real NCCL training groups) ------------------


@requires_torch
@pytest.mark.parametrize("gate", sorted(GATES))
def test_combination_cuda_nccl(tmp_path, gate):
    _skip_if_few_gpus(GATES[gate][1])
    results = _spawn(gate, "nccl", tmp_path)
    for result in results:
        assert result["is_cuda"], result
    # An idle rank holds no training group; every rank that holds one is on NCCL.
    backends = {result["backend"] for result in results} - {None}
    assert backends == {"nccl"}, [result["backend"] for result in results]
    GATES[gate][2](results, f"{gate} NCCL")
