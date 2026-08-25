"""The single fail-stop recovery path (T14).

Plan section 3.6 fixes exactly one recovery chain and forbids a second:

    fail-stop -> read checkpoint / collect from healthy replicas -> restore the full
    logical state -> replan -> rebuild groups -> continue.

This module is **orchestration only**, and it orchestrates what the plan already
decided. The planner emits :class:`~resihp.plan.StateRoute` entries -- which state
group each rank must acquire, from which donor, under which source TP layout, into
which target layout -- and :func:`recover` executes them. There is no second routing
policy here: nothing recomputes donors, targets, or layouts from the old and new
topologies. What remains are execution-time safety checks the plan cannot make
(a donor set that turns out to be incomplete falls back to the checkpoint; a stage
whose state the routes fail to cover is a structured error, not a KeyError).

Every mechanism it drives already exists and is owned elsewhere:

* TP members and degree -> T11 :func:`~resihp.parallel.reshard.reshard_tp_state`
  (collect from healthy peers, fall back to the checkpoint, re-chunk);
* micro-batch/stage executors and the 1F1B schedule -> T12/T13
  :class:`~resihp.parallel.pp.PipelineRuntime`, driven by the plan's own
  :class:`~resihp.planner.dp.DPAssignment`;
* the stage itself -> T10 :class:`~resihp.parallel.tp.TensorParallelStage`, cut to the
  plan's ``layer_range`` and sharded over the plan's TP group.

Nothing here re-implements TP, PP, or DP, and nothing infers a layout from an older
one: PP ownership, TP degree, TP membership, and executors are read from the current
plan alone. The intermediate recovery form is the full logical tensor -- the
checkpoint's own form -- but what is finally instantiated is one stage's layer subset
at one rank's TP shard, never the whole model per rank.

**What is recovered.** ``param``, ``exp_avg``, ``exp_avg_sq``, ``step``, plus the
iteration count / data cursor and the RNG state the checkpoint carries. Gradients are
not persistent state: a safe point is reached only after the current iteration has
completed *and* its AdamW step has been applied, and the next iteration calls
``zero_grad`` before recomputing them, so the gradients resident at a safe point have
no remaining semantic value. They are in no checkpoint, in no route, and in no
transfer.
"""

import torch

from .checkpoint import load_anchor, save_checkpoint
from .parallel.pp import PipelineRuntime
from .parallel.reshard import ReshardError, reshard_tp_state, shard_dims, shard_logical_state
from .parallel.tp import TensorParallelStage
from .model import ReferenceTransformer
from .planner.dp import DPAssignment
from .reference import ReferenceRun, _digest, _OPTIM_STATES, _token_stream


#: Resharding to degree 1 *is* reconstructing the full logical tensor.
_FULL_DEGREE = 1
#: Logical tensors that belong to a replica's first / last executable stage (plan 3.4),
#: keyed by the boundary group name the plan's routes use.
_BOUNDARY_NAMES = {
    "embedding": ("token_embedding.weight", "position_embedding.weight"),
    "head": ("final_norm.weight", "final_norm.bias", "lm_head.weight"),
}


# --- reading the plan (the only authority on layout) ------------------------------


def stage_of(plan, rank):
    """The plan's stage for ``rank``, or ``None`` when it places the rank nowhere."""
    for stage in plan.stages:
        if rank in stage.tp_members:
            return stage
    return None


def _replica_stages(plan, replica):
    return sorted(
        (stage for stage in plan.stages if stage.replica_id == replica),
        key=lambda stage: stage.stage_id,
    )


def stage_layout(plan, stage) -> dict[str, int | None]:
    """Logical names ``stage`` owns, with their TP shard dims.

    ``layer_range`` is the plan's PP ownership; the embeddings and the LM head belong
    to the replica's first and last *executable* stage (plan 3.4), which the plan
    gives as the lowest and highest surviving stage id of that replica.
    """
    dims = shard_dims(range(*stage.layer_range))
    names = [name for name in dims if name.startswith("layers.")]
    ends = _replica_stages(plan, stage.replica_id)
    if stage.stage_id == ends[0].stage_id:
        names += list(_BOUNDARY_NAMES["embedding"])
    if stage.stage_id == ends[-1].stage_id:
        names += list(_BOUNDARY_NAMES["head"])
    return {name: dims[name] for name in names}


def route_names(route) -> dict[str, int | None]:
    """The logical tensors one :class:`~resihp.plan.StateRoute` carries, with shard dims.

    A route names a state group -- one global layer, or one boundary group -- and this
    expands it into the logical names that group is made of. The plan stays free of the
    tensor layout (it is torch-free by construction); the layout stays in
    :func:`resihp.parallel.reshard.shard_dims`, its one definition.
    """
    if route.boundary:
        dims = shard_dims(())
        return {name: dims[name] for name in _BOUNDARY_NAMES[route.boundary]}
    prefix = f"layers.{route.layer}."
    return {
        name: dim for name, dim in shard_dims((route.layer,)).items() if name.startswith(prefix)
    }


def routes_for(plan, rank) -> tuple:
    """The plan's state routes that name ``rank`` among their target ranks."""
    return tuple(route for route in plan.state_routes if rank in route.target_ranks)


def acquire_layout(plan, rank) -> dict[str, int | None]:
    """Everything ``rank`` must fetch under this plan, straight from its routes.

    Empty when the plan routed nothing to this rank, which is exactly the case where
    everything the stage owns it already holds in the shape the plan wants.
    """
    layout: dict[str, int | None] = {}
    for route in routes_for(plan, rank):
        layout.update(route_names(route))
    return layout


def dp_assignment(plan) -> DPAssignment:
    """The plan's micro-batch/stage executor assignment, as the runtime takes it."""
    return DPAssignment(
        step=plan.step, failure_signature=plan.failed_ranks, placements=plan.placements
    )


def _gather_degree(plan) -> int:
    """Degree to interpret gathered shards under; replicas may differ, so prefer this one."""
    return max(stage.tp_degree for stage in plan.stages)


def _model_layout(config) -> dict[str, int | None]:
    return shard_dims(range(config.num_layers))


def _donor_degrees(plan, fallback: int) -> tuple[int, ...]:
    """Source layouts the plan's routes read under, in one order every rank shares.

    Each becomes one collective acquisition pass. The sequence is derived from the plan
    alone, so every rank makes the same number of passes in the same order however few
    routes target it -- a rank with nothing to acquire still joins, because its own
    shards are what the other ranks are collecting. ``donor_degree`` is 0 for a
    checkpoint restore, which has no source layout; it rides ``fallback``, where the
    absent peer shards make the checkpoint the only remaining source anyway.
    """
    return tuple(sorted({route.donor_degree or fallback for route in plan.state_routes}))


# --- this rank's state under one plan ---------------------------------------------


class PlannedRun:
    """One rank's training state under one ExecutionPlan: its stage and its runtime.

    The stage owns the plan's ``layer_range`` sharded over the plan's TP group; the
    iteration is executed by the 1F1B :class:`~resihp.parallel.pp.PipelineRuntime` over
    the plan's assignment, and the optimizer is that runtime's. ``cursor`` is the number
    of completed iterations, which is also the data cursor into the fixed token stream
    (plan 3.1).
    """

    def __init__(
        self,
        plan,
        *,
        rank,
        vocab_size,
        sequence_length,
        local_state,
        moments,
        tp_group,
        executor_group,
        boundary_groups,
        device,
        cursor,
    ):
        stage_plan = stage_of(plan, rank)
        ends = _replica_stages(plan, stage_plan.replica_id)
        config = plan.config
        self.device = torch.device("cpu") if device is None else device
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.stage = TensorParallelStage(
            config,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            layer_ids=range(*stage_plan.layer_range),
            is_first=stage_plan.stage_id == ends[0].stage_id,
            is_last=stage_plan.stage_id == ends[-1].stage_id,
            local_state=local_state,
            group=tp_group,
        ).to(self.device)
        self.stage.train()
        self.runtime = PipelineRuntime(
            self.stage,
            rank=rank,
            replica_id=stage_plan.replica_id,
            assignment=dp_assignment(plan),
            boundary_groups=boundary_groups,
            group=executor_group,
        )
        self._install_moments(moments)
        self.batches = _token_stream(
            vocab_size, sequence_length, config.batch_size, config.iterations, config.seed
        )
        self.cursor = cursor

    def _install_moments(self, moments) -> None:
        """Seed the runtime's AdamW with the recovered moments (empty on a fresh run)."""
        state = self.runtime.optimizer.state
        state.clear()
        for name, (param, _dim) in self.stage.local_shards().items():
            entry = moments.get(name)
            if entry is None:
                continue
            state[param] = {
                "exp_avg": entry["exp_avg"].to(self.device).clone(),
                "exp_avg_sq": entry["exp_avg_sq"].to(self.device).clone(),
                # AdamW keeps its step count on the CPU; leave it there.
                "step": entry["step"].clone(),
            }

    def step(self):
        """Run one iteration through the plan's assignment; loss on the last stage."""
        loss = self.runtime.train_step(self.batches[self.cursor].to(self.device))
        self.cursor += 1
        return loss

    def local_state(self) -> dict:
        """This rank's shards and AdamW moments, tagged with the degree they use.

        Gradients are deliberately absent: they are not persistent state (see the module
        docstring), so nothing carries them across a safe point.
        """
        moments = self.runtime.optimizer.state
        state = {}
        for name, (param, _dim) in self.stage.local_shards().items():
            entry = {
                "shard_index": self.stage.tp_rank,
                "shard_count": self.stage.tp_size,
                "param": param.detach(),
            }
            held = moments.get(param)
            if held:
                entry.update({key: held[key] for key in _OPTIM_STATES})
            state[name] = entry
        return state


def initial_run(
    plan,
    *,
    rank,
    vocab_size,
    sequence_length,
    tp_group,
    executor_group,
    boundary_groups=None,
    device=None,
):
    """The run a rank starts with, before any fail-stop.

    Sharded from the same fixed initialization the single-process reference uses
    (plan 3.1), cut to the plan's stage, so a distributed run and the reference start
    from identical weights.
    """
    stage = stage_of(plan, rank)
    if stage is None:
        return None
    config = plan.config
    torch.manual_seed(config.seed)
    source = ReferenceTransformer(
        config, vocab_size=vocab_size, sequence_length=sequence_length
    ).logical_state_dict()
    local = shard_logical_state(
        {name: param.detach() for name, param in source.items()},
        layout=stage_layout(plan, stage),
        tp_rank=stage.tp_members.index(rank),
        tp_size=stage.tp_degree,
    )
    return PlannedRun(
        plan,
        rank=rank,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        local_state=local,
        moments={},
        tp_group=tp_group,
        executor_group=executor_group,
        boundary_groups=boundary_groups,
        device=device,
        cursor=0,
    )


# --- the single recovery path -----------------------------------------------------


def logical_digest(full) -> str:
    """sha256 over a full logical state -- equal on every rank once recovery is done."""
    return _digest(
        sorted(
            (f"{name}.{field}", tensor)
            for name, fields in full.items()
            for field, tensor in fields.items()
        )
    )


def _anchor_run(config, vocab_size, sequence_length, full, cursor) -> ReferenceRun:
    """A :class:`ReferenceRun` carrying ``full``, ready for :func:`save_checkpoint`.

    Reusing ``ReferenceRun`` keeps one checkpoint format for the whole project. Its
    constructor re-seeds the global RNG, so the caller's RNG is saved and restored
    around it rather than being perturbed by taking a checkpoint.
    """
    rng_state = torch.get_rng_state()
    try:
        run = ReferenceRun(config, vocab_size=vocab_size, sequence_length=sequence_length)
    finally:
        torch.set_rng_state(rng_state)

    param_by_name = {name: param for param, name in run.name_by_param.items()}
    with torch.no_grad():
        for name, param in run.model.logical_state_dict().items():
            param.copy_(full[name]["param"])
    run.optimizer.state.clear()
    for name, fields in full.items():
        if "exp_avg" in fields:
            run.optimizer.state[param_by_name[name]] = {key: fields[key] for key in _OPTIM_STATES}
    run.cursor = cursor
    return run


def commit_checkpoint(
    run, path, *, plan, vocab_size, sequence_length, group, writer
) -> None:
    """Safe-point step 2: gather every shard into one atomic full logical checkpoint.

    Every process in ``group`` joins the gather -- a rank holding no state contributes
    nothing -- so the union of the stages and replicas is the whole model. Only
    ``writer`` puts the single canonical file on disk (plan 3.6).

    The control plane runs this at the end of *every* iteration, not only when a
    fail-stop is about to happen. A killed process cannot contribute to anything
    afterwards, so the only checkpoint that can still hold the dead rank's shards is
    one written while it was alive -- and that is what makes "the state before resume
    is exactly the pre-failure checkpoint" true of a real kill rather than of a rank
    that was merely excluded.
    """
    full = reshard_tp_state(
        {} if run is None else run.local_state(),
        layout=_model_layout(plan.config),
        old_size=_gather_degree(plan),
        new_size=_FULL_DEGREE,
        new_rank=0,
        group=group,
    )
    if not writer:
        return
    save_checkpoint(
        path,
        _anchor_run(plan.config, vocab_size, sequence_length, full, 0 if run is None else run.cursor),
        plan_version=plan.version,
    )


def recover(
    run,
    *,
    plan,
    previous,
    rank,
    vocab_size,
    sequence_length,
    checkpoint_path,
    world_group,
    tp_group,
    executor_group,
    boundary_groups=None,
    device=None,
):
    """Safe-point step 7: execute the new plan's state routes.

    Reads the checkpoint anchor, then runs one collective acquisition pass per source
    layout the plan's routes declare, fetching exactly the state groups those routes
    target at this rank. Healthy peers supply what they still hold and the checkpoint
    supplies only what survives nowhere -- the order the plan itself chose. The stage is
    then instantiated at the plan's PP ownership and TP layout, and the result is
    re-verified: the installed shards are gathered back into the full logical state and
    checked tensor by tensor against the anchor (plan 3.3 step 5), which is also
    safe-point step 8's state digest -- identical on every rank by construction.

    Returns ``(run, digest)``; ``run`` is ``None`` for a rank the new plan places on no
    stage. Every *surviving* rank calls this, and only those: the failed rank is a dead
    process, so what it held is genuinely gone and has to be rebuilt from a healthy
    peer replica or from the checkpoint written while it was still alive.
    """
    anchor, completed_steps = load_anchor(checkpoint_path)
    stage = stage_of(plan, rank)
    local = {} if run is None else run.local_state()

    fallback = _gather_degree(previous)
    routes = routes_for(plan, rank) if stage is not None else ()
    acquired: dict[str, dict] = {}
    for degree in _donor_degrees(plan, fallback):
        layout: dict[str, int | None] = {}
        for route in routes:
            if (route.donor_degree or fallback) == degree:
                layout.update(route_names(route))
        acquired.update(
            reshard_tp_state(
                local,
                layout=layout,
                old_size=degree,
                new_size=1 if stage is None else stage.tp_degree,
                new_rank=0 if stage is None else stage.tp_members.index(rank),
                group=world_group,
                checkpoint=anchor,
                verify=False,  # verified below, on what actually landed
            )
        )

    recovered = None
    if stage is not None:
        owned = stage_layout(plan, stage)
        # The plan decides what moves, so it also decides what stays: a name it did not
        # route is one this rank must already hold in the right shape. If it holds
        # neither, the routes cover less than the stage owns -- name that rather than
        # letting it surface as a KeyError inside the stage constructor.
        wanted = acquire_layout(plan, rank)
        missing = [name for name in owned if name not in wanted and name not in local]
        if missing:
            raise ReshardError(f"no state route covers {sorted(missing)} for rank {rank}")
        state = {name: local[name] for name in owned if name not in wanted}
        state.update({name: fields for name, fields in acquired.items() if name in owned})
        recovered = PlannedRun(
            plan,
            rank=rank,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            local_state={name: fields["param"] for name, fields in state.items()},
            moments={name: fields for name, fields in state.items() if "exp_avg" in fields},
            tp_group=tp_group,
            executor_group=executor_group,
            boundary_groups=boundary_groups,
            device=device,
            # The checkpoint is the resume point, so it is also the cursor -- a rank the
            # previous plan left idle has no cursor of its own to carry forward.
            cursor=completed_steps,
        )

    full = reshard_tp_state(
        {} if recovered is None else recovered.local_state(),
        layout=_model_layout(plan.config),
        old_size=_gather_degree(plan),
        new_size=_FULL_DEGREE,
        new_rank=0,
        group=world_group,
        checkpoint=anchor,
        verify=True,
    )
    return recovered, logical_digest(full)
