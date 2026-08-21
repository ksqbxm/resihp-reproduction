"""The single fail-stop recovery path (T14).

Plan section 3.6 fixes exactly one recovery chain and forbids a second:

    fail-stop -> read checkpoint / collect from healthy replicas -> restore the full
    logical state -> replan -> rebuild groups -> continue.

This module is **orchestration only**. Every mechanism it drives already exists and
is owned elsewhere; T14 wires them to the new :class:`~resihp.plan.ExecutionPlan`:

* TP members and degree -> T11 :func:`~resihp.parallel.reshard.reshard_tp_state`
  (collect from healthy peers, fall back to the checkpoint, re-chunk);
* stage layer ownership and its changes -> T12
  :class:`~resihp.parallel.pp.LayerPlacement` /
  :func:`~resihp.parallel.pp.reshard_layout`, read from the old and new plans;
* micro-batch/stage executors -> T13 :class:`~resihp.parallel.dp.DataParallelRuntime`,
  driven by the plan's own :class:`~resihp.planner.dp.DPAssignment`;
* the stage itself -> T10 :class:`~resihp.parallel.tp.TensorParallelStage`, cut to
  the plan's ``layer_range`` and sharded over the plan's TP group.

Nothing here re-implements TP, PP, or DP, and nothing infers a layout from an older
one: PP ownership, TP degree, TP membership, and executors are read from the current
plan alone. The intermediate recovery form is the full logical tensor -- the
checkpoint's own form -- but what is finally instantiated is one stage's layer subset
at one rank's TP shard, never the whole model per rank.
"""

import torch

from .checkpoint import load_anchor, save_checkpoint
from .parallel.dp import DataParallelRuntime
from .parallel.pp import LayerPlacement, reshard_layout
from .parallel.reshard import reshard_tp_state, shard_dims, shard_logical_state
from .parallel.tp import TensorParallelStage
from .model import ReferenceTransformer
from .planner.dp import DPAssignment
from .reference import ReferenceRun, _digest, _OPTIM_STATES, _token_stream


#: Resharding to degree 1 *is* reconstructing the full logical tensor.
_FULL_DEGREE = 1
#: Logical tensors that belong to a replica's first / last executable stage (plan 3.4).
_EMBEDDING_NAMES = ("token_embedding.weight", "position_embedding.weight")
_HEAD_NAMES = ("final_norm.weight", "final_norm.bias", "lm_head.weight")


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
        names += list(_EMBEDDING_NAMES)
    if stage.stage_id == ends[-1].stage_id:
        names += list(_HEAD_NAMES)
    return {name: dims[name] for name in names}


def _owner_degree(plan, replica) -> dict[int, tuple[int, int]]:
    return {
        layer: (stage.stage_id, stage.tp_degree)
        for stage in _replica_stages(plan, replica)
        for layer in range(*stage.layer_range)
    }


def layer_placements(previous, plan, replica) -> tuple[LayerPlacement, ...]:
    """Every layer's old and new ``(owner stage, TP degree)`` for one replica.

    T12's migration view of the repartition, *read from the two plans* rather than
    recomputed, so the ExecutionPlan stays the only authority on who owns what.
    """
    old = _owner_degree(previous, replica)
    new = _owner_degree(plan, replica)
    return tuple(
        LayerPlacement(
            layer=layer,
            old_owner=old[layer][0],
            new_owner=new[layer][0],
            old_degree=old[layer][1],
            new_degree=new[layer][1],
        )
        for layer in sorted(new)
    )


def _keeps_its_shard(previous, plan, stage, rank) -> bool:
    """True when ``rank`` already holds this stage's state in the shape the plan wants.

    Only then does "not in the acquire set" mean "already correct". A rank the previous
    plan placed on another stage, at another degree, or at another index within the
    group holds nothing reusable -- and a rank it left idle holds nothing at all. The
    planner takes the largest power-of-two prefix of a stage's survivors, so a later
    failure can shift indices or draw in a rank that sat out the previous plan.
    """
    before = stage_of(previous, rank)
    return (
        before is not None
        and before.stage_id == stage.stage_id
        and before.tp_degree == stage.tp_degree
        and before.tp_members.index(rank) == stage.tp_members.index(rank)
    )


def acquire_layout(previous, plan, stage, rank) -> dict[str, int | None]:
    """Names ``rank`` must fetch state for -- empty when nothing about its seat changed.

    A rank that did not keep its seat fetches the whole stage: nothing it holds is in
    the right shape. Otherwise the layers come from T12's
    :func:`~resihp.parallel.pp.reshard_layout` -- a layer needs work when it changed
    stage or changed TP degree -- and the boundary tensors are added when this stage
    has *become* the replica's first or last executable stage, which ``reshard_layout``
    cannot express because it is per-layer by construction. That boundary owner move is
    the case T12 left to this task. Everything else the stage owns, this rank already
    holds in the right shape, so it is not moved at all.
    """
    if not _keeps_its_shard(previous, plan, stage, rank):
        return dict(stage_layout(plan, stage))

    placements = layer_placements(previous, plan, stage.replica_id)
    layout = dict(reshard_layout(placements, new_owner=stage.stage_id))
    dims = shard_dims(())
    for names, end in ((_EMBEDDING_NAMES, 0), (_HEAD_NAMES, -1)):
        # A degree change would already have failed ``_keeps_its_shard``, so the only
        # boundary move left to catch is the owner stage itself changing.
        owner_before = _replica_stages(previous, stage.replica_id)[end].stage_id
        owner_now = _replica_stages(plan, stage.replica_id)[end].stage_id
        if owner_now == stage.stage_id and owner_before != owner_now:
            layout.update({name: dims[name] for name in names})
    return layout


def dp_assignment(plan) -> DPAssignment:
    """The plan's micro-batch/stage executor assignment, as T13's runtime takes it."""
    return DPAssignment(
        step=plan.step, failure_signature=plan.failed_ranks, placements=plan.placements
    )


def _gather_degree(plan) -> int:
    """Degree to interpret gathered shards under; replicas may differ, so prefer this one."""
    return max(stage.tp_degree for stage in plan.stages)


def _model_layout(config) -> dict[str, int | None]:
    return shard_dims(range(config.num_layers))


# --- this rank's state under one plan ---------------------------------------------


class PlannedRun:
    """One rank's training state under one ExecutionPlan: its stage and its runtime.

    The stage owns the plan's ``layer_range`` sharded over the plan's TP group; the
    iteration is executed by T13's assignment-driven runtime and the optimizer is
    that runtime's. ``cursor`` is the number of completed iterations, which is also
    the data cursor into the fixed token stream (plan 3.1).
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
        self.runtime = DataParallelRuntime(
            self.stage,
            replica_id=stage_plan.replica_id,
            assignment=dp_assignment(plan),
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

        Gradients are deliberately absent. A safe point is reached only after the
        current iteration has completed and its AdamW step has been applied, so
        ``param.grad`` holds spent values that the next iteration recomputes; the
        checkpoint stores none for the same reason.
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
    plan, *, rank, vocab_size, sequence_length, tp_group, executor_group, device=None
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
    control_group,
    tp_group,
    executor_group,
    device=None,
):
    """Safe-point step 7: the whole recovery chain, driven by the new plan.

    Reads the checkpoint anchor, fetches what this rank's new stage must acquire
    (healthy peers first, checkpoint only for what survives nowhere), instantiates the
    stage at the plan's PP ownership and TP layout, and re-verifies the result: the
    installed shards are gathered back into the full logical state and checked tensor
    by tensor against the anchor (plan 3.3 step 5), which is also safe-point step 8's
    state digest -- identical on every rank by construction.

    Returns ``(run, digest)``; ``run`` is ``None`` for a rank the new plan places on
    no stage. Every rank calls this: the collectives inside are how a dead rank's
    shard reaches the peers that must rebuild it.
    """
    anchor, completed_steps = load_anchor(checkpoint_path)
    stage = stage_of(plan, rank)
    holds_state = run is not None and rank not in set(plan.failed_ranks)
    local = run.local_state() if holds_state else {}

    acquire = {} if stage is None else acquire_layout(previous, plan, stage, rank)
    acquired = reshard_tp_state(
        local,
        layout=acquire,
        old_size=_gather_degree(previous),
        new_size=1 if stage is None else stage.tp_degree,
        new_rank=0 if stage is None else stage.tp_members.index(rank),
        group=control_group,
        checkpoint=anchor,
        verify=False,  # verified below, on what actually landed
    )

    recovered = None
    if stage is not None:
        # What this stage keeps: names it already holds in the right shape, i.e. every
        # owned name that neither moved stage nor changed degree.
        state = {name: local[name] for name in stage_layout(plan, stage) if name not in acquire}
        state.update(acquired)
        recovered = PlannedRun(
            plan,
            rank=rank,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            local_state={name: fields["param"] for name, fields in state.items()},
            moments={name: fields for name, fields in state.items() if "exp_avg" in fields},
            tp_group=tp_group,
            executor_group=executor_group,
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
        group=control_group,
        checkpoint=anchor,
        verify=True,
    )
    return recovered, logical_digest(full)
