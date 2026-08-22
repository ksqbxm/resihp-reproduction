"""Immutable, versioned execution plan assembled from the planner primitives.

A plan is a deterministic function of ``(config, step, version, cumulative
failed ranks, previous plan)``. It is rebuilt from scratch on every fail-stop
event; the ``previous`` plan supplies the *old* layout so that layer and state
migrations describe ``previous -> current`` rather than ``initial config ->
current``. Communication ranks, TP shards, and PP owners are read only from the
plan, never inferred from an older layout at run time.

Rank layout convention: ``rank = ((replica * PP + stage) * TP + tp_local)``, so
each ``(replica, stage)`` owns a contiguous block of ``TP`` ranks.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Iterable

from .config import TrainConfig
from .planner.dp import DPPlacement, DPStage, DPTopology, InfeasibleDP, assign
from .planner.pp import InfeasiblePP, balanced_layers, peak_in_flight, repartition_pp
from .planner.tp import InfeasibleTP, choose_tp


#: The persistent training state a migrating logical tensor carries. ``grad`` is
#: deliberately absent: a safe point is reached only after the current iteration's
#: AdamW step has been applied, so the gradients then resident are spent values the
#: next iteration recomputes from scratch. They are in no checkpoint and in no route.
RECOVERED_STATES = ("param", "exp_avg", "exp_avg_sq", "step")
#: State groups that belong to a replica's first / last executable stage rather than
#: to any one layer (plan 3.4), and so are routed as units of their own.
BOUNDARY_GROUPS = ("embedding", "head")
_DONOR_KINDS = frozenset({"prev_owner", "peer_replica", "checkpoint"})
_RESHARD_METHODS = frozenset({"peer_copy", "gather_reshard", "checkpoint_restore"})


@dataclass(frozen=True)
class StagePlan:
    """TP membership and contiguous layer ownership for one active stage."""

    replica_id: int
    stage_id: int
    tp_degree: int
    tp_members: tuple[int, ...]
    layer_range: tuple[int, int]

    @property
    def stage_layers(self) -> int:
        return self.layer_range[1] - self.layer_range[0]


@dataclass(frozen=True)
class StateRoute:
    """How one group of a replica's logical state reaches its new owner ranks.

    The plan's executable recovery instruction, not a description of one:
    :func:`resihp.recovery.recover` reads these routes to decide what this rank must
    fetch, under which source layout, into which target layout. It performs no routing
    decision of its own.

    A route covers one **state group** -- either a single global layer (``layer`` set,
    ``boundary`` empty) or one of :data:`BOUNDARY_GROUPS` (``layer`` ``None``), which
    follow the replica's first / last executable stage rather than any layer.
    ``donor_degree`` / ``target_degree`` are the TP layouts the state is read under and
    written into; ``donor_degree`` is 0 exactly for a checkpoint restore.
    """

    replica_id: int
    layer: int | None
    boundary: str  # "" for a layer route, else one of BOUNDARY_GROUPS
    target_ranks: tuple[int, ...]
    target_degree: int
    donor_kind: str  # prev_owner | peer_replica | checkpoint
    donor_ranks: tuple[int, ...]  # empty only for a checkpoint restore
    donor_degree: int
    reshard: str  # peer_copy | gather_reshard | checkpoint_restore
    states: tuple[str, ...] = RECOVERED_STATES


@dataclass(frozen=True)
class PlanInfeasibleReason:
    code: str
    message: str


class InfeasiblePlan(ValueError):
    """Raised when no executable plan exists for the current failure signature."""

    def __init__(self, reason: PlanInfeasibleReason):
        super().__init__(reason.message)
        self.reason = reason


class PlanInvariantError(ValueError):
    """Raised when an ExecutionPlan violates a locked structural invariant."""


@dataclass(frozen=True)
class ExecutionPlan:
    version: int
    step: int
    config: TrainConfig
    failed_ranks: tuple[int, ...]
    active_ranks: tuple[int, ...]  # ranks assigned to an active stage this plan
    stages: tuple[StagePlan, ...]
    placements: tuple[DPPlacement, ...]
    state_routes: tuple[StateRoute, ...]

    def _canonical(self) -> dict:
        config = self.config
        return {
            "version": self.version,
            "step": self.step,
            "config": {
                "model_dim": config.model_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "batch_size": config.batch_size,
                "micro_batch_size": config.micro_batch_size,
                "tp": config.tp,
                "pp": config.pp,
                "dp": config.dp,
            },
            "failed_ranks": list(self.failed_ranks),
            "active_ranks": list(self.active_ranks),
            "stages": [
                [s.replica_id, s.stage_id, s.tp_degree, list(s.tp_members), list(s.layer_range)]
                for s in self.stages
            ],
            "placements": [
                [p.micro_batch, p.stage_id, p.replica_id, list(p.executor_ranks)]
                for p in self.placements
            ],
            "state_routes": [
                [
                    r.replica_id,
                    r.layer,
                    r.boundary,
                    list(r.target_ranks),
                    r.target_degree,
                    r.donor_kind,
                    list(r.donor_ranks),
                    r.donor_degree,
                    r.reshard,
                    list(r.states),
                ]
                for r in self.state_routes
            ],
        }

    @property
    def digest(self) -> str:
        """Stable hash over the whole normalized plan, shared by every rank."""
        canonical = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def live_ranks(self) -> tuple[int, ...]:
        """All ranks that have not failed (a superset of ``active_ranks``)."""
        return tuple(rank for rank in range(self.config.world_size) if rank not in set(self.failed_ranks))


def _owner_ranges(stage_layers: tuple[int, ...]) -> tuple[tuple[int, int] | None, ...]:
    ranges = []
    start = 0
    for count in stage_layers:
        ranges.append((start, start + count) if count else None)
        start += count
    return tuple(ranges)


def _initial_stage_layers(num_layers: int, pp: int) -> tuple[int, ...]:
    """Layer counts of the pristine partition, from the planner's one definition."""
    return tuple(len(stage) for stage in balanced_layers(num_layers, pp))


def _stage_domains(config: TrainConfig) -> dict[int, dict[int, tuple[int, ...]]]:
    domains: dict[int, dict[int, tuple[int, ...]]] = {}
    for replica in range(config.dp):
        for stage in range(config.pp):
            first = (replica * config.pp + stage) * config.tp
            domains.setdefault(replica, {})[stage] = tuple(range(first, first + config.tp))
    return domains


def _normalize_failed(failed_ranks: Iterable[int], world_size: int) -> tuple[int, ...]:
    ranks = []
    for rank in failed_ranks:
        if not isinstance(rank, Integral) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError(f"failed_ranks contains an out-of-range rank: {rank!r}")
        ranks.append(int(rank))
    if len(set(ranks)) != len(ranks):
        raise ValueError("failed_ranks must not contain duplicate ranks")
    return tuple(sorted(ranks))


def _old_layout(config: TrainConfig, previous: "ExecutionPlan | None") -> dict[int, dict[int, tuple[tuple[int, ...], tuple[int, int]]]]:
    """Return ``replica -> stage -> (members, layer_range)`` for the old plan.

    With no previous plan the old layout is the pristine initial topology.
    """
    if previous is not None:
        layout: dict[int, dict[int, tuple[tuple[int, ...], tuple[int, int]]]] = {}
        for stage in previous.stages:
            layout.setdefault(stage.replica_id, {})[stage.stage_id] = (stage.tp_members, stage.layer_range)
        return layout

    domains = _stage_domains(config)
    ranges = _owner_ranges(_initial_stage_layers(config.num_layers, config.pp))
    return {
        replica: {stage: (domains[replica][stage], ranges[stage]) for stage in range(config.pp)}
        for replica in range(config.dp)
    }


def _owner_of(stage_layout: dict[int, tuple[tuple[int, ...], tuple[int, int]]], layer: int) -> tuple[int, ...]:
    """Return the TP members of the stage owning ``layer``."""
    for members, layer_range in stage_layout.values():
        if layer_range[0] <= layer < layer_range[1]:
            return members
    raise AssertionError("layer is not owned by any stage in the layout")


def _boundary_owner_of(
    stage_layout: dict[int, tuple[tuple[int, ...], tuple[int, int]]], boundary: str
) -> tuple[int, ...]:
    """TP members of the stage owning ``boundary`` -- the first / last executable one."""
    ordered = sorted(stage_layout)
    return stage_layout[ordered[0 if boundary == "embedding" else -1]][0]


def _group_owner(
    stage_layout: dict[int, tuple[tuple[int, ...], tuple[int, int]]],
    layer: int | None,
    boundary: str,
) -> tuple[int, ...]:
    """TP members currently owning one state group, layer or boundary alike."""
    return _owner_of(stage_layout, layer) if boundary == "" else _boundary_owner_of(stage_layout, boundary)


def _repartition(old_layers, old_tp, new_tp):
    """``repartition_pp``, with its structured reason carried into the plan's own."""
    try:
        return repartition_pp(old_layers, old_tp, new_tp)
    except InfeasiblePP as error:
        raise InfeasiblePlan(
            PlanInfeasibleReason(error.reason.code, error.reason.message)
        ) from error


def _state_route(
    old_layout: dict[int, dict[int, tuple[tuple[int, ...], tuple[int, int]]]],
    replica: int,
    layer: int | None,
    boundary: str,
    old_members: tuple[int, ...],
    new_members: tuple[int, ...],
    failed_set: set[int],
) -> StateRoute:
    """Pick the donor for one state group: own previous owner, healthy peer, checkpoint.

    Plan 3.3/3.6 fix this order and admit no other: the state is collected from a
    healthy replica whenever one still holds it, and the pre-failure checkpoint is
    consulted only when it survives nowhere. Recovery executes the choice made here.
    """

    def route(kind: str, donors: tuple[int, ...], method: str) -> StateRoute:
        return StateRoute(
            replica_id=replica,
            layer=layer,
            boundary=boundary,
            target_ranks=new_members,
            target_degree=len(new_members),
            donor_kind=kind,
            donor_ranks=donors,
            donor_degree=len(donors),
            reshard=method,
        )

    if all(rank not in failed_set for rank in old_members):
        method = "peer_copy" if len(old_members) == len(new_members) else "gather_reshard"
        return route("prev_owner", old_members, method)
    for peer in sorted(old_layout):
        if peer == replica:
            continue
        peer_members = _group_owner(old_layout[peer], layer, boundary)
        if all(rank not in failed_set for rank in peer_members):
            return route("peer_replica", peer_members, "gather_reshard")
    return route("checkpoint", (), "checkpoint_restore")


def _state_groups(num_layers: int) -> tuple[tuple[int | None, str], ...]:
    """Every routable state group of one replica: each layer, then the two boundaries."""
    return tuple((layer, "") for layer in range(num_layers)) + tuple(
        (None, boundary) for boundary in BOUNDARY_GROUPS
    )


def boundary_pairs(plan: "ExecutionPlan") -> tuple[tuple[int, int], ...]:
    """Sorted leader pairs of every pipeline hop this plan's assignment creates.

    The pipeline boundaries are read from the assignment, so a hop exists exactly where
    a micro-batch actually moves between two stages. Each pair gets its own two-rank
    process group: the 1F1B steady state issues a fused ``batch_isend_irecv``, which
    NCCL runs on the group's collective communicator, so the group must hold exactly
    the two ranks taking part (see :mod:`resihp.parallel.pp`).
    """
    pipelines: dict[int, list[DPPlacement]] = {}
    for placement in plan.placements:
        pipelines.setdefault(placement.micro_batch, []).append(placement)
    pairs = set()
    for places in pipelines.values():
        ordered = sorted(places, key=lambda placement: placement.stage_id)
        for upstream, downstream in zip(ordered, ordered[1:]):
            leaders = (upstream.executor_ranks[0], downstream.executor_ranks[0])
            pairs.add((min(leaders), max(leaders)))
    return tuple(sorted(pairs))


def build_plan(
    config: TrainConfig,
    *,
    step: int,
    version: int,
    failed_ranks: Iterable[int] = (),
    previous: "ExecutionPlan | None" = None,
    memory_budget: int | None = None,
    sequence_length: int = 1,
    vocab_size: int | None = None,
) -> ExecutionPlan:
    """Chain TP, PP, and DP planning into one immutable, verified plan.

    Each stage's surviving ranks fix its TP degree/members via ``choose_tp``
    (memory-gated when a budget is given, so an oversized degree is rejected at
    the TP stage rather than masked as a DP failure). ``repartition_pp`` then
    redistributes contiguous layers for the new degrees relative to the previous
    layout -- a fully dead stage becomes zero layers and its layers move to the
    replica's survivors, so a single dead stage never discards a whole replica.
    ``assign`` finally distributes micro-batches across the surviving replicas.

    ``vocab_size`` is a topology constraint, not only a memory input: the token
    embedding and the LM head are sharded over vocabulary, so a degree that does not
    divide it cannot be built. Pass it and no such degree is ever selected; leave it
    out (as pure planning tests do) and only the dimension constraints apply. A
    ``memory_budget`` requires it either way.

    Peak activation memory is never guessed: each stage's in-flight micro-batch count
    comes from :func:`resihp.planner.pp.peak_in_flight`, replayed from the very 1F1B
    schedule :class:`resihp.parallel.pp.PipelineRuntime` executes.
    """
    if not isinstance(step, Integral) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(version, Integral) or isinstance(version, bool) or version < 0:
        raise ValueError("version must be a non-negative integer")
    if previous is not None and not isinstance(previous, ExecutionPlan):
        raise TypeError("previous must be an ExecutionPlan or None")

    failed = _normalize_failed(failed_ranks, config.world_size)
    failed_set = set(failed)
    micro_batches = config.batch_size // config.micro_batch_size
    domains = _stage_domains(config)
    old_layout = _old_layout(config, previous)

    stages: list[StagePlan] = []
    dp_stages: list[DPStage] = []
    state_routes: list[StateRoute] = []
    wiped_out: list[tuple[list[int], list[int], list[int]]] = []
    for replica in range(config.dp):
        old_layers = [0] * config.pp
        old_tp = [0] * config.pp
        for stage, (members, layer_range) in old_layout.get(replica, {}).items():
            old_layers[stage] = layer_range[1] - layer_range[0]
            old_tp[stage] = len(members)

        # The old pipeline is the only shape available when TP degrees are chosen --
        # the repartition needs those degrees before it can produce the new one -- so a
        # stage's 1F1B position is taken from it, and the in-flight count from the whole
        # global micro-batch set, an upper bound on any replica's share. Both over-
        # rather than under-state the peak, and the DP gate below re-checks every stage
        # against the layout that is actually published.
        old_active = [stage for stage in range(config.pp) if old_tp[stage] > 0]

        new_members: dict[int, tuple[int, ...]] = {}
        new_tp = [0] * config.pp
        for stage in range(config.pp):
            survivors = tuple(rank for rank in domains[replica][stage] if rank not in failed_set)
            if not survivors:
                continue
            try:
                choice = choose_tp(
                    config,
                    active_ranks=survivors,
                    min_degree=1,
                    stage_layers=max(1, old_layers[stage]),
                    micro_batches=micro_batches,
                    sequence_length=sequence_length,
                    vocab_size=vocab_size,
                    memory_budget=memory_budget,
                    in_flight_micro_batches=peak_in_flight(
                        micro_batches,
                        stage_index=old_active.index(stage),
                        num_stages=len(old_active),
                    ),
                )
            except InfeasibleTP as error:
                raise InfeasiblePlan(
                    PlanInfeasibleReason(error.reason.code, error.reason.message)
                ) from error
            new_members[stage] = choice.members
            new_tp[stage] = choice.degree

        if not any(new_tp):
            # Every stage of this replica lost all its ranks. Other replicas may still
            # carry the run, so remember the input rather than deciding here -- but
            # only from a replica that still had a pipeline, since one emptied by an
            # earlier event holds no layers and says nothing about this one.
            if any(old_tp):
                wiped_out.append((old_layers, old_tp, new_tp))
            continue

        pp_plan = _repartition(old_layers, old_tp, new_tp)

        for stage in range(config.pp):
            layer_range = pp_plan.layer_ranges[stage]
            if layer_range is None:
                continue
            members = new_members[stage]
            stages.append(StagePlan(replica, stage, new_tp[stage], members, layer_range))
            dp_stages.append(DPStage(replica, stage, members, layer_range[1] - layer_range[0], new_tp[stage]))

        new_stage_layout = {
            stage: (new_members[stage], pp_plan.layer_ranges[stage])
            for stage in range(config.pp)
            if pp_plan.layer_ranges[stage] is not None
        }
        # A state group needs a route exactly when the TP members owning it change:
        # different stages hold disjoint rank domains, so that single test covers a
        # layer moving stage, a stage changing degree, a rank changing seat within its
        # stage, and a boundary tensor following a new first / last executable stage.
        for layer, boundary in _state_groups(config.num_layers):
            old_members = _group_owner(old_layout[replica], layer, boundary)
            new_owner = _group_owner(new_stage_layout, layer, boundary)
            if old_members != new_owner:
                state_routes.append(
                    _state_route(
                        old_layout, replica, layer, boundary, old_members, new_owner, failed_set
                    )
                )

    if not dp_stages:
        # No replica survived, so there is no pipeline left to lay out. That is the PP
        # planner's own condition, so ask it rather than inventing a second name: with
        # every stage at zero capacity it always raises ``no_executable_pp``. There is
        # always something to ask about, because the previous plan had at least one
        # replica with a pipeline and it either survives here or was just recorded.
        _repartition(*wiped_out[0])

    # ``choose_tp`` gates memory with each stage's *old* layer count -- it has to, the
    # repartition needs the new degrees first -- so ``repartition_pp`` can hand a stage
    # more layers than were ever checked. The DP planner is the first point that sees
    # the final layout, which is why plan 3.5 puts a ``MemoryFeasible`` gate here, on
    # the same calculator: a replica whose stages no longer fit stops being a target,
    # and when none is left the plan is infeasible.
    topology = DPTopology(
        config=config,
        stages=tuple(dp_stages),
        micro_batches=micro_batches,
        sequence_length=sequence_length,
        vocab_size=1 if vocab_size is None else vocab_size,
        memory_budget=memory_budget,
    )
    try:
        assignment = assign(step, failed, topology)
    except InfeasibleDP as error:
        raise InfeasiblePlan(
            PlanInfeasibleReason(error.reason.code, error.reason.message)
        ) from error

    active_ranks = tuple(sorted({rank for stage in stages for rank in stage.tp_members}))
    plan = ExecutionPlan(
        version=version,
        step=step,
        config=config,
        failed_ranks=failed,
        active_ranks=active_ranks,
        stages=tuple(stages),
        placements=assignment.placements,
        state_routes=tuple(state_routes),
    )
    assert_invariants(plan, previous=previous)
    return plan


def assert_invariants(plan: ExecutionPlan, previous: ExecutionPlan | None = None) -> None:
    """Assert every locked structural invariant (plan §4.B) for one plan.

    ``active_ranks`` are the ranks assigned to an active stage in this plan. When
    ``previous`` is given, also assert the cross-plan invariants: the version strictly
    increases and failures accumulate, which together with the single-plan check that
    active and failed ranks never overlap is exactly "a failed rank never re-enters
    the training path".
    """
    config = plan.config
    pp = config.pp
    micro_batches = config.batch_size // config.micro_batch_size

    active = set(plan.active_ranks)
    failed = set(plan.failed_ranks)
    if len(plan.active_ranks) != len(active):
        raise PlanInvariantError("active_ranks contains duplicate ranks")
    if active & failed:
        raise PlanInvariantError("active and failed ranks overlap")

    assigned = [rank for stage in plan.stages for rank in stage.tp_members]
    if len(assigned) != len(set(assigned)):
        raise PlanInvariantError("a rank is assigned to multiple stages")
    if set(assigned) != active:
        raise PlanInvariantError("assigned ranks do not match active ranks")

    stages_by_replica: dict[int, list[StagePlan]] = {}
    active_stage_ids: dict[int, set[int]] = {}
    for stage in plan.stages:
        if stage.tp_degree != len(stage.tp_members):
            raise PlanInvariantError("stage tp_degree does not match its member count")
        stages_by_replica.setdefault(stage.replica_id, []).append(stage)
        active_stage_ids.setdefault(stage.replica_id, set()).add(stage.stage_id)
    for replica, replica_stages in stages_by_replica.items():
        cursor = 0
        for stage in sorted(replica_stages, key=lambda s: s.stage_id):
            if stage.stage_layers < 1 or stage.layer_range != (cursor, cursor + stage.stage_layers):
                raise PlanInvariantError(f"replica {replica} layer ranges are not contiguous")
            cursor = stage.layer_range[1]
        if cursor != config.num_layers:
            raise PlanInvariantError(f"replica {replica} layers do not cover the model")

    members_by_key = {(s.replica_id, s.stage_id): s.tp_members for s in plan.stages}
    placements_by_mb: dict[int, list[DPPlacement]] = {}
    for placement in plan.placements:
        if members_by_key.get((placement.replica_id, placement.stage_id)) != placement.executor_ranks:
            raise PlanInvariantError("a placement executor does not match its stage TP members")
        placements_by_mb.setdefault(placement.micro_batch, []).append(placement)
    if set(placements_by_mb) != set(range(micro_batches)):
        raise PlanInvariantError("micro-batch coverage is incomplete")
    for micro_batch, places in placements_by_mb.items():
        replicas = {placement.replica_id for placement in places}
        if len(replicas) != 1:
            raise PlanInvariantError(f"micro-batch {micro_batch} is split across replicas")
        replica = replicas.pop()
        ran = [placement.stage_id for placement in places]
        if len(ran) != len(set(ran)):
            raise PlanInvariantError(f"micro-batch {micro_batch} runs a stage more than once")
        if set(ran) != active_stage_ids[replica]:
            raise PlanInvariantError(f"micro-batch {micro_batch} does not run its replica's stages exactly once")

    routed = set()
    for route in plan.state_routes:
        if route.donor_kind not in _DONOR_KINDS or route.reshard not in _RESHARD_METHODS:
            raise PlanInvariantError("state route has an unknown donor kind or reshard method")
        if (route.layer is None) == (route.boundary == ""):
            raise PlanInvariantError("state route must name exactly one layer or boundary")
        if route.boundary not in ("",) + BOUNDARY_GROUPS:
            raise PlanInvariantError("state route names an unknown boundary group")
        if (route.replica_id, route.layer, route.boundary) in routed:
            raise PlanInvariantError("a state group is routed more than once")
        routed.add((route.replica_id, route.layer, route.boundary))
        if not route.target_ranks or set(route.target_ranks) - active:
            raise PlanInvariantError("state route target ranks are empty or not active")
        if route.target_degree != len(route.target_ranks):
            raise PlanInvariantError("state route target degree does not match its target ranks")
        if route.donor_degree != len(route.donor_ranks):
            raise PlanInvariantError("state route donor degree does not match its donor ranks")
        if not route.states:
            raise PlanInvariantError("state route migrates no state")
        if "grad" in route.states:
            raise PlanInvariantError("gradients are not persistent state and are never routed")
        if route.donor_kind == "checkpoint":
            if route.donor_ranks or route.reshard != "checkpoint_restore":
                raise PlanInvariantError("checkpoint route must have no donor ranks")
        elif not route.donor_ranks or set(route.donor_ranks) & failed:
            raise PlanInvariantError("state route donor ranks are empty or failed")

    if previous is not None:
        if plan.version <= previous.version:
            raise PlanInvariantError("plan version must strictly increase")
        # Fail-stop's cross-plan guarantee is that a *failed* rank never returns to
        # the training path. This check plus the ``active & failed`` one above carry
        # it between them: failures accumulate, and no accumulated failure is active.
        # It is deliberately not "the active set only ever shrinks" -- a healthy rank
        # the previous plan left idle (``choose_tp`` takes the largest power-of-two
        # prefix of a stage's survivors, so it can leave one out) is a resource a later
        # plan may legitimately pick up again, which is not a resurrection.
        if not set(previous.failed_ranks) <= failed:
            raise PlanInvariantError("failed ranks must grow monotonically")
