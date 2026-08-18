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
from .planner.dp import DPPlacement, DPStage, DPTopology, assign
from .planner.pp import InfeasiblePP, repartition_pp
from .planner.tp import InfeasibleTP, choose_tp


#: Optimizer/parameter state that travels with a migrating logical layer.
MIGRATED_STATES = ("param", "grad", "exp_avg", "exp_avg_sq", "step")
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
    """How one replica's logical layer state reaches its new owner ranks."""

    replica_id: int
    layer: int
    target_ranks: tuple[int, ...]
    donor_kind: str  # prev_owner | peer_replica | checkpoint
    donor_ranks: tuple[int, ...]  # empty only for a checkpoint restore
    reshard: str  # peer_copy | gather_reshard | checkpoint_restore
    states: tuple[str, ...] = MIGRATED_STATES


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
                [r.replica_id, r.layer, list(r.target_ranks), r.donor_kind, list(r.donor_ranks), r.reshard, list(r.states)]
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

    @property
    def by_micro_batch(self) -> dict[int, tuple[DPPlacement, ...]]:
        result: dict[int, list[DPPlacement]] = {}
        for placement in self.placements:
            result.setdefault(placement.micro_batch, []).append(placement)
        return {key: tuple(value) for key, value in result.items()}


def _owner_ranges(stage_layers: tuple[int, ...]) -> tuple[tuple[int, int] | None, ...]:
    ranges = []
    start = 0
    for count in stage_layers:
        ranges.append((start, start + count) if count else None)
        start += count
    return tuple(ranges)


def _initial_stage_layers(num_layers: int, pp: int) -> tuple[int, ...]:
    base, extra = divmod(num_layers, pp)
    return tuple(base + (1 if stage < extra else 0) for stage in range(pp))


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


def _owner_of(stage_layout: dict[int, tuple[tuple[int, ...], tuple[int, int]]], layer: int):
    """Return ``(members, layer_range)`` of the stage owning ``layer``."""
    for members, layer_range in stage_layout.values():
        if layer_range[0] <= layer < layer_range[1]:
            return members
    raise AssertionError("layer is not owned by any stage in the layout")


def _state_route(
    config: TrainConfig,
    old_layout: dict[int, dict[int, tuple[tuple[int, ...], tuple[int, int]]]],
    replica: int,
    layer: int,
    old_members: tuple[int, ...],
    new_members: tuple[int, ...],
    failed_set: set[int],
) -> StateRoute:
    if all(rank not in failed_set for rank in old_members):
        reshard = "peer_copy" if len(old_members) == len(new_members) else "gather_reshard"
        return StateRoute(replica, layer, new_members, "prev_owner", old_members, reshard)
    for peer in sorted(old_layout):
        if peer == replica:
            continue
        peer_members = _owner_of(old_layout[peer], layer)
        if peer_members and all(rank not in failed_set for rank in peer_members):
            return StateRoute(replica, layer, new_members, "peer_replica", peer_members, "gather_reshard")
    return StateRoute(replica, layer, new_members, "checkpoint", (), "checkpoint_restore")


def build_plan(
    config: TrainConfig,
    *,
    step: int,
    version: int,
    failed_ranks: Iterable[int] = (),
    previous: "ExecutionPlan | None" = None,
    memory_budget: int | None = None,
    sequence_length: int = 1,
    vocab_size: int = 1,
    in_flight_micro_batches: int = 1,
) -> ExecutionPlan:
    """Chain TP, PP, and DP planning into one immutable, verified plan.

    Each stage's surviving ranks fix its TP degree/members via ``choose_tp``
    (memory-gated when a budget is given, so an oversized degree is rejected at
    the TP stage rather than masked as a DP failure). ``repartition_pp`` then
    redistributes contiguous layers for the new degrees relative to the previous
    layout -- a fully dead stage becomes zero layers and its layers move to the
    replica's survivors, so a single dead stage never discards a whole replica.
    ``assign`` finally distributes micro-batches across the surviving replicas.
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
    for replica in range(config.dp):
        old_layers = [0] * config.pp
        old_tp = [0] * config.pp
        for stage, (members, layer_range) in old_layout.get(replica, {}).items():
            old_layers[stage] = layer_range[1] - layer_range[0]
            old_tp[stage] = len(members)

        new_members: dict[int, tuple[int, ...]] = {}
        new_tp = [0] * config.pp
        for stage in range(config.pp):
            survivors = tuple(rank for rank in domains[replica][stage] if rank not in failed_set)
            if not survivors:
                continue
            if memory_budget is None:
                choice = choose_tp(config, active_ranks=survivors, min_degree=1)
            else:
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
                        in_flight_micro_batches=in_flight_micro_batches,
                    )
                except InfeasibleTP as error:
                    raise InfeasiblePlan(
                        PlanInfeasibleReason(error.reason.code, error.reason.message)
                    ) from error
            new_members[stage] = choice.members
            new_tp[stage] = choice.degree

        if not any(new_tp):
            continue  # every stage lost all ranks: this replica is truly dead

        try:
            pp_plan = repartition_pp(old_layers, old_tp, new_tp)
        except InfeasiblePP as error:
            raise InfeasiblePlan(PlanInfeasibleReason(error.reason.code, error.reason.message)) from error

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
        for layer in range(config.num_layers):
            old_members = _owner_of(old_layout[replica], layer)
            new_owner = _owner_of(new_stage_layout, layer)
            if old_members != new_owner:
                state_routes.append(
                    _state_route(config, old_layout, replica, layer, old_members, new_owner, failed_set)
                )

    if not dp_stages:
        raise InfeasiblePlan(
            PlanInfeasibleReason(
                "no_surviving_replica",
                "every DP replica lost all of its ranks",
            )
        )

    # Memory feasibility is owned by the TP stage above (choose_tp); the DP
    # layer only distributes micro-batches across the surviving replicas.
    topology = DPTopology(config=config, stages=tuple(dp_stages), micro_batches=micro_batches)
    assignment = assign(step, failed, topology)

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
    ``previous`` is given, also assert the cross-plan invariants: the version
    strictly increases, failed ranks grow, and active ranks shrink.
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

    for route in plan.state_routes:
        if route.donor_kind not in _DONOR_KINDS or route.reshard not in _RESHARD_METHODS:
            raise PlanInvariantError("state route has an unknown donor kind or reshard method")
        if not route.target_ranks or set(route.target_ranks) - active:
            raise PlanInvariantError("state route target ranks are empty or not active")
        if not route.states:
            raise PlanInvariantError("state route migrates no state")
        if route.donor_kind == "checkpoint":
            if route.donor_ranks or route.reshard != "checkpoint_restore":
                raise PlanInvariantError("checkpoint route must have no donor ranks")
        elif not route.donor_ranks or set(route.donor_ranks) & failed:
            raise PlanInvariantError("state route donor ranks are empty or failed")

    if previous is not None:
        if plan.version <= previous.version:
            raise PlanInvariantError("plan version must strictly increase")
        if not active <= set(previous.active_ranks):
            raise PlanInvariantError("active ranks must shrink monotonically")
        if not set(previous.failed_ranks) <= failed:
            raise PlanInvariantError("failed ranks must grow monotonically")
