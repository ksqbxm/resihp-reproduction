"""Deterministic data-parallel micro-batch rerouting."""

from dataclasses import dataclass
from typing import Iterable

from ..config import TrainConfig
from ..memory import memory_feasible
from ..validate import non_negative_int, positive_int, unique_ranks
from .pp import peak_in_flight


@dataclass(frozen=True)
class DPStage:
    """One logical PP stage in one DP replica."""

    replica_id: int
    stage_id: int
    ranks: tuple[int, ...]
    stage_layers: int
    tp_degree: int
    capacity: int | None = None


@dataclass(frozen=True)
class DPTopology:
    """The active topology and inputs needed for deterministic rerouting."""

    config: TrainConfig
    stages: tuple[DPStage, ...]
    micro_batches: int
    sequence_length: int = 1
    vocab_size: int = 1
    memory_budget: int | None = None


@dataclass(frozen=True)
class DPPlacement:
    """The executor for one logical micro-batch and PP stage."""

    micro_batch: int
    stage_id: int
    replica_id: int
    executor_ranks: tuple[int, ...]


@dataclass(frozen=True)
class DPAssignment:
    """Complete, immutable assignment for one training iteration."""

    step: int
    failure_signature: tuple[int, ...]
    placements: tuple[DPPlacement, ...]

    @property
    def by_micro_batch(self) -> dict[int, tuple[DPPlacement, ...]]:
        result: dict[int, list[DPPlacement]] = {}
        for placement in self.placements:
            result.setdefault(placement.micro_batch, []).append(placement)
        return {key: tuple(value) for key, value in result.items()}

    @property
    def micro_batches_per_replica(self) -> dict[int, int]:
        per_replica: dict[int, set[int]] = {}
        for placement in self.placements:
            per_replica.setdefault(placement.replica_id, set()).add(placement.micro_batch)
        return {replica: len(micro_batches) for replica, micro_batches in per_replica.items()}


@dataclass(frozen=True)
class DPInfeasibleReason:
    code: str
    message: str


class InfeasibleDP(ValueError):
    """Raised when no healthy DP replica can accept the micro-batch workload."""

    def __init__(self, reason: DPInfeasibleReason):
        super().__init__(reason.message)
        self.reason = reason


def _validate_topology(topology: DPTopology) -> None:
    if not isinstance(topology, DPTopology):
        raise TypeError("active_topology must be a DPTopology")
    positive_int("micro_batches", topology.micro_batches)
    positive_int("sequence_length", topology.sequence_length)
    positive_int("vocab_size", topology.vocab_size)
    if topology.memory_budget is not None:
        positive_int("memory_budget", topology.memory_budget)
    if not topology.stages:
        raise ValueError("active_topology must contain at least one stage")
    seen: set[tuple[int, int]] = set()
    for stage in topology.stages:
        non_negative_int("replica_id", stage.replica_id)
        non_negative_int("stage_id", stage.stage_id)
        positive_int("stage_layers", stage.stage_layers)
        positive_int("tp_degree", stage.tp_degree)
        if len(unique_ranks("stage ranks", stage.ranks)) != stage.tp_degree:
            raise ValueError("stage ranks length must equal tp_degree")
        if (stage.replica_id, stage.stage_id) in seen:
            raise ValueError("active_topology contains duplicate replica/stage")
        seen.add((stage.replica_id, stage.stage_id))
        if stage.capacity is not None:
            positive_int("capacity", stage.capacity)


def _stage_capacity(topology: DPTopology, stage: DPStage, *, index: int, num_stages: int) -> int:
    """Micro-batches this stage can take, or 0 when the budget cannot hold it.

    This is the first point that sees the layout actually being published, which is why
    plan 3.5 puts the ``MemoryFeasible`` gate here -- on the very calculator the TP
    ``k_min`` search uses, never a second formula. The stage's in-flight activation
    count is replayed from its own position in the 1F1B schedule the runtime executes,
    so the budget and the runtime speak about the same peak.
    """
    if topology.memory_budget is not None and not memory_feasible(
        topology.config,
        tp_degree=stage.tp_degree,
        stage_layers=stage.stage_layers,
        micro_batches=topology.micro_batches,
        sequence_length=topology.sequence_length,
        vocab_size=topology.vocab_size,
        memory_budget=topology.memory_budget,
        in_flight_micro_batches=peak_in_flight(
            topology.micro_batches, stage_index=index, num_stages=num_stages
        ),
    ):
        return 0
    return stage.capacity if stage.capacity is not None else topology.micro_batches


def _allocate(total: int, capacities: tuple[int, ...]) -> tuple[int, ...]:
    """Allocate by capacity proportion, using candidate order for all ties."""
    capacity_sum = sum(capacities)
    if capacity_sum <= 0:
        return tuple(0 for _ in capacities)
    base = tuple(total * capacity // capacity_sum for capacity in capacities)
    remaining = total - sum(base)
    order = sorted(
        range(len(capacities)),
        key=lambda index: (-(total * capacities[index] % capacity_sum), index),
    )
    result = list(base)
    for index in order[:remaining]:
        result[index] += 1
    return tuple(result)


def assign(
    step: int,
    failure_signature: Iterable[int],
    active_topology: DPTopology,
) -> DPAssignment:
    """Assign every micro-batch exactly once to each logical PP stage.

    The failure signature is cumulative. Each DP replica runs its own set of
    active PP stages (replicas may differ after a PP repartition has moved a dead
    stage's layers elsewhere); a replica is a valid target only when none of its
    stages have a failed TP member and every stage fits the memory budget.
    Surviving replicas are considered in replica-ID order. Each micro-batch flows
    through a single replica across that replica's stages, so its executor is
    consistent along the whole pipeline. No runtime timing or progress information
    participates in the result.
    """
    step = non_negative_int("step", step)
    _validate_topology(active_topology)
    failures = unique_ranks("failure_signature", failure_signature)
    failed = set(failures)

    replicas: dict[int, dict[int, DPStage]] = {}
    for stage in active_topology.stages:
        replicas.setdefault(stage.replica_id, {})[stage.stage_id] = stage

    candidates: list[tuple[tuple[DPStage, ...], int]] = []
    for replica_id in sorted(replicas):
        pipeline = tuple(stage for _, stage in sorted(replicas[replica_id].items()))
        if any(failed.intersection(stage.ranks) for stage in pipeline):
            continue
        capacity = min(
            _stage_capacity(active_topology, stage, index=index, num_stages=len(pipeline))
            for index, stage in enumerate(pipeline)
        )
        if capacity > 0:
            candidates.append((pipeline, capacity))

    if not candidates:
        reason = DPInfeasibleReason(
            code="no_feasible_dp_target",
            message="no healthy DP replica can accept the micro-batches",
        )
        raise InfeasibleDP(reason)

    capacities = tuple(capacity for _, capacity in candidates)
    allocation = _allocate(active_topology.micro_batches, capacities)

    placements: list[DPPlacement] = []
    micro_batch = 0
    for (pipeline, _), count in zip(candidates, allocation):
        for _ in range(count):
            for stage in pipeline:
                placements.append(
                    DPPlacement(
                        micro_batch=micro_batch,
                        stage_id=stage.stage_id,
                        replica_id=stage.replica_id,
                        executor_ranks=stage.ranks,
                    )
                )
            micro_batch += 1

    placements.sort(key=lambda placement: (placement.micro_batch, placement.stage_id))
    return DPAssignment(
        step=step,
        failure_signature=failures,
        placements=tuple(placements),
    )

