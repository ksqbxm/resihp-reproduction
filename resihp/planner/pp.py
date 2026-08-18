"""Deterministic pipeline-parallel repartition planning."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class PPInfeasibleReason:
    code: str
    message: str


class InfeasiblePP(ValueError):
    """Raised when no executable PP partition can be produced."""

    def __init__(self, reason: PPInfeasibleReason):
        super().__init__(reason.message)
        self.reason = reason


@dataclass(frozen=True)
class PPPlan:
    stage_layers: tuple[int, ...]
    layer_ranges: tuple[tuple[int, int] | None, ...]
    migrations: tuple[tuple[int, int, int], ...]
    embedding_owner: int
    lm_head_owner: int


def _integer_sequence(name: str, values) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in result):
        raise ValueError(f"{name} must contain integers")
    return tuple(int(value) for value in result)


def _owner_ranges(stage_layers: tuple[int, ...]) -> tuple[tuple[int, int] | None, ...]:
    ranges = []
    start = 0
    for count in stage_layers:
        ranges.append((start, start + count) if count else None)
        start += count
    return tuple(ranges)


def _stage_for_layer(ranges: tuple[tuple[int, int] | None, ...], layer: int) -> int:
    for stage, bounds in enumerate(ranges):
        if bounds is not None and bounds[0] <= layer < bounds[1]:
            return stage
    raise AssertionError("layer ranges do not cover the model")


def _ratio_less(left: int, right: int, target: list[int], tp_degrees: tuple[int, ...]) -> bool:
    return target[left] * tp_degrees[right] < target[right] * tp_degrees[left]


def _least_loaded(stages: list[int], target: list[int], tp_degrees: tuple[int, ...]) -> int:
    best = stages[0]
    for stage in stages[1:]:
        if _ratio_less(stage, best, target, tp_degrees) or (
            not _ratio_less(best, stage, target, tp_degrees) and stage < best
        ):
            best = stage
    return best


def _most_loaded(stages: list[int], target: list[int], tp_degrees: tuple[int, ...]) -> int:
    best = stages[0]
    for stage in stages[1:]:
        if _ratio_less(best, stage, target, tp_degrees) or (
            not _ratio_less(stage, best, target, tp_degrees) and stage < best
        ):
            best = stage
    return best


def _adjust_to_total(
    target: list[int],
    *,
    total_layers: int,
    tp_degrees: tuple[int, ...],
) -> None:
    active = [stage for stage, degree in enumerate(tp_degrees) if degree > 0]
    if not active:
        raise InfeasiblePP(
            PPInfeasibleReason("no_executable_pp", "all PP stages have zero TP capacity")
        )
    if len(active) > total_layers:
        raise InfeasiblePP(
            PPInfeasibleReason(
                "no_executable_pp",
                "active PP stages outnumber model layers",
            )
        )

    difference = total_layers - sum(target)
    while difference > 0:
        target[_least_loaded(active, target, tp_degrees)] += 1
        difference -= 1
    while difference < 0:
        removable = [stage for stage in active if target[stage] > 1]
        if not removable:
            raise InfeasiblePP(
                PPInfeasibleReason(
                    "no_executable_pp",
                    "active PP stages cannot retain at least one layer each",
                )
            )
        target[_most_loaded(removable, target, tp_degrees)] -= 1
        difference += 1


def repartition_pp(old_stage_layers, old_tp_degrees, new_tp_degrees) -> PPPlan:
    """Repartition contiguous layers after a TP-degree change.

    ``new_tp_degrees`` contains new TP capacities, not target layer counts.
    Old stages may already be empty after an earlier repartition; an empty
    stage must be represented by ``old_stage_layers == old_tp_degrees == 0``.
    """
    old_layers = _integer_sequence("old_stage_layers", old_stage_layers)
    old_tp = _integer_sequence("old_tp_degrees", old_tp_degrees)
    new_tp = _integer_sequence("new_tp_degrees", new_tp_degrees)
    if len(old_layers) != len(old_tp) or len(old_layers) != len(new_tp):
        raise ValueError("stage layer and TP degree sequences must have the same length")
    if any(count < 0 for count in old_layers):
        raise ValueError("old_stage_layers must contain non-negative integers")
    if any(degree < 0 for degree in old_tp):
        raise ValueError("old_tp_degrees must contain non-negative integers")
    if any(degree < 0 for degree in new_tp):
        raise ValueError("new_tp_degrees must contain non-negative integers")
    if any((count == 0) != (degree == 0) for count, degree in zip(old_layers, old_tp)):
        raise ValueError("empty old stages must have zero TP degree")

    total_layers = sum(old_layers)
    if total_layers == 0:
        raise ValueError("old_stage_layers must contain at least one layer")

    target = [
        0
        if new_degree == 0 or old_degree == 0
        else max(1, old_count * new_degree // old_degree)
        for old_count, old_degree, new_degree in zip(old_layers, old_tp, new_tp)
    ]
    _adjust_to_total(target, total_layers=total_layers, tp_degrees=new_tp)
    new_layers = tuple(target)

    if sum(new_layers) != total_layers:
        raise AssertionError("PP layer count is not conserved")
    if any(
        (degree == 0 and count != 0) or (degree > 0 and count < 1)
        for count, degree in zip(new_layers, new_tp)
    ):
        raise AssertionError("PP stage executability invariant failed")

    old_ranges = _owner_ranges(old_layers)
    new_ranges = _owner_ranges(new_layers)
    migrations = tuple(
        (layer, old_owner, new_owner)
        for layer in range(total_layers)
        for old_owner in (_stage_for_layer(old_ranges, layer),)
        for new_owner in (_stage_for_layer(new_ranges, layer),)
        if old_owner != new_owner
    )
    active = [stage for stage, degree in enumerate(new_tp) if degree > 0]
    return PPPlan(
        stage_layers=new_layers,
        layer_ranges=new_ranges,
        migrations=migrations,
        embedding_owner=active[0],
        lm_head_owner=active[-1],
    )
