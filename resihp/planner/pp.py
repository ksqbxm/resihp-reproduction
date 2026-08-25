"""Deterministic pipeline-parallel repartition planning and the 1F1B schedule.

The 1F1B schedule lives here, not in the runtime, because three callers must agree
on it and only one of them may import torch: :class:`resihp.parallel.pp.PipelineRuntime`
executes it, :mod:`resihp.plan` and :mod:`resihp.planner.dp` size peak activation
memory from it. Deriving :func:`peak_in_flight` from :func:`pipeline_schedule` rather
than from a closed form is what keeps the memory model and the runtime one semantics
(plan 3.4/3.5): change the schedule and the budget follows automatically.
"""

from dataclasses import dataclass

from ..validate import non_negative_int


def balanced_layers(num_layers: int, num_stages: int) -> tuple[tuple[int, ...], ...]:
    """Contiguous global layer ids per stage, the remainder going to earlier stages.

    The base partition a run starts from, before any fail-stop repartition. The one
    definition of that split: :func:`resihp.plan.build_plan` derives the initial
    layer counts from it too, so the pristine topology cannot drift between the
    planner and the plan.
    """
    if num_stages < 1 or num_layers < num_stages:
        raise ValueError("need at least one layer per stage")
    base, remainder = divmod(num_layers, num_stages)
    stages = []
    start = 0
    for stage in range(num_stages):
        count = base + (1 if stage < remainder else 0)
        stages.append(tuple(range(start, start + count)))
        start += count
    return tuple(stages)


def pipeline_phases(num_micro_batches: int, *, stage_index: int, num_stages: int) -> tuple[int, int]:
    """``(warmup, steady)`` micro-batch counts of one stage's 1F1B schedule.

    A stage warms up with one forward per stage still downstream of it (capped by the
    micro-batches it runs), then alternates one forward with one backward, then drains
    the warmup backwards. The last stage has no warmup and alternates throughout.
    """
    if num_stages < 1 or not 0 <= stage_index < num_stages:
        raise ValueError("stage_index must identify a stage of the pipeline")
    non_negative_int("num_micro_batches", num_micro_batches)
    warmup = min(num_stages - 1 - stage_index, num_micro_batches)
    return warmup, num_micro_batches - warmup


def pipeline_schedule(num_micro_batches: int, *, stage_index: int, num_stages: int) -> tuple[str, ...]:
    """The forward/backward order one stage issues: warmup -> steady 1F1B -> cooldown.

    ``"F"``/``"B"`` in issue order, which is exactly what
    :meth:`resihp.parallel.pp.PipelineRuntime.train_step` emits (a torch-gated test
    compares the two element for element, so they cannot drift).
    """
    warmup, steady = pipeline_phases(
        num_micro_batches, stage_index=stage_index, num_stages=num_stages
    )
    return ("F",) * warmup + ("F", "B") * steady + ("B",) * warmup


def peak_in_flight(num_micro_batches: int, *, stage_index: int, num_stages: int) -> int:
    """Most activations one stage holds at once, replayed from :func:`pipeline_schedule`.

    An activation is retained at its forward and released at its matching backward
    (plan 3.5), so replaying the schedule as +1/-1 gives the high-water mark the
    analytical memory model must budget. Derived rather than asserted, so the figure
    is the runtime's own behaviour and not a constant that can fall out of date.
    """
    live = peak = 0
    for operation in pipeline_schedule(
        num_micro_batches, stage_index=stage_index, num_stages=num_stages
    ):
        live += 1 if operation == "F" else -1
        peak = max(peak, live)
    return peak


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


def _stage_counts(name: str, values) -> tuple[int, ...]:
    """A per-stage sequence of non-negative counts: layer counts or TP degrees alike."""
    result = tuple(non_negative_int(name, value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def owner_ranges(stage_layers: tuple[int, ...]) -> tuple[tuple[int, int] | None, ...]:
    """Contiguous ``(start, end)`` layer range per stage, ``None`` for an empty stage.

    The one definition of "layer counts laid out into ownership ranges":
    :func:`resihp.plan.build_plan` lays the pristine partition out with it too, so the
    repartitioned layout and the initial one cannot be built two different ways.
    """
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
    old_layers = _stage_counts("old_stage_layers", old_stage_layers)
    old_tp = _stage_counts("old_tp_degrees", old_tp_degrees)
    new_tp = _stage_counts("new_tp_degrees", new_tp_degrees)
    if len(old_layers) != len(old_tp) or len(old_layers) != len(new_tp):
        raise ValueError("stage layer and TP degree sequences must have the same length")
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

    old_ranges = owner_ranges(old_layers)
    new_ranges = owner_ranges(new_layers)
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
