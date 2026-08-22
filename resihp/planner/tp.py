"""Deterministic tensor-parallel candidate selection."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, NamedTuple

from ..config import TrainConfig
from ..memory import memory_feasible


@dataclass(frozen=True)
class TPInfeasibleReason:
    code: str
    message: str
    active_ranks: tuple[int, ...]


class InfeasibleTP(ValueError):
    """Raised when no TP degree satisfies the locked constraints."""

    def __init__(self, reason: TPInfeasibleReason):
        super().__init__(reason.message)
        self.reason = reason


@dataclass(frozen=True)
class TPChoice:
    degree: int
    members: tuple[int, ...]


class _MemoryInputs(NamedTuple):
    sequence_length: int
    memory_budget: int
    in_flight_micro_batches: int


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _normalize_active_ranks(active_ranks: Iterable[int]) -> tuple[int, ...]:
    ranks = tuple(_non_negative_integer("active_ranks", rank) for rank in active_ranks)
    if len(set(ranks)) != len(ranks):
        raise ValueError("active_ranks must not contain duplicate ranks")
    return tuple(sorted(ranks))


def _powers_of_two_between(limit: int, minimum: int) -> tuple[int, ...]:
    candidates = []
    degree = 1
    while degree <= limit:
        if degree >= minimum:
            candidates.append(degree)
        degree *= 2
    return tuple(candidates)


def _validated_inputs(
    config: TrainConfig,
    *,
    active_ranks: Iterable[int],
    min_degree: int,
    stage_layers: int | None,
    micro_batches: int | None,
    sequence_length: int | None,
    vocab_size: int | None,
    memory_budget: int | None,
    in_flight_micro_batches: int | None,
) -> tuple[tuple[int, ...], int, int, int | None, _MemoryInputs | None]:
    min_degree = _positive_integer("min_degree", min_degree)
    ranks = _normalize_active_ranks(active_ranks)
    if config.model_dim % config.num_heads:
        raise ValueError("model_dim must be divisible by num_heads")

    layers = config.num_layers // config.pp if stage_layers is None else stage_layers
    batches = config.batch_size // config.micro_batch_size if micro_batches is None else micro_batches
    layers = _positive_integer("stage_layers", layers)
    batches = _positive_integer("micro_batches", batches)

    vocab = None if vocab_size is None else _positive_integer("vocab_size", vocab_size)

    if memory_budget is None:
        if sequence_length is not None:
            _positive_integer("sequence_length", sequence_length)
        if in_flight_micro_batches is not None:
            _positive_integer("in_flight_micro_batches", in_flight_micro_batches)
        return ranks, layers, batches, vocab, None

    budget = _positive_integer("memory_budget", memory_budget)
    if sequence_length is None or vocab is None or in_flight_micro_batches is None:
        raise ValueError(
            "sequence_length, vocab_size, and in_flight_micro_batches are required when memory_budget is set"
        )
    memory_inputs = _MemoryInputs(
        sequence_length=_positive_integer("sequence_length", sequence_length),
        memory_budget=budget,
        in_flight_micro_batches=_positive_integer(
            "in_flight_micro_batches", in_flight_micro_batches
        ),
    )
    return ranks, layers, batches, vocab, memory_inputs


def _feasible_degrees(
    config: TrainConfig,
    *,
    ranks: tuple[int, ...],
    min_degree: int,
    stage_layers: int,
    micro_batches: int,
    vocab_size: int | None,
    memory_inputs: _MemoryInputs | None,
) -> tuple[int, ...]:
    """Degrees a stage could actually be built at, ascending.

    Every static topology constraint that would make the new TP layout unbuildable is
    applied here, so an infeasible degree becomes the structured ``no_feasible_tp``
    reason instead of a ``ValueError`` when the runtime tries to construct the stage:
    the attention heads and the TP-sharded model dimensions must divide by the degree,
    and so must the vocabulary, because the token embedding and the LM head are
    vocab-parallel (:class:`resihp.parallel.tp.TensorParallelStage`).
    """
    result = []
    for degree in _powers_of_two_between(len(ranks), min_degree):
        if config.model_dim % degree or config.num_heads % degree:
            continue
        if vocab_size is not None and vocab_size % degree:
            continue
        # Every resident term the budget checks -- including the embedding/LM-head
        # boundary_parameters -- is sharded by ``degree``, so the whole footprint
        # scales with k. None of it is a fixed per-rank overhead in this k_min search.
        if memory_inputs is not None and not memory_feasible(
            config,
            tp_degree=degree,
            stage_layers=stage_layers,
            micro_batches=micro_batches,
            sequence_length=memory_inputs.sequence_length,
            vocab_size=vocab_size,
            memory_budget=memory_inputs.memory_budget,
            in_flight_micro_batches=memory_inputs.in_flight_micro_batches,
        ):
            continue
        result.append(degree)
    return tuple(result)


def feasible_degrees(
    config: TrainConfig,
    *,
    active_ranks: Iterable[int],
    min_degree: int,
    stage_layers: int | None = None,
    micro_batches: int | None = None,
    sequence_length: int | None = None,
    vocab_size: int | None = None,
    memory_budget: int | None = None,
    in_flight_micro_batches: int | None = None,
) -> tuple[int, ...]:
    """Return eligible TP degrees in ascending order.

    ``active_ranks`` must already be the live rank set for one stage / physical
    communication domain: ``G' = stage_ranks - F_stop``. This function only
    sorts that set and chooses deterministic members from it; it does not infer
    stage membership or physical domains from global ranks. ``min_degree`` is a
    lower bound, so non-power-of-two values select the next power-of-two TP
    candidate that satisfies ``k >= min_degree``. ``vocab_size``, when given, also
    filters degrees that do not divide the vocabulary -- the embedding and LM head are
    vocab-parallel, so such a degree cannot be built at all. When ``memory_budget`` is
    set, callers must explicitly pass ``sequence_length``, ``vocab_size``, and the
    stage-specific 1F1B ``in_flight_micro_batches`` to avoid undercounting peak
    activation memory.
    """
    ranks, layers, batches, vocab, memory_inputs = _validated_inputs(
        config,
        active_ranks=active_ranks,
        min_degree=min_degree,
        stage_layers=stage_layers,
        micro_batches=micro_batches,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        memory_budget=memory_budget,
        in_flight_micro_batches=in_flight_micro_batches,
    )
    return _feasible_degrees(
        config,
        ranks=ranks,
        min_degree=min_degree,
        stage_layers=layers,
        micro_batches=batches,
        vocab_size=vocab,
        memory_inputs=memory_inputs,
    )


def choose_tp(
    config: TrainConfig,
    *,
    active_ranks: Iterable[int],
    min_degree: int,
    stage_layers: int | None = None,
    micro_batches: int | None = None,
    sequence_length: int | None = None,
    vocab_size: int | None = None,
    memory_budget: int | None = None,
    in_flight_micro_batches: int | None = None,
) -> TPChoice:
    """Choose the maximum eligible degree and ascending stage-local members."""
    ranks, layers, batches, vocab, memory_inputs = _validated_inputs(
        config,
        active_ranks=active_ranks,
        min_degree=min_degree,
        stage_layers=stage_layers,
        micro_batches=micro_batches,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        memory_budget=memory_budget,
        in_flight_micro_batches=in_flight_micro_batches,
    )
    degrees = _feasible_degrees(
        config,
        ranks=ranks,
        min_degree=min_degree,
        stage_layers=layers,
        micro_batches=batches,
        vocab_size=vocab,
        memory_inputs=memory_inputs,
    )
    if not degrees:
        reason = TPInfeasibleReason(
            code="no_feasible_tp",
            message=(
                "no TP degree satisfies active-rank, divisibility (heads, model "
                "dimension, vocabulary), and memory constraints"
            ),
            active_ranks=ranks,
        )
        raise InfeasibleTP(reason)
    degree = degrees[-1]
    return TPChoice(degree=degree, members=ranks[:degree])
