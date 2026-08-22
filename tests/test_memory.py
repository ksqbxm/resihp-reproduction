"""Tests for the single analytical memory calculator.

``in_flight_micro_batches`` has no default, so every call states the schedule position
it is budgeting for; the 1F1B gates below take it from
:func:`resihp.planner.pp.peak_in_flight` rather than a literal, which is what ties the
budget to the schedule the runtime actually executes.
"""

from dataclasses import replace

import pytest

from resihp.config import TrainConfig
from resihp.memory import memory_feasible, estimate_memory
from resihp.planner.pp import peak_in_flight


CONFIG = TrainConfig(
    model_dim=8,
    num_layers=2,
    num_heads=2,
    batch_size=4,
    micro_batch_size=2,
    seed=1,
    tp=2,
    pp=1,
    dp=1,
    iterations=1,
)


def test_known_memory_breakdown_matches_independent_hand_calculation():
    result = estimate_memory(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=1,
    )

    # Independent constants: layer shard 1600 B; fixed embedding/head shards 320 B.
    assert result.sharded_parameters == 1600
    assert result.boundary_parameters == 320
    assert result.gradients == 1920
    assert result.adam_exp_avg == 1920
    assert result.adam_exp_avg_sq == 1920
    # One in-flight micro-batch: 64 B sharded intermediate + 128 B boundary.
    assert result.activation == 192
    assert result.total == 7872


def test_second_known_configuration_matches_independent_hand_calculation():
    config = replace(CONFIG, model_dim=4)
    result = estimate_memory(
        config,
        tp_degree=1,
        stage_layers=2,
        micro_batches=3,
        sequence_length=5,
        vocab_size=6,
        in_flight_micro_batches=2,
    )

    assert result.sharded_parameters == 1664
    assert result.boundary_parameters == 192
    assert result.gradients == 1856
    assert result.adam_exp_avg == 1856
    assert result.adam_exp_avg_sq == 1856
    # Two in-flight micro-batches, both shard and complete boundary are 320 B.
    assert result.activation == 640


def test_activation_peak_uses_in_flight_not_total_micro_batches():
    one = estimate_memory(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=1,
    )
    many = estimate_memory(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=8,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=1,
    )
    assert many.activation == one.activation


def test_memory_feasible_inclusive_boundary_and_one_byte_over():
    result = estimate_memory(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=1,
    )

    assert memory_feasible(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        memory_budget=result.total,
        in_flight_micro_batches=1,
    )
    assert not memory_feasible(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        memory_budget=result.total - 1,
        in_flight_micro_batches=1,
    )


@pytest.mark.parametrize("field, value", [("tp_degree", 0), ("stage_layers", 0), ("micro_batches", 0)])
def test_invalid_shape_inputs_are_rejected(field, value):
    kwargs = {
        "tp_degree": 2,
        "stage_layers": 1,
        "micro_batches": 1,
        "sequence_length": 2,
        "vocab_size": 10,
        "in_flight_micro_batches": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, **kwargs)


# --- the budget follows the real 1F1B schedule ------------------------------------


def _stage_budget(*, num_stages, stage_index, micro_batches):
    """Resident bytes for one stage, budgeting its own 1F1B in-flight peak."""
    return estimate_memory(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=micro_batches,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=peak_in_flight(
            micro_batches, stage_index=stage_index, num_stages=num_stages
        ),
    ).total


def test_single_stage_budgets_exactly_one_in_flight_activation():
    """One stage retires each micro-batch immediately, so more of them cost nothing."""
    assert _stage_budget(num_stages=1, stage_index=0, micro_batches=1) == _stage_budget(
        num_stages=1, stage_index=0, micro_batches=8
    )


def test_two_stages_budget_the_warmup_the_runtime_actually_holds():
    """Stage 0 warms up one extra forward before its first backward; stage 1 does not.

    This is the P3 case: budgeting a flat one in-flight micro-batch would understate
    stage 0 by exactly one activation, because the runtime genuinely holds two.
    """
    first = _stage_budget(num_stages=2, stage_index=0, micro_batches=4)
    last = _stage_budget(num_stages=2, stage_index=1, micro_batches=4)
    one_activation = (
        estimate_memory(
            CONFIG, tp_degree=2, stage_layers=1, micro_batches=4, sequence_length=2,
            vocab_size=10, in_flight_micro_batches=1,
        ).activation
    )
    assert first - last == one_activation
    assert last == _stage_budget(num_stages=1, stage_index=0, micro_batches=4)


def test_deep_pipeline_budgets_warmup_steady_and_cooldown_per_stage():
    """Every stage of a 3-deep pipeline budgets its own peak, decreasing downstream."""
    budgets = [
        _stage_budget(num_stages=3, stage_index=index, micro_batches=4) for index in range(3)
    ]
    assert budgets[0] > budgets[1] > budgets[2]
    assert [peak_in_flight(4, stage_index=i, num_stages=3) for i in range(3)] == [3, 2, 1]


@pytest.mark.parametrize(
    "num_stages, stage_index, micro_batches",
    [(1, 0, 1), (1, 0, 4), (2, 0, 4), (2, 1, 4), (3, 0, 8), (3, 2, 8)],
)
def test_budget_boundary_is_inclusive_and_one_byte_short_is_rejected(
    num_stages, stage_index, micro_batches
):
    in_flight = peak_in_flight(
        micro_batches, stage_index=stage_index, num_stages=num_stages
    )
    gate = dict(
        tp_degree=2,
        stage_layers=1,
        micro_batches=micro_batches,
        sequence_length=2,
        vocab_size=10,
        in_flight_micro_batches=in_flight,
    )
    total = estimate_memory(CONFIG, **gate).total
    assert memory_feasible(CONFIG, **gate, memory_budget=total)
    assert not memory_feasible(CONFIG, **gate, memory_budget=total - 1)
