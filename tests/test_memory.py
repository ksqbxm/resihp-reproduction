"""Tests for the single analytical memory calculator."""

from dataclasses import replace

import pytest

from resihp.config import TrainConfig
from resihp.memory import memory_feasible, estimate_memory


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
    )

    # Independent constants: layer shard 1600 B; fixed embedding/head shards 320 B.
    assert result.sharded_parameters == 1600
    assert result.replicated_parameters == 320
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
    assert result.replicated_parameters == 192
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
    )

    assert memory_feasible(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        memory_budget=result.total,
    )
    assert not memory_feasible(
        CONFIG,
        tp_degree=2,
        stage_layers=1,
        micro_batches=1,
        sequence_length=2,
        vocab_size=10,
        memory_budget=result.total - 1,
    )


@pytest.mark.parametrize("field, value", [("tp_degree", 0), ("stage_layers", 0), ("micro_batches", 0)])
def test_invalid_shape_inputs_are_rejected(field, value):
    kwargs = {
        "tp_degree": 2,
        "stage_layers": 1,
        "micro_batches": 1,
        "sequence_length": 2,
        "vocab_size": 10,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        estimate_memory(CONFIG, **kwargs)
