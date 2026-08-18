"""Tests for deterministic TP candidate planning."""

from dataclasses import replace

import pytest

from resihp.config import TrainConfig
from resihp.planner.tp import InfeasibleTP, choose_tp, feasible_degrees


CONFIG = TrainConfig(
    model_dim=16,
    num_layers=4,
    num_heads=8,
    batch_size=8,
    micro_batch_size=2,
    seed=1,
    tp=2,
    pp=2,
    dp=2,
    iterations=8,
)


def test_candidates_are_powers_of_two_and_respect_divisibility():
    assert feasible_degrees(CONFIG, active_ranks=range(8), min_degree=1) == (1, 2, 4, 8)
    assert feasible_degrees(
        CONFIG,
        active_ranks=range(8),
        min_degree=1,
        memory_budget=1,
        sequence_length=8,
        vocab_size=32,
        in_flight_micro_batches=1,
    ) == ()


def test_minimum_degree_lower_bound_rounds_to_next_power_candidate():
    assert feasible_degrees(CONFIG, active_ranks=range(8), min_degree=3) == (4, 8)
    assert feasible_degrees(CONFIG, active_ranks=range(8), min_degree=5) == (8,)
    assert feasible_degrees(CONFIG, active_ranks=range(8), min_degree=9) == ()


def test_memory_budget_requires_explicit_shape_and_inflight_inputs():
    with pytest.raises(ValueError, match="required when memory_budget is set"):
        feasible_degrees(CONFIG, active_ranks=range(8), min_degree=1, memory_budget=1)


def test_maximum_degree_and_ascending_stage_local_members_are_deterministic():
    # active_ranks is already one stage-local live communication domain.
    result = choose_tp(CONFIG, active_ranks={7, 2, 5, 0}, min_degree=1)

    assert result.degree == 4
    assert result.members == (0, 2, 5, 7)
    assert result == choose_tp(CONFIG, active_ranks={0, 2, 5, 7}, min_degree=1)


def test_single_and_cumulative_stage_local_failures_remove_ranks():
    first = choose_tp(CONFIG, active_ranks={0, 2, 3}, min_degree=1)
    second = choose_tp(CONFIG, active_ranks={0, 3}, min_degree=1)

    assert first.degree == 2
    assert first.members == (0, 2)
    assert second.degree == 2
    assert second.members == (0, 3)
    assert 1 not in first.members and 2 not in second.members


def test_no_feasible_degree_has_structured_reason():
    with pytest.raises(InfeasibleTP) as error:
        choose_tp(CONFIG, active_ranks={0, 1}, min_degree=4)

    assert error.value.reason.code == "no_feasible_tp"
    assert error.value.reason.active_ranks == (0, 1)


def test_invalid_model_head_config_is_rejected_as_bad_input():
    config = replace(CONFIG, num_heads=6)
    with pytest.raises(ValueError, match="model_dim must be divisible by num_heads"):
        choose_tp(config, active_ranks=range(8), min_degree=1)


def test_invalid_inputs_are_not_hidden_by_empty_rank_set():
    with pytest.raises(ValueError, match="stage_layers"):
        feasible_degrees(CONFIG, active_ranks=(), min_degree=1, stage_layers=0)


@pytest.mark.parametrize("bad_rank", [-1, True, 1.5])
def test_invalid_active_rank_values_are_rejected(bad_rank):
    with pytest.raises(ValueError, match="active_ranks"):
        feasible_degrees(CONFIG, active_ranks=[0, bad_rank], min_degree=1)


def test_duplicate_active_ranks_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        feasible_degrees(CONFIG, active_ranks=[0, 0, 1], min_degree=1)
