"""Tests for deterministic TP candidate planning."""

from dataclasses import replace

import pytest

from resihp.config import TrainConfig
from resihp.planner.tp import InfeasibleTP, choose_tp


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
    # Five live ranks: only the powers of two are candidates, so the choice is 4.
    assert choose_tp(CONFIG, active_ranks=range(5), min_degree=1).degree == 4
    assert choose_tp(CONFIG, active_ranks=range(8), min_degree=1).degree == 8


def test_a_budget_no_degree_fits_is_a_structured_reason():
    with pytest.raises(InfeasibleTP) as error:
        choose_tp(
            CONFIG,
            active_ranks=range(8),
            min_degree=1,
            memory_budget=1,
            sequence_length=8,
            vocab_size=32,
            in_flight_micro_batches=1,
        )

    assert error.value.reason.code == "no_feasible_tp"


def test_minimum_degree_lower_bound_rounds_to_next_power_candidate():
    # A non-power-of-two bound selects the next power-of-two candidate above it: six
    # live ranks and min_degree 3 give 4, eight ranks and min_degree 5 give 8.
    assert choose_tp(CONFIG, active_ranks=range(6), min_degree=3).degree == 4
    assert choose_tp(CONFIG, active_ranks=range(8), min_degree=5).degree == 8
    with pytest.raises(InfeasibleTP):
        choose_tp(CONFIG, active_ranks=range(8), min_degree=9)


def test_memory_budget_requires_explicit_shape_and_inflight_inputs():
    with pytest.raises(ValueError, match="required when memory_budget is set"):
        choose_tp(CONFIG, active_ranks=range(8), min_degree=1, memory_budget=1)


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
        choose_tp(CONFIG, active_ranks=(), min_degree=1, stage_layers=0)


@pytest.mark.parametrize("bad_rank", [-1, True, 1.5])
def test_invalid_active_rank_values_are_rejected(bad_rank):
    with pytest.raises(ValueError, match="active_ranks"):
        choose_tp(CONFIG, active_ranks=[0, bad_rank], min_degree=1)


def test_duplicate_active_ranks_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        choose_tp(CONFIG, active_ranks=[0, 0, 1], min_degree=1)


def test_vocab_parallel_divisibility_filters_candidate_degrees():
    """A degree that does not divide the vocabulary cannot build the stage at all.

    The token embedding and the LM head are vocab-parallel, so
    ``TensorParallelStage`` rejects such a degree outright. Catching it here is what
    turns a runtime ``ValueError`` into the structured ``no_feasible_tp`` reason.
    """
    # 24 = 8 * 3: divisible by 1, 2, 4, 8; 12 = 4 * 3: only by 1, 2, 4.
    assert choose_tp(CONFIG, active_ranks=range(8), min_degree=1, vocab_size=24).degree == 8
    assert choose_tp(CONFIG, active_ranks=range(8), min_degree=1, vocab_size=12).degree == 4
    # Left out entirely, only the dimension constraints apply -- pure planning tests
    # that never build a stage keep working unchanged.
    assert choose_tp(CONFIG, active_ranks=range(8), min_degree=1).degree == 8


def test_an_indivisible_vocabulary_is_a_structured_reason_not_a_crash():
    """A vocabulary no candidate divides must stop the run, not crash the runtime."""
    with pytest.raises(InfeasibleTP) as error:
        choose_tp(CONFIG, active_ranks=range(8), min_degree=2, vocab_size=7)

    assert error.value.reason.code == "no_feasible_tp"
    assert "vocabulary" in error.value.reason.message
