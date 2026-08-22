"""Tests for deterministic data-parallel micro-batch rerouting."""

from dataclasses import replace

import pytest

from resihp.config import TrainConfig
from resihp.planner.dp import DPStage, DPTopology, InfeasibleDP, assign


CONFIG = TrainConfig(
    model_dim=8,
    num_layers=2,
    num_heads=2,
    batch_size=8,
    micro_batch_size=2,
    seed=1,
    tp=1,
    pp=2,
    dp=2,
    iterations=4,
)


def topology(*, budget=None, stages=None):
    return DPTopology(
        config=CONFIG,
        stages=tuple(stages or (
            DPStage(0, 0, (0,), 1, 1, capacity=1),
            DPStage(1, 0, (1,), 1, 1, capacity=3),
            DPStage(0, 1, (2,), 1, 1, capacity=2),
            DPStage(1, 1, (3,), 1, 1, capacity=2),
        )),
        micro_batches=4,
        sequence_length=2,
        vocab_size=8,
        memory_budget=budget,
    )


def test_capacity_proportional_assignment_uses_replica_id_tie_order():
    result = assign(3, (), topology())

    assert [(p.micro_batch, p.stage_id, p.replica_id) for p in result.placements] == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
        (2, 0, 1),
        (2, 1, 1),
        (3, 0, 1),
        (3, 1, 1),
    ]
    assert result.micro_batches_per_replica == {0: 1, 1: 3}

    # A micro-batch must flow through a single replica across every stage.
    for placements in result.by_micro_batch.values():
        assert len({p.replica_id for p in placements}) == 1


def test_failed_stage_is_rerouted_to_healthy_peer():
    result = assign(1, {0}, topology())

    stage_zero = [p for p in result.placements if p.stage_id == 0]
    assert all(p.replica_id == 1 for p in stage_zero)
    assert [p.micro_batch for p in stage_zero] == [0, 1, 2, 3]
    assert next(p for p in stage_zero if p.micro_batch == 0).replica_id == 1


def test_every_micro_batch_has_exactly_one_executor_per_stage():
    result = assign(0, (), topology())
    pairs = [(p.micro_batch, p.stage_id) for p in result.placements]

    assert len(pairs) == 8
    assert len(set(pairs)) == 8
    assert set(pairs) == {(micro_batch, stage) for micro_batch in range(4) for stage in range(2)}


def test_same_inputs_are_idempotent_and_failure_signature_is_normalized():
    result = assign(2, {2, 0}, topology())

    assert result == assign(2, [0, 2], topology())
    assert result.failure_signature == (0, 2)


def test_memory_budget_rejects_all_targets_with_structured_reason():
    with pytest.raises(InfeasibleDP) as error:
        assign(0, (), topology(budget=1))

    assert error.value.reason.code == "no_feasible_dp_target"


def test_memory_budget_boundary_is_accepted():
    """The gate sizes each stage by its own 1F1B peak, so stage 0 is the binding one.

    Two stages, four micro-batches: stage 0 holds two activations at its peak and
    stage 1 holds one, so a budget cut to stage 1's footprint would wrongly admit
    stage 0. Budgeting the deeper peak is what makes the boundary meaningful.
    """
    from resihp.memory import estimate_memory
    from resihp.planner.pp import peak_in_flight

    def total(stage_index):
        return estimate_memory(
            CONFIG,
            tp_degree=1,
            stage_layers=1,
            micro_batches=4,
            sequence_length=2,
            vocab_size=8,
            in_flight_micro_batches=peak_in_flight(4, stage_index=stage_index, num_stages=2),
        ).total

    budget = max(total(0), total(1))
    assert total(0) > total(1)  # the warmup stage really is the expensive one
    assert len(assign(0, (), topology(budget=budget)).placements) == 8
    # One byte short of the deepest stage and no replica can host the pipeline.
    with pytest.raises(InfeasibleDP) as error:
        assign(0, (), topology(budget=budget - 1))
    assert error.value.reason.code == "no_feasible_dp_target"


def test_invalid_failure_signature_and_duplicate_stage_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        assign(0, [0, 0], topology())

    with pytest.raises(ValueError, match="duplicate replica/stage"):
        assign(
            0,
            (),
            topology(stages=(
                DPStage(0, 0, (0,), 1, 1),
                DPStage(0, 0, (1,), 1, 1),
            )),
        )


def test_stage_failure_is_based_on_any_failed_tp_member():
    multi_tp = topology(stages=(
        DPStage(0, 0, (0, 4), 1, 2, capacity=4),
        DPStage(1, 0, (1, 5), 1, 2, capacity=4),
    ))

    result = assign(0, {4}, multi_tp)

    assert {p.replica_id for p in result.placements} == {1}


def test_replicas_with_different_active_stage_sets_are_both_scheduled():
    # After a PP repartition, replica 0 runs a single stage that owns every layer
    # while replica 1 keeps two stages. Both must receive micro-batches, and each
    # micro-batch runs only its own replica's active stages.
    heterogeneous = DPTopology(
        config=CONFIG,
        stages=(
            DPStage(0, 0, (0,), 4, 1),
            DPStage(1, 0, (4, 5), 2, 2),
            DPStage(1, 1, (6, 7), 2, 2),
        ),
        micro_batches=4,
    )

    result = assign(0, (), heterogeneous)

    stages_by_replica = {}
    for placement in result.placements:
        stages_by_replica.setdefault(placement.replica_id, set()).add(placement.stage_id)
    assert stages_by_replica == {0: {0}, 1: {0, 1}}
    assert result.micro_batches_per_replica == {0: 2, 1: 2}
    for placements in result.by_micro_batch.values():
        assert len({p.replica_id for p in placements}) == 1


def test_step_is_recorded_but_does_not_change_assignment():
    assert assign(0, (), topology()).placements == assign(99, (), topology()).placements
    assert assign(99, (), topology()).step == 99
