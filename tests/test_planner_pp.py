"""Tests for deterministic PP repartition planning and the 1F1B schedule.

The schedule gates here are the project's one definition of 1F1B: the runtime issues
exactly :func:`pipeline_schedule`'s order (a torch-gated test in
``test_parallel_pp.py`` compares them element for element) and the memory model sizes
peak activations from :func:`peak_in_flight`. Locking the order here therefore locks
both, and rules out the GPipe order -- all forwards, then all backwards -- which is
what this project previously ran in production.
"""

import pytest

from resihp.planner.pp import (
    InfeasiblePP,
    balanced_layers,
    peak_in_flight,
    pipeline_phases,
    pipeline_schedule,
    repartition_pp,
)


def owners(ranges):
    return {
        layer: stage
        for stage, bounds in enumerate(ranges)
        if bounds is not None
        for layer in range(*bounds)
    }


def test_new_tp_degrees_are_capacities_not_target_layer_counts():
    plan = repartition_pp(
        old_stage_layers=(6, 2, 4),
        old_tp_degrees=(4, 2, 4),
        new_tp_degrees=(2, 4, 2),
    )

    assert plan.stage_layers == (3, 6, 3)
    assert plan.layer_ranges == ((0, 3), (3, 9), (9, 12))


def test_positive_difference_uses_exact_ratio_and_stage_id_tie_break():
    plan = repartition_pp(
        old_stage_layers=(6, 2, 4),
        old_tp_degrees=(4, 2, 4),
        new_tp_degrees=(2, 4, 2),
    )

    assert plan.stage_layers == (3, 6, 3)
    assert plan.layer_ranges == ((0, 3), (3, 9), (9, 12))
    assert sum(plan.stage_layers) == 12


def test_equal_ratios_use_lower_stage_id_for_the_remainder():
    plan = repartition_pp(
        old_stage_layers=(2, 2, 3),
        old_tp_degrees=(2, 1, 3),
        new_tp_degrees=(1, 1, 3),
    )

    assert plan.stage_layers == (2, 2, 3)


def test_negative_difference_is_deterministic_and_preserves_active_stages():
    plan = repartition_pp(
        old_stage_layers=(2, 1, 1),
        old_tp_degrees=(1, 1, 1),
        new_tp_degrees=(2, 2, 2),
    )

    assert plan.stage_layers == (1, 1, 2)
    assert all(count >= 1 for count in plan.stage_layers)


def test_empty_old_stage_can_be_used_for_continuous_repartition():
    first = repartition_pp((2, 2, 2), (2, 2, 2), (0, 2, 2))
    second = repartition_pp(first.stage_layers, (0, 2, 2), (0, 0, 2))

    assert first.stage_layers == (0, 3, 3)
    assert second.stage_layers == (0, 0, 6)
    assert second.layer_ranges == (None, None, (0, 6))
    assert (second.embedding_owner, second.lm_head_owner) == (2, 2)


def test_empty_stage_and_boundary_owners_follow_new_tp_capacity():
    first = repartition_pp((2, 2, 2), (2, 2, 2), (0, 2, 2))
    middle = repartition_pp((2, 2, 2), (2, 2, 2), (2, 0, 2))
    last = repartition_pp((2, 2, 2), (2, 2, 2), (2, 2, 0))

    assert first.layer_ranges == (None, (0, 3), (3, 6))
    assert (first.embedding_owner, first.lm_head_owner) == (1, 2)
    assert (middle.embedding_owner, middle.lm_head_owner) == (0, 2)
    assert (last.embedding_owner, last.lm_head_owner) == (0, 1)


def test_ranges_and_migrations_are_complete_and_consistent():
    plan = repartition_pp((2, 3, 4), (2, 3, 4), (4, 2, 3))
    old = owners(((0, 2), (2, 5), (5, 9)))
    new = owners(plan.layer_ranges)

    assert set(new) == set(range(9))
    assert len(new) == 9
    assert len({layer for layer, _, _ in plan.migrations}) == len(plan.migrations)
    assert all(old[layer] == source and new[layer] == target for layer, source, target in plan.migrations)
    assert all(source != target for _, source, target in plan.migrations)


def test_large_integer_ratio_comparison_does_not_use_float():
    plan = repartition_pp((10, 1), (10**309, 1), (10**309, 2))

    assert plan.stage_layers == (10, 1)


@pytest.mark.parametrize(
    ("old_layers", "old_tp", "new_tp"),
    [
        ((), (), ()),
        ((-1, 1), (1, 1), (1, 1)),
        ((0, 1), (1, 1), (1, 1)),
        ((0, 1), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (-1, 1)),
        ((1, 1), (1, 1), (0, 0)),
    ],
)
def test_invalid_inputs_are_rejected(old_layers, old_tp, new_tp):
    expected = InfeasiblePP if new_tp == (0, 0) else ValueError
    with pytest.raises(expected):
        repartition_pp(old_layers, old_tp, new_tp)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        repartition_pp((2,), (2,), (2, 2))


def test_active_stage_count_cannot_exceed_model_layers():
    with pytest.raises(InfeasiblePP) as error:
        repartition_pp((1, 1, 0), (1, 1, 0), (2, 2, 2))

    assert error.value.reason.code == "no_executable_pp"


# --- the base partition -----------------------------------------------------------


def test_balanced_layers_covers_every_layer_once():
    assert balanced_layers(4, 2) == ((0, 1), (2, 3))
    assert balanced_layers(5, 2) == ((0, 1, 2), (3, 4))  # remainder to earlier stages
    assert balanced_layers(3, 3) == ((0,), (1,), (2,))
    flat = [gid for stage in balanced_layers(7, 3) for gid in stage]
    assert flat == list(range(7))  # contiguous, unique, complete


def test_balanced_layers_rejects_more_stages_than_layers():
    with pytest.raises(ValueError):
        balanced_layers(2, 3)


def test_the_plan_lays_its_pristine_partition_out_from_balanced_layers():
    """One definition of the initial split, so plan and planner cannot disagree.

    Asserted on the layer ranges ``build_plan`` actually publishes, so the plan has no
    private copy of either the split or the way counts become ownership ranges.
    """
    from resihp.config import TrainConfig
    from resihp.plan import build_plan

    for num_layers, pp in ((4, 2), (6, 2), (7, 3), (6, 4)):
        config = TrainConfig(
            model_dim=16,
            num_layers=num_layers,
            num_heads=8,
            batch_size=4,
            micro_batch_size=2,
            seed=1,
            tp=1,
            pp=pp,
            dp=1,
            iterations=1,
        )
        plan = build_plan(config, step=0, version=0)
        assert tuple(stage.layer_range for stage in plan.stages) == tuple(
            (stage[0], stage[-1] + 1) for stage in balanced_layers(num_layers, pp)
        )


# --- the 1F1B schedule ------------------------------------------------------------


def test_single_stage_alternates_forward_and_backward():
    """With nothing downstream there is no warmup: every forward retires immediately."""
    assert pipeline_phases(4, stage_index=0, num_stages=1) == (0, 4)
    assert pipeline_schedule(4, stage_index=0, num_stages=1) == ("F", "B") * 4
    assert peak_in_flight(4, stage_index=0, num_stages=1) == 1


def test_two_stages_warm_up_steady_and_cool_down():
    """The canonical 2-stage / 4-micro-batch schedule, stage by stage."""
    assert pipeline_phases(4, stage_index=0, num_stages=2) == (1, 3)
    assert pipeline_schedule(4, stage_index=0, num_stages=2) == (
        "F", "F", "B", "F", "B", "F", "B", "B",
    )
    assert pipeline_phases(4, stage_index=1, num_stages=2) == (0, 4)
    assert pipeline_schedule(4, stage_index=1, num_stages=2) == ("F", "B") * 4


def test_schedule_is_not_gpipe():
    """No stage runs every forward before its first backward once micro-batches exceed
    the warmup -- that ordering is exactly what this project had to stop doing."""
    for num_stages in (1, 2, 3, 4):
        for stage_index in range(num_stages):
            order = pipeline_schedule(8, stage_index=stage_index, num_stages=num_stages)
            gpipe = ("F",) * 8 + ("B",) * 8
            assert order != gpipe, (num_stages, stage_index)


def test_every_micro_batch_is_forwarded_and_backwarded_exactly_once():
    for num_stages in (1, 2, 3, 4):
        for micro in (1, 2, 4, 8):
            for stage_index in range(num_stages):
                order = pipeline_schedule(micro, stage_index=stage_index, num_stages=num_stages)
                assert order.count("F") == micro
                assert order.count("B") == micro
                # A backward can only retire something already forwarded.
                live = 0
                for operation in order:
                    live += 1 if operation == "F" else -1
                    assert live >= 0, (num_stages, stage_index, micro, order)
                assert live == 0


def test_peak_in_flight_is_replayed_from_the_schedule():
    """Earlier stages hold more, and the micro-batch count caps every stage."""
    assert [peak_in_flight(4, stage_index=i, num_stages=3) for i in range(3)] == [3, 2, 1]
    # Two micro-batches cannot produce three in flight, however deep the pipeline is.
    assert [peak_in_flight(2, stage_index=i, num_stages=4) for i in range(4)] == [2, 2, 2, 1]
    for num_stages in (1, 2, 3, 4, 5):
        for micro in (1, 2, 3, 8):
            for stage_index in range(num_stages):
                order = pipeline_schedule(micro, stage_index=stage_index, num_stages=num_stages)
                live = peak = 0
                for operation in order:
                    live += 1 if operation == "F" else -1
                    peak = max(peak, live)
                assert peak == peak_in_flight(
                    micro, stage_index=stage_index, num_stages=num_stages
                )


def test_schedule_rejects_a_stage_outside_its_pipeline():
    for bad in ((4, 2, 2), (4, -1, 2), (4, 0, 0)):
        with pytest.raises(ValueError):
            pipeline_phases(bad[0], stage_index=bad[1], num_stages=bad[2])
    with pytest.raises(ValueError):
        pipeline_phases(-1, stage_index=0, num_stages=1)
