"""Tests for deterministic PP repartition planning."""

import pytest

from resihp.planner.pp import InfeasiblePP, repartition_pp


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
