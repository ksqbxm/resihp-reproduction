"""Tests for the immutable, versioned ExecutionPlan and its invariants."""

from dataclasses import FrozenInstanceError, replace

import pytest

from resihp.config import TrainConfig
from resihp.plan import (
    InfeasiblePlan,
    PlanInvariantError,
    assert_invariants,
    build_plan,
)


CONFIG = TrainConfig(
    model_dim=128,
    num_layers=4,
    num_heads=8,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=6,
)


def stage_members(plan):
    return {(s.replica_id, s.stage_id): s.tp_members for s in plan.stages}


def test_initial_plan_covers_the_full_topology_and_is_stable():
    plan = build_plan(CONFIG, step=0, version=1)
    again = build_plan(CONFIG, step=0, version=1)

    assert plan.active_ranks == tuple(range(8))
    assert plan.failed_ranks == ()
    assert stage_members(plan) == {
        (0, 0): (0, 1),
        (0, 1): (2, 3),
        (1, 0): (4, 5),
        (1, 1): (6, 7),
    }
    assert all(s.layer_range in {(0, 2), (2, 4)} for s in plan.stages)
    assert plan.state_routes == ()
    assert len(plan.placements) == 8
    assert plan == again
    assert plan.digest == again.digest


def test_step_is_passed_to_dp_and_version_is_metadata():
    plan = build_plan(CONFIG, step=5, version=2)

    assert plan.step == 5
    assert plan.version == 2
    # Version never reaches the DP assignment; only the step can.
    assert plan.placements == build_plan(CONFIG, step=5, version=99).placements


def test_a_fully_failed_stage_relayers_instead_of_dropping_the_replica():
    # Ranks 0 and 1 are replica 0's whole stage 0. Its layers must move to the
    # surviving stage 1 (ranks 2, 3); the healthy ranks must not be discarded.
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(0, 1))

    replica0 = {s.stage_id: s for s in plan.stages if s.replica_id == 0}
    assert set(replica0) == {1}
    assert replica0[1].tp_members == (2, 3)
    assert replica0[1].layer_range == (0, 4)
    assert 2 in plan.active_ranks and 3 in plan.active_ranks
    # Replica 1 stays intact and both replicas still take work.
    assert {p.replica_id for p in plan.placements} == {0, 1}


def test_losing_a_stage_in_every_replica_is_still_feasible():
    # Ranks 0,1 kill replica 0 stage 0 and ranks 4,5 kill replica 1 stage 0.
    # Both replicas relayer onto their stage 1; the plan stays feasible.
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(0, 1, 4, 5))

    assert {(s.replica_id, s.stage_id) for s in plan.stages} == {(0, 1), (1, 1)}
    assert plan.active_ranks == (2, 3, 6, 7)


def test_only_a_total_wipeout_is_no_surviving_replica():
    with pytest.raises(InfeasiblePlan) as error:
        build_plan(CONFIG, step=0, version=1, failed_ranks=tuple(range(8)))

    assert error.value.reason.code == "no_surviving_replica"


def test_tp_memory_infeasibility_is_reported_at_the_tp_stage():
    with pytest.raises(InfeasiblePlan) as error:
        build_plan(
            CONFIG,
            step=0,
            version=1,
            memory_budget=1,
            sequence_length=2,
            vocab_size=8,
        )

    assert error.value.reason.code == "no_feasible_tp"


def test_tp_change_emits_a_donor_target_reshard_state_route():
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(1,))

    route = next(r for r in plan.state_routes if r.replica_id == 0 and r.layer == 0)
    assert route.target_ranks == (0,)
    assert route.donor_kind == "peer_replica"
    assert route.donor_ranks == (4, 5)
    assert route.reshard == "gather_reshard"
    assert set(route.states) == {"param", "grad", "exp_avg", "exp_avg_sq", "step"}


def test_consecutive_failures_route_from_the_previous_plan_not_the_initial_config():
    first = build_plan(CONFIG, step=0, version=1)
    second = build_plan(CONFIG, step=2, version=2, failed_ranks=(1,), previous=first)
    third = build_plan(CONFIG, step=4, version=3, failed_ranks=(1, 3), previous=second)

    # Layers 0-1 have stayed on rank 0 since the second plan, so nothing routes
    # them; only the freshly degraded stage (layers 2-3) migrates.
    assert {r.layer for r in third.state_routes if r.replica_id == 0} == {2, 3}

    # Rebuilding the same failure from the initial config would wrongly claim the
    # already-settled layers 0-1 migrate again.
    from_initial = build_plan(CONFIG, step=4, version=3, failed_ranks=(1, 3))
    assert {r.layer for r in from_initial.state_routes if r.replica_id == 0} == {0, 1, 2, 3}


def test_cross_replica_heterogeneous_stages_pass_the_invariants():
    # build_plan runs assert_invariants internally, so a successful return means
    # the heterogeneous topology (replica 0 single-stage, replica 1 two-stage)
    # already satisfies every invariant.
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(0, 1))

    assert_invariants(plan)
    assert {len({s.stage_id for s in plan.stages if s.replica_id == r}) for r in (0, 1)} == {1, 2}


def test_build_plan_runs_invariants_and_rejects_a_non_increasing_version():
    high = build_plan(CONFIG, step=0, version=5)

    with pytest.raises(PlanInvariantError, match="version"):
        build_plan(CONFIG, step=1, version=3, failed_ranks=(1,), previous=high)


def test_invariants_reject_overlapping_layer_ranges():
    plan = build_plan(CONFIG, step=0, version=1)
    broken_stage = replace(plan.stages[1], layer_range=(1, 3))
    broken = replace(plan, stages=(plan.stages[0], broken_stage) + plan.stages[2:])

    with pytest.raises(PlanInvariantError, match="layer"):
        assert_invariants(broken)


def test_invariants_reject_a_missing_micro_batch():
    plan = build_plan(CONFIG, step=0, version=1)
    broken = replace(plan, placements=tuple(p for p in plan.placements if p.micro_batch != 3))

    with pytest.raises(PlanInvariantError, match="coverage"):
        assert_invariants(broken)


def test_invariants_reject_a_rank_assigned_to_two_stages():
    plan = build_plan(CONFIG, step=0, version=1)
    clash = replace(plan.stages[1], tp_members=(0, 3))
    broken = replace(plan, stages=(plan.stages[0], clash) + plan.stages[2:])

    with pytest.raises(PlanInvariantError, match="multiple stages"):
        assert_invariants(broken)


def test_invariants_reject_tp_degree_and_member_count_mismatch():
    plan = build_plan(CONFIG, step=0, version=1)
    broken_stage = replace(plan.stages[0], tp_degree=1)
    broken = replace(plan, stages=(broken_stage,) + plan.stages[1:])

    with pytest.raises(PlanInvariantError, match="tp_degree"):
        assert_invariants(broken)


def test_digest_changes_with_version_step_config_and_failures():
    base = build_plan(CONFIG, step=0, version=1)

    assert base.digest != build_plan(CONFIG, step=0, version=2).digest
    assert base.digest != build_plan(CONFIG, step=1, version=1).digest
    assert base.digest != build_plan(CONFIG, step=0, version=1, failed_ranks=(1,)).digest
    assert base.digest != build_plan(replace(CONFIG, model_dim=256), step=0, version=1).digest


def test_execution_plan_and_nested_structures_are_immutable():
    plan = build_plan(CONFIG, step=0, version=1)

    with pytest.raises(FrozenInstanceError):
        plan.version = 9
    with pytest.raises(FrozenInstanceError):
        plan.stages[0].tp_degree = 1
    with pytest.raises(TypeError):
        plan.placements[0] = plan.placements[1]


def test_out_of_range_and_duplicate_failures_are_rejected():
    with pytest.raises(ValueError, match="out-of-range"):
        build_plan(CONFIG, step=0, version=1, failed_ranks=(99,))

    with pytest.raises(ValueError, match="duplicate"):
        build_plan(CONFIG, step=0, version=1, failed_ranks=(1, 1))
