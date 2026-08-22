"""Tests for the immutable, versioned ExecutionPlan and its invariants."""

from dataclasses import FrozenInstanceError, replace

import pytest

from resihp.config import TrainConfig
from resihp.control import STOP_CODES
from resihp.memory import estimate_memory
from resihp.plan import (
    InfeasiblePlan,
    PlanInvariantError,
    assert_invariants,
    boundary_pairs,
    build_plan,
)


VOCAB = 32
SEQLEN = 8


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


def test_a_total_wipeout_is_named_by_the_pp_planner():
    # No replica survived, so there is no pipeline left to lay out. That is the PP
    # planner's own condition and it reports it under its own code -- the plan layer
    # does not invent a second name for it.
    with pytest.raises(InfeasiblePlan) as error:
        build_plan(CONFIG, step=0, version=1, failed_ranks=tuple(range(8)))

    assert error.value.reason.code == "no_executable_pp"


def test_a_replica_emptied_by_an_earlier_event_is_not_the_evidence():
    # Replica 0 dies first, then replica 1. At the last event replica 0 has no layers
    # left to reason about, so the reason must come from the replica this event killed.
    config = replace(CONFIG, pp=1, dp=2)
    plan = build_plan(config, step=0, version=0)
    for version, failed in enumerate([(0,), (0, 1), (0, 1, 2)], start=1):
        plan = build_plan(config, step=version, version=version, failed_ranks=failed, previous=plan)

    with pytest.raises(InfeasiblePlan) as error:
        build_plan(config, step=4, version=4, failed_ranks=(0, 1, 2, 3), previous=plan)
    assert error.value.reason.code == "no_executable_pp"


def test_dp_rejects_a_final_layout_that_no_longer_fits():
    # ``choose_tp`` gates memory with each stage's *old* layer count, so a stage that
    # absorbs a dead stage's layers can end up needing more than was ever checked. The
    # DP planner sees the final layout and is the one that says so.
    config = replace(CONFIG, tp=1, pp=2, dp=1, num_layers=4)
    budget = estimate_memory(
        config,
        tp_degree=1,
        stage_layers=3,
        micro_batches=config.batch_size // config.micro_batch_size,
        sequence_length=SEQLEN,
        vocab_size=VOCAB,
        in_flight_micro_batches=1,
    ).total
    gated = dict(memory_budget=budget, vocab_size=VOCAB, sequence_length=SEQLEN)

    plan = build_plan(config, step=0, version=0, **gated)  # two 2-layer stages fit
    with pytest.raises(InfeasiblePlan) as error:
        # Rank 1 was stage 1's only rank; stage 0 absorbs all four layers and no longer fits.
        build_plan(config, step=1, version=1, failed_ranks=(1,), previous=plan, **gated)
    assert error.value.reason.code == "no_feasible_dp_target"


# --- cumulative fail-stop: what the invariant may and may not forbid ---------------


def test_a_failed_rank_never_becomes_active_again():
    pristine = build_plan(CONFIG, step=0, version=0)  # every rank active
    after = build_plan(CONFIG, step=1, version=1, failed_ranks=(1,), previous=pristine)
    assert 1 not in after.active_ranks

    # A resurrection can only be expressed two ways, and both are rejected. Either the
    # plan drops rank 1 from the failed set to make room for it...
    with pytest.raises(PlanInvariantError, match="failed ranks must grow"):
        assert_invariants(replace(pristine, version=2), previous=after)
    # ...or it keeps rank 1 failed and schedules it anyway.
    with pytest.raises(PlanInvariantError, match="active and failed ranks overlap"):
        assert_invariants(replace(pristine, version=2, failed_ranks=(1,)), previous=after)


def test_a_healthy_rank_left_idle_can_be_selected_again():
    # TP4: after rank 0 dies the largest power-of-two prefix of the survivors is
    # (1, 2), leaving healthy rank 3 idle. When rank 1 then dies, the planner must be
    # free to pick rank 3 back up -- being unused is not being failed.
    config = replace(CONFIG, tp=4, pp=1, dp=1, num_layers=2)
    plan = build_plan(config, step=0, version=0)

    v1 = build_plan(config, step=1, version=1, failed_ranks=(0,), previous=plan)
    assert v1.active_ranks == (1, 2) and 3 in v1.live_ranks

    v2 = build_plan(config, step=2, version=2, failed_ranks=(0, 1), previous=v1)
    assert v2.active_ranks == (2, 3)  # idle rank 3 is back in the training path
    assert v2.failed_ranks == (0, 1)


def test_every_cumulative_failure_sequence_stays_inside_the_taxonomy():
    # Each event either yields a plan or one of the six consistent-stop conditions.
    # Nothing may escape as an unclassified error.
    for config in (
        replace(CONFIG, tp=4, pp=1, dp=1, num_layers=2),
        replace(CONFIG, tp=2, pp=2, dp=1, num_layers=4),
        CONFIG,
    ):
        plan = build_plan(config, step=0, version=0)
        failed = ()
        for version in range(1, config.world_size + 1):
            failed = tuple(range(version))
            try:
                plan = build_plan(
                    config, step=version, version=version, failed_ranks=failed, previous=plan
                )
            except InfeasiblePlan as error:
                assert error.reason.code in STOP_CODES, error.reason.code
                break
            assert set(plan.active_ranks) <= set(plan.live_ranks)


def test_the_same_failure_sequence_yields_the_same_digests():
    def sequence():
        plan = build_plan(CONFIG, step=0, version=0)
        digests = [plan.digest]
        for version, failed in enumerate([(1,), (1, 5), (1, 3, 5)], start=1):
            plan = build_plan(
                CONFIG, step=version, version=version, failed_ranks=failed, previous=plan
            )
            digests.append(plan.digest)
        return digests

    assert sequence() == sequence()


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
    """A route carries both layouts, so recovery can execute it without re-deriving one."""
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(1,))

    route = next(r for r in plan.state_routes if r.replica_id == 0 and r.layer == 0)
    assert route.boundary == ""
    assert route.target_ranks == (0,) and route.target_degree == 1
    assert route.donor_kind == "peer_replica"
    assert route.donor_ranks == (4, 5) and route.donor_degree == 2
    assert route.reshard == "gather_reshard"
    # Gradients are not persistent state: they are recomputed by the next iteration.
    assert set(route.states) == {"param", "exp_avg", "exp_avg_sq", "step"}
    assert "grad" not in route.states


def test_boundary_tensors_are_routed_when_their_owning_stage_changes():
    """The embedding and LM head follow the first / last executable stage (plan 3.4).

    They belong to no layer, so a per-layer route table cannot express them -- and a
    recovery that never fetched them would leave a resharded boundary stage holding the
    old layout.
    """
    plan = build_plan(CONFIG, step=2, version=2, failed_ranks=(1,))

    boundaries = {r.boundary: r for r in plan.state_routes if r.replica_id == 0 and r.layer is None}
    # Replica 0's stage 0 halved to TP1, so its embedding shard must be rebuilt.
    assert boundaries["embedding"].target_ranks == (0,)
    assert boundaries["embedding"].donor_kind == "peer_replica"
    # Its last stage was untouched, so the head does not move at all.
    assert "head" not in boundaries


def test_consecutive_failures_route_from_the_previous_plan_not_the_initial_config():
    first = build_plan(CONFIG, step=0, version=1)
    second = build_plan(CONFIG, step=2, version=2, failed_ranks=(1,), previous=first)
    third = build_plan(CONFIG, step=4, version=3, failed_ranks=(1, 3), previous=second)

    # Layers 0-1 have stayed on rank 0 since the second plan, so nothing routes
    # them; only the freshly degraded stage (layers 2-3, and the LM head it owns)
    # migrates.
    groups = {(r.layer, r.boundary) for r in third.state_routes if r.replica_id == 0}
    assert groups == {(2, ""), (3, ""), (None, "head")}

    # Rebuilding the same failure from the initial config would wrongly claim the
    # already-settled layers 0-1 -- and the embedding with them -- migrate again.
    from_initial = build_plan(CONFIG, step=4, version=3, failed_ranks=(1, 3))
    assert {(r.layer, r.boundary) for r in from_initial.state_routes if r.replica_id == 0} == {
        (0, ""), (1, ""), (2, ""), (3, ""), (None, "embedding"), (None, "head"),
    }


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


def test_an_indivisible_vocabulary_stops_the_plan_instead_of_crashing_the_runtime():
    """Every static constraint that would make the layout unbuildable is a plan reason.

    ``TensorParallelStage`` refuses a TP degree that does not divide the vocabulary, so
    a planner blind to it would publish a plan the runtime cannot construct. With
    ``vocab_size`` passed -- as the launched entrypoint always does -- no such degree is
    ever selected, and when none survives the plan is infeasible with a root cause.
    """
    # 6 is divisible by 2 but not by 4: TP2 is buildable, TP4 is not.
    config = replace(CONFIG, tp=4, pp=1, dp=1)
    plan = build_plan(config, step=0, version=0, vocab_size=6, sequence_length=SEQLEN)
    assert [s.tp_degree for s in plan.stages] == [2]

    # Without the vocabulary the planner would have taken the largest degree the
    # dimensions allow -- 4 -- which ``TensorParallelStage`` then refuses to build.
    # That difference is the whole point of the gate.
    blind = build_plan(config, step=0, version=0, sequence_length=SEQLEN)
    assert [s.tp_degree for s in blind.stages] == [4]

    # Degree 1 divides every vocabulary, so a plan is never infeasible on vocabulary
    # alone; the structured ``no_feasible_tp`` reason belongs to a stage whose ``k_min``
    # rules degree 1 out as well, which ``test_planner_tp.py`` covers directly.
    odd = build_plan(config, step=0, version=0, vocab_size=7, sequence_length=SEQLEN)
    assert [s.tp_degree for s in odd.stages] == [1]


def test_boundary_pairs_name_every_pipeline_hop_the_assignment_creates():
    """The two-rank groups the 1F1B boundary transfers ride on, read from the plan."""
    plan = build_plan(CONFIG, step=0, version=0)
    # TP2 x PP2 x DP2: one hop per replica, between the two stages' leaders.
    assert boundary_pairs(plan) == ((0, 2), (4, 6))

    # Rank 1 dies: replica 0's stage 0 keeps leader 0, so its hop is unchanged.
    degraded = build_plan(CONFIG, step=1, version=1, failed_ranks=(1,), previous=plan)
    assert boundary_pairs(degraded) == ((0, 2), (4, 6))

    # Rank 4 dies too: replica 1's stage 0 leader becomes rank 5, and the hop follows.
    moved = build_plan(CONFIG, step=2, version=2, failed_ranks=(1, 4), previous=degraded)
    assert boundary_pairs(moved) == ((0, 2), (5, 6))


def test_a_single_stage_replica_has_no_pipeline_hop():
    """Nothing to transfer when a replica runs one stage, so no boundary group exists."""
    config = replace(CONFIG, pp=1, dp=2, tp=2)
    assert boundary_pairs(build_plan(config, step=0, version=0)) == ()
