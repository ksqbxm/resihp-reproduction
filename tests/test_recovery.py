"""Atomic reconfiguration and consistent stop (T14).

Three families of gates.

**Plan reading** (pure) -- that PP ownership, TP layout, and "what actually changed"
are all read from the ExecutionPlan, and specifically that the acquire set *is* the
plan's own ``state_routes``: recovery makes no routing decision of its own, so a layer
is fetched when the plan routed it, the boundary tensors when the plan routed them, and
a replica nothing happened to fetches nothing at all.

**Recovery** -- the single path of plan 3.6 driven through the real safe point, opened
by a real kill: every victim below sends itself ``SIGKILL``, so its shards are gone
with its process and have to come back from a peer replica or from the checkpoint
written while it was still alive. A victim writes no result, so the results are keyed
by rank. The run executes through the plan's own micro-batch assignment on the 1F1B
``PipelineRuntime``, over stages cut to the plan's ``layer_range`` and sharded to its TP
degree (T10's ``TensorParallelStage``):

* ``*_pipeline`` (4 ranks, ``TP2 x PP2 x DP1``, 6 layers) -- dropping one rank of
  stage 0 halves its degree *and* moves layer 2 across the stage boundary, so one
  event exercises a TP reshard, a real PP layer migration, an embedding re-shard,
  and layers that stay put and are therefore not moved at all. With no peer replica,
  the dead rank's shard exists nowhere else: this is the checkpoint-fallback branch.
* ``*_replicated`` (4 ranks, ``TP2 x PP1 x DP2``) -- two consecutive fail-stops, one
  rank at a time. At the first, the dead rank's shard is collected from the peer
  replica while that replica itself acquires nothing; at the second the two replicas
  are running *different* TP degrees (1 and 2), so donor sets must be grouped by the
  degree they were sharded under.
* ``*_reseat`` (4 ranks, ``TP4 x PP1 x DP1``) -- the planner keeps the largest
  power-of-two prefix of a stage's survivors, so the first failure leaves healthy rank
  3 idle. The second re-seats it and shifts rank 2 from shard index 1 to 0: a rank can
  therefore hold state at the right degree and still hold the *wrong slice*, so
  "unchanged degree" is not on its own a licence to keep it.

The acceptance criterion is principle A's before-resume half: after recovery each
rank holds exactly the names its new stage owns -- no more -- and each is
``torch.equal`` to the checkpoint's slice under the new degree.

**Consistent stop** -- one gate per condition of plan 3.6, each triggered where the
control plane really has to handle it, then checked for: every rank stopping with the
same single root cause, no new plan published, no group left over from the stopped
plan, no residual process group, the pre-failure checkpoint still reloadable, and
everyone exiting before the timeout rather than blocking in a collective.

Every stop condition a surviving rank can observe is reached through the real
``safe_point`` -> ``build_plan`` path; ``STOP_SCENARIOS`` says how. Only
``plan_disagreement`` injects a fault, because deterministic replanning cannot disagree
with itself -- and what is under test there is the control plane's reaction, not the
disagreement. ``no_executable_pp`` is the one condition no rank can report under a real
kill, because it means every process is dead; it has its own gate.

Every distributed gate runs on CPU/**Gloo** and on GPU/**NCCL** with real device
tensors and real NCCL training groups (the world group stays Gloo on both, as it must).
NCCL gates skip only when there are fewer GPUs than the case needs.
"""

import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from harness import assert_killed, read_results, run_ranks
from resihp.checkpoint import CheckpointError, load_anchor
from resihp.config import TrainConfig
from resihp.control import STOP_CODES, ConsistentStop, ControlPlane
from resihp.parallel.reshard import reconstruct_full
from resihp.plan import build_plan
from resihp.recovery import acquire_layout, stage_layout, stage_of
from resihp.verify import matches_checkpoint


#: TP2 x PP2 x DP1 over 6 layers: dropping a rank of stage 0 both halves its degree
#: and pushes layer 2 across the stage boundary.
PIPELINE = dict(
    model_dim=16,
    num_layers=6,
    num_heads=4,
    batch_size=4,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=1,
    iterations=4,
)
#: TP2 x PP1 x DP2: a peer replica can donate the dead rank's shard.
REPLICATED = dict(PIPELINE, num_layers=2, pp=1, dp=2)
#: TP2 x PP1 x DP1: the smallest layout that can lose a rank, for the stop gates.
PAIR = dict(PIPELINE, num_layers=2, pp=1, dp=1)
#: TP1 x PP2 x DP1: losing stage 1 makes stage 0 absorb every layer, so the final
#: layout needs more than ``choose_tp`` ever checked it for.
ABSORB = dict(PIPELINE, num_layers=4, tp=1, pp=2, dp=1)
#: TP4 x PP1 x DP1: the first failure leaves a healthy rank idle, the second re-seats
#: it and moves another rank's shard index.
RESEAT = dict(PIPELINE, num_layers=2, tp=4, pp=1, dp=1)
VOCAB = 32
SEQLEN = 8
#: Victims per scenario, applied one rank at a time (principle B). ``pair`` kills both
#: of its ranks, which is the end of the job rather than a recovery.
VICTIMS = {"pipeline": (1,), "replicated": (1, 3), "reseat": (0, 1), "pair": (1, 0)}
#: ``code -> (layout, victims, analytic budget as (tp_degree, stage_layers) or None)``.
#: The budget is one the pristine layout fits and the post-failure layout does not.
#:
#: ``no_executable_pp`` is not here, and cannot be: the planner raises it only when
#: *every* replica is gone, and a fail-stop that kills the last process leaves nobody
#: to observe a stop, publish a plan, or exit consistently. Under real kills that
#: condition is the end of the job, so it has its own gate
#: (:func:`test_losing_every_rank_ends_the_job`) rather than a rank-observed one; the
#: planner code itself is still locked by
#: :func:`test_the_planner_stop_codes_come_out_of_a_real_replan`.
STOP_SCENARIOS = {
    "no_feasible_tp": ("pair", (1,), (2, 2)),
    "no_feasible_dp_target": ("absorb", (1,), (1, 3)),
    "checkpoint_unusable": ("pair", (1,), None),
    # Four ranks, so that the rank made to diverge is a *survivor*: a disagreement
    # needs two live ranks that replanned differently.
    "plan_disagreement": ("replicated", (1,), None),
    "state_mismatch": ("pair", (1,), None),
}
#: Which rank ``plan_disagreement`` makes replan differently -- never a victim.
DIVERGING_RANK = 2

#: A blocked rank is invisible from outside, so each one dumps its own Python stack
#: after this long. A hang then names the exact operation every rank is sitting in.
STACK_DUMP_AFTER = 60.0

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _config(kind):
    return TrainConfig(
        **{"pipeline": PIPELINE, "replicated": REPLICATED, "absorb": ABSORB, "reseat": RESEAT}
        .get(kind, PAIR)
    )


def _budget(config, tp_degree, stage_layers) -> int:
    """The analytic budget for one stage at ``(tp_degree, stage_layers)``.

    The in-flight count here is the *gate's* knob, not the planner's: what each
    scenario needs is a budget the pristine layout fits and the post-failure one does
    not, and one in-flight activation gives that. The planner derives its own in-flight
    counts from the 1F1B schedule (``resihp.planner.pp.peak_in_flight``), which is
    strictly larger for a warmup stage -- so these budgets are, if anything, the harder
    side of the boundary.
    """
    from resihp.memory import estimate_memory

    return estimate_memory(
        config,
        tp_degree=tp_degree,
        stage_layers=stage_layers,
        micro_batches=config.batch_size // config.micro_batch_size,
        sequence_length=SEQLEN,
        vocab_size=VOCAB,
        in_flight_micro_batches=1,
    ).total


def _scenario(code):
    kind, victims, budget = STOP_SCENARIOS[code]
    config = _config(kind)
    return config, victims, None if budget is None else _budget(config, *budget)


def _layers_in(names):
    return sorted({int(name.split(".")[1]) for name in names if name.startswith("layers.")})


def _file_digest(path):
    """Hash a checkpoint file's bytes, or ``None`` when it is not there."""
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# --- pure: the plan is the only layout authority ----------------------------------


@requires_torch
def test_stage_layout_follows_the_plan_not_the_whole_model():
    config = _config("pipeline")
    plan = build_plan(config, step=0, version=0)
    first, last = stage_of(plan, 0), stage_of(plan, 2)

    assert _layers_in(stage_layout(plan, first)) == [0, 1, 2]
    assert _layers_in(stage_layout(plan, last)) == [3, 4, 5]
    # Embeddings on the first executable stage, LM head on the last (plan 3.4).
    assert "token_embedding.weight" in stage_layout(plan, first)
    assert "lm_head.weight" not in stage_layout(plan, first)
    assert "lm_head.weight" in stage_layout(plan, last)
    # Disjoint and complete: no rank's stage is the whole model.
    assert not set(stage_layout(plan, first)) & set(stage_layout(plan, last))


@requires_torch
def test_the_acquire_set_is_exactly_what_the_plan_routed():
    """Recovery fetches the plan's routes and nothing else -- one authority, not two."""
    config = _config("pipeline")
    before = build_plan(config, step=0, version=0)
    after = build_plan(config, step=1, version=1, failed_ranks=(1,), previous=before)

    shrunk, grown = stage_of(after, 0), stage_of(after, 2)
    assert (shrunk.tp_degree, list(range(*shrunk.layer_range))) == (1, [0, 1])
    assert (grown.tp_degree, list(range(*grown.layer_range))) == (2, [2, 3, 4, 5])

    # Stage 0 lost a TP rank: the layers it keeps must be re-chunked, and so must the
    # embeddings it owns -- their owner stage is unchanged but its degree is not.
    acquired = acquire_layout(after, 0)
    assert _layers_in(acquired) == [0, 1]
    assert "token_embedding.weight" in acquired and "lm_head.weight" not in acquired

    # Stage 1 fetches only the layer that moved into it; its own layers and its LM
    # head neither moved nor changed degree, so they are not moved at all.
    arriving = acquire_layout(after, 2)
    assert _layers_in(arriving) == [2]
    assert "lm_head.weight" not in arriving  # its owner stage and degree are unchanged
    assert set(arriving) < set(stage_layout(after, grown))  # it keeps the rest in place

    # Every name came from a route that names this rank as a target, and every route
    # carries the two layouts the transfer needs -- nothing is re-derived at run time.
    from resihp.recovery import route_names, routes_for

    for rank, expected in ((0, acquired), (2, arriving)):
        routes = routes_for(after, rank)
        assert {name for route in routes for name in route_names(route)} == set(expected)
        for route in routes:
            assert route.donor_degree > 0 or route.donor_kind == "checkpoint"
            assert route.target_degree == stage_of(after, rank).tp_degree


@requires_torch
def test_an_untouched_replica_acquires_nothing():
    config = _config("replicated")
    before = build_plan(config, step=0, version=0)
    after = build_plan(config, step=1, version=1, failed_ranks=(1,), previous=before)

    assert acquire_layout(after, 2) == {}  # replica 1 untouched
    assert acquire_layout(after, 0) != {}  # replica 0 halved


@requires_torch
def test_a_rank_that_lost_its_seat_keeps_nothing():
    # TP4: rank 2 goes from shard index 1 to index 0 at the same degree, and idle rank 3
    # is seated with no state at all. Neither may keep anything -- "same degree" is not
    # "same slice".
    config = _config("reseat")
    plan = build_plan(config, step=0, version=0)
    v1 = build_plan(config, step=1, version=1, failed_ranks=(0,), previous=plan)
    v2 = build_plan(config, step=2, version=2, failed_ranks=(0, 1), previous=v1)

    assert stage_of(v1, 3) is None and 3 in v1.live_ranks  # healthy, just unused
    for rank in (2, 3):
        stage = stage_of(v2, rank)
        assert acquire_layout(v2, rank) == stage_layout(v2, stage)


@requires_torch
def test_peer_shards_win_over_the_checkpoint():
    """Healthy peers first: the checkpoint is consulted only for what survives nowhere."""
    left = torch.arange(6.0).reshape(2, 3)
    right = torch.arange(6.0, 12.0).reshape(2, 3)
    stale = torch.zeros(2, 6)

    full, source = reconstruct_full(
        "w", shard_dim=1, old_size=2, contributions={0: left, 1: right}, checkpoint={"w": stale}
    )
    assert source == "peer"
    assert torch.equal(full, torch.cat([left, right], dim=1))

    full, source = reconstruct_full(
        "w", shard_dim=1, old_size=2, contributions={0: left}, checkpoint={"w": stale}
    )
    assert source == "checkpoint"
    assert torch.equal(full, stale)


@requires_torch
def test_missing_checkpoint_names_its_own_root_cause(tmp_path):
    with pytest.raises(CheckpointError, match="缺失"):
        load_anchor(tmp_path / "absent.pt")


@requires_torch
def test_unreadable_checkpoint_names_its_own_root_cause(tmp_path):
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"not a torch payload")
    with pytest.raises(CheckpointError, match="损坏"):
        load_anchor(path)


@requires_torch
def test_tampered_checkpoint_fails_the_stored_digest(tmp_path):
    from resihp.checkpoint import _torch_load, save_checkpoint
    from resihp.reference import ReferenceRun

    path = tmp_path / "ckpt.pt"
    run = ReferenceRun(_config("pair"), vocab_size=VOCAB, sequence_length=SEQLEN)
    run.step()
    save_checkpoint(path, run, plan_version=0)
    assert load_anchor(path)[0]  # intact payload loads

    payload = _torch_load(path)
    payload["params"]["lm_head.weight"] = payload["params"]["lm_head.weight"] + 1.0
    torch.save(payload, path)  # right shapes, right cursor, wrong bytes
    with pytest.raises(CheckpointError, match="摘要不匹配"):
        load_anchor(path)


@requires_torch
def test_the_planner_stop_codes_come_out_of_a_real_replan():
    """Each planner code is produced by replanning a real failure, not by a stub call.

    This also keeps ``STOP_SCENARIOS`` honest: the distributed gates drive exactly
    these sequences through ``safe_point``.
    """
    from resihp.plan import InfeasiblePlan

    planner_codes = ("no_feasible_tp", "no_executable_pp", "no_feasible_dp_target")
    seen = {}
    for code in planner_codes:
        config, victims, budget = _scenario(code)
        gated = dict(memory_budget=budget, vocab_size=VOCAB, sequence_length=SEQLEN)
        plan = build_plan(config, step=0, version=0, **gated)
        failed = ()
        for version, victim in enumerate(victims, start=1):
            failed = tuple(sorted(set(failed) | {victim}))
            try:
                plan = build_plan(
                    config, step=version, version=version, failed_ranks=failed,
                    previous=plan, **gated
                )
            except InfeasiblePlan as error:
                seen[code] = error.reason.code
                break

    assert seen == {code: code for code in planner_codes}
    assert set(seen) <= set(STOP_CODES)


# --- distributed: logic shared by the Gloo and NCCL gates --------------------------


def _matches_checkpoint(control, plan, rank) -> bool:
    """Principle A, before-resume half -- the project's one definition of it."""
    return matches_checkpoint(control.training_run, plan, rank, control.checkpoint_path)


def _snapshot(control, plan, rank):
    """Per-event view of what this rank ended up owning and what it had to fetch."""
    stage = stage_of(plan, rank)
    return {
        "degree": None if stage is None else stage.tp_degree,
        "layers": None if stage is None else _layers_in(stage_layout(plan, stage)),
        "owned": None if stage is None else sorted(stage_layout(plan, stage)),
        # Read from the plan alone -- there is no "previous" to consult, because the
        # plan already carries the routes recovery executed.
        "acquired": None if stage is None else sorted(acquire_layout(plan, rank)),
        "donors": sorted({route.donor_kind for route in plan.state_routes}),
        "matches_checkpoint": _matches_checkpoint(control, plan, rank),
    }


def _attach(control, plan, rank, device, checkpoint):
    from resihp.recovery import initial_run

    control.attach_run(
        initial_run(
            plan,
            rank=rank,
            vocab_size=VOCAB,
            sequence_length=SEQLEN,
            tp_group=control.tp_group,
            executor_group=control.executor_group,
            boundary_groups=control.boundary_groups,
            device=device,
        ),
        checkpoint_path=checkpoint,
        device=device,
    )


def _run_recovery(rank, device, kind, backend, result_dir):
    """Drive real safe points -- each opened by a real kill -- and report the outcome."""
    import torch.distributed as dist

    from resihp.checkpoint import load_checkpoint
    from resihp.reference import ReferenceRun
    from resihp.train import fail_stop

    config = _config(kind)
    checkpoint = result_dir / "ckpt.pt"
    control = ControlPlane.initialize(
        training_backend=backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
    _attach(control, plan, rank, device, checkpoint)

    result = {
        "backend": dist.get_backend(control.tp_group),
        "is_cuda": bool(next(control.training_run.stage.parameters()).is_cuda),
        "all_names": sorted(_model_names(config)),
        "events": [],
        "losses": [],
    }
    control.training_step()  # iteration 1, on the pristine topology

    failed: tuple[int, ...] = ()
    for event, victim in enumerate(VICTIMS[kind], start=1):
        # Step 2 first, while the victim is still alive: it is the last checkpoint that
        # can hold the shards nobody else has.
        control.commit_checkpoint(plan)
        fail_stop({event: victim}, rank, event)  # the victim's process ends here
        plan, failed = control.safe_point(
            config, plan, failed, control.observe(), next_step=event
        )
        assert plan.version == event
        result["events"].append(_snapshot(control, plan, rank))
        result["losses"].append(control.training_step())  # keep training after recovery

    result["cursor"] = None if control.training_run is None else control.training_run.cursor
    result["failed"] = list(failed)
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    # Every surviving rank reloads it: which ranks are left is decided by the schedule,
    # so no single rank can be the designated reporter any more.
    completed, version = load_checkpoint(
        checkpoint, ReferenceRun(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    )
    result["checkpoint"] = [completed, version]
    return result


def _model_names(config):
    from resihp.parallel.reshard import shard_dims

    return shard_dims(range(config.num_layers))


class _FaultInjector(ControlPlane):
    """A control plane that injects one fault at the safe point's own call points.

    Injection lands *after* step 2 has written the pre-failure checkpoint, so what the
    safe point then meets is exactly the condition plan 3.6 names -- not a run that was
    broken before it started. It runs on the same rank the plan makes the writer, so
    the fault always lands on the process that just produced the file.
    """

    kind = ""
    corrupted = None
    injected = False

    def commit_checkpoint(self, plan):
        super().commit_checkpoint(plan)
        if self.rank != plan.active_ranks[0] or self.injected:
            return
        self.injected = True  # the commit now runs every iteration; inject once
        if self.kind == "checkpoint_unusable":
            # Corrupt the payload the commit just wrote. The file still loads, so what
            # rejects it is the stored digest. Corrupting rather than deleting also
            # leaves a file whose bytes the stop path can be *shown* not to touch --
            # deleting the only checkpoint would make the "preserved" claim vacuous.
            from resihp.checkpoint import _torch_load

            payload = _torch_load(Path(self.checkpoint_path))
            payload["params"]["lm_head.weight"] = payload["params"]["lm_head.weight"] + 1.0
            torch.save(payload, self.checkpoint_path)
            self.corrupted = _file_digest(self.checkpoint_path)
        elif self.kind == "state_mismatch":
            # A replicated logical tensor now disagrees with the anchor it was just
            # written from, which is what the post-reshard verification has to catch.
            with torch.no_grad():
                self.training_run.stage.final_norm_weight.add_(1.0)


def _diverge_one_rank(rank):
    """Make one survivor replan to a different -- and perfectly valid -- plan.

    Deterministic replanning cannot disagree with itself, so this one condition has to
    be injected. What is under test is the control plane's reaction to a disagreement,
    not the disagreement itself. The rank chosen is one the schedule never kills: a
    dead process cannot disagree with anybody.
    """
    if rank != DIVERGING_RANK:
        return
    import resihp.control as control_module

    real = control_module.reconfigure
    control_module.reconfigure = lambda *args, **kwargs: real(
        *args, **{**kwargs, "step": kwargs["step"] + 1}
    )


def _run_stop(rank, device, code, backend, result_dir):
    """Walk a scenario's failures until one stops the run; report the aftermath."""
    import torch.distributed as dist

    from resihp.checkpoint import load_checkpoint
    from resihp.reference import ReferenceRun
    from resihp.train import fail_stop

    config, victims, budget = _scenario(code)
    checkpoint = result_dir / "ckpt.pt"
    if code == "plan_disagreement":
        _diverge_one_rank(rank)

    control = _FaultInjector.initialize(
        training_backend=backend,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        memory_budget=budget,
    )
    control.kind = code
    plan = build_plan(
        config, step=0, version=0, memory_budget=budget, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    control.build_training_groups(plan)
    _attach(control, plan, rank, device, checkpoint)
    control.training_step()  # an iteration completes and commits before each fault

    before = (control.tp_group, control.executor_group)
    result = {
        "backend": dist.get_backend(control.tp_group),
        "is_cuda": bool(next(control.training_run.stage.parameters()).is_cuda),
        "published": [],
        "stopped": None,
        "message": None,
    }
    failed: tuple[int, ...] = ()
    for event, victim in enumerate(victims, start=1):
        control.commit_checkpoint(plan)  # step 2, and where the fault is injected
        fail_stop({event: victim}, rank, event)  # the victim's process ends here
        try:
            plan, failed = control.safe_point(
                config, plan, failed, control.observe(), next_step=event
            )
        except ConsistentStop as stop:
            result["stopped"] = stop.reason.code
            result["message"] = stop.reason.message
            break
        result["published"].append(plan.version)
        before = (control.tp_group, control.executor_group)
        control.training_step()

    result["plan_version"] = plan.version
    # Either the pre-failure groups are untouched (the stop landed before any group
    # work) or they were torn down: never a live group belonging to the stopped plan.
    result["groups_are_prefailure"] = (control.tp_group, control.executor_group) == before
    result["groups_alive"] = any(group is not None for group in (control.tp_group, control.executor_group))

    control.shutdown()
    result["initialized"] = dist.is_initialized()
    result["checkpoint"] = None
    if rank == 0:
        # Atomicity: the stop path must not have written, replaced, or removed the
        # checkpoint, and must not have left a half-written temporary behind.
        result["tmp_left"] = Path(str(checkpoint) + ".tmp").exists()
        result["checkpoint_untouched"] = (
            control.corrupted is None or _file_digest(checkpoint) == control.corrupted
        )
        try:
            completed, version = load_checkpoint(
                checkpoint, ReferenceRun(config, vocab_size=VOCAB, sequence_length=SEQLEN)
            )
            result["checkpoint"] = [completed, version]
        except CheckpointError as error:
            result["checkpoint_error"] = str(error)
    return result


# --- process entry point ----------------------------------------------------------

#: ``multiprocessing`` pickles the entry point by name, so the runner travels as a key.
RUNNERS = {"recovery": _run_recovery, "stop": _run_stop}


def _worker(rank, env, runner, kind, backend, result_dir):
    """One spawned rank: join the world from the store, then run the scenario."""
    os.environ.update(env)
    import torch.distributed as dist

    # Kept open for the process's lifetime: faulthandler writes into it from a timer.
    stack_file = Path(result_dir, f"stack_{rank}.txt").open("w")
    faulthandler.dump_traceback_later(STACK_DUMP_AFTER, repeat=True, file=stack_file)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        assert torch.cuda.current_device() == rank
    # The world group is Gloo on both backends -- it carries object collectives and is
    # re-formed on every fail-stop; only the training groups switch to NCCL.
    result = RUNNERS[runner](rank, device, kind, backend, Path(result_dir))
    faulthandler.cancel_dump_traceback_later()
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(runner, kind, backend, tmp_path, *, world_size, killed):
    """Run the scenario for real; return the survivors' results, keyed by rank."""
    exit_codes = run_ranks(_worker, world_size, runner, kind, backend, str(tmp_path))
    label = f"{kind}/{backend}"
    assert_killed(exit_codes, killed, tmp_path, label)
    return read_results(tmp_path, [r for r in range(world_size) if r not in set(killed)])


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- shared assertions ------------------------------------------------------------


def _assert_recovery(results, *, kind, label):
    # A victim is a dead process: it has no result at all, and the harness has already
    # matched the survivors against the schedule.
    assert sorted(results) == [
        rank for rank in range(_config(kind).world_size) if rank not in VICTIMS[kind]
    ], sorted(results)
    for rank, result in sorted(results.items()):
        for event in result["events"]:
            # Principle A: exactly the checkpoint under the new plan, and only the
            # names the new plan says this stage owns.
            assert event["matches_checkpoint"], (label, rank, event)
        assert result["initialized"] is False, (label, rank, result)

    every_name = set(next(iter(results.values()))["all_names"])
    if kind == "pipeline":
        shrunk, grown = results[0]["events"][0], results[2]["events"][0]
        assert (shrunk["degree"], shrunk["layers"]) == (1, [0, 1]), shrunk
        assert (grown["degree"], grown["layers"]) == (2, [2, 3, 4, 5]), grown
        # A real layer migration: layer 2 left stage 0 and was fetched by stage 1,
        # while the layers stage 1 already had were not moved.
        assert _layers_in(shrunk["acquired"]) == [0, 1], shrunk
        assert _layers_in(grown["acquired"]) == [2], grown
        assert set(grown["acquired"]) < set(grown["owned"]), grown
        # No rank holds the whole model; the two stages tile it exactly.
        assert set(shrunk["owned"]) | set(grown["owned"]) == every_name
        assert not set(shrunk["owned"]) & set(grown["owned"])
        assert results[3]["events"][0]["owned"] == grown["owned"]  # its TP peer agrees
        assert results[2]["losses"][0] is not None  # the last stage produces the loss
        assert results[0]["losses"][0] is None
    elif kind == "replicated":
        degrees = {
            rank: [event["degree"] for event in result["events"]]
            for rank, result in results.items()
        }
        assert degrees[0] == [1, 1], degrees  # replica 0 halved at the first event
        assert degrees[2] == [2, 1], degrees  # replica 1 only at the second
        # The untouched replica moves no state at all at the first event.
        assert results[2]["events"][0]["acquired"] == [], results[2]
        assert results[0]["events"][0]["acquired"] != [], results[0]
        for rank in (0, 2):
            assert set(results[rank]["events"][-1]["owned"]) == every_name  # PP1
    else:  # reseat
        degrees = {
            rank: [event["degree"] for event in result["events"]]
            for rank, result in results.items()
        }
        assert degrees[2] == [2, 2], degrees  # rank 2 stays at the same degree
        # A healthy rank the first plan left idle is picked back up by the second.
        assert degrees[3] == [None, 2], degrees
        # Neither may keep its old state: rank 2's shard index moved and rank 3 has none.
        for rank in (2, 3):
            last = results[rank]["events"][1]
            assert last["acquired"] == last["owned"], (rank, last)
        # And everyone resumes from the same iteration -- including the rank that had
        # no cursor of its own to carry forward.
        cursors = {r["cursor"] for r in results.values() if r["cursor"] is not None}
        assert len(cursors) == 1, cursors
    for result in results.values():  # the last checkpoint still reloads, on every rank
        assert result["checkpoint"] is not None, (label, result)


def _assert_consistent_stop(results, code, label, tmp_path):
    events = len(STOP_SCENARIOS[code][1])
    assert {result["stopped"] for result in results.values()} == {code}, (label, results)
    # One agreed root cause, identical on every *surviving* rank -- not each rank's
    # local view. The victims are dead processes and report nothing at all.
    assert len({result["message"] for result in results.values()}) == 1, (label, results)
    for result in results.values():
        # The message points at this condition only -- no second guess, no fallback.
        # (The temp path is stripped first: pytest names it after the test itself.)
        message = result["message"].replace(str(tmp_path), "")
        for other in STOP_CODES:
            if other != code:
                assert other not in message, (label, result)
        # Every event before the last published a plan; the one that stopped did not.
        assert result["published"] == list(range(1, events)), (label, result)
        assert result["plan_version"] == events - 1, (label, result)
        assert result["groups_are_prefailure"] or not result["groups_alive"], (label, result)
        assert result["initialized"] is False, (label, result)

    assert not results[0]["tmp_left"], results[0]  # the write stayed atomic
    assert results[0]["checkpoint_untouched"], results[0]  # the stop path wrote nothing
    if code == "checkpoint_unusable":
        # The subject of this condition *is* the one checkpoint, and plan 3.6 keeps
        # exactly one file -- so "the last valid checkpoint reloads" cannot apply here
        # without contradicting the fault. What the stop path owes is the two
        # assertions above: it left the file exactly as it found it. That it no longer
        # loads is the condition, reported by its own root cause.
        assert results[0]["checkpoint"] is None, results[0]
        assert "摘要不匹配" in results[0]["checkpoint_error"], results[0]
    else:
        # One iteration ran before each event, so the last commit holds them all.
        assert results[0]["checkpoint"] == [events, events - 1], results[0]


# --- Gloo gates (always run where torch is installed) -----------------------------


def _recovery_results(kind, backend, tmp_path):
    return _spawn(
        "recovery",
        kind,
        backend,
        tmp_path,
        world_size=_config(kind).world_size,
        killed=VICTIMS[kind],
    )


def _stop_results(code, backend, tmp_path):
    config, victims, _budget = _scenario(code)
    return _spawn(
        "stop", code, backend, tmp_path, world_size=config.world_size, killed=victims
    )


@requires_torch
@pytest.mark.parametrize("kind", ["pipeline", "replicated", "reseat"])
def test_recovery_gloo(tmp_path, kind):
    results = _recovery_results(kind, "gloo", tmp_path)
    _assert_recovery(results, kind=kind, label=f"{kind} Gloo")


@requires_torch
@pytest.mark.parametrize("code", sorted(STOP_SCENARIOS))
def test_consistent_stop_gloo(tmp_path, code):
    _assert_consistent_stop(
        _stop_results(code, "gloo", tmp_path), code, f"{code} Gloo", tmp_path
    )


@requires_torch
def test_losing_every_rank_ends_the_job(tmp_path):
    """The last rank's death ends the job -- and leaves the checkpoint intact.

    ``no_executable_pp`` is the planner's name for "no replica survived", and under a
    real fail-stop that state has nobody left to observe it: there is no rank to agree
    with, no plan to publish, and no process to exit cleanly. What the run owes in that
    case is what it owes in every other: the last completed iteration is on disk, whole
    and reloadable, so the job can be restarted from it. Both ranks here are killed,
    the second one after it has already recovered from the first, so the file under
    test was written by a topology that a fail-stop had already reconfigured.
    """
    from resihp.checkpoint import load_checkpoint
    from resihp.reference import ReferenceRun

    config = _config("pair")
    exit_codes = run_ranks(
        _worker, config.world_size, "recovery", "pair", "gloo", str(tmp_path)
    )
    assert_killed(exit_codes, VICTIMS["pair"], tmp_path, "pair")

    completed, version = load_checkpoint(
        Path(tmp_path, "ckpt.pt"),
        ReferenceRun(config, vocab_size=VOCAB, sequence_length=SEQLEN),
    )
    # Two iterations completed and one plan version was published before the last
    # rank died, and the file that survives is the one committed after both.
    assert (completed, version) == (2, 1), (completed, version)
    assert not Path(str(Path(tmp_path, "ckpt.pt")) + ".tmp").exists()


# --- NCCL gates (real GPU tensors and real NCCL training groups) ------------------


@requires_torch
@pytest.mark.parametrize("kind", ["pipeline", "replicated", "reseat"])
def test_recovery_cuda_nccl(tmp_path, kind):
    _skip_if_few_gpus(_config(kind).world_size)
    results = _recovery_results(kind, "nccl", tmp_path)
    for result in results.values():
        assert result["is_cuda"] and result["backend"] == "nccl", result
    _assert_recovery(results, kind=kind, label=f"{kind} NCCL")


@requires_torch
@pytest.mark.parametrize("code", sorted(STOP_SCENARIOS))
def test_consistent_stop_cuda_nccl(tmp_path, code):
    _skip_if_few_gpus(_scenario(code)[0].world_size)
    results = _stop_results(code, "nccl", tmp_path)
    for result in results.values():
        assert result["is_cuda"] and result["backend"] == "nccl", result
    _assert_consistent_stop(results, code, f"{code} NCCL", tmp_path)
