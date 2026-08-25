"""Random, repeated, and exhaustive fail-stop sequences (T17).

Plan sections 四.C (multi failure-point coverage and sequence regressions), 四.D
(a fixed-seed random sequence injected to exhaustion, and repeated injection without
leaks) and 四.E (the one resource case the T14 stop gates structurally cannot reach).

Every case is one uninterrupted job driven through the real safe point -- every
failure below is a real ``SIGKILL``, and a killed rank writes no result at all -- over
the same runtime T16 uses; the cases differ only in *when* each failure lands and
*which* rank it takes, which is exactly the axis 四.C asks to cover:

* ``random_a`` / ``random_b`` -- fixed-seed random safe points and random victims, one
  rank per event, injected until only one rank is left running: every event is a real
  ``SIGKILL``, and the survivor still has to end up holding the whole model and
  training on. Two independent sequences over one topology are 四.C's "多次序列回归";
  between them the replicas lose their first stage, their last stage, and finally all
  but one rank.
* ``interval_n`` -- one rank every ``N = 2`` iterations, four events (四.D's repeated
  injection). Victims ``1, 5, 0, 4`` empty *stage 0* of both replicas in turn, so the
  surviving stage absorbs every layer and the embedding changes owner twice.
* ``interval_2n`` -- one rank every ``2N = 4`` iterations, and its victims ``3, 7`` sit
  on the *last* stage, so it is the LM-head end that is resharded and a layer moves
  toward the embedding end rather than away from it.
* ``donor_exhaustion`` -- 四.E's "健康 donor 全失但 checkpoint 可用", the only listed
  scenario T14 does not gate. Its other six are the consistent-stop conditions, each
  already driven through a real ``safe_point`` in ``tests/test_recovery.py``;
  this one is not a stop but a *successful* recovery, and it needs a replica to be
  wiped out entirely before the last surviving replica loses a rank. ``TP2 x PP1 x
  DP2`` losing ranks 2, 3 and then 1: at event 1 the shard comes from the peer
  replica, at event 3 that replica no longer exists and rank 0 is the only rank still
  running, so the plan routes the state from the checkpoint -- and the run trains on.

What every case locks (plan section 四.B, asserted for the *whole* sequence rather
than for one event):

* the published plan stream is byte-for-byte the pure planner's own stream for that
  victim sequence, recomputed twice -- determinism and idempotence together;
* versions increase by exactly one per event, every rank agrees on every digest,
  failures accumulate, and no failed rank is ever active or trains again;
* each replica's stages tile the model contiguously, and every ``(micro-batch,
  stage)`` pair is executed exactly once, by exactly the ranks the plan placed it on;
* principle A on both sides of every recovery: exact ``torch.equal`` equality with the
  committed checkpoint before resuming, and ``allclose`` agreement with the
  "same checkpoint + new topology + new configuration's actual batch + same seed"
  reference afterwards;
* the data cursor advances once per iteration, with no replay and no skip.

**Leaks** (四.D): after every event a rank holds exactly the training groups the
current plan gives it -- its TP group, its executor group, and one union group per
pipeline hop it takes part in -- on top of the baseline measured before any training
group was built, and a dropped rank holds nothing beyond that baseline, so a rebuild
neither accumulates groups nor leaves half a set. The expected count is read from the
control plane itself rather than written down, because it varies with the topology: a
PP1 replica has no hop at all, a rank on an end stage of a PP2 pipeline has one, and a
rank on a middle stage of a deeper pipeline has two; the checkpoint directory holds
exactly one
file and never a leftover
``.tmp``; and no process group survives shutdown. That a reconfiguration *replaces*
a generation rather than accumulating one is asserted on the generation itself -- the
one each event superseded must be unreachable once the collector has run, and so must
the last one when the run ends. Device bytes are reported but never asserted on:
the allocator's floor is process-wide library workspace tens of times larger than any
state this model holds, and its run-to-run spread is the size of a whole generation,
so it cannot resolve what a weakref answers exactly (see ``_resources``).

The gates run on CPU/**Gloo** and on GPU/**NCCL** with real device tensors and real
NCCL training groups; the world group stays Gloo on both, and is re-formed over the
survivors at every fail-stop. The NCCL gate needs one GPU per rank -- eight for the
three-dimensional cases, four for ``donor_exhaustion`` -- and skips with fewer.

Tolerance is T10's reassociation band for anything arithmetic; the ``torch.equal``
comparisons are the checkpoint ones, where no arithmetic is involved.
"""

import faulthandler
import gc
import json
import os
import types
import weakref
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
from torch.nn import functional as F

from harness import assert_killed, read_results, run_ranks
from resihp.checkpoint import load_anchor, load_checkpoint
from resihp.config import TrainConfig
from resihp.control import ConsistentStop, ControlPlane
from resihp.train import fail_stop
from resihp.plan import InfeasiblePlan, build_plan
from resihp.planner.pp import peak_in_flight
from resihp.recovery import initial_run, stage_of
from resihp.reference import ReferenceRun
from resihp import verify


VOCAB = 32
SEQLEN = 8
#: Reassociation band: real TP all-reduce plus micro-batch splitting (T10/T12/T13).
#: Taken from ``resihp.verify``, the one place the numerical contract is defined.
RTOL = verify.RTOL
ATOL = verify.ATOL

#: TP2 x PP2 x DP2 over six layers -- the topology plan section 四.D names, with the
#: layer count that makes a degree change also move a layer across a stage boundary.
THREE_D = TrainConfig(
    model_dim=16,
    num_layers=6,
    num_heads=4,
    batch_size=4,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=10,
)
#: TP2 x PP1 x DP2: two replicas that each hold the whole model, so one can be wiped
#: out while the other still runs -- the shape donor exhaustion needs.
DONOR = TrainConfig(
    model_dim=16,
    num_layers=2,
    num_heads=4,
    batch_size=4,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=1,
    dp=2,
    iterations=5,
)

#: A blocked rank is invisible from outside, so each dumps its own stack after this long.
STACK_DUMP_AFTER = 120.0


def _random_events(seed, world_size, iterations):
    """A fixed-seed random failure sequence: random safe points, random victims.

    One rank per event (principle B), never a rank that already failed, and the gaps
    between safe points are one or two iterations rather than a fixed cadence. The
    generator is a plain LCG instead of :mod:`random` on purpose: the point of a
    fixed seed is that this exact sequence replays anywhere, and ``random.choice``'s
    internals are not part of the language's compatibility guarantee.

    The last rank is never taken. Killing it would leave no process to observe
    anything, so the sequence would end with nothing to assert; exhaustion all the way
    to an empty job is its own gate
    (``tests/test_recovery.py::test_losing_every_rank_ends_the_job``).
    """
    state = seed

    def draw(bound):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 33) % bound

    events, at, live = [], 0, list(range(world_size))
    while len(live) > 1:
        at += 1 + draw(2)
        if at >= iterations:
            break
        events.append((at, live.pop(draw(len(live)))))
    return tuple(events)


@dataclass(frozen=True)
class _Case:
    """One failure sequence: what to run, when to break it, how it must end."""

    config: TrainConfig
    events: tuple[tuple[int, int], ...]  # (after_iteration, failed rank)
    stop: str | None  # the consistent-stop code it must end on, or None to run out
    #: The donor kind each published plan routes its state from, for a case that is
    #: *about* which source recovery draws on; ``None`` for the cases that are not.
    donors: tuple[tuple[str, ...], ...] | None = None


CASES = {
    # Injected until only one rank is left: every victim is a killed process.
    "random_a": _Case(THREE_D, _random_events(42, THREE_D.world_size, 10), None),
    "random_b": _Case(THREE_D, _random_events(112, THREE_D.world_size, 10), None),
    # Every N = 2 iterations, emptying stage 0 of each replica in turn.
    "interval_n": _Case(THREE_D, ((2, 1), (4, 5), (6, 0), (8, 4)), None),
    # Every 2N = 4 iterations, on the last stage instead of the first.
    "interval_2n": _Case(THREE_D, ((4, 3), (8, 7)), None),
    # Both peer replicas gone before the survivor loses a rank (plan section 四.E):
    # a peer donates, then that replica is wiped out, then nothing is left to ask.
    "donor_exhaustion": _Case(
        DONOR,
        ((1, 2), (2, 3), (3, 1)),
        None,
        donors=((), ("peer_replica",), (), ("checkpoint",)),
    ),
}


def _segment_end(case, iteration) -> int:
    """The last iteration that runs under the plan in force just after ``iteration``."""
    later = [after for after, _victim in case.events if after > iteration]
    return min(later) if later else case.config.iterations


def _batch(config, index, device):
    """Iteration ``index``'s fixed token batch -- the same stream the run consumes."""
    return verify.batch(config, index, vocab_size=VOCAB, sequence_length=SEQLEN, device=device)


# --- reference anchors ------------------------------------------------------------


def _reference_steps(config, device, count):
    """The no-failure reference: ``count`` iterations from the fixed initialization."""
    return verify.reference_steps(
        config, vocab_size=VOCAB, sequence_length=SEQLEN, count=count, device=device
    )


def _steps_from_anchor(config, anchor, device, *, start, count):
    """Principle A's after-resume reference: ``count`` steps from the checkpoint anchor.

    The baseline is *not* an uninterrupted run from iteration 0 but "the same
    checkpoint + the new topology + the new configuration's actual batch + the same
    seed" -- which is what :func:`resihp.verify.steps_from_anchor` builds.
    """
    return verify.steps_from_anchor(
        config,
        anchor,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        start=start,
        count=count,
        device=device,
    )


# --- comparisons ------------------------------------------------------------------

#: Principle A lives in one place for the whole project (``resihp.verify``); this is
#: only a local spelling of it, so this gate and the combination / end-to-end gates
#: cannot drift into three slightly different contracts.
_compare_shards = verify.compare_shards


def _matches_anchor(run, plan, rank, checkpoint):
    """Principle A's before-resume half -- the project's one definition of it."""
    return verify.matches_checkpoint(run, plan, rank, checkpoint)


# --- the run ----------------------------------------------------------------------


def _plan_record(plan) -> dict:
    """The parts of a plan this gate checks: identity, layout, placements, donors."""
    return {
        "version": plan.version,
        "digest": plan.digest,
        "failed": list(plan.failed_ranks),
        "active": list(plan.active_ranks),
        "stages": sorted(
            [
                stage.replica_id,
                stage.stage_id,
                stage.tp_degree,
                list(stage.tp_members),
                list(stage.layer_range),
            ]
            for stage in plan.stages
        ),
        "placements": sorted(
            [place.micro_batch, place.stage_id, place.replica_id, list(place.executor_ranks)]
            for place in plan.placements
        ),
        "donors": sorted({route.donor_kind for route in plan.state_routes}),
    }


def _resources(label, control, result_dir, device) -> dict:
    """What must not accumulate across reconfigurations: groups and files.

    ``cuda_bytes`` is carried for diagnosis only, never asserted on. cuBLAS and its
    friends take their workspaces from the same caching allocator on first use and
    hold them for the life of the process: on the eight-GPU target that floor is
    ~16.4 MiB, which is *fifty times* the largest state a rank of this model could
    hold, identical to the byte between a four-rank two-layer job and an eight-rank
    six-layer one, and no larger after four fail-stops than after two. Worse, the
    floor's run-to-run spread (~236 KiB) is the same size as one generation of state
    (~312 KiB), so an absolute byte count cannot resolve the very thing it would be
    asked to detect. What a leaked generation actually means -- the one a
    reconfiguration replaced is still alive -- a weakref answers exactly (``_sentinel``).
    """
    # ``_world.pg_map`` is the process groups this rank currently holds. It is torch's
    # own private registry and there is no public equivalent -- a leaked group is
    # invisible from anywhere else, which is exactly what has to be checked here.
    from torch.distributed.distributed_c10d import _world

    # What the current plan entitles this rank to hold, counted from the control plane
    # rather than assumed: TP group + executor group + one per pipeline hop it joins.
    # Reading it live is also what makes the snapshot right at a *stopped* plan, where
    # the groups still held are the previous plan's.
    training_group_count = (
        int(control.tp_group is not None)
        + int(control.executor_group is not None)
        + len(control.boundary_groups)
    )
    return {
        "at": label,
        "groups": len(_world.pg_map),
        "training_group_count": training_group_count,
        "files": sorted(path.name for path in Path(result_dir).glob("ckpt.pt*")),
        "cuda_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
    }


def _iteration(control, plan, rank, iteration, wanted):
    """Run one iteration and describe it, holding on to nothing when it returns.

    The whole per-iteration body lives here for one reason: it touches the runtime,
    the stage and their parameters, and *any* of those left bound in the caller would
    outlive the next safe point and keep the generation it replaced alive -- which is
    exactly what ``_sentinel`` must be free to observe. A function frame makes that
    structural: every reference dies with the call, rather than depending on nobody
    ever adding one more local to a loop body.
    """
    record = {"iteration": iteration, "trained": False}
    if control.training_run is None:
        return record
    stage = stage_of(plan, rank)
    record["cursor"] = control.training_run.cursor  # the batch about to be read
    record["replica"] = stage.replica_id
    record["layers"] = list(range(*stage.layer_range))
    record["loss"] = control.training_step()  # step 1: complete this iteration
    record["trained"] = True
    runtime = control.training_run.runtime
    record["processed"] = [[route["micro_batch"], route["stage_id"]] for route in runtime.routes]
    record["activation_peak"] = runtime.activation_log.peak
    record["activation_drained"] = runtime.activation_log.live == set()
    record["stage_index"] = runtime.stage_index
    record["num_stages"] = runtime.num_stages
    record.update(_compare_shards(control.training_run.stage, wanted))
    return record


def _owners(sentinel):
    """What is still holding a generation that should be gone, named well enough to fix.

    A bounded walk out from the surviving parameter: the objects that reference it,
    then the objects that reference *those*, so the answer is the retaining chain
    (``PipelineRuntime``, ``TensorParallelStage``, ...) rather than the parameter's
    nearest container. Frames do not appear -- CPython keeps a function's locals in an
    array the collector does not walk -- but a live runtime is already the whole story:
    something is still bound to the generation.
    """
    alive = sentinel()
    if alive is None:
        return []
    seen, frontier, found = {id(alive)}, [alive], set()
    for _ in range(3):
        step = []
        for obj in frontier:
            for referrer in gc.get_referrers(obj):
                if id(referrer) in seen or referrer is frontier or referrer is step:
                    continue
                seen.add(id(referrer))
                if isinstance(referrer, (dict, list, tuple, set)):
                    step.append(referrer)  # a container: keep walking to its owner
                elif not isinstance(referrer, types.FrameType):
                    found.add(type(referrer).__name__)
                    step.append(referrer)
        frontier = step
    return sorted(found)


def _sentinel(run):
    """A weakref to one of ``run``'s parameters, or ``None`` when it holds no state.

    A parameter is the strongest sentinel for "this generation is gone": the stage
    holds it, the runtime holds it through the stage, and the optimizer holds it twice
    over -- in its parameter group and as the key of its moment state -- so anything
    that retains any part of the generation keeps this object alive. A weakref is used
    so that watching it cannot itself be what keeps it alive.
    """
    return None if run is None else weakref.ref(next(run.stage.parameters()))


def _run_sequence(rank, device, backend, case_name, result_dir):
    """One rank's whole run: train, and take every safe point a real kill opens."""
    case = CASES[case_name]
    config = case.config
    schedule = dict(case.events)
    checkpoint = result_dir / "ckpt.pt"

    control = ControlPlane.initialize(
        training_backend=backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    # Taken before any training group exists: the world group on its own is the
    # baseline every later group count is measured against, so the leak check does not
    # have to assume how torch counts the default group. A fail-stop re-forms that
    # group, so the baseline is also what proves the *old* world was released.
    resources = [_resources("init", control, result_dir, device)]
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
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
    on_gpu = bool(next(control.training_run.stage.parameters()).is_cuda)

    # Segment 1's baseline is the no-failure reference; each recovery replaces it with
    # the reference for the topology that recovery produced.
    reference = _reference_steps(config, device, _segment_end(case, 0))
    base = 1  # the iteration ``reference[0]`` describes

    plans = [_plan_record(plan)]
    resources.append(_resources("start", control, result_dir, device))
    iterations, events, stop = [], [], None
    failed: tuple[int, ...] = ()
    # Written after every iteration rather than once at the end: a rank about to be
    # killed gets no chance to report afterwards, and its account of the iterations it
    # *did* run is what lets the "(micro-batch, stage) executed exactly once" invariant
    # still be checked over the iterations it took part in.
    result = {
        "rank": rank,
        "plans": plans,
        "events": events,
        "iterations": iterations,
        "stop": stop,
    }
    partial = Path(result_dir, f"result_{rank}.json")

    def flush():
        result["stop"] = stop
        partial.write_text(json.dumps(result))

    flush()
    for step in range(config.iterations):
        iteration = step + 1
        iterations.append(_iteration(control, plan, rank, iteration, reference[iteration - base]))
        flush()

        # Step 2 runs every iteration, while everyone the schedule is about to kill is
        # still alive; step 3 is the boundary, where the survivors find out who is not.
        control.commit_checkpoint(plan)
        fail_stop(schedule, rank, iteration)  # a scheduled rank's process ends here
        replaced = _sentinel(control.training_run)
        lost = control.observe()
        if not lost:
            continue
        try:
            plan, failed = control.safe_point(config, plan, failed, lost, next_step=iteration)
        except ConsistentStop as stopped:
            stop = {"code": stopped.reason.code, "message": stopped.reason.message}
            resources.append(_resources("stop", control, result_dir, device))
            break
        gc.collect()
        anchor, completed = load_anchor(checkpoint)
        plans.append(_plan_record(plan))
        events.append(
            {
                "iteration": iteration,
                "victim": lost[0],
                "failed": list(failed),
                "completed": completed,
                "matches_checkpoint": _matches_anchor(control.training_run, plan, rank, checkpoint),
                "cursor": None if control.training_run is None else control.training_run.cursor,
                # Reconfiguration must *replace* a generation, not accumulate one: the
                # one this event superseded has to be unreachable once the collector has
                # run, whether this rank was rebuilt or dropped entirely.
                "released_previous": replaced is None or replaced() is None,
                "held_by": [] if replaced is None else _owners(replaced),
            }
        )
        resources.append(_resources(f"event{iteration}", control, result_dir, device))
        flush()
        reference = _steps_from_anchor(
            config, anchor, device, start=completed, count=_segment_end(case, iteration) - iteration
        )
        base = iteration + 1

    if control.training_run is not None:
        # Recovery rebuilds the stage, so the device has to be re-checked afterwards:
        # this is where it lands, not where it was asked to go.
        on_gpu &= bool(next(control.training_run.stage.parameters()).is_cuda)

    result.update(
        stop=stop,
        backend=None if control.tp_group is None else dist.get_backend(control.tp_group),
        is_cuda=on_gpu,
        # Every surviving rank reloads it: which ranks survive is the schedule's
        # choice, so no fixed rank can be the designated reporter.
        checkpoint=_reload_checkpoint(config, checkpoint),
    )
    control.shutdown()
    result["initialized"] = dist.is_initialized()

    # The generation still standing at the end has to go the same way every replaced
    # one did, so the run leaves nothing of itself behind.
    final = _sentinel(control.training_run)
    control.training_run = None
    gc.collect()
    result["released_final"] = final is None or final() is None
    result["final_held_by"] = [] if final is None else _owners(final)
    resources.append(_resources("shutdown", control, result_dir, device))
    result["resources"] = resources
    flush()
    return result


def _reload_checkpoint(config, path):
    """The last checkpoint, reloaded end to end: ``[completed_steps, plan_version]``."""
    completed, version = load_checkpoint(
        path, ReferenceRun(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    )
    return [completed, version]


# --- assertions -------------------------------------------------------------------


def _pure_stream(case):
    """The plan stream the *pure* planner produces for this victim sequence.

    Recomputed twice and required to agree: same ``(config, step, version, cumulative
    failures, previous plan)`` in, same plans out, which is plan section 四.B's
    idempotence requirement stated over a whole sequence rather than a single call.
    """

    def once():
        digests, plan, failed = [], build_plan(case.config, step=0, version=0), ()
        digests.append(plan.digest)
        for iteration, victim in case.events:
            failed = tuple(sorted(set(failed) | {victim}))
            try:
                plan = build_plan(
                    case.config,
                    step=iteration,
                    version=plan.version + 1,
                    failed_ranks=failed,
                    previous=plan,
                    vocab_size=VOCAB,
                    sequence_length=SEQLEN,
                )
            except InfeasiblePlan as error:
                return digests, error.reason.code
            digests.append(plan.digest)
        return digests, None

    first, second = once(), once()
    assert first == second, "replanning the same failure sequence gave a different plan"
    return first


def _assert_plan_stream(results, case, label):
    """One new version per event, agreed by every rank, equal to the pure planner's."""
    digests, stop_code = _pure_stream(case)
    # The case table's expected ending is the planner's, not a hand-written guess.
    assert stop_code == case.stop, (label, stop_code, case.stop)
    published = len(case.events) + 1 if stop_code is None else len(case.events)
    complete = _complete(results, case)
    for result in results:
        # A rank that was killed stops mid-stream, and everything it did record has to
        # be exactly the survivors' prefix -- same versions, same digests, same events.
        acted = len(result["plans"])
        assert acted == published or result["rank"] in set(_killed(case)), (
            label,
            result["rank"],
            acted,
        )
        versions = [entry["version"] for entry in result["plans"]]
        assert versions == list(range(acted)), (label, result["rank"], versions)
        assert [entry["digest"] for entry in result["plans"]] == digests[:acted], (
            label,
            result["rank"],
        )
        assert [entry["iteration"] for entry in result["events"]] == [
            after for after, _victim in case.events[: acted - 1]
        ], (label, result["rank"])
    for result in _survivors(results, case):
        assert len(result["iterations"]) == len(complete["iterations"]), (label, result["rank"])
        # The failure set the broadcast confirmed is the one the plan was built from.
        for order, entry in enumerate(result["events"]):
            assert entry["failed"] == result["plans"][order + 1]["failed"], (
                label,
                result["rank"],
                entry,
            )
        # Failures accumulate and never re-enter the training path.
        seen: set[int] = set()
        for entry in result["plans"]:
            assert seen <= set(entry["failed"]), (label, result["rank"], entry)
            seen = set(entry["failed"])
            assert not seen & set(entry["active"]), (label, result["rank"], entry)
            members = [rank for row in entry["stages"] for rank in row[3]]
            assert sorted(members) == entry["active"], (label, result["rank"], entry)
            assert len(members) == len(set(members)), (label, result["rank"], entry)


def _assert_layers_and_placements(results, case, label):
    """Layers tile each replica; every (micro-batch, stage) runs once, where planned."""
    config = case.config
    micro = config.batch_size // config.micro_batch_size
    # One rank's copy of the plans is every rank's: the digest covers the stages and
    # the placements in full, and ``_assert_plan_stream`` has already required the
    # digests to be identical on every rank.
    for version, entry in enumerate(_complete(results, case)["plans"]):
        for replica in {row[0] for row in entry["stages"]}:
            ranges = sorted(row[4] for row in entry["stages"] if row[0] == replica)
            covered = [layer for low, high in ranges for layer in range(low, high)]
            assert covered == list(range(config.num_layers)), (label, version, replica, ranges)
        assert {place[0] for place in entry["placements"]} == set(range(micro)), (label, version)

    plan_of = _plan_by_iteration(_complete(results, case), case)
    for iteration in range(1, config.iterations + 1):
        entry = plan_of.get(iteration)
        if entry is None:
            continue  # the run stopped before this iteration
        # Every rank that reached this iteration, killed ones included: what the plan
        # placed on a rank that later died still had to be executed while it was alive.
        executed: dict[tuple[int, int], list[int]] = {}
        for result in results:
            if len(result["iterations"]) < iteration:
                continue
            for pair in result["iterations"][iteration - 1].get("processed", []):
                executed.setdefault((pair[0], pair[1]), []).append(result["rank"])
        planned = {(place[0], place[1]): sorted(place[3]) for place in entry["placements"]}
        assert {key: sorted(value) for key, value in executed.items()} == planned, (
            label,
            iteration,
            executed,
            planned,
        )

    for result in results:
        for record in result["iterations"]:
            if not record["trained"]:
                continue
            entry = plan_of[record["iteration"]]
            mine = [row for row in entry["stages"] if result["rank"] in row[3]]
            assert len(mine) == 1, (label, result["rank"], record["iteration"])
            assert record["layers"] == list(range(*mine[0][4])), (label, record)
            # Activations are all retired, and none is held past its own backward.
            assert record["activation_drained"], (label, result["rank"], record["iteration"])
            # 1F1B, not GPipe: the peak is one activation per stage still downstream,
            # capped by the micro-batches this rank runs -- never one per micro-batch.
            assert record["activation_peak"] == peak_in_flight(
                len(record["processed"]),
                stage_index=record["stage_index"],
                num_stages=record["num_stages"],
            ), (label, record)


def _plan_by_iteration(result, case) -> dict[int, dict]:
    """Which published plan each iteration ran under, by iteration number."""
    plans, mapping, index = result["plans"], {}, 0
    for iteration in range(1, case.config.iterations + 1):
        if iteration > len(result["iterations"]):
            break
        mapping[iteration] = plans[index]
        if any(after == iteration for after, _victim in case.events):
            index += 1
        if index >= len(plans):
            break
    return mapping


def _assert_failed_ranks_and_data(results, case, label):
    """A failed rank never trains again; the token stream is walked once, in order."""
    plan_of = _plan_by_iteration(_complete(results, case), case)
    for result in results:
        for record in result["iterations"]:
            entry = plan_of[record["iteration"]]
            expected = result["rank"] in entry["active"]
            assert record["trained"] == expected, (label, result["rank"], record["iteration"])
            if not record["trained"]:
                assert "processed" not in record, (label, result["rank"], record)
        for event in result["events"]:
            # The recovered cursor is the checkpoint's completed-step count: the resumed
            # run reads the next batch, neither replaying nor skipping one.
            assert event["completed"] == event["iteration"], (label, result["rank"], event)
            assert event["cursor"] in (None, event["completed"]), (label, result["rank"], event)
    # In any one iteration, every training rank consumes the same batch index.
    for iteration in plan_of:
        seen = {
            result["iterations"][iteration - 1]["cursor"]
            for result in results
            if len(result["iterations"]) >= iteration
            and result["iterations"][iteration - 1]["trained"]
        }
        assert seen == {iteration - 1}, (label, iteration, seen)


def _assert_principle_a(results, case, label):
    """Exactly the checkpoint before resuming; the new-configuration reference after."""
    for result in results:
        for event in result["events"]:
            assert event["matches_checkpoint"], (label, result["rank"], event)
        for record in result["iterations"]:
            if not record["trained"]:
                continue
            # The worst element's whole story: which tensor, how far the gradient is
            # off in absolute and relative terms, how large the reference gradient is
            # there, what the parameter did, and AdamW's divisor at that element.
            detail = (
                label,
                f"rank {result['rank']} iteration {record['iteration']}",
                record["worst_grad"],
                record["worst_param"],
            )
            assert record["grad_close"], detail
            assert record["step_close"], detail
    # The replicas' losses partition the global batch, so they sum to the reference's.
    for iteration in _plan_by_iteration(_complete(results, case), case):
        by_replica, reference_loss = {}, None
        for result in results:
            if len(result["iterations"]) < iteration:
                continue
            record = result["iterations"][iteration - 1]
            if not record["trained"] or record["loss"] is None:
                continue
            # A replica contributes one loss: its last stage's TP peers each compute
            # the identical value, so they must agree before one of them is counted.
            seen = by_replica.setdefault(record["replica"], record["loss"])
            assert seen == pytest.approx(record["loss"]), (label, iteration, record)
            reference_loss = record["reference_loss"]
        assert reference_loss is not None, (label, iteration)
        assert abs(sum(by_replica.values()) - reference_loss) < 1e-4, (
            label,
            iteration,
            by_replica,
            reference_loss,
        )


def _assert_no_leaks(results, case, label):
    """Groups, files, and generations never accumulate, and the checkpoint reloads.

    The survivors only: releasing what it held on the way out is something a killed
    process never gets to do, and its shutdown snapshot does not exist.
    """
    for result in _survivors(results, case):
        # Every reconfiguration released the generation it replaced, and the run
        # released the last one on the way out (``_resources`` says why this is a
        # weakref rather than a byte count).
        for event in result["events"]:
            assert event["released_previous"], (label, result["rank"], event["held_by"], event)
        assert result["released_final"], (label, result["rank"], result["final_held_by"])
        baseline = result["resources"][0]["groups"]  # the control group, before any build
        for snapshot in result["resources"]:
            if snapshot["at"] == "shutdown":
                assert snapshot["groups"] == 0, (label, result["rank"], snapshot)
            else:
                # Exactly the training groups this rank is entitled to on top of the
                # control group, and none when the plan places it nowhere: one
                # rebuild's worth, never two, and never half a set.
                expected = baseline + snapshot["training_group_count"]
                assert snapshot["groups"] == expected, (label, result["rank"], snapshot)
            expected = [] if snapshot["at"] in ("init", "start") else ["ckpt.pt"]
            assert snapshot["files"] == expected, (label, result["rank"], snapshot)
        assert result["initialized"] is False, (label, result["rank"])
    # The checkpoint the last safe point committed is still the one on disk, and it
    # still reloads end to end.
    # It is committed every iteration, so the file that survives the run is the last
    # iteration's, under the last plan the events published -- and every rank agrees.
    expected = [case.config.iterations, len(case.events)]
    for result in _survivors(results, case):
        assert result["checkpoint"] == expected, (label, result["rank"], result["checkpoint"])


def _assert_no_stop(results, case, label):
    """Every sequence here recovers and trains on -- none of them may end in a stop.

    The stop conditions themselves are T14's gates; what these cases are about is the
    other outcome, so a stop appearing here is a failure to recover, not a scenario.
    """
    assert case.stop is None, (label, case.stop)
    for result in results:
        assert result["stop"] is None, (label, result["rank"], result["stop"])
    # And the survivors really did reach the last iteration rather than stopping early.
    for result in _survivors(results, case):
        assert len(result["iterations"]) == case.config.iterations, (label, result["rank"])


def _assert_donor_stream(results, case, label):
    """Where each event draws its state from, for a case that is about the donors.

    ``state_routes`` is the *plan's* choice of source, which recovery does not read --
    :func:`resihp.parallel.reshard.reshard_tp_state` decides per name from what the
    healthy ranks actually contribute. So the checkpoint fallback is not proven by
    this list alone: it is proven by the list plus the fact that at that event only
    one rank of the whole job is still alive, so nothing *could* have donated, plus
    ``_assert_principle_a``'s ``torch.equal`` with the anchor, which a silently
    zero-filled or half-kept shard could not satisfy.
    """
    plans = _complete(results, case)["plans"]
    assert [tuple(entry["donors"]) for entry in plans] == list(case.donors), (
        label,
        [entry["donors"] for entry in plans],
    )
    if "checkpoint" in case.donors[-1]:
        alive = case.config.world_size - len(plans[-1]["failed"])
        assert alive == 1, (label, alive, plans[-1])


def _assert_sequence(results, case, label):
    _assert_plan_stream(results, case, label)
    _assert_layers_and_placements(results, case, label)
    _assert_failed_ranks_and_data(results, case, label)
    _assert_principle_a(results, case, label)
    _assert_no_leaks(results, case, label)
    _assert_no_stop(results, case, label)
    if case.donors is not None:
        _assert_donor_stream(results, case, label)


# --- process entry point ----------------------------------------------------------


def _entry(rank, env, backend, case_name, result_dir):
    """One spawned rank: join the world from the store, then run the case."""
    os.environ.update(env)
    # Kept open for the process's lifetime: faulthandler writes into it from a timer.
    stack_file = Path(result_dir, f"stack_{rank}.txt").open("w")
    faulthandler.dump_traceback_later(STACK_DUMP_AFTER, repeat=True, file=stack_file)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        assert torch.cuda.current_device() == rank
    # The world group is Gloo on both backends -- it carries the object collectives and
    # is re-formed on every fail-stop; only the training groups switch to NCCL.
    _run_sequence(rank, device, backend, case_name, Path(result_dir))
    faulthandler.cancel_dump_traceback_later()
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(backend, case_name, tmp_path):
    """Run the case for real; return every rank's account, in rank order.

    A killed rank's file stops at the iteration it died in (see :func:`_run_sequence`),
    so the invariants that are about *what ran* still see the whole job, while the ones
    that are about how the run ended ask :func:`_survivors`.
    """
    case = CASES[case_name]
    world_size = case.config.world_size
    exit_codes = run_ranks(_entry, world_size, backend, case_name, str(tmp_path))
    assert_killed(exit_codes, _killed(case), tmp_path, f"{case_name}/{backend}")
    accounts = read_results(tmp_path, range(world_size))
    return [accounts[rank] for rank in sorted(accounts)]


def _killed(case):
    """The ranks this case really kills, in the order it kills them."""
    return [victim for _after, victim in case.events]


def _survivors(results, case):
    """The ranks still running at the end -- the only ones with a full account."""
    return [result for result in results if result["rank"] not in set(_killed(case))]


def _complete(results, case):
    """One surviving rank's account: every plan, every iteration, every event."""
    return _survivors(results, case)[0]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- gates ------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_fault_sequence_gloo(tmp_path, case_name):
    """Each failure sequence, end to end, on the CPU/Gloo configuration."""
    results = _spawn("gloo", case_name, tmp_path)
    _assert_sequence(results, CASES[case_name], f"{case_name} Gloo")


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_fault_sequence_cuda_nccl(tmp_path, case_name):
    """The same sequences on real GPU tensors and real NCCL training groups."""
    case = CASES[case_name]
    _skip_if_few_gpus(case.config.world_size)
    results = _spawn("nccl", case_name, tmp_path)
    for result in results:
        assert result["is_cuda"], result["rank"]
    # An idle rank holds no training group; every rank that holds one is on NCCL.
    backends = {result["backend"] for result in results} - {None}
    assert backends == {"nccl"}, [result["backend"] for result in results]
    _assert_sequence(results, case, f"{case_name} NCCL")
