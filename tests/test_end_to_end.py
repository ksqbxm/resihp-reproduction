"""Full 3D end-to-end fail-stop scenario (T16).

Plan section 四.D's "完整 3D" bullet, run as one uninterrupted eight-process job at
``TP2 x PP2 x DP2``:

    no-failure reference -> fail a TP rank after iteration 2 -> TP reshard / PP layer
    migration / DP reroute -> recover -> fail a rank of the *other* DP replica after
    iteration 4 -> reconfigure and recover again -> train to the end.

Six layers rather than four, so halving stage 0's degree also pushes a layer across
the stage boundary: one event exercises all three planes at once instead of only the
TP one. The two events hit ranks 1 and 5 -- one per DP replica -- so the second event
lands on a replica that was untouched by the first, and between the events the two
replicas run *different* PP layerings at *different* TP degrees.

What the gate locks (the bullet's own checklist):

* **no collective-order error, no deadlock** -- a mismatched collective order under
  Gloo/NCCL either hangs or raises; every rank is required to finish and write its
  result before a hard timeout, and a blocked rank dumps its own stack into the
  failure message rather than hanging the suite.
* **a failed rank never trains again** -- the iterations each rank actually executed.
* **exactly one new plan per failure** -- versions ``[1, 2]``, strictly increasing,
  with every rank agreeing on each event's digest.
* **before resume, the state is exactly the checkpoint** -- principle A's first half:
  every rank's shards equal ``torch.equal`` slices of the committed anchor, and a
  dropped rank holds nothing at all.
* **after resume, the run matches the new-configuration reference** -- principle A's
  second half: the baseline is "the same checkpoint + the new topology + the new
  configuration's actual batch + the same seed", never an uninterrupted run from
  iteration 0. Iterations 1-2 are compared against the no-failure reference, 3-4
  against the reference restarted from event 1's anchor, 5-6 from event 2's.
* **data is neither repeated nor skipped** -- the cursor each rank consumed, iteration
  by iteration, across both recoveries.
* **layers and micro-batch/stage pairs are neither duplicated nor dropped** -- each
  plan's stages tile the model contiguously, and the pairs the ranks *actually*
  executed match the plan's placements exactly, once each.

The gate runs on CPU/**Gloo** (the eight-process configuration plan section 四.D
names) and on GPU/**NCCL** with real device tensors and real NCCL training groups;
the world group stays Gloo on both, because it is the control plane's always-alive
control group (plan 3.2). The NCCL gate needs eight GPUs -- one rank per device --
and skips when there are fewer.

Tolerance: TP all-reduce and micro-batch splitting reorder FP32 accumulation relative
to the reference's single full-batch pass, so the numeric comparison is the
``allclose`` band T10 fixed, not bit-for-bit. The ``torch.equal`` comparisons are the
checkpoint ones, where no arithmetic is involved.
"""

import faulthandler
import json
import os
import socket
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
from torch.nn import functional as F

from resihp.checkpoint import load_anchor
from resihp.config import TrainConfig
from resihp.control import ControlPlane
from resihp.plan import build_plan
from resihp.planner.pp import peak_in_flight
from resihp.recovery import initial_run, stage_of
from resihp import verify


VOCAB = 32
SEQLEN = 8
#: Reassociation band: real TP all-reduce plus micro-batch splitting (T10/T12/T13).
#: Taken from ``resihp.verify``, the one place the numerical contract is defined.
RTOL = verify.RTOL
ATOL = verify.ATOL

#: Six layers so one event both halves a degree and moves a layer across a boundary.
CONFIG = TrainConfig(
    model_dim=16,
    num_layers=6,
    num_heads=4,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=6,
)
WORLD_SIZE = CONFIG.world_size  # 8
MICRO = CONFIG.batch_size // CONFIG.micro_batch_size  # 4 micro-batches
#: ``after_iteration -> failed rank``: a TP rank of replica 0, then one of replica 1.
FAILURES = {2: 1, 4: 5}

#: ``[replica, stage, tp_degree, tp_members, layer_range]`` per plan version, in order.
#: Version 0 is the pristine layout; version 1 is event 1 (rank 1 dies: replica 0's
#: stage 0 halves to TP1 and gives layer 2 to stage 1, replica 1 is untouched);
#: version 2 is event 2 (rank 5 dies: the same happens to replica 1, and replica 0 --
#: already reconfigured -- is not moved by a single byte).
EXPECTED_PLANS = [
    [
        [0, 0, 2, [0, 1], [0, 3]],
        [0, 1, 2, [2, 3], [3, 6]],
        [1, 0, 2, [4, 5], [0, 3]],
        [1, 1, 2, [6, 7], [3, 6]],
    ],
    [
        [0, 0, 1, [0], [0, 2]],
        [0, 1, 2, [2, 3], [2, 6]],
        [1, 0, 2, [4, 5], [0, 3]],
        [1, 1, 2, [6, 7], [3, 6]],
    ],
    [
        [0, 0, 1, [0], [0, 2]],
        [0, 1, 2, [2, 3], [2, 6]],
        [1, 0, 1, [4], [0, 2]],
        [1, 1, 2, [6, 7], [2, 6]],
    ],
]
#: Which plan version each iteration ran under (iteration 1 is index 0).
PLAN_BY_ITERATION = [0, 0, 1, 1, 2, 2]
#: The iterations each rank executes: rank 1 stops after event 1, rank 5 after event 2.
EXPECTED_TRAINED = {
    1: [1, 2],
    5: [1, 2, 3, 4],
    **{rank: [1, 2, 3, 4, 5, 6] for rank in (0, 2, 3, 4, 6, 7)},
}

#: A rank stuck in a collective would hang the suite forever; fail the gate instead.
JOIN_TIMEOUT = 600.0
#: A blocked rank is invisible from outside, so each dumps its own stack after this long.
STACK_DUMP_AFTER = 120.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _batch(index, device):
    """Iteration ``index``'s fixed token batch -- the same stream the run consumes."""
    return verify.batch(CONFIG, index, vocab_size=VOCAB, sequence_length=SEQLEN, device=device)


# --- reference anchors ------------------------------------------------------------


def _reference_steps(device, count):
    """The no-failure reference: ``count`` iterations from the fixed initialization."""
    return verify.reference_steps(
        CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN, count=count, device=device
    )


def _steps_from_anchor(anchor, device, *, start, count):
    """Principle A's after-resume reference: ``count`` steps from the checkpoint anchor.

    The baseline is *not* an uninterrupted run from iteration 0 but "the same
    checkpoint + the new topology + the new configuration's actual batch + the same
    seed" -- which is what :func:`resihp.verify.steps_from_anchor` builds.
    """
    return verify.steps_from_anchor(
        CONFIG,
        anchor,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        start=start,
        count=count,
        device=device,
    )


# --- comparisons ------------------------------------------------------------------

#: Principle A lives in one place for the whole project (``resihp.verify``); these are
#: only local spellings of it, so this gate and the combination / fault-sequence gates
#: cannot drift into three slightly different contracts.
_compare_shards = verify.compare_shards


def _matches_anchor(run, plan, rank, checkpoint):
    """Principle A's before-resume half: exactly the checkpoint, re-sharded by the plan."""
    return verify.matches_checkpoint(run, plan, rank, checkpoint)


# --- the scenario -----------------------------------------------------------------


def _plan_record(plan) -> dict:
    """The parts of a plan this gate checks: its identity, its stages, its placements."""
    return {
        "version": plan.version,
        "digest": plan.digest,
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
    }


def _segment_end(iteration: int) -> int:
    """The last iteration that runs under the plan in force just after ``iteration``."""
    later = [after for after in FAILURES if after > iteration]
    return min(later) if later else CONFIG.iterations


def _run_end_to_end(rank, world_size, device, backend, result_dir):
    """One rank's whole run: six iterations across two fail-stop safe points."""
    checkpoint = result_dir / "ckpt.pt"
    control = ControlPlane(
        rank, world_size, dist.group.WORLD, backend, vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(CONFIG, step=0, version=0)
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
    reference = _reference_steps(device, _segment_end(0))
    base = 1  # the iteration ``reference[0]`` describes
    all_names = sorted(reference[0]["params"])  # the whole logical model

    plans = [_plan_record(plan)]
    iterations, events = [], []
    failed: tuple[int, ...] = ()
    for step in range(CONFIG.iterations):
        iteration = step + 1
        record = {"iteration": iteration, "trained": False}
        if control.training_run is not None:
            stage = stage_of(plan, rank)
            record["cursor"] = control.training_run.cursor  # the batch about to be read
            record["replica"] = stage.replica_id
            record["layers"] = list(range(*stage.layer_range))
            record["loss"] = control.training_step()  # step 1: complete this iteration
            record["trained"] = True
            runtime = control.training_run.runtime
            record["processed"] = [
                [route["micro_batch"], route["stage_id"]] for route in runtime.routes
            ]
            record["activation_peak"] = runtime.activation_log.peak
            record["activation_drained"] = runtime.activation_log.live == set()
            record["stage_index"] = runtime.stage_index
            record["num_stages"] = runtime.num_stages
            record["schedule"] = list(runtime.schedule)
            wanted = reference[iteration - base]
            record.update(_compare_shards(control.training_run.stage, wanted))
        iterations.append(record)

        failed_rank = FAILURES.get(iteration)
        if failed_rank is None:
            continue
        plan, failed = control.safe_point(CONFIG, plan, failed, failed_rank, next_step=iteration)
        anchor, completed = load_anchor(checkpoint)
        plans.append(_plan_record(plan))
        events.append(
            {
                "iteration": iteration,
                "failed_rank": failed_rank,
                "failed": list(failed),
                "completed": completed,
                "matches_checkpoint": _matches_anchor(control.training_run, plan, rank, checkpoint),
                "cursor": None if control.training_run is None else control.training_run.cursor,
            }
        )
        reference = _steps_from_anchor(
            anchor, device, start=completed, count=_segment_end(iteration) - iteration
        )
        base = iteration + 1

    if control.training_run is not None:
        # Recovery rebuilds the stage, so the device has to be re-checked afterwards:
        # this is where it lands, not where it was asked to go.
        on_gpu &= bool(next(control.training_run.stage.parameters()).is_cuda)

    result = {
        "rank": rank,
        "plans": plans,
        "events": events,
        "iterations": iterations,
        "final_failed": list(plan.failed_ranks),
        "final_live": list(plan.live_ranks),
        "all_names": all_names,
        "backend": None if control.tp_group is None else dist.get_backend(control.tp_group),
        "is_cuda": on_gpu,
    }
    control.shutdown()
    result["initialized"] = dist.is_initialized()
    return result


# --- assertions -------------------------------------------------------------------


def _iteration(results, iteration):
    """Every rank's record for one iteration, indexed by rank."""
    return [result["iterations"][iteration - 1] for result in results]


def _assert_one_plan_per_failure(results, label):
    """Each event yields exactly one new version, and every rank agrees on it."""
    for result in results:
        assert [entry["version"] for entry in result["plans"]] == [0, 1, 2], result["rank"]
        assert [entry["iteration"] for entry in result["events"]] == sorted(FAILURES)
        assert [entry["failed_rank"] for entry in result["events"]] == [
            FAILURES[after] for after in sorted(FAILURES)
        ]
        assert result["final_failed"] == [1, 5], result["rank"]
        assert result["final_live"] == [0, 2, 3, 4, 6, 7], result["rank"]
    for version in range(3):
        digests = {result["plans"][version]["digest"] for result in results}
        assert len(digests) == 1, (label, version, digests)


def _assert_planes_reconfigured(results, label):
    """Each event really moved TP, PP, and DP -- not just the plan's version number."""
    plans = results[0]["plans"]
    for version, expected in enumerate(EXPECTED_PLANS):
        assert plans[version]["stages"] == expected, (label, version, plans[version]["stages"])

    def executors(version, micro, stage_id):
        return [
            place[3]
            for place in plans[version]["placements"]
            if place[0] == micro and place[1] == stage_id
        ]

    # Event 1: replica 0's stage 0 is resharded TP2 -> TP1, layer 2 migrates to stage 1,
    # and its micro-batches are rerouted onto the surviving executor.
    assert executors(0, 0, 0) == [[0, 1]] and executors(1, 0, 0) == [[0]], label
    # Event 2 does the same to replica 1 and leaves replica 0 alone.
    assert executors(1, 2, 0) == [[4, 5]] and executors(2, 2, 0) == [[4]], label
    assert [row for row in plans[1]["stages"] if row[0] == 0] == [
        row for row in plans[2]["stages"] if row[0] == 0
    ], label
    # Between the events the two replicas run different PP layerings at different
    # TP degrees -- the heterogeneous case the DP combine has to handle.
    assert [row[2] for row in plans[1]["stages"]] == [1, 2, 2, 2], label


def _assert_layers_tile_the_model(results, label):
    """Every global layer is owned by exactly one stage of each replica, contiguously."""
    for version, plan in enumerate(results[0]["plans"]):
        for replica in (0, 1):
            ranges = sorted(row[4] for row in plan["stages"] if row[0] == replica)
            covered = [layer for low, high in ranges for layer in range(low, high)]
            assert covered == list(range(CONFIG.num_layers)), (label, version, replica, ranges)
    # And every rank really holds only its own stage's layers while it trains.
    for result in results:
        for record in result["iterations"]:
            if not record["trained"]:
                continue
            stages = result["plans"][PLAN_BY_ITERATION[record["iteration"] - 1]]["stages"]
            mine = [row for row in stages if result["rank"] in row[3]]
            assert len(mine) == 1, (label, result["rank"], record["iteration"])
            assert record["layers"] == list(range(*mine[0][4])), (label, record)


def _assert_micro_batch_stage_executed_once(results, label):
    """Each ``(micro-batch, stage)`` ran exactly once per iteration, on its planned ranks."""
    for iteration in range(1, CONFIG.iterations + 1):
        plan = results[0]["plans"][PLAN_BY_ITERATION[iteration - 1]]
        executed: dict[tuple[int, int], list[int]] = {}
        for result in results:
            for micro, stage_id in result["iterations"][iteration - 1].get("processed", []):
                executed.setdefault((micro, stage_id), []).append(result["rank"])
        planned = {
            (place[0], place[1]): sorted(place[3]) for place in plan["placements"]
        }
        assert len(planned) == MICRO * CONFIG.pp, (label, iteration, planned)
        assert {key: sorted(value) for key, value in executed.items()} == planned, (
            label,
            iteration,
            executed,
            planned,
        )
    # Activations are all retired, and none is held past its own backward.
    for result in results:
        for record in result["iterations"]:
            if not record["trained"]:
                continue
            assert record["activation_drained"], (label, result["rank"], record["iteration"])
            # 1F1B, not GPipe: the peak is one activation per stage still downstream,
            # capped by the micro-batches this rank runs -- never one per micro-batch.
            assert record["activation_peak"] == peak_in_flight(
                len(record["processed"]),
                stage_index=record["stage_index"],
                num_stages=record["num_stages"],
            ), (label, record)


def _assert_failed_ranks_stop_training(results, label):
    """A rank marked failed leaves the training path permanently."""
    for result in results:
        trained = [record["iteration"] for record in result["iterations"] if record["trained"]]
        assert trained == EXPECTED_TRAINED[result["rank"]], (label, result["rank"], trained)
        for record in result["iterations"]:
            if not record["trained"]:
                assert "processed" not in record, (label, result["rank"], record)


def _assert_data_is_neither_repeated_nor_skipped(results, label):
    """Every rank walks the fixed token stream once, and recovery resumes at the cursor."""
    for result in results:
        cursors = [record["cursor"] for record in result["iterations"] if record["trained"]]
        assert cursors == list(range(len(cursors))), (label, result["rank"], cursors)
        # The recovered cursor is the checkpoint's completed-step count: the resumed
        # run reads the next batch, neither replaying nor skipping one.
        for event in result["events"]:
            assert event["completed"] == event["iteration"], (label, result["rank"], event)
            assert event["cursor"] in (None, event["completed"]), (label, result["rank"], event)
    # In any one iteration, every training rank consumes the same batch index.
    for iteration in range(1, CONFIG.iterations + 1):
        seen = {record["cursor"] for record in _iteration(results, iteration) if record["trained"]}
        assert seen == {iteration - 1}, (label, iteration, seen)


def _assert_state_equals_checkpoint_before_resume(results, label):
    """Principle A, before resume: exact equality with the committed anchor."""
    for result in results:
        for event in result["events"]:
            assert event["matches_checkpoint"], (label, result["rank"], event)


def _assert_matches_reference_after_resume(results, label):
    """Principle A, after resume, plus the no-failure reference for iterations 1-2."""
    for result in results:
        for record in result["iterations"]:
            if not record["trained"]:
                continue
            worst = record["worst_grad"]
            print(
                f"{label} rank {result['rank']} iter {record['iteration']}: "
                f"grad={record['max_grad_diff']:.2e} param={record['max_param_diff']:.2e} "
                f"worst={worst['name']} rel={worst['rel']:.2e} "
                f"ref_max={worst['reference_grad_max']:.2e}"
            )
            # The worst element's whole story: which tensor, how far the gradient is
            # off in absolute and relative terms, how large the reference gradient is
            # there, what the parameter did, and AdamW's divisor at that element.
            detail = (
                label,
                f"rank {result['rank']} iter {record['iteration']}",
                worst,
                record["worst_param"],
            )
            assert record["grad_close"], detail
            assert record["step_close"], detail
    # The replicas' losses partition the global batch, so they sum to the reference's.
    for iteration in range(1, CONFIG.iterations + 1):
        by_replica, reference_loss = {}, None
        for record in _iteration(results, iteration):
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


def _assert_no_residual_state(results, label):
    """No process group survives the run, and the model was really pipelined."""
    every = set(results[0]["all_names"])
    for result in results:
        assert result["initialized"] is False, (label, result["rank"])
        for record in result["iterations"]:
            if record["trained"]:
                assert set(record["owned"]) < every, (label, result["rank"], record)


def _assert_end_to_end(results, label):
    _assert_one_plan_per_failure(results, label)
    _assert_planes_reconfigured(results, label)
    _assert_layers_tile_the_model(results, label)
    _assert_micro_batch_stage_executed_once(results, label)
    _assert_failed_ranks_stop_training(results, label)
    _assert_data_is_neither_repeated_nor_skipped(results, label)
    _assert_state_equals_checkpoint_before_resume(results, label)
    _assert_matches_reference_after_resume(results, label)
    _assert_no_residual_state(results, label)


# --- process entry point ----------------------------------------------------------


def _entry(rank, world_size, backend, result_dir, port):
    """One spawned rank: Gloo world, training groups on ``backend``, then the scenario."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    # Kept open for the process's lifetime: faulthandler writes into it from a timer.
    stack_file = Path(result_dir, f"stack_{rank}.txt").open("w")
    faulthandler.dump_traceback_later(STACK_DUMP_AFTER, repeat=True, file=stack_file)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        assert torch.cuda.current_device() == rank
    # The world group is Gloo on both backends -- it is the control plane's
    # always-alive group (plan 3.2); only the training groups switch to NCCL.
    dist.init_process_group(backend="gloo")
    result = _run_end_to_end(rank, world_size, device, backend, Path(result_dir))
    faulthandler.cancel_dump_traceback_later()
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(backend, tmp_path):
    """Spawn the eight ranks and require *all* of them to exit before the timeout."""
    import torch.multiprocessing as mp

    context = mp.spawn(
        _entry,
        args=(WORLD_SIZE, backend, str(tmp_path), _free_port()),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + JOIN_TIMEOUT
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            stacks = "\n".join(
                f"--- rank {peer} ---\n{Path(tmp_path, f'stack_{peer}.txt').read_text()}"
                for peer in range(WORLD_SIZE)
                if Path(tmp_path, f"stack_{peer}.txt").exists()
            )
            pytest.fail(f"end-to-end/{backend}: a rank is blocked and did not exit\n{stacks}")
    return [
        json.loads(Path(tmp_path, f"result_{rank}.json").read_text()) for rank in range(WORLD_SIZE)
    ]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- gates ------------------------------------------------------------------------


def test_end_to_end_three_d_gloo(tmp_path):
    """The eight-process Gloo scenario plan section 四.D specifies."""
    results = _spawn("gloo", tmp_path)
    _assert_end_to_end(results, "end-to-end Gloo")


def test_end_to_end_three_d_cuda_nccl(tmp_path):
    """The same scenario on real GPU tensors and real NCCL training groups."""
    _skip_if_few_gpus(WORLD_SIZE)
    results = _spawn("nccl", tmp_path)
    for result in results:
        assert result["is_cuda"], result["rank"]
    # An idle rank holds no training group; every rank that holds one is on NCCL.
    backends = {result["backend"] for result in results} - {None}
    assert backends == {"nccl"}, [result["backend"] for result in results]
    _assert_end_to_end(results, "end-to-end NCCL")
