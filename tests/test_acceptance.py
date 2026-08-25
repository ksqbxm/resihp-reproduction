"""GPU/NCCL acceptance of the launched command, plus the banned-construct scan (T18).

Two gates, deliberately independent:

* :func:`test_no_banned_constructs` -- plan section 六's last completion criterion
  ("代码中不存在 Detector、pᵢ、速度/降速分支、standby、Algorithm 1、旧入口或前向兼容
  逻辑"). It imports no torch, so it runs on every machine, every time.
* :func:`test_launcher_nccl_acceptance` -- plan section 五 step 6: the single
  documented command, launched for real under NCCL on eight GPUs, required to take
  **at least two consecutive fail-stops and keep training**. The fail-stops are real
  kills, so the evidence includes the killed ranks' exit codes: ``-9``, reported by
  the operating system.

The acceptance gate does not re-derive what T16/T17 already lock (per-tensor equality
with the checkpoint, agreement with the new-configuration reference). Those run against
hand-built topologies. This one asks the different question that only the real launcher
can answer: *does the command in the plan document, with the configs shipped in the
repo, actually survive its failure schedule?* So it checks the outside-visible facts --
exit status, which processes died and how, which iterations each surviving rank truly
executed, which plan versions and digests every rank acted on, and the checkpoint left
on disk -- and checks them against the **pure planner's** own output for the same
configs, so a run that silently trained the wrong ranks cannot pass.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from resihp.config import load_config
from resihp.plan import build_plan
from resihp.train import CHECKPOINT_PATH, SEQUENCE_LENGTH, VOCAB_SIZE


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/train.json"
FAILURES_PATH = "configs/failures.json"
#: Section 六: "GPU/NCCL 下 ≥2 次连续 fail-stop 并继续".
REQUIRED_FAILSTOPS = 2
#: A handful of tiny iterations plus one reconfiguration per shipped event; the same
#: bound the other multi-process gates use, so a hang fails the suite instead of
#: hanging it.
RUN_TIMEOUT = 600.0

#: What must not exist in the implementation, and the phrase that would betray it.
#: Word-anchored on purpose: ``degree``/``_step_int``/``tp_index`` are legitimate and
#: must not be dragged in by a loose substring.
BANNED = (
    ("Detector", r"detector"),
    ("pᵢ", r"\bp_i(?![a-z])|p_\{i\}|\bpi\b|pᵢ"),
    ("speed / slowdown branch", r"\bspeed|slowdown|slower|fail.?slow|nvidia.?smi|heartbeat"),
    ("standby", r"standby"),
    ("Algorithm 1", r"algorithm\s*1(?!\d)"),
    ("old entrypoint", r"hello_dist|nccl_test"),
    ("forward-compatibility layer", r"\blegacy|\bdeprecated|backward.?compat|forward.?compat"),
)
#: Files that must not exist anywhere in the repository (plan section 一: 唯一入口).
BANNED_FILES = ("hello_dist.py", "nccl_test.py")


# --- banned-construct scan ---------------------------------------------------------


def _scanned_files():
    """The implementation and its configs -- what "代码" means in the criterion.

    ``tests/`` is out of scope by construction, and that is not a loophole: the only
    occurrences there are ``test_config.py`` parametrizing ``speed``/``p_i``/
    ``detector`` as fields the loader must *reject*, and ``test_parallel_reshard.py``
    naming a TP-degree reduction ``degrade``. Scanning them would force an
    allow-list keyed on file names, which decays the moment a test moves.
    """
    return [
        *sorted(ROOT.joinpath("resihp").rglob("*.py")),
        *sorted(ROOT.joinpath("configs").glob("*.json")),
        *sorted(ROOT.glob("*.py")),
    ]


def test_no_banned_constructs():
    """No detector, pᵢ, speed branch, standby, Algorithm 1, old entrypoint, or shim."""
    hits = []
    for path in _scanned_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in BANNED:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    hits.append(f"{path.relative_to(ROOT)}:{number}: {label}: {line.strip()}")
    assert hits == [], "\n".join(hits)

    present = [name for name in BANNED_FILES if list(ROOT.rglob(name))]
    assert present == [], f"old entrypoint still in the repository: {present}"


# --- expected plan sequence (pure planner, no torch) -------------------------------


def _expected_plans(config, events):
    """Replay the failure schedule through ``build_plan`` alone.

    This is the same call the entrypoint and ``ControlPlane.safe_point`` make, with the
    same run constants and the same (config-supplied) memory budget, so the digests it
    produces are exactly the ones the launched job must publish. Comparing against it
    -- rather than against a table copied into this file -- is what turns "the run
    finished" into "the run executed the plan the pure planner defines".
    """
    plan = build_plan(
        config,
        step=0,
        version=0,
        memory_budget=config.memory_budget_bytes,
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    plans = [plan]
    failed: tuple[int, ...] = ()
    for version, event in enumerate(events, start=1):
        failed = tuple(sorted(set(failed) | {event.failed_rank}))
        plan = build_plan(
            config,
            step=event.after_iteration,
            version=version,
            failed_ranks=failed,
            previous=plan,
            memory_budget=config.memory_budget_bytes,
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        plans.append(plan)
    return plans


def _plan_in_force(plans, events, iteration):
    """The plan iteration ``iteration`` runs under: an event after N applies from N+1."""
    return plans[sum(1 for event in events if event.after_iteration < iteration)]


def _expected_trained(plans, events, config, rank):
    return [
        iteration
        for iteration in range(1, config.iterations + 1)
        if rank in _plan_in_force(plans, events, iteration).active_ranks
    ]


def _dimension_changes(plans):
    """Which of TP, PP and DP actually change across a plan sequence.

    Read from the published plans themselves: the TP membership of each stage, the
    layer range of each stage, and which micro-batches each replica owns.
    """
    def tp(plan):
        return {(s.replica_id, s.stage_id): (s.tp_degree, s.tp_members) for s in plan.stages}

    def pp(plan):
        return {(s.replica_id, s.stage_id): s.layer_range for s in plan.stages}

    def dp(plan):
        owned: dict[int, set[int]] = {}
        for placement in plan.placements:
            owned.setdefault(placement.replica_id, set()).add(placement.micro_batch)
        return {replica: sorted(micro) for replica, micro in sorted(owned.items())}

    return {
        name: any(view(a) != view(b) for a, b in zip(plans, plans[1:]))
        for name, view in (("tp", tp), ("pp", pp), ("dp", dp))
    }


def test_the_shipped_schedule_exercises_tp_pp_and_dp():
    """The default acceptance case must move all three dimensions, not just TP.

    Nothing here needs a GPU: the plan sequence the launched job publishes is a pure
    function of the two shipped config files, so whether the demonstration is real can
    be settled on any machine. Without this, a config whose repartition happens to hand
    every stage back the layers it already had would still "pass" the NCCL gate while
    demonstrating only a TP degree change.
    """
    loaded = load_config(ROOT / CONFIG_PATH, ROOT / FAILURES_PATH)
    plans = _expected_plans(loaded.train, loaded.failures)

    assert _dimension_changes(plans) == {"tp": True, "pp": True, "dp": True}

    # And each is a *real* change, not a relabelling: a stage really drops a TP rank,
    # a layer really changes owning stage, and a micro-batch really changes replica.
    degrees = [sorted(stage.tp_degree for stage in plan.stages) for plan in plans]
    assert min(min(row) for row in degrees) < max(max(row) for row in degrees)

    def owner_of_layer(plan, replica, layer):
        for stage in plan.stages:
            if stage.replica_id == replica and stage.layer_range[0] <= layer < stage.layer_range[1]:
                return stage.stage_id
        return None

    moved = {
        (replica, layer)
        for replica in range(loaded.train.dp)
        for layer in range(loaded.train.num_layers)
        if len({owner_of_layer(plan, replica, layer) for plan in plans} - {None}) > 1
    }
    assert moved, "no global layer ever changes owning stage"

    def replica_of(plan, micro):
        return next(p.replica_id for p in plan.placements if p.micro_batch == micro)

    rerouted = {
        micro
        for micro in range(loaded.train.batch_size // loaded.train.micro_batch_size)
        if len({replica_of(plan, micro) for plan in plans}) > 1
    }
    assert rerouted, "no micro-batch is ever rerouted to another replica"


# --- launching the documented command ----------------------------------------------


def _skip_unless_gpus(count):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


def _launch():
    """Run the plan's acceptance command and return its completed process.

    ``python -m resihp.launch`` is the documented argv, verbatim, run from the
    repository root through ``sys.executable`` -- so the job provably runs in the
    interpreter under test, and the relative config paths are exercised too.
    ``torchrun`` cannot host this run: its agent kills the surviving workers the
    moment one of them dies from a signal, which is the very thing the schedule does
    on purpose (see :mod:`resihp.launch`).

    The launcher gets the shorter deadline of the two, so that a stuck job is cleaned
    up by the supervisor that owns the processes rather than by killing the launcher
    and leaving eight orphans holding GPUs.
    """
    command = [
        sys.executable,
        "-m",
        "resihp.launch",
        "--config",
        CONFIG_PATH,
        "--failures",
        FAILURES_PATH,
        "--timeout",
        str(RUN_TIMEOUT - 60),
    ]
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = process.communicate(timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=60)
        pytest.fail(f"the acceptance run did not finish in {RUN_TIMEOUT}s\n{stdout}\n{stderr}")
    return stdout, stderr, process.returncode


def _launch_record(stdout):
    """The launcher's own summary line: which processes died, and how."""
    decoder = json.JSONDecoder()
    for line in stdout.splitlines():
        start = line.find('{"launch"')
        if start >= 0:
            return decoder.raw_decode(line[start:])[0]["launch"]
    raise AssertionError(f"the launcher printed no summary\n{stdout}")


def _acceptance_records(stdout):
    """One record per surviving rank, parsed out of the merged stdout.

    ``raw_decode`` rather than a line regex: eight ranks write into one pipe, so a
    line may carry another rank's text after the object. Decoding from the opening
    brace takes exactly one value and ignores whatever follows. A killed rank prints
    nothing here -- it never reaches the end of the run.
    """
    records = {}
    decoder = json.JSONDecoder()
    for line in stdout.splitlines():
        start = line.find('{"acceptance"')
        if start < 0:
            continue
        record = decoder.raw_decode(line[start:])[0]["acceptance"]
        assert record["rank"] not in records, f"rank {record['rank']} reported twice"
        records[record["rank"]] = record
    return records


# --- gate ---------------------------------------------------------------------------


def test_launcher_nccl_acceptance():
    """The documented command under NCCL: two real kills, then training carries on.

    Every assertion is on evidence the job itself emitted, cross-checked against the
    pure planner:

    * the launcher exits 0 and no rank printed a consistent stop -- the run reached
      the end rather than ending on a structured root cause;
    * exactly the scheduled ranks died, and died on ``SIGKILL`` (exit ``-9``), while
      every other process exited cleanly: the fail-stop is a dead process, not a name
      struck off a list, and the survivors kept running anyway;
    * every surviving rank ran on a CUDA device with NCCL training groups, so this is
      the real GPU path and not a CPU run that happened to pass;
    * each of them acted on plan versions ``0..N`` with digests identical across ranks
      and equal to the planner's -- one new plan per fail-stop, strictly increasing,
      unanimous, and free of any runtime input -- and finished in a world holding
      exactly the surviving ranks, which is the rebuild having really happened;
    * the iterations each rank *actually executed* equal the active sets of those
      plans, which together with the exit codes is "≥2 次连续 fail-stop 并继续训练";
    * the one canonical checkpoint is on disk with no ``.tmp`` beside it.
    """
    loaded = load_config(ROOT / CONFIG_PATH, ROOT / FAILURES_PATH)
    config, events = loaded.train, loaded.failures
    # Preconditions on the *shipped* schedule: without them "kept training" could be
    # satisfied vacuously by a config whose last event lands on the final iteration.
    assert len(events) >= REQUIRED_FAILSTOPS, f"shipped schedule has only {len(events)} fail-stop"
    assert max(event.after_iteration for event in events) < config.iterations, (
        "the schedule leaves no iteration after the last fail-stop"
    )
    _skip_unless_gpus(config.world_size)

    checkpoint = ROOT / CHECKPOINT_PATH
    # The command writes this file itself; clearing it first is what lets its presence
    # afterwards mean "this run committed it" instead of "some earlier run did".
    checkpoint.unlink(missing_ok=True)
    checkpoint.with_name(checkpoint.name + ".tmp").unlink(missing_ok=True)

    stdout, stderr, code = _launch()
    assert code == 0, f"exit {code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    assert '"stopped"' not in stdout, f"the run ended on a consistent stop\n{stdout}"

    expected_failed = sorted(event.failed_rank for event in events)
    survivors = [rank for rank in range(config.world_size) if rank not in expected_failed]

    # The kills were kills: the operating system reports signal 9 for exactly the
    # scheduled ranks, and a clean exit for every other process.
    launched = _launch_record(stdout)
    assert launched["killed_ranks"] == expected_failed, launched
    assert launched["scheduled_kills"] == expected_failed, launched
    assert {int(rank): code for rank, code in launched["exit_codes"].items()} == {
        rank: (-9 if rank in expected_failed else 0) for rank in range(config.world_size)
    }, launched

    records = _acceptance_records(stdout)
    assert sorted(records) == survivors, f"{sorted(records)}\n{stdout}"

    plans = _expected_plans(config, events)
    for rank, record in sorted(records.items()):
        where = f"rank {rank}"
        assert record["device"] == "cuda", f"{where}: ran on {record['device']}"
        assert record["training_backend"] == "nccl", f"{where}: {record['training_backend']}"
        assert record["plan_versions"] == list(range(len(plans))), f"{where}: {record}"
        assert record["plan_digests"] == [plan.digest for plan in plans], where
        assert record["failed_ranks"] == expected_failed, f"{where}: {record['failed_ranks']}"
        assert record["world_members"] == survivors, f"{where}: {record['world_members']}"
        assert record["trained_iterations"] == _expected_trained(plans, events, config, rank), (
            f"{where}: executed {record['trained_iterations']}"
        )

    assert checkpoint.exists(), "the safe point committed no checkpoint"
    assert not checkpoint.with_name(checkpoint.name + ".tmp").exists(), "atomic write left a .tmp"
