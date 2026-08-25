"""Real fail-stop harness: one process per rank, and ``SIGKILL`` means gone.

Every multi-process gate in this suite runs here, and it is the same machinery the
launcher uses -- :class:`resihp.membership.Supervisor` hosting the run's store, one
child process per plan rank, and no attempt whatsoever to keep the survivors in step
with a child that has died. That is the point: a rank injected with a fail-stop calls
``resihp.train.fail_stop``, which sends itself ``SIGKILL``, so the gate exercises the
same dead process, the same unusable world group, and the same rebuild that the real
run does.

``torch.multiprocessing.spawn`` cannot host these gates: it raises as soon as a child
exits on a signal, which is exactly the situation under test.

A killed rank cannot write anything after it dies, so a gate either asserts over the
survivors' files alone, or has each rank rewrite its file as it goes so that a killed
rank's own account of the iterations it lived through outlives it. Either way the kill
itself shows up in the exit code, as ``-9``.
"""

import json
from pathlib import Path

from resihp.membership import Supervisor


#: Every gate's bug catcher. A killed rank costs one poll interval, so reaching this
#: means a *live* rank is stuck; the harness kills the job and fails the gate rather
#: than hanging the suite.
JOIN_TIMEOUT = 600.0
#: ``Popen.poll``/``Process.exitcode`` for a process that died on ``SIGKILL``.
KILLED = -9


def run_ranks(entry, world_size, *extra, timeout=JOIN_TIMEOUT):
    """Run ``entry(rank, env, *extra)`` on ``world_size`` processes; return exit codes.

    ``env`` carries the store coordinates and this rank's identity; the entry point
    installs it into ``os.environ`` before touching torch, exactly as the launcher's
    workers receive it. The supervisor is the run's process table, so a rank that kills
    itself simply stops being expected at the next boundary.
    """
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    supervisor = Supervisor(range(world_size))
    shared = {**supervisor.env(), "WORLD_SIZE": str(world_size)}
    for rank in range(world_size):
        env = {**shared, "RANK": str(rank), "LOCAL_RANK": str(rank)}
        child = context.Process(target=entry, args=(rank, env, *extra))
        child.start()
        supervisor.register(rank, child)
    return supervisor.serve(timeout=timeout)


def read_results(tmp_path, ranks):
    """The result file each surviving rank wrote, keyed by plan rank."""
    return {
        rank: json.loads(Path(tmp_path, f"result_{rank}.json").read_text()) for rank in ranks
    }


def assert_killed(exit_codes, killed, tmp_path, label=""):
    """The scheduled ranks really died on ``SIGKILL``; every other rank exited cleanly.

    This is the assertion a logically-excluded rank could never satisfy: what happened
    to the process is reported by the operating system, not by the run. ``tmp_path`` is
    where a gate keeps its per-rank files, and is carried here so a failure message can
    point at them.
    """
    expected = {rank: (KILLED if rank in set(killed) else 0) for rank in sorted(exit_codes)}
    assert exit_codes == expected, (label, tmp_path, exit_codes, expected)
