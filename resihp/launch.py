"""The run's single entrypoint: spawn the ranks, host the store, outlive the dead.

``torchrun`` cannot launch this reproduction. Its elastic agent treats a worker that
dies from a signal as a job failure and tears down or restarts every sibling -- which
is precisely the behaviour a fail-stop reproduction must not have, because ResiHP's
recovery is an *in-place* reconfiguration of the survivors, not a restart of the job.
So the launcher is here instead, and it is deliberately small:

* it hosts the run's ``TCPStore`` -- in the launcher rather than in rank 0, so that no
  rank's death can take the rendezvous down with it;
* it spawns one worker process per rank and leaves the survivors strictly alone when
  one of them dies;
* it is the run's authority on which ranks are still running, by the only mechanism
  that cannot be wrong: the operating system reported the child's exit
  (:class:`resihp.membership.Supervisor`).

Usage::

    python3 -m resihp.launch --config configs/train.json --failures configs/failures.json

The world size is ``tp * pp * dp`` from the training config -- one process per rank,
one GPU per process. The launcher exits non-zero unless every rank the failure schedule
names was really killed and every rank it does not name exited cleanly.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .membership import Supervisor


#: Bug catcher for the whole job. Nothing in the failure model reaches it: a killed
#: rank is reaped within one poll interval, so hitting this means a live worker is stuck.
DEFAULT_TIMEOUT = 900.0
#: What ``Popen.poll`` reports for a process that died on ``SIGKILL`` (signal 9).
KILLED = -9


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ResiHP fail-stop launcher")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser.parse_args(argv)


def launch(config_path: Path, failures_path: Path, *, timeout: float = DEFAULT_TIMEOUT):
    """Run the whole job and return ``(exit codes by rank, ranks the schedule killed)``."""
    loaded = load_config(config_path, failures_path)
    world_size = loaded.train.world_size
    scheduled = sorted(event.failed_rank for event in loaded.failures)

    supervisor = Supervisor(range(world_size))
    for rank in range(world_size):
        environ = {
            **os.environ,
            **supervisor.env(),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
        supervisor.register(
            rank,
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "resihp.train",
                    "--config",
                    str(config_path),
                    "--failures",
                    str(failures_path),
                ],
                env=environ,
            ),
        )
    return supervisor.serve(timeout=timeout), scheduled


def main(argv=None) -> int:
    args = parse_args(argv)
    exit_codes, scheduled = launch(args.config, args.failures, timeout=args.timeout)

    killed = sorted(rank for rank, code in exit_codes.items() if code == KILLED)
    survivors = {rank: code for rank, code in exit_codes.items() if rank not in scheduled}
    ok = killed == scheduled and set(survivors.values()) <= {0}
    print(
        json.dumps(
            {
                "launch": {
                    "world_size": len(exit_codes),
                    "killed_ranks": killed,
                    "scheduled_kills": scheduled,
                    "exit_codes": {str(rank): exit_codes[rank] for rank in sorted(exit_codes)},
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
