"""Store-backed membership: the one channel a killed rank cannot block.

A real fail-stop kills the process. From the moment it dies, every process group the
dead rank belonged to is unusable -- a collective issued on one blocks forever -- so
neither the detection of the failure nor the agreement on who is left can ride a
process group. Both live outside torch.distributed entirely:

* the **supervisor** is the parent process that spawned the ranks. It owns the run's
  ``TCPStore`` and is the run's only authority on who is still running: a rank is gone
  when the operating system says its process has exited. That is ground truth, so
  nothing here polls a rank for signs of life, times it, or infers anything about a
  process that is still running -- and no live rank can ever be declared dead.
* the **boundary rendezvous** runs once per iteration over that store. Every live rank
  announces its arrival and blocks until the supervisor publishes the membership of
  that boundary. A rank killed at a safe point never announces, so the membership the
  survivors read already excludes it -- which is exactly what keeps any training
  collective from ever being issued with a dead peer.

Ranks are named by their **plan rank**: the identity they are launched with, stable
for the whole run. A rank's torch rank is its index in the current membership and
changes every time the world is re-formed; :class:`~resihp.control.ControlPlane` owns
that translation.
"""

import json
import socket
import time
from datetime import timedelta


#: Store keys. ``arrivals`` is a counter rather than one key per rank: the membership
#: comes from the supervisor's own process table, so all the rendezvous needs to know
#: is that everyone still alive has reached the boundary.
_ARRIVALS = "boundary/{iteration}/arrivals"
_MEMBERS = "boundary/{iteration}/members"
#: Every re-formed world rendezvouses under its own prefix, so a new world can never
#: read a key the world it replaced left behind.
PG_PREFIX = "world/{epoch}/"

#: Environment the supervisor hands every worker.
HOST_ENV = "RESIHP_STORE_HOST"
PORT_ENV = "RESIHP_STORE_PORT"

DEFAULT_TIMEOUT = timedelta(seconds=300)
#: How often the supervisor re-reads the arrival counter and the process table.
POLL_SECONDS = 0.02


def free_port() -> int:
    """A port the supervisor's store can bind right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect(host: str, port: int, timeout: timedelta = DEFAULT_TIMEOUT):
    """A worker's client handle on the supervisor's store."""
    from torch.distributed import TCPStore

    return TCPStore(host, int(port), is_master=False, timeout=timeout)


def boundary(store, iteration: int, *, timeout: timedelta = DEFAULT_TIMEOUT):
    """Announce arrival at ``iteration``'s safe point; return that boundary's membership.

    The returned tuple is the supervisor's verdict, identical on every rank that reads
    it, so the survivors cannot disagree about who was lost. A rank that arrives here
    is by definition still alive, and one that was killed before arriving is already
    missing from the tuple.
    """
    store.add(_ARRIVALS.format(iteration=iteration), 1)
    key = _MEMBERS.format(iteration=iteration)
    store.wait([key], timeout)
    return tuple(json.loads(store.get(key)))


class Supervisor:
    """Parent-side process manager, store server, and liveness authority.

    Owns the ``TCPStore`` every rank talks to, and the run's process table. It is not
    a rank: it holds no process group, runs no collective, and survives any number of
    worker deaths, which is what lets it be the one authority on who is still running.
    What it knows about a rank is exactly what ``wait``-ing on a child process tells
    it -- running, or exited with this status.

    A child is any spawned process object exposing ``pid`` and an exit code, so the
    same supervisor drives the launcher's ``subprocess.Popen`` workers and the tests'
    ``multiprocessing.Process`` workers.
    """

    def __init__(self, ranks, *, host: str = "127.0.0.1", port: int | None = None):
        from torch.distributed import TCPStore

        self.host = host
        self.port = free_port() if port is None else int(port)
        self.ranks = tuple(sorted(ranks))
        self.store = TCPStore(
            self.host,
            self.port,
            is_master=True,
            wait_for_workers=False,
            timeout=DEFAULT_TIMEOUT,
        )
        self._children: dict[int, object] = {}
        self.exit_codes: dict[int, int] = {}

    # --- process table -------------------------------------------------------------

    def env(self) -> dict[str, str]:
        """Store coordinates for a worker's environment."""
        return {HOST_ENV: self.host, PORT_ENV: str(self.port)}

    def register(self, rank: int, child) -> None:
        self._children[int(rank)] = child

    @property
    def alive(self) -> tuple[int, ...]:
        return tuple(rank for rank in self.ranks if rank not in self.exit_codes)

    def _reap(self) -> None:
        """Record every child the operating system says has exited.

        ``Popen.poll`` and ``Process.exitcode`` are the same question asked of the two
        kinds of child; both return ``None`` while the process is running.
        """
        for rank in self.alive:
            child = self._children[rank]
            code = child.poll() if hasattr(child, "poll") else child.exitcode
            if code is not None:
                self.exit_codes[rank] = int(code)

    # --- the rendezvous ------------------------------------------------------------

    def serve(self, *, timeout: float) -> dict[int, int]:
        """Publish one membership per boundary until every worker has exited.

        A boundary is published as soon as everyone still running has arrived: the
        supervisor never waits on a process the operating system has already reaped,
        so a killed rank costs the survivors nothing beyond one poll interval. The
        ``timeout`` is a bug catcher, not part of the failure model -- reaching it
        means a live worker is stuck, so every child is killed and the wait is
        reported rather than left to hang.
        """
        deadline = time.monotonic() + timeout
        iteration = 1
        while True:
            self._reap()
            if not self.alive:
                return dict(self.exit_codes)
            arrived = self.store.add(_ARRIVALS.format(iteration=iteration), 0)
            if arrived >= len(self.alive):
                self.store.set(
                    _MEMBERS.format(iteration=iteration), json.dumps(list(self.alive))
                )
                iteration += 1
                continue
            if time.monotonic() > deadline:
                self.kill_all()
                raise TimeoutError(
                    f"boundary {iteration}: {arrived} of {len(self.alive)} live ranks "
                    f"arrived within {timeout}s (alive={list(self.alive)})"
                )
            time.sleep(POLL_SECONDS)

    def kill_all(self) -> None:
        """Last resort for a stuck run: nothing here is part of the failure model."""
        import os
        import signal

        for rank in self.alive:
            try:
                os.kill(self._children[rank].pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
