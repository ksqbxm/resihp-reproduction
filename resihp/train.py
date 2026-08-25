"""One rank's training process: the fail-stop control loop, or a config echo.

This module is the **worker**. :mod:`resihp.launch` is the run's entrypoint and spawns
one of these per rank; run plainly (no ``RANK`` in the environment) it stays a
torch-free config echo, so the CLI and the configuration files are inspectable without
a distributed launcher.

**The fail-stop is a real kill.** At the safe point the deterministic schedule names,
the scheduled rank sends itself ``SIGKILL``: the process dies where it stands, with no
cleanup, no farewell collective, and no further participation in anything. It happens
after the iteration is complete and its checkpoint is on disk, and before the rank
announces itself at the next boundary -- so the last checkpoint holds the dead rank's
shards, and the survivors learn it is gone before they issue another collective. The
supervisor reaps the process and publishes the new membership; the survivors dissolve
the world the dead rank was in, re-form one over themselves, replan, rebuild their
training groups, recover the lost shards, and carry on.

``VOCAB_SIZE`` / ``SEQUENCE_LENGTH`` are run constants rather than config fields: the
training schema (plan section 1) admits only the fields it lists, and these two fix the
token stream and the embedding shapes for the whole run. Both reach the planner,
because both are topology constraints: no TP degree that fails to divide the vocabulary
can be built, and the sequence length sizes the activation budget.

``memory_budget_bytes`` *is* a config field, and the only way to switch the analytical
memory gate on for a launched run. Left ``null``, no artificial ceiling applies.
"""

import argparse
import json
import os
from pathlib import Path

from .config import load_config


#: Token stream shape for a launched run; ``VOCAB_SIZE`` must divide by every TP degree.
VOCAB_SIZE = 256
SEQUENCE_LENGTH = 16
#: The single canonical checkpoint every rank shares (plan 3.6: exactly one file).
CHECKPOINT_PATH = Path("checkpoint.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ResiHP training worker")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    return parser.parse_args(argv)


def _select_device():
    """Bind this process to its own GPU under NCCL; CPU runs need nothing."""
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return torch.device("cuda", torch.cuda.current_device())


def fail_stop(schedule, rank: int, iteration: int) -> None:
    """Kill this process if the schedule fails ``rank`` after ``iteration``.

    ``SIGKILL`` and not an exception, an exit, or a flag: a fail-stop the process can
    observe is not a fail-stop. Nothing runs afterwards -- no ``atexit`` hook, no
    process-group teardown, no buffered write -- which is exactly the state the
    survivors have to recover from, and the reason the checkpoint has to be committed
    before this point rather than after it.
    """
    import signal

    if schedule.get(iteration) == rank:
        os.kill(os.getpid(), signal.SIGKILL)


def run_distributed(loaded):
    """Drive the versioned control loop for this rank until the run ends.

    Each iteration completes, commits its checkpoint, and then rendezvouses at the
    boundary. A boundary that comes back short of a rank is a fail-stop: it triggers
    exactly one safe point, which produces exactly one new plan version, re-forms the
    world and the training groups over the survivors, and recovers this rank's stage
    along the single recovery path. Returns the final plan, or ``None`` when a
    consistent stop ended the run: every surviving rank then keeps the pre-failure
    checkpoint, prints the one structured root cause, and exits normally (plan 3.6).

    A rank that reaches the end prints one ``{"acceptance": ...}`` line: the iterations
    it actually executed, every plan version and digest it acted on, the ranks that
    ended up failed, and the membership of the world it finished in. That is what makes
    the launched command checkable from outside the job (T18): the exit code alone
    cannot show that the scheduled kills happened and that training carried on past
    them.
    """
    from .control import ConsistentStop, ControlPlane
    from .recovery import initial_run

    config = loaded.train
    schedule = {event.after_iteration: event.failed_rank for event in loaded.failures}

    device = _select_device()
    control = ControlPlane.initialize(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        memory_budget=config.memory_budget_bytes,
    )
    plan = build_initial_plan(
        config,
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        memory_budget=config.memory_budget_bytes,
    )
    control.build_training_groups(plan)
    control.attach_run(
        initial_run(
            plan,
            rank=control.rank,
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            tp_group=control.tp_group,
            executor_group=control.executor_group,
            boundary_groups=control.boundary_groups,
            device=device,
        ),
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )
    failed: tuple[int, ...] = ()
    trained: list[int] = []
    versions = [plan.version]
    digests = [plan.digest]

    try:
        for step in range(config.iterations):
            iteration = step + 1
            # Holding a run *is* being on the training path: a rank the current plan
            # places nowhere holds none and executes nothing (plan 3.2, step 4).
            if control.training_run is not None:
                trained.append(iteration)
            control.training_step()  # step 1: complete this iteration
            control.commit_checkpoint(plan)  # step 2: atomic checkpoint, while all alive
            fail_stop(schedule, control.rank, iteration)  # the scheduled process dies here
            lost = control.observe()  # steps 3-4: notice the loss and mark it failed
            if lost:
                plan, failed = control.safe_point(
                    config, plan, failed, lost, next_step=iteration
                )
                versions.append(plan.version)
                digests.append(plan.digest)
    except ConsistentStop as stop:
        print(json.dumps({"stopped": stop.reason.code, "reason": stop.reason.message}, ensure_ascii=False))
        control.shutdown()
        return None

    print(
        json.dumps(
            {
                "acceptance": {
                    "rank": control.rank,
                    "device": device.type,
                    "training_backend": control.training_backend,
                    "trained_iterations": trained,
                    "plan_versions": versions,
                    "plan_digests": digests,
                    "failed_ranks": list(failed),
                    "world_members": list(control.members),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,  # one small write per rank, so the merged stdout stays parseable
    )
    control.shutdown()
    return plan


def build_initial_plan(config, *, vocab_size, sequence_length, memory_budget=None):
    """The pristine plan a run starts from, gated by the inputs every replan uses.

    Passing the run constants and the budget here rather than defaulting them is what
    makes the launched run's first plan subject to the same feasibility rules as the
    ones a fail-stop produces: one planner, one set of constraints, no privileged
    starting topology.
    """
    from .plan import build_plan

    return build_plan(
        config,
        step=0,
        version=0,
        memory_budget=memory_budget,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
    )


def main(argv=None):
    args = parse_args(argv)
    if "RANK" in os.environ:
        run_distributed(load_config(args.config, args.failures))
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    print(json.dumps({"config": config, "failures": failures}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
