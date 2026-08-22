"""Training entrypoint: fail-stop control loop under torchrun, config echo otherwise.

Launched under ``torchrun`` (``RANK`` is set) it runs the distributed control loop
in :func:`run_distributed`. Run plainly (no ``RANK``) it stays a torch-free config
echo, so the CLI is inspectable without a distributed launcher.

``VOCAB_SIZE`` / ``SEQUENCE_LENGTH`` are run constants rather than config fields:
the training schema (plan section 1) admits only the fields it lists, and these two
fix the token stream and the embedding shapes for the whole run.
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
    parser = argparse.ArgumentParser(description="ResiHP training entrypoint")
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


def run_distributed(loaded):
    """Drive the versioned control loop across the torchrun process group.

    Each scheduled fail-stop triggers exactly one safe point, which produces exactly
    one new plan version, rebuilds the training groups over the surviving ranks, and
    recovers this rank's stage along the single recovery path. Returns the final
    plan, or ``None`` when a consistent stop ended the run: every rank then keeps the
    pre-failure checkpoint, prints the one structured root cause, and exits normally
    (plan 3.6).

    A run that reaches the end prints one ``{"acceptance": ...}`` line per rank: the
    iterations this rank actually executed, every plan version and digest it acted
    on, and the ranks that ended up failed. That is what makes the launched command
    checkable from outside the job (T18): the exit code alone cannot show that the
    scheduled fail-stops happened and that training carried on past them.
    """
    from .control import ConsistentStop, ControlPlane
    from .recovery import initial_run

    config = loaded.train
    failures_by_iteration = {event.after_iteration: event.failed_rank for event in loaded.failures}

    device = _select_device()
    control = ControlPlane.initialize(vocab_size=VOCAB_SIZE, sequence_length=SEQUENCE_LENGTH)
    plan = build_initial_plan(config)
    control.build_training_groups(plan)
    control.attach_run(
        initial_run(
            plan,
            rank=control.rank,
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            tp_group=control.tp_group,
            executor_group=control.executor_group,
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
            control.training_step()  # step 1: complete and commit this iteration
            failed_rank = failures_by_iteration.get(iteration)
            if failed_rank is not None:
                plan, failed = control.safe_point(
                    config, plan, failed, failed_rank, next_step=iteration
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
                }
            },
            ensure_ascii=False,
        ),
        flush=True,  # one small write per rank, so the merged stdout stays parseable
    )
    control.shutdown()
    return plan


def build_initial_plan(config):
    from .plan import build_plan

    return build_plan(config, step=0, version=0)


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
