"""Training entrypoint: fail-stop control loop under torchrun, config echo otherwise.

Launched under ``torchrun`` (``RANK`` is set) it runs the distributed control
loop in :func:`run_distributed`. Run plainly (no ``RANK``) it stays a torch-free
config echo, so the CLI is inspectable without a distributed launcher.
"""

import argparse
import json
import os
from pathlib import Path

from .config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ResiHP training entrypoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    return parser.parse_args(argv)


def run_distributed(loaded):
    """Drive the versioned control loop across the torchrun process group.

    Each scheduled fail-stop triggers exactly one safe point, which produces
    exactly one new plan version and rebuilds the training group over the
    surviving ranks. Returns the final plan.
    """
    from .control import ControlPlane

    config = loaded.train
    failures_by_iteration = {event.after_iteration: event.failed_rank for event in loaded.failures}

    control = ControlPlane.initialize()
    plan = build_initial_plan(config)
    control.build_training_group(plan)
    failed: tuple[int, ...] = ()

    for step in range(config.iterations):
        iteration = step + 1
        control.training_step(plan)  # step 1: complete and commit this iteration
        failed_rank = failures_by_iteration.get(iteration)
        if failed_rank is not None:
            plan, failed = control.safe_point(
                config, plan, failed, failed_rank, next_step=iteration
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
