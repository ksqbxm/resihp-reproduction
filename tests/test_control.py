"""Tests for the fail-stop control plane and nine-step safe point (T9).

The pure ``reconfigure`` test runs anywhere. The eight-process Gloo test needs
torch and is skipped without it (the plan forbids installing torch on this
machine, so that gate runs on the target box).
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

from resihp.config import TrainConfig
from resihp.control import reconfigure
from resihp.plan import build_plan


CONFIG = TrainConfig(
    model_dim=16,
    num_layers=4,
    num_heads=4,
    batch_size=8,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=6,
)
FAILURES = {2: 1, 4: 5}  # after_iteration -> failed_rank
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def test_reconfigure_is_deterministic_and_increments_version():
    initial = build_plan(CONFIG, step=0, version=0)
    v1 = reconfigure(CONFIG, initial, (1,), version=1, step=2)
    v1_again = reconfigure(CONFIG, initial, (1,), version=1, step=2)
    v2 = reconfigure(CONFIG, v1, (1, 5), version=2, step=4)

    assert v1.version == 1 and v2.version == 2
    assert v1.digest == v1_again.digest  # deterministic across identical inputs
    assert v1.failed_ranks == (1,) and v2.failed_ranks == (1, 5)
    assert v1.live_ranks == (0, 2, 3, 4, 5, 6, 7)
    assert v2.live_ranks == (0, 2, 3, 4, 6, 7)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, result_dir, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    import torch.distributed as dist

    from resihp.control import ControlPlane
    from resihp.train import build_initial_plan

    control = ControlPlane.initialize(training_backend="gloo")
    plan = build_initial_plan(CONFIG)
    control.build_training_group(plan)
    failed = ()
    versions, digests, trained = [], [], []
    for step in range(CONFIG.iterations):
        iteration = step + 1
        if control.training_step(plan) is not None:
            trained.append(iteration)
        failed_rank = FAILURES.get(iteration)
        if failed_rank is not None:
            plan, failed = control.safe_point(
                CONFIG, plan, failed, failed_rank, next_step=iteration
            )
            versions.append(plan.version)
            digests.append(plan.digest)
    control.shutdown()

    Path(result_dir, f"result_{rank}.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "digests": digests,
                "trained": trained,
                "final_failed": list(plan.failed_ranks),
                "final_live": list(plan.live_ranks),
                "still_initialized": dist.is_initialized(),
            }
        )
    )


@requires_torch
def test_control_plane_eight_process_gloo(tmp_path):
    import torch.multiprocessing as mp

    world_size = CONFIG.world_size
    assert world_size == 8
    mp.spawn(
        _worker,
        args=(world_size, str(tmp_path), _free_port()),
        nprocs=world_size,
        join=True,
    )
    results = {
        rank: json.loads(Path(tmp_path, f"result_{rank}.json").read_text())
        for rank in range(world_size)
    }

    # Each fail-stop event yields exactly one new, strictly increasing version.
    for result in results.values():
        assert result["versions"] == [1, 2]

    # Failed ranks permanently leave the training path at their safe point.
    assert results[1]["trained"] == [1, 2]
    assert results[5]["trained"] == [1, 2, 3, 4]
    for rank in (0, 2, 3, 4, 6, 7):
        assert results[rank]["trained"] == [1, 2, 3, 4, 5, 6]

    # Every rank agrees on each event's plan digest.
    for event in range(2):
        assert len({results[rank]["digests"][event] for rank in results}) == 1

    # Consistent final topology and no residual process group anywhere.
    for result in results.values():
        assert result["final_failed"] == [1, 5]
        assert result["final_live"] == [0, 2, 3, 4, 6, 7]
        assert result["still_initialized"] is False
