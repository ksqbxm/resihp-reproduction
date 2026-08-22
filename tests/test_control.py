"""Tests for the fail-stop control plane and nine-step safe point (T9).

The pure ``reconfigure`` test runs anywhere. The eight-process Gloo test runs a real
``TP2 x PP2 x DP2`` layout -- every rank holds the stage its plan gives it and steps
through the plan's own micro-batch assignment -- and needs torch, so it is skipped
without it (the plan forbids installing torch on this machine, so that gate runs on
the target box).
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
VOCAB = 32
SEQLEN = 8
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
    from resihp.recovery import initial_run, stage_of
    from resihp.train import build_initial_plan

    control = ControlPlane.initialize(
        training_backend="gloo", vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_initial_plan(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
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
        ),
        checkpoint_path=Path(result_dir, "ckpt.pt"),
    )

    failed = ()
    versions, digests, trained, layers = [], [], [], []
    for step in range(CONFIG.iterations):
        iteration = step + 1
        if control.training_run is not None:
            control.training_step()
            trained.append(iteration)
        failed_rank = FAILURES.get(iteration)
        if failed_rank is not None:
            plan, failed = control.safe_point(
                CONFIG, plan, failed, failed_rank, next_step=iteration
            )
            versions.append(plan.version)
            digests.append(plan.digest)
            stage = stage_of(plan, rank)
            layers.append(None if stage is None else list(range(*stage.layer_range)))
    control.shutdown()

    Path(result_dir, f"result_{rank}.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "digests": digests,
                "trained": trained,
                "layers": layers,
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

    # Real PP ownership: each rank keeps only its stage's contiguous layers, and a
    # rank the plan drops keeps none at all.
    for rank in (0, 2, 4, 6):
        assert results[rank]["layers"] == [[0, 1], [0, 1]] or results[rank]["layers"] == [
            [2, 3],
            [2, 3],
        ], results[rank]
    assert results[1]["layers"] == [None, None]
    assert results[5]["layers"][1] is None

    # Consistent final topology and no residual process group anywhere.
    for result in results.values():
        assert result["final_failed"] == [1, 5]
        assert result["final_live"] == [0, 2, 3, 4, 6, 7]
        assert result["still_initialized"] is False
