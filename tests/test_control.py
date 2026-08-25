"""Tests for the fail-stop control plane and the safe point (T9).

The pure replan test runs anywhere. The eight-process Gloo test runs a real
``TP2 x PP2 x DP2`` layout -- every rank holds the stage its plan gives it and steps
through the plan's own micro-batch assignment -- and needs torch, so it is skipped
without it (the plan forbids installing torch on this machine, so that gate runs on
the target box).

**The fail-stop is a real kill.** Ranks 1 and 5 send themselves ``SIGKILL`` at their
scheduled safe points and are gone: no result file, an exit code of ``-9``, and a world
group the survivors can no longer use. What the gate locks is that the survivors notice
without being told by the dead rank, dissolve that world, re-form one over themselves,
and carry the run to the end -- which is the whole difference between killing a process
and merely excluding it from a list.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from harness import assert_killed, read_results, run_ranks
from resihp.config import TrainConfig
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
SURVIVORS = (0, 2, 3, 4, 6, 7)
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def test_replanning_is_deterministic_and_increments_version():
    initial = build_plan(CONFIG, step=0, version=0)
    v1 = build_plan(CONFIG, step=2, version=1, failed_ranks=(1,), previous=initial)
    v1_again = build_plan(CONFIG, step=2, version=1, failed_ranks=(1,), previous=initial)
    v2 = build_plan(CONFIG, step=4, version=2, failed_ranks=(1, 5), previous=v1)

    assert v1.version == 1 and v2.version == 2
    assert v1.digest == v1_again.digest  # deterministic across identical inputs
    assert v1.failed_ranks == (1,) and v2.failed_ranks == (1, 5)
    assert v1.live_ranks == (0, 2, 3, 4, 5, 6, 7)
    assert v2.live_ranks == (0, 2, 3, 4, 6, 7)


def _worker(rank, env, result_dir):
    os.environ.update(env)

    import torch.distributed as dist

    from resihp.control import ControlPlane
    from resihp.recovery import initial_run, stage_of
    from resihp.train import fail_stop

    control = ControlPlane.initialize(
        training_backend="gloo", vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(
        CONFIG, step=0, version=0, vocab_size=VOCAB, sequence_length=SEQLEN
    )
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
    versions, digests, trained, layers, epochs = [], [], [], [], []
    for step in range(CONFIG.iterations):
        iteration = step + 1
        if control.training_run is not None:
            control.training_step()
            trained.append(iteration)
        control.commit_checkpoint(plan)
        fail_stop(FAILURES, rank, iteration)  # the scheduled rank dies here, for real
        lost = control.observe()
        if lost:
            plan, failed = control.safe_point(
                CONFIG, plan, failed, lost, next_step=iteration
            )
            versions.append(plan.version)
            digests.append(plan.digest)
            epochs.append(control.epoch)
            stage = stage_of(plan, rank)
            layers.append(None if stage is None else list(range(*stage.layer_range)))
    control.shutdown()

    Path(result_dir, f"result_{rank}.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "digests": digests,
                "epochs": epochs,
                "trained": trained,
                "layers": layers,
                "members": list(control.members),
                "final_failed": list(plan.failed_ranks),
                "final_live": list(plan.live_ranks),
                "still_initialized": dist.is_initialized(),
            }
        )
    )


@requires_torch
def test_control_plane_survives_two_real_kills(tmp_path):
    exit_codes = run_ranks(_worker, CONFIG.world_size, str(tmp_path))

    # The failed ranks are dead processes, not excluded list entries: the operating
    # system reports the kill, and nothing of theirs was written after it.
    assert_killed(exit_codes, killed=(1, 5), tmp_path=tmp_path, label="control")
    for rank in (1, 5):
        assert not Path(tmp_path, f"result_{rank}.json").exists(), rank
    survivors = read_results(tmp_path, SURVIVORS)

    # Each fail-stop event yields exactly one new, strictly increasing version, and one
    # newly formed world -- the old one contained a killed process and is unusable.
    for result in survivors.values():
        assert result["versions"] == [1, 2]
        assert result["epochs"] == [1, 2]

    # The survivors trained through both kills, right to the end.
    for rank in SURVIVORS:
        assert survivors[rank]["trained"] == [1, 2, 3, 4, 5, 6]

    # Every rank agrees on each event's plan digest, and on the membership it ended in.
    for event in range(2):
        assert len({survivors[rank]["digests"][event] for rank in SURVIVORS}) == 1
    for result in survivors.values():
        assert result["members"] == list(SURVIVORS)

    # Real PP ownership: each rank keeps only its stage's contiguous layers.
    for rank in (0, 2, 4, 6):
        assert survivors[rank]["layers"] == [[0, 1], [0, 1]] or survivors[rank][
            "layers"
        ] == [[2, 3], [2, 3]], survivors[rank]

    # Consistent final topology and no residual process group anywhere.
    for result in survivors.values():
        assert result["final_failed"] == [1, 5]
        assert result["final_live"] == [0, 2, 3, 4, 6, 7]
        assert result["still_initialized"] is False
