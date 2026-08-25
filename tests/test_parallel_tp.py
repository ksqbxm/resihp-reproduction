"""Real sharded tensor parallelism, matched to the reference within tolerance (T10).

Genuine TP all-reduce reorders FP32 accumulation, so TP2 matches the
single-process reference within ``torch.allclose`` rather than bit-for-bit; TP1
(a no-op group) matches exactly. Two gates run the identical comparison:

* CPU / **Gloo** (``test_tp_matches_reference_within_tolerance``) -- always
  available where torch is installed.
* GPU / **NCCL** (``test_tp_cuda_nccl_matches_reference``) -- real device tensors
  and NCCL collectives; skipped only when CUDA is missing or there are fewer GPUs
  than the TP degree, so on a 2-GPU box it actually runs.

Each rank builds the full reference (identical under a fixed seed), shards its own
init from it, runs one reference step and one TP step on the same batch, and
checks that its local shards -- gradients and post-AdamW-step weights -- match the
corresponding slice of the reference, plus that the gathered logits and loss match.
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resihp.config import TrainConfig


CONFIG = TrainConfig(
    model_dim=16,
    num_layers=4,
    num_heads=4,
    batch_size=4,
    micro_batch_size=2,
    seed=1234,
    tp=2,
    pp=2,
    dp=2,
    iterations=3,
)
VOCAB = 32
SEQLEN = 8
# Genuine TP all-reduce reorders FP32 accumulation, so results match the
# single-process reference within floating-point tolerance, not bit-for-bit. The
# default ``allclose`` atol (1e-8) is far tighter than one 2-way FP32 reduction
# (~1e-6 absolute, and larger through several layers), so use a tolerance that
# reflects the real reassociation error. A genuine bug is orders of magnitude
# larger than this and still fails.
RTOL = 1e-4
ATOL = 1e-5
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _compare(rank, world_size, device):
    """Run one reference step and one TP step on ``device``; return diff summary.

    Backend-agnostic: the caller has already initialized the process group and
    (for CUDA) selected the device. ``device`` moves the reference, the TP shards,
    and the input batch onto the same device, so Gloo/CPU and NCCL/GPU exercise the
    identical numerics and comparison.
    """
    import torch
    from torch.nn import functional as F

    from resihp.model import ReferenceTransformer
    from resihp.reference import adamw
    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage

    def close(a, b):
        return torch.allclose(a, b, rtol=RTOL, atol=ATOL)

    torch.manual_seed(CONFIG.seed)
    reference = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    # A whole-model TP run is the stage that owns every layer and both boundaries.
    layer_ids = range(CONFIG.num_layers)
    tp = TensorParallelStage(
        CONFIG,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        layer_ids=layer_ids,
        is_first=True,
        is_last=True,
        local_state=shard_logical_state(
            reference.logical_state_dict(),
            layout=shard_dims(layer_ids),
            tp_rank=rank,
            tp_size=world_size,
        ),
    )
    reference = reference.to(device).train()
    tp = tp.to(device).train()
    ref_opt = adamw(reference.parameters())
    tp_opt = adamw(tp.parameters())

    generator = torch.Generator().manual_seed(99)
    tokens = torch.randint(0, VOCAB, (CONFIG.batch_size, SEQLEN), generator=generator).to(device)

    def loss_of(logits):
        return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))

    ref_logits = reference(tokens)
    ref_loss = loss_of(ref_logits)
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_grads = {name: param.grad.detach().clone() for name, param in reference.logical_state_dict().items()}
    ref_opt.step()
    ref_updated = {name: param.detach().clone() for name, param in reference.logical_state_dict().items()}

    tp_logits = tp(tokens=tokens)
    tp_loss = loss_of(tp_logits)
    tp_opt.zero_grad()
    tp_loss.backward()
    tp_grads = {name: param.grad.detach().clone() for name, (param, _) in tp.local_shards().items()}
    tp_opt.step()

    def expected(full, cat_dim):
        return full if cat_dim is None else full.chunk(world_size, dim=cat_dim)[rank]

    grad_close = step_close = True
    max_grad_diff = max_param_diff = 0.0
    for name, (param, cat_dim) in tp.local_shards().items():
        want_grad = expected(ref_grads[name], cat_dim)
        want_param = expected(ref_updated[name], cat_dim)
        grad_close &= close(tp_grads[name], want_grad)
        step_close &= close(param.detach(), want_param)
        max_grad_diff = max(max_grad_diff, (tp_grads[name] - want_grad).abs().max().item())
        max_param_diff = max(max_param_diff, (param.detach() - want_param).abs().max().item())

    return {
        "forward_close": close(tp_logits, ref_logits),
        "loss_close": close(tp_loss, ref_loss),
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "tested_is_cuda": bool(tp_logits.is_cuda),
        "max_forward_diff": (tp_logits - ref_logits).abs().max().item(),
        "max_loss_diff": (tp_loss - ref_loss).abs().item(),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
    }


def _gloo_worker(rank, world_size, result_dir, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _compare(rank, world_size, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_worker(rank, world_size, result_dir, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    assert torch.cuda.current_device() == rank
    result = _compare(rank, world_size, device)
    assert result["tested_is_cuda"]
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _read_results(result_dir, degree):
    return [json.loads(Path(result_dir, f"result_{rank}.json").read_text()) for rank in range(degree)]


def _assert_all_close(results, degree, label):
    assert len(results) == degree
    for rank, result in enumerate(results):
        # Surfaced so a failure shows the actual magnitude (reassociation vs bug).
        print(f"{label} rank {rank} diffs: {result}")
    for result in results:
        assert result["forward_close"], result
        assert result["loss_close"], result
        assert result["grad_close"], result
        assert result["step_close"], result


@requires_torch
@pytest.mark.parametrize("degree", [1, 2])
def test_tp_matches_reference_within_tolerance(tmp_path, degree):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_worker, args=(degree, str(tmp_path), _free_port()), nprocs=degree, join=True)
    _assert_all_close(_read_results(tmp_path, degree), degree, f"TP{degree} Gloo")


@requires_torch
@pytest.mark.parametrize("degree", [1, 2])
def test_tp_cuda_nccl_matches_reference(tmp_path, degree):
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < degree:
        pytest.skip(f"needs {degree} GPU(s), found {torch.cuda.device_count()}")

    mp.spawn(_nccl_worker, args=(degree, str(tmp_path), _free_port()), nprocs=degree, join=True)
    results = _read_results(tmp_path, degree)
    for result in results:
        assert result["tested_is_cuda"], result
    _assert_all_close(results, degree, f"TP{degree} NCCL")


@requires_torch
def test_indivisible_degree_is_rejected():
    from resihp.parallel.tp import _require_divisible

    with pytest.raises(ValueError):
        _require_divisible(CONFIG, VOCAB, 3)  # 4 heads / 16 dim / 32 vocab not divisible by 3


def _commit_worker(rank, env, result_dir):
    """Safe-point step 2 over a real (single-rank) plan, on real planned state."""
    import os

    os.environ.update(env)
    from dataclasses import replace

    import torch.distributed as dist

    from resihp.checkpoint import load_checkpoint
    from resihp.control import ControlPlane
    from resihp.plan import build_plan
    from resihp.recovery import initial_run
    from resihp.reference import ReferenceRun

    config = replace(CONFIG, tp=1, pp=1, dp=1)
    control = ControlPlane.initialize(
        training_backend="gloo", vocab_size=VOCAB, sequence_length=SEQLEN
    )
    plan = build_plan(config, step=0, version=0)
    control.build_training_groups(plan)
    path = Path(result_dir) / "ckpt.pt"

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
        checkpoint_path=path,
    )
    control.training_step()
    control.commit_checkpoint(plan)  # gathers the shards into one logical checkpoint
    assert path.exists()

    fresh = ReferenceRun(config, vocab_size=VOCAB, sequence_length=SEQLEN)
    assert load_checkpoint(path, fresh) == (1, plan.version)
    dist.destroy_process_group()


@requires_torch
def test_control_plane_commits_real_state(tmp_path):
    """Step 2 writes the attached planned run, not a second checkpoint path.

    Recovery, the state digest, and the stop conditions that read this file are
    exercised in ``tests/test_recovery.py``; this pins the commit call point only.
    """
    from harness import assert_killed, run_ranks

    assert_killed(run_ranks(_commit_worker, 1, str(tmp_path)), (), tmp_path, "commit")
