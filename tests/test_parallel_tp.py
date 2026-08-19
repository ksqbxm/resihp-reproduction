"""Real sharded tensor parallelism, matched to the reference within tolerance (T10).

Genuine TP all-reduce reorders FP32 accumulation, so TP2 matches the
single-process reference within ``torch.allclose`` rather than bit-for-bit; TP1
(a no-op group) matches exactly. The multi-process gate needs torch and is
skipped without it (the plan forbids installing torch here, so it runs on the
target box); ``torchrun``/NCCL acceptance on GPU is T18.

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
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


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
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F

    from resihp.model import ReferenceTransformer
    from resihp.reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY
    from resihp.parallel.tp import TensorParallelTransformer

    def adamw(params):
        return torch.optim.AdamW(
            params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
        )

    dist.init_process_group(backend="gloo")
    torch.manual_seed(CONFIG.seed)
    reference = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    reference.train()
    tp = TensorParallelTransformer(
        CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN, source_state=reference.logical_state_dict()
    )
    tp.train()
    ref_opt = adamw(reference.parameters())
    tp_opt = adamw(tp.parameters())

    generator = torch.Generator().manual_seed(99)
    tokens = torch.randint(0, VOCAB, (CONFIG.batch_size, SEQLEN), generator=generator)

    def loss_of(logits):
        return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))

    ref_logits = reference(tokens)
    ref_loss = loss_of(ref_logits)
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_grads = {name: param.grad.detach().clone() for name, param in reference.logical_state_dict().items()}
    ref_opt.step()
    ref_updated = {name: param.detach().clone() for name, param in reference.logical_state_dict().items()}

    tp_logits = tp(tokens)
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
        grad_close &= torch.allclose(tp_grads[name], want_grad)
        step_close &= torch.allclose(param.detach(), want_param)
        max_grad_diff = max(max_grad_diff, (tp_grads[name] - want_grad).abs().max().item())
        max_param_diff = max(max_param_diff, (param.detach() - want_param).abs().max().item())

    Path(result_dir, f"result_{rank}.json").write_text(
        json.dumps(
            {
                "forward_close": torch.allclose(tp_logits, ref_logits),
                "loss_close": torch.allclose(tp_loss, ref_loss),
                "grad_close": bool(grad_close),
                "step_close": bool(step_close),
                "max_grad_diff": max_grad_diff,
                "max_param_diff": max_param_diff,
            }
        )
    )
    dist.destroy_process_group()


@requires_torch
@pytest.mark.parametrize("degree", [1, 2])
def test_tp_matches_reference_within_tolerance(tmp_path, degree):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(degree, str(tmp_path), _free_port()), nprocs=degree, join=True)
    results = [json.loads(Path(tmp_path, f"result_{rank}.json").read_text()) for rank in range(degree)]

    assert len(results) == degree
    for result in results:
        assert result["forward_close"]
        assert result["loss_close"]
        assert result["grad_close"]
        assert result["step_close"]


@requires_torch
def test_indivisible_degree_is_rejected():
    from resihp.parallel.tp import _require_divisible

    with pytest.raises(ValueError):
        _require_divisible(CONFIG, VOCAB, 3)  # 4 heads / 16 dim / 32 vocab not divisible by 3


@requires_torch
def test_control_plane_commits_real_state(tmp_path):
    from resihp.checkpoint import load_checkpoint
    from resihp.control import ControlPlane
    from resihp.reference import ReferenceRun, logical_state_digest

    run = ReferenceRun(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    run.step()
    control = ControlPlane(0, 1, None, "gloo")

    # With no state attached the call points stay the T9 no-op / empty digest.
    assert control._state_digest() == ""
    control._commit_checkpoint(0)  # no path attached -> no file written

    # Attaching real training state (T10) wires safe-point steps 2 and 8 to it.
    path = tmp_path / "ckpt.pt"
    control.training_run = run
    control.checkpoint_path = path
    control._commit_checkpoint(7)
    assert path.exists()
    assert control._state_digest() == logical_state_digest(run.model)

    fresh = ReferenceRun(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    assert load_checkpoint(path, fresh) == (1, 7)
