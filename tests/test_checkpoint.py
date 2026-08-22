"""Tests for atomic save/restore of a resumable reference run (T8)."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from resihp.checkpoint import (
    CheckpointError,
    _torch_load,
    load_checkpoint,
    save_checkpoint,
)
from resihp.config import TrainConfig
from resihp.reference import ReferenceRun, run_reference


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


def _fresh_run():
    return ReferenceRun(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)


def test_resume_matches_uninterrupted_run(tmp_path):
    reference = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)

    interrupted = _fresh_run()
    interrupted.step()
    interrupted.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, interrupted, plan_version=7)

    resumed = _fresh_run()
    completed, plan_version = load_checkpoint(path, resumed)
    assert (completed, plan_version) == (2, 7)

    # The step taken after resuming is byte-identical to the uninterrupted one.
    assert resumed.step() == reference[2]


def test_save_leaves_single_atomic_file(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    save_checkpoint(path, run)

    assert path.exists()
    assert not (tmp_path / "ckpt.pt.tmp").exists()  # no leftover temp file
    assert list(tmp_path.iterdir()) == [path]  # exactly one, the canonical file


def test_failed_replace_keeps_last_valid_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    first = _fresh_run()
    first.step()
    save_checkpoint(path, first, plan_version=1)

    def boom(src, dst):
        raise OSError("simulated crash during atomic replace")

    monkeypatch.setattr("resihp.checkpoint.os.replace", boom)
    second = _fresh_run()
    second.step()
    second.step()
    with pytest.raises(OSError):
        save_checkpoint(path, second, plan_version=2)

    # The original checkpoint is untouched and no temp file was left behind.
    assert not (tmp_path / "ckpt.pt.tmp").exists()
    completed, plan_version = load_checkpoint(path, _fresh_run())
    assert (completed, plan_version) == (1, 1)


def test_missing_parameter_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    save_checkpoint(path, run)

    payload = _torch_load(path)
    del payload["params"]["lm_head.weight"]
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="缺少参数"):
        load_checkpoint(path, _fresh_run())


def test_shape_mismatch_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    save_checkpoint(path, run)

    payload = _torch_load(path)
    payload["params"]["lm_head.weight"] = payload["params"]["lm_head.weight"][:, :-1].clone()
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="形状不匹配"):
        load_checkpoint(path, _fresh_run())


def test_cursor_inconsistency_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    run.step()
    save_checkpoint(path, run)

    payload = _torch_load(path)
    payload["completed_steps"] = 5  # disagrees with the AdamW step counts (2)
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="游标不一致"):
        load_checkpoint(path, _fresh_run())


def test_config_mismatch_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    save_checkpoint(path, run)

    other = ReferenceRun(
        replace(CONFIG, seed=CONFIG.seed + 1),
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
    )
    with pytest.raises(CheckpointError, match="配置与当前 run 不一致"):
        load_checkpoint(path, other)


def test_identity_ignores_topology_fields(tmp_path):
    # A checkpoint is a topology-free logical anchor: restoring it under a
    # different TP/PP/DP layout (the whole point of recovery) must be allowed.
    path = tmp_path / "ckpt.pt"
    saver = _fresh_run()
    saver.step()
    saver.step()
    save_checkpoint(path, saver, plan_version=3)

    retopo = replace(CONFIG, tp=4, pp=1, dp=1, micro_batch_size=4)
    loader = ReferenceRun(retopo, vocab_size=VOCAB, sequence_length=SEQLEN)
    completed, plan_version = load_checkpoint(path, loader)
    assert (completed, plan_version) == (2, 3)


def test_missing_optimizer_state_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    save_checkpoint(path, run)

    payload = _torch_load(path)
    payload["optim"].pop(next(iter(payload["optim"])))
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="优化器状态与参数集合不匹配"):
        load_checkpoint(path, _fresh_run())


def test_negative_plan_version_is_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    run = _fresh_run()
    run.step()
    with pytest.raises(CheckpointError, match="plan_version"):
        save_checkpoint(path, run, plan_version=-1)
