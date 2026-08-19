"""Atomic training checkpoint: save and restore a :class:`ReferenceRun`.

A checkpoint captures the full logical model parameters, the AdamW state
(``exp_avg``/``exp_avg_sq``/``step``), the completed-step count (which is also
the data cursor), the CPU/CUDA RNG state, and the execution-plan version -- the
exact set plan section 3.1 requires so that a resumed run continues bit-for-bit
with an uninterrupted one.

Writes are atomic (plan section 3.6): the payload is written to a temporary
file, fully re-read and verified against the in-memory state, and only then
``os.replace``-d over the single canonical path. Any failure removes the temp
file and leaves the last valid checkpoint untouched; there is exactly one file.
"""

from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from .reference import ReferenceRun, _OPTIM_STATES


#: Checkpoint payload layout version (distinct from the execution-plan version).
_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing state, mismatched, or inconsistent."""


def _identity(run: ReferenceRun) -> dict:
    """Fingerprint of what fixes the saved logical state and the token stream.

    Only fields that determine parameter shapes (``model_dim``/``num_layers``/
    ``num_heads``/``vocab_size``/``sequence_length``) or the fixed token stream
    (``seed``/``batch_size`` and the two shape fields) belong here. The 3D
    topology (``tp``/``pp``/``dp``/``micro_batch_size``) is deliberately absent:
    a checkpoint is a topology-free logical anchor that recovery restores and
    then re-shards under a new plan (plan principle A), so topology is tracked
    by ``plan_version``, never enforced as config equality on load.
    """
    config = run.config
    return {
        "model_dim": config.model_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "vocab_size": run.vocab_size,
        "sequence_length": run.sequence_length,
    }


def _step_int(value) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


def _clone(value):
    return value.detach().clone() if torch.is_tensor(value) else value


def _build_payload(run: ReferenceRun, plan_version: int) -> dict:
    params = {name: param.detach().clone() for name, param in run.model.logical_state_dict().items()}
    optim = {
        run.name_by_param[param]: {key: _clone(state[key]) for key in _OPTIM_STATES}
        for param, state in run.optimizer.state.items()
    }
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "format_version": _FORMAT_VERSION,
        "completed_steps": run.cursor,
        "plan_version": plan_version,
        "identity": _identity(run),
        "params": params,
        "optim": optim,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_rng,
    }


def _payload_digest(payload: dict) -> str:
    """Hash the scalar fields and every tensor's exact bytes (for write verify)."""
    hasher = sha256()
    scalars = {key: payload[key] for key in ("format_version", "completed_steps", "plan_version", "identity")}
    hasher.update(json.dumps(scalars, sort_keys=True).encode("utf-8"))

    def add(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(repr(array.dtype).encode("utf-8"))
        hasher.update(repr(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())

    for name in sorted(payload["params"]):
        hasher.update(name.encode("utf-8"))
        add(payload["params"][name])
    for name in sorted(payload["optim"]):
        for key in _OPTIM_STATES:
            value = payload["optim"][name][key]
            hasher.update(f"{name}:{key}".encode("utf-8"))
            add(value if torch.is_tensor(value) else torch.tensor(value))
    add(payload["cpu_rng_state"])
    cuda = payload["cuda_rng_state"]
    hasher.update(b"cuda:none" if cuda is None else b"cuda")
    for tensor in cuda or ():
        add(tensor)
    return hasher.hexdigest()


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch too old to know weights_only
        return torch.load(path)


def save_checkpoint(path: str | Path, run: ReferenceRun, *, plan_version: int = 0) -> Path:
    """Atomically write ``run``'s state to ``path``, keeping only the latest."""
    if type(plan_version) is not int or plan_version < 0:
        raise CheckpointError("plan_version 必须是非负整数")

    payload = _build_payload(run, plan_version)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        if _payload_digest(_torch_load(tmp)) != _payload_digest(payload):
            raise CheckpointError("checkpoint 写入校验失败：临时文件与内存状态不一致")
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _validate(payload: dict, run: ReferenceRun) -> None:
    if payload.get("format_version") != _FORMAT_VERSION:
        raise CheckpointError(f"checkpoint 格式版本不支持: {payload.get('format_version')}")

    identity = _identity(run)
    if payload["identity"] != identity:
        raise CheckpointError(f"checkpoint 配置与当前 run 不一致: {payload['identity']} != {identity}")

    model_state = run.model.logical_state_dict()
    saved = payload["params"]
    missing = set(model_state) - set(saved)
    if missing:
        raise CheckpointError(f"checkpoint 缺少参数: {sorted(missing)}")
    extra = set(saved) - set(model_state)
    if extra:
        raise CheckpointError(f"checkpoint 含有未知参数: {sorted(extra)}")
    for name, param in model_state.items():
        if tuple(saved[name].shape) != tuple(param.shape):
            raise CheckpointError(
                f"checkpoint 参数形状不匹配 {name}: {tuple(saved[name].shape)} != {tuple(param.shape)}"
            )

    completed = payload["completed_steps"]
    optim = payload["optim"]
    if completed == 0:
        if optim:
            raise CheckpointError("游标不一致：completed_steps 为 0 但优化器状态非空")
        return
    # A stepped checkpoint must carry optimizer state for exactly every
    # parameter; an empty or partial ``optim`` would otherwise slip past the
    # per-entry step check below and restore silently incomplete state.
    if set(optim) != set(saved):
        missing = sorted(set(saved) - set(optim))
        extra = sorted(set(optim) - set(saved))
        raise CheckpointError(f"游标不一致：优化器状态与参数集合不匹配 missing={missing} extra={extra}")
    for name, state in optim.items():
        step = _step_int(state["step"])
        if step != completed:
            raise CheckpointError(f"游标不一致：completed_steps={completed} 但 {name}.step={step}")


def _restore(payload: dict, run: ReferenceRun) -> None:
    param_by_name = {name: param for param, name in run.name_by_param.items()}
    with torch.no_grad():
        for name, param in run.model.logical_state_dict().items():
            param.copy_(payload["params"][name])
    run.optimizer.state.clear()
    for name, state in payload["optim"].items():
        run.optimizer.state[param_by_name[name]] = {key: _clone(state[key]) for key in _OPTIM_STATES}
    torch.set_rng_state(payload["cpu_rng_state"])
    if payload["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    run.cursor = payload["completed_steps"]


def load_checkpoint(path: str | Path, run: ReferenceRun) -> tuple[int, int]:
    """Validate and load a checkpoint into ``run``; return ``(completed_steps, plan_version)``."""
    payload = _torch_load(Path(path))
    _validate(payload, run)
    _restore(payload, run)
    return payload["completed_steps"], payload["plan_version"]
