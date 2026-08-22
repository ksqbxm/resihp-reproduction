"""The two Principle A contracts, in one place (plan section 2, T8).

Principle A fixes acceptance in two halves, and this module is the only definition of
either. Before this existed each distributed gate carried its own copy, which is how
three subtly different comparisons came to exist for one contract.

**A1 -- before resume** (:func:`assert_equals_checkpoint`). Once reconfiguration is
done and before training continues, this rank's complete logical state -- ``param``,
``exp_avg``, ``exp_avg_sq``, ``step`` -- equals the pre-failure checkpoint, tensor by
tensor, under ``torch.equal`` with matching dtype. No tolerance: recovery moves bytes,
it does not compute, so anything but exact equality is a lost or corrupted tensor.
Gradients are not part of it -- they are not persistent state (see
:mod:`resihp.recovery`), so the checkpoint carries none and nothing is compared.

**A2 -- after resume** (:func:`assert_matches_reference`). The resumed run is compared
against a reference rebuilt from "the same checkpoint + the new topology + the new
configuration's actual batch + the same seed" (:func:`steps_from_anchor`) -- never
against an uninterrupted run from iteration 0, which diverges by construction once the
batch assignment and the reduction structure change.

The numerical contract of A2 is the one the project already validated on real NCCL
hardware, and it is reused verbatim rather than re-derived:

* **gradients** are compared inside the reassociation band (``rtol=1e-4``,
  ``atol=1e-5``) that real TP all-reduce and micro-batch splitting produce. They are
  what the distributed algorithm is responsible for computing, so they are checked
  directly.
* **parameters** carry an extra per-element allowance, because AdamW's update is
  ``lr * m_hat / (sqrt(v_hat) + eps)``: a gradient difference ``d`` reaches the
  parameter scaled by ``lr / denom``, ~1e-10 where the gradient is healthy but rising
  to ``lr`` itself where ``sqrt(v_hat)`` has decayed to ``eps`` and the update
  degenerates into ``lr * sign(g)``. A fixed band there would assert that AdamW is well
  conditioned, not that the run is correct. The allowance is driven by the *measured*
  gradient difference, so it stays tight exactly where the check earns its keep: a
  moment that did not survive recovery, a wrong step count, or a parameter the
  optimizer never touched each move the parameter by order ``lr`` while leaving the
  gradient -- and therefore the allowance -- where it was.

The comparison is split into a measurement (:func:`compare_shards`, returning a plain
JSON-serializable summary) and an assertion (:func:`assert_matches_reference`), because
the distributed gates measure inside spawned ranks and assert in the parent process.
"""

import math

import torch
from torch.nn import functional as F

from .checkpoint import load_anchor
from .model import ReferenceTransformer
from .parallel.reshard import local_slice
from .recovery import stage_layout, stage_of
from .reference import (
    ADAM_BETAS,
    ADAM_EPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _OPTIM_STATES,
    _token_stream,
)


#: Reassociation band: real TP all-reduce plus micro-batch splitting (T10/T12/T13).
RTOL = 1e-4
ATOL = 1e-5


def adamw(params):
    """The project's one optimizer configuration, so a reference cannot drift from a run."""
    return torch.optim.AdamW(
        params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )


def batch(config, index, *, vocab_size, sequence_length, device):
    """Iteration ``index``'s fixed token batch -- the same stream a run consumes.

    ``device`` is required throughout this module: a reference that silently ran
    somewhere other than the run it is compared against is the one mistake these
    helpers exist to prevent.
    """
    stream = _token_stream(
        vocab_size, sequence_length, config.batch_size, config.iterations, config.seed
    )
    return stream[index].to(device)


def denominator(state):
    """AdamW's own ``sqrt(v / bias_correction2) + eps`` for the step it just took."""
    bias_correction2 = 1 - ADAM_BETAS[1] ** float(state["step"])
    return state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2) + ADAM_EPS


# --- building the reference -------------------------------------------------------


def step_record(model, optimizer, tokens, *, vocab_size) -> dict:
    """One full-batch reference iteration: its gradients, new weights, new moments."""
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size), tokens[:, 1:].reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    named = model.logical_state_dict()
    grads = {name: param.grad.detach().clone() for name, param in named.items()}
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grads": grads,
        "params": {name: param.detach().clone() for name, param in named.items()},
        "moments": {
            name: {key: optimizer.state[param][key].detach().clone() for key in _OPTIM_STATES}
            for name, param in named.items()
            if param in optimizer.state
        },
        # Read back from the optimizer that just used it: this is the divisor that
        # turns a gradient difference into a parameter difference (``compare_shards``).
        "denoms": {name: denominator(optimizer.state[param]) for name, param in named.items()},
    }


def reference_steps(config, *, vocab_size, sequence_length, count, device) -> list[dict]:
    """The no-failure reference: ``count`` iterations from the fixed initialization."""
    torch.manual_seed(config.seed)
    model = ReferenceTransformer(
        config, vocab_size=vocab_size, sequence_length=sequence_length
    ).to(device)
    model.train()
    optimizer = adamw(model.parameters())
    return [
        step_record(
            model,
            optimizer,
            batch(config, index, vocab_size=vocab_size, sequence_length=sequence_length, device=device),
            vocab_size=vocab_size,
        )
        for index in range(count)
    ]


def steps_from_anchor(
    config, anchor, *, vocab_size, sequence_length, start, count, device
) -> list[dict]:
    """Principle A2's reference: ``count`` steps from the checkpoint anchor.

    The baseline is *not* an uninterrupted run from iteration 0 but "the same checkpoint
    + the new topology + the new configuration's actual batch + the same seed" -- the
    checkpoint's full logical parameters and AdamW moments, stepped on the batches the
    resumed run consumes from cursor ``start``.
    """
    model = ReferenceTransformer(
        config, vocab_size=vocab_size, sequence_length=sequence_length
    ).to(device)
    model.train()
    named = model.logical_state_dict()
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(anchor[name]["param"].to(device))
    optimizer = adamw(model.parameters())
    optimizer.state.clear()
    for name, fields in anchor.items():
        if "exp_avg" not in fields:
            continue
        optimizer.state[named[name]] = {
            "exp_avg": fields["exp_avg"].to(device).clone(),
            "exp_avg_sq": fields["exp_avg_sq"].to(device).clone(),
            "step": fields["step"].clone(),  # AdamW keeps its step count on the CPU
        }
    return [
        step_record(
            model,
            optimizer,
            batch(
                config,
                start + offset,
                vocab_size=vocab_size,
                sequence_length=sequence_length,
                device=device,
            ),
            vocab_size=vocab_size,
        )
        for offset in range(count)
    ]


# --- A1: recovery equals the checkpoint -------------------------------------------


def checkpoint_mismatches(run, *, plan, rank, checkpoint_path) -> list[str]:
    """Every way this rank's state differs from the checkpoint, re-sharded by the plan.

    Empty means A1 holds. A rank the plan places on no stage must hold no state at all,
    which is its own entry in the list when violated.
    """
    stage = stage_of(plan, rank)
    if stage is None:
        return [] if run is None else [f"rank {rank} is placed nowhere but still holds a run"]
    layout = stage_layout(plan, stage)
    shards = run.stage.local_shards()
    if set(shards) != set(layout):
        extra = sorted(set(shards) - set(layout))
        absent = sorted(set(layout) - set(shards))
        return [f"stage holds the wrong names: extra={extra} missing={absent}"]

    anchor, _completed = load_anchor(checkpoint_path)
    moments = run.runtime.optimizer.state
    index = stage.tp_members.index(rank)
    problems = []
    for name, dim in layout.items():
        param = shards[name][0]
        if not torch.equal(
            param.detach().cpu(), local_slice(anchor[name]["param"], dim, index, stage.tp_degree)
        ):
            problems.append(f"{name}.param")
        held = moments.get(param)
        if held is None:
            problems.append(f"{name}: AdamW state was not installed")
            continue
        for field in ("exp_avg", "exp_avg_sq"):
            if not torch.equal(
                held[field].detach().cpu(),
                local_slice(anchor[name][field], dim, index, stage.tp_degree),
            ):
                problems.append(f"{name}.{field}")
        if not torch.equal(held["step"].detach().cpu(), anchor[name]["step"]):
            problems.append(f"{name}.step")
    return problems


def matches_checkpoint(run, plan, rank, checkpoint_path) -> bool:
    """A1 as a bool, for a worker that must ship its verdict through JSON."""
    return not checkpoint_mismatches(run, plan=plan, rank=rank, checkpoint_path=checkpoint_path)


def assert_equals_checkpoint(run, *, plan, rank, checkpoint_path) -> None:
    """A1: raise unless this rank's recovered state is exactly the checkpoint."""
    problems = checkpoint_mismatches(run, plan=plan, rank=rank, checkpoint_path=checkpoint_path)
    if problems:
        raise AssertionError(
            f"rank {rank} does not equal the checkpoint under plan v{plan.version}: {problems}"
        )


# --- A2: the resumed run matches the new-topology reference -----------------------


def compare_shards(stage, record) -> dict:
    """Measure this rank's shards against a reference record, sliced by its TP layout.

    Returns a JSON-serializable summary; :func:`assert_matches_reference` turns it into
    a verdict. See the module docstring for why gradients and parameters are held to
    different contracts.
    """
    grad_close = step_close = True
    max_grad_diff = max_param_diff = 0.0
    worst_grad = worst_param = None
    for name, (param, dim) in stage.local_shards().items():
        if param.grad is None:
            grad_close = False  # every owned parameter must have taken a gradient
            continue
        sliced = (dim, stage.tp_rank, stage.tp_size)
        want_param = local_slice(record["params"][name], *sliced)
        want_grad = local_slice(record["grads"][name], *sliced)
        denom = local_slice(record["denoms"][name], *sliced)
        grad_diff = (param.grad - want_grad).abs()
        param_diff = (param.detach() - want_param).abs()
        # The factor 2 is what makes this an upper bound rather than a first-order
        # estimate: at ``g ~ 0`` the update is ``lr * sign(g)``, so two runs can differ
        # by the whole ``2 * lr`` while the linear term alone would allow only ``lr``.
        allowed = ATOL + RTOL * want_param.abs() + 2 * LEARNING_RATE * grad_diff / denom

        grad_close &= torch.allclose(param.grad, want_grad, rtol=RTOL, atol=ATOL)
        step_close &= bool((param_diff <= allowed).all())
        max_grad_diff = max(max_grad_diff, grad_diff.max().item())
        max_param_diff = max(max_param_diff, param_diff.max().item())
        element = (name, stage, grad_diff, param_diff, want_grad, denom)
        if worst_grad is None or grad_diff.max().item() > worst_grad["abs"]:
            worst_grad = _worst(*element, int(grad_diff.argmax()))
        excess = (param_diff - allowed).max().item()
        if worst_param is None or excess > worst_param["excess"]:
            index = int((param_diff - allowed).argmax())
            worst_param = dict(_worst(*element, index), excess=excess)
    return {
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "worst_grad": worst_grad,
        "worst_param": worst_param,
        "owned": sorted(stage.logical_state_dict()),
        "tp_size": stage.tp_size,
        "reference_loss": record["loss"],
    }


def _worst(name, stage, grad_diff, param_diff, want_grad, denom, index) -> dict:
    """One element's full numeric story, for a failure message that explains itself."""
    return {
        "name": name,
        "abs": grad_diff.max().item(),
        "grad_diff_here": grad_diff.flatten()[index].item(),
        "param_diff_here": param_diff.flatten()[index].item(),
        "reference_grad_here": want_grad.flatten()[index].abs().item(),
        "reference_grad_max": want_grad.abs().max().item(),
        # How far the gradient is off relative to the tensor's own scale, and AdamW's
        # divisor at this element -- ``eps``-sized means the update is ``lr * sign(g)``.
        "rel": grad_diff.max().item() / max(want_grad.abs().max().item(), ATOL),
        "denom_here": denom.flatten()[index].item(),
        "tp_size": stage.tp_size,
    }


def assert_matches_reference(comparison, *, label="") -> None:
    """A2: raise unless a :func:`compare_shards` summary satisfies the contract."""
    where = f"{label}: " if label else ""
    if not comparison["grad_close"]:
        raise AssertionError(f"{where}gradients left the reassociation band: {comparison}")
    if not comparison["step_close"]:
        raise AssertionError(f"{where}post-AdamW parameters differ beyond AdamW's own conditioning: {comparison}")
