"""Single-process deterministic reference training run.

For each iteration this records the exact input tokens, the loss, a parameter
summary, and an optimizer summary. Given the same config, vocab size, and
sequence length, two runs produce byte-identical records -- the fixed anchor
that every later distributed run is compared against (plan section 3.1 and
principle A). AdamW is the only optimizer; loss is next-token cross-entropy.
"""

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch.nn import functional as F

from .config import TrainConfig
from .model import ReferenceTransformer


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

#: AdamW per-parameter state that the optimizer summary covers.
_OPTIM_STATES = ("exp_avg", "exp_avg_sq", "step")


@dataclass(frozen=True)
class StepRecord:
    """One iteration's fixed inputs and post-update summaries."""

    step: int
    tokens: tuple[tuple[int, ...], ...]
    loss: float
    param_digest: str
    optim_digest: str


def _digest(items) -> str:
    """Hash ``(name, tensor)`` pairs by their exact FP32 bytes."""
    hasher = sha256()
    for name, tensor in items:
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(repr(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def logical_state_digest(model) -> str:
    """sha256 over a model's full logical parameters.

    The per-step parameter summary and the safe-point step-8 state digest. Works
    for any model exposing ``logical_state_dict`` -- the single-process reference
    and the tensor-parallel transformer alike -- so every rank that holds the same
    logical state agrees on the digest.
    """
    return _digest(sorted(model.logical_state_dict().items()))


def _optim_digest(optimizer, name_by_param) -> str:
    items = []
    for param, state in optimizer.state.items():
        name = name_by_param[param]
        for key in _OPTIM_STATES:
            value = state[key]
            tensor = value if torch.is_tensor(value) else torch.tensor(value)
            items.append((f"{name}.{key}", tensor))
    return _digest(sorted(items))


def _token_stream(vocab_size, sequence_length, batch_size, count, seed):
    """Fixed, reproducible sequence of token batches (the data order)."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (batch_size, sequence_length), generator=generator)
        for _ in range(count)
    ]


class ReferenceRun:
    """A resumable single-process reference training run.

    Construction fixes the model init, optimizer, and token stream from
    ``config.seed`` exactly as before; :meth:`step` advances one iteration and
    :attr:`cursor` is the number of completed steps (also the data cursor). The
    run object is the unit that :mod:`resihp.checkpoint` saves and restores, so
    that resuming from a checkpoint continues bit-for-bit with an uninterrupted
    run (plan section 3.1).
    """

    def __init__(self, config: TrainConfig, *, vocab_size: int, sequence_length: int):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for next-token loss")

        torch.manual_seed(config.seed)
        self.config = config
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model = ReferenceTransformer(config, vocab_size=vocab_size, sequence_length=sequence_length)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        self.batches = _token_stream(
            vocab_size, sequence_length, config.batch_size, config.iterations, config.seed
        )
        self.name_by_param = {param: name for name, param in self.model.named_parameters()}
        self.cursor = 0

    def step(self) -> StepRecord:
        """Run one iteration on the next fixed batch and record the result."""
        tokens = self.batches[self.cursor]
        logits = self.model(tokens)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        record = StepRecord(
            step=self.cursor,
            tokens=tuple(tuple(int(token) for token in row) for row in tokens.tolist()),
            loss=float(loss.detach()),
            param_digest=logical_state_digest(self.model),
            optim_digest=_optim_digest(self.optimizer, self.name_by_param),
        )
        self.cursor += 1
        return record


def run_reference(
    config: TrainConfig,
    *,
    vocab_size: int,
    sequence_length: int,
) -> list[StepRecord]:
    """Train the reference model deterministically and record every step."""
    run = ReferenceRun(config, vocab_size=vocab_size, sequence_length=sequence_length)
    return [run.step() for _ in range(config.iterations)]
