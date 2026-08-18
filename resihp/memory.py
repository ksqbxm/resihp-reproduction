"""Analytical memory budget used by all planning decisions."""

from dataclasses import dataclass

from .config import TrainConfig


_BYTES_PER_FP32 = 4


@dataclass(frozen=True)
class MemoryBreakdown:
    sharded_parameters: int
    replicated_parameters: int
    gradients: int
    adam_exp_avg: int
    adam_exp_avg_sq: int
    activation: int

    @property
    def total(self) -> int:
        return (
            self.sharded_parameters
            + self.replicated_parameters
            + self.gradients
            + self.adam_exp_avg
            + self.adam_exp_avg_sq
            + self.activation
        )


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def estimate_memory(
    config: TrainConfig,
    *,
    tp_degree: int,
    stage_layers: int,
    micro_batches: int,
    sequence_length: int,
    vocab_size: int,
    in_flight_micro_batches: int = 1,
) -> MemoryBreakdown:
    """Return the resident FP32 bytes for one stage and TP rank.

    Transformer weights are sharded over TP. Section 3.3 calls embedding and
    LM-head layouts the unique fixed shards, so those boundary weights are
    also divided by ``tp_degree`` rather than copied in full on every rank.
    The activation estimate follows section 3.5: only micro-batches still
    awaiting backward count toward the peak. TP-sharded attention/MLP
    intermediates are divided by ``tp_degree``; the layer input/output after
    TP all-reduce is complete on every rank and is therefore not divided.
    """
    for name, value in (
        ("tp_degree", tp_degree),
        ("stage_layers", stage_layers),
        ("micro_batches", micro_batches),
        ("sequence_length", sequence_length),
        ("vocab_size", vocab_size),
        ("in_flight_micro_batches", in_flight_micro_batches),
    ):
        _positive(name, value)
    if in_flight_micro_batches > micro_batches:
        raise ValueError("in_flight_micro_batches cannot exceed micro_batches")
    if config.model_dim % tp_degree or config.num_heads % tp_degree:
        raise ValueError("tp_degree must divide model_dim and num_heads")

    # Per Transformer layer: Q/K/V/O and two MLP matrices, plus two LayerNorms.
    layer_parameters = 12 * config.model_dim**2 + 4 * config.model_dim
    sharded_parameters = stage_layers * layer_parameters * _BYTES_PER_FP32 // tp_degree
    # Section 3.3 requires embedding/LM head to use one fixed TP shard layout.
    # They are therefore sharded by tp_degree, not replicated full-size copies.
    replicated_parameters = 2 * vocab_size * config.model_dim * _BYTES_PER_FP32 // tp_degree
    gradients = sharded_parameters + replicated_parameters
    adam_exp_avg = gradients
    adam_exp_avg_sq = gradients

    # Section 3.5 requires retaining activations until their matching backward.
    # Only in-flight micro-batches contribute. Internal attention/MLP tensors
    # are TP-sharded, while the all-reduced layer boundary is complete per rank.
    tokens = in_flight_micro_batches * config.micro_batch_size * sequence_length
    sharded_activation = tokens * config.model_dim * _BYTES_PER_FP32 // tp_degree
    replicated_boundary_activation = tokens * config.model_dim * _BYTES_PER_FP32
    activation = sharded_activation + replicated_boundary_activation
    return MemoryBreakdown(
        sharded_parameters=sharded_parameters,
        replicated_parameters=replicated_parameters,
        gradients=gradients,
        adam_exp_avg=adam_exp_avg,
        adam_exp_avg_sq=adam_exp_avg_sq,
        activation=activation,
    )


def memory_feasible(
    config: TrainConfig,
    *,
    tp_degree: int,
    stage_layers: int,
    micro_batches: int,
    sequence_length: int,
    vocab_size: int,
    memory_budget: int,
    in_flight_micro_batches: int = 1,
) -> bool:
    _positive("memory_budget", memory_budget)
    return estimate_memory(
        config,
        tp_degree=tp_degree,
        stage_layers=stage_layers,
        micro_batches=micro_batches,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        in_flight_micro_batches=in_flight_micro_batches,
    ).total <= memory_budget
