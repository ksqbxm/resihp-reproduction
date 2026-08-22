"""Analytical memory budget used by all planning decisions.

The single calculator: the TP ``k_min`` search (:mod:`resihp.planner.tp`) and the DP
``MemoryFeasible`` gate (:mod:`resihp.planner.dp`) both call :func:`memory_feasible`,
and there is no second formula anywhere. ``in_flight_micro_batches`` has no default on
purpose -- a silent ``1`` would under-count the activation peak of every stage the 1F1B
schedule warms up; callers derive it from :func:`resihp.planner.pp.peak_in_flight`.
"""

from dataclasses import dataclass

from .config import TrainConfig


_BYTES_PER_FP32 = 4


@dataclass(frozen=True)
class MemoryBreakdown:
    sharded_parameters: int
    boundary_parameters: int
    gradients: int
    adam_exp_avg: int
    adam_exp_avg_sq: int
    activation: int

    @property
    def total(self) -> int:
        return (
            self.sharded_parameters
            + self.boundary_parameters
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
    in_flight_micro_batches: int,
) -> MemoryBreakdown:
    """Return the resident FP32 bytes for one stage and TP rank.

    Every Transformer weight is sharded over TP. Per section 3.3 the embedding
    and LM head use the same fixed TP shard layout as ordinary layers, so their
    ``boundary_parameters`` term is divided by ``tp_degree`` just like the rest.
    The activation estimate follows section 3.5: only micro-batches still
    awaiting backward count toward the peak. TP-sharded attention/MLP
    intermediates are divided by ``tp_degree``; the layer input/output after
    TP all-reduce is complete on every rank and is therefore not divided. That
    boundary activation is counted for a single layer rather than expanded over
    ``stage_layers``, so the activation figure is a lower-bound estimate.
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
    # Section 3.3: embedding/LM head share one fixed TP shard layout, so this
    # boundary term is divided by tp_degree exactly like the per-layer weights.
    boundary_parameters = 2 * vocab_size * config.model_dim * _BYTES_PER_FP32 // tp_degree
    gradients = sharded_parameters + boundary_parameters
    adam_exp_avg = gradients
    adam_exp_avg_sq = gradients

    # Section 3.5 requires retaining activations until their matching backward.
    # Only in-flight micro-batches contribute. Internal attention/MLP tensors
    # are TP-sharded, while the all-reduced layer boundary is complete per rank.
    # This counts one layer's boundary activation, not stage_layers of them, so
    # the result is a deliberate lower bound on the true activation peak.
    tokens = in_flight_micro_batches * config.micro_batch_size * sequence_length
    sharded_activation = tokens * config.model_dim * _BYTES_PER_FP32 // tp_degree
    replicated_boundary_activation = tokens * config.model_dim * _BYTES_PER_FP32
    activation = sharded_activation + replicated_boundary_activation
    return MemoryBreakdown(
        sharded_parameters=sharded_parameters,
        boundary_parameters=boundary_parameters,
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
    in_flight_micro_batches: int,
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
