"""Tests for the single-process deterministic reference training run."""

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F

from resihp.config import TrainConfig
from resihp.model import ReferenceTransformer
from resihp.reference import run_reference, _token_stream


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


def _expected_logical_names(num_layers):
    names = {
        "token_embedding.weight",
        "position_embedding.weight",
        "final_norm.weight",
        "final_norm.bias",
        "lm_head.weight",
    }
    for gid in range(num_layers):
        names.update(
            {
                f"layers.{gid}.attn_norm.weight",
                f"layers.{gid}.attn_norm.bias",
                f"layers.{gid}.attn.q_proj.weight",
                f"layers.{gid}.attn.k_proj.weight",
                f"layers.{gid}.attn.v_proj.weight",
                f"layers.{gid}.attn.out_proj.weight",
                f"layers.{gid}.mlp_norm.weight",
                f"layers.{gid}.mlp_norm.bias",
                f"layers.{gid}.mlp.fc1.weight",
                f"layers.{gid}.mlp.fc2.weight",
            }
        )
    return names


def test_run_is_repeatable():
    first = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    second = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)

    assert len(first) == CONFIG.iterations
    assert first == second  # tokens, loss, and both digests are byte-identical


def test_data_order_is_fixed():
    records = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)

    for record in records:
        assert len(record.tokens) == CONFIG.batch_size
        for row in record.tokens:
            assert len(row) == SEQLEN
            assert all(0 <= token < VOCAB for token in row)
    # Distinct steps consume distinct batches (fixed, advancing data cursor).
    assert len({record.tokens for record in records}) == CONFIG.iterations


def test_single_step_update_is_repeatable_and_real():
    first = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    second = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)

    # The post-step summaries are reproducible...
    assert first[0].param_digest == second[0].param_digest
    assert first[0].optim_digest == second[0].optim_digest
    # ...and the optimizer actually moved the weights each iteration.
    digests = [record.param_digest for record in first]
    assert len(set(digests)) == len(digests)


def test_loss_is_finite():
    records = run_reference(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    for record in records:
        assert record.loss == record.loss  # not NaN
        assert abs(record.loss) != float("inf")


def test_parameter_names_and_layer_ids_are_stable():
    model = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)

    assert model.layer_ids == (0, 1, 2, 3)
    assert set(model.logical_state_dict()) == _expected_logical_names(CONFIG.num_layers)


def test_logical_state_dict_returns_live_references():
    model = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    params = dict(model.named_parameters())

    for name, tensor in model.logical_state_dict().items():
        assert tensor is params[name]  # references, not copies


def test_layer_parameter_count_matches_memory_model():
    model = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    state = model.logical_state_dict()

    layer0 = sum(
        state[name].numel() for name in state if name.startswith("layers.0.")
    )
    expected = 12 * CONFIG.model_dim**2 + 4 * CONFIG.model_dim
    assert layer0 == expected


def test_short_sequence_is_rejected():
    with pytest.raises(ValueError):
        run_reference(CONFIG, vocab_size=VOCAB, sequence_length=1)


def test_cuda_and_cpu_agree_at_float32():
    """Control for principle A's numeric band: identical math, different device.

    Nothing distributed happens here -- one model, one batch, one backward, run on two
    devices from the same seed. What it measures is therefore only what the device's
    kernels do to identical FP32 arithmetic, which is the floor under every
    distributed comparison in the suite. That makes it the guard on the precision
    ``resihp.model`` pins: TF32 answers a GEMM to ten mantissa bits and puts this at
    ~1e-3 relative, three orders of magnitude above IEEE FP32, and the eight-process
    gates fail with it. The band here is ~8 ulps of the largest gradient.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    def gradients(device):
        torch.manual_seed(CONFIG.seed)
        model = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN).to(device)
        tokens = _token_stream(VOCAB, SEQLEN, CONFIG.batch_size, 1, CONFIG.seed)[0].to(device)
        logits = model(tokens)
        F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1)
        ).backward()
        return {
            name: param.grad.detach().cpu()
            for name, param in model.logical_state_dict().items()
        }

    cpu, cuda = gradients(torch.device("cpu")), gradients(torch.device("cuda"))
    worst = max((cuda[name] - cpu[name]).abs().max().item() for name in cpu)
    scale = max(tensor.abs().max().item() for tensor in cpu.values())
    assert worst <= 1e-6 * scale, (
        f"CUDA and CPU float32 disagree by {worst:.3e} on gradients up to {scale:.3e} "
        f"(matmul precision: {torch.get_float32_matmul_precision()})"
    )
