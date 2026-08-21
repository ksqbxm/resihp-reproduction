"""Pipeline-parallel 1F1B runtime and layer-state migration (T12).

Pure gates (single process) cover the base partition and the placement planner --
including the case a migrations-only view misses: a layer that stays on its stage
but must still reshard because the stage lost a TP rank.

The distributed gates run identical logic on two backends, exactly like T10/T11:
CPU/**Gloo** (always available where torch is installed) and GPU/**NCCL** (real
device tensors and real point-to-point transfers, skipped when there are fewer GPUs
than the case needs), so a 2-GPU box actually runs them.

* ``*_matches_reference`` -- two stages, four micro-batches: each rank owns only its
  slice of the model, activations and gradients cross the stage boundary as real
  transfers, and every stage's gradients and post-AdamW weights match the
  single-process reference for the names it owns. The emitted primitive order is
  asserted to be genuine 1F1B, not all-forwards-then-all-backwards.
* ``*_layer_migration_is_lossless`` -- the layer that both moves stage and changes
  TP degree carries ``param``/``grad``/``exp_avg``/``exp_avg_sq``/``step`` through a
  real collective and lands tensor-for-tensor equal to the checkpoint anchor.
"""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from resihp.config import TrainConfig
from resihp.parallel.pp import (
    LayerPlacement,
    balanced_layers,
    plan_migration,
    reshard_layout,
)


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
    iterations=3,
)
VOCAB = 32
SEQLEN = 8
MICRO = CONFIG.batch_size // CONFIG.micro_batch_size  # 4 micro-batches
# Micro-batching reorders the FP32 loss reduction relative to the reference's single
# full-batch pass, so the same tolerance the TP path fixed in T10 applies here.
RTOL = 1e-4
ATOL = 1e-5

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)

#: The 1F1B primitive order each stage must emit at 2 stages / 4 micro-batches:
#: stage 0 warms up one forward then alternates, stage 1 alternates throughout.
EXPECTED_SCHEDULE = {
    0: ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3", "W"],
    1: ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3", "W"],
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- pure: base partition and placement planning ----------------------------------


@requires_torch
def test_balanced_layers_covers_every_layer_once():
    assert balanced_layers(4, 2) == ((0, 1), (2, 3))
    assert balanced_layers(5, 2) == ((0, 1, 2), (3, 4))  # remainder to earlier stages
    assert balanced_layers(3, 3) == ((0,), (1,), (2,))
    flat = [gid for stage in balanced_layers(7, 3) for gid in stage]
    assert flat == list(range(7))  # contiguous, unique, complete


@requires_torch
def test_balanced_layers_rejects_more_stages_than_layers():
    with pytest.raises(ValueError):
        balanced_layers(2, 3)


@requires_torch
def test_placements_cover_every_layer_with_both_sides():
    """Every global layer gets an old and a new (owner, degree) -- none omitted."""
    plan, placements = plan_migration((2, 2, 2), (2, 2, 2), (2, 1, 1))
    assert [place.layer for place in placements] == list(range(6))
    assert plan.stage_layers == (3, 2, 1)
    assert plan.embedding_owner == 0 and plan.lm_head_owner == 2

    by_layer = {place.layer: place for place in placements}
    assert by_layer[2] == LayerPlacement(2, old_owner=1, new_owner=0, old_degree=2, new_degree=2)
    assert by_layer[2].moved and not by_layer[2].resharded
    # Layer 4 both changes stage and changes degree: moved *and* resharded.
    assert by_layer[4] == LayerPlacement(4, old_owner=2, new_owner=1, old_degree=2, new_degree=1)
    assert by_layer[4].moved and by_layer[4].resharded
    # Layer 5 stays on stage 2 but its degree dropped, so it still reshards.
    assert not by_layer[5].moved and by_layer[5].resharded


@requires_torch
def test_stationary_layers_still_reshard_when_their_stage_loses_a_rank():
    """No layer moves, yet stage 0's layers must be re-chunked from TP2 to TP1.

    ``PPPlan.migrations`` is empty here, so a plan that only tracked moved layers
    would silently leave stage 0's state in the old two-way layout.
    """
    plan, placements = plan_migration((2, 2), (2, 2), (1, 2))
    assert plan.stage_layers == (2, 2)
    assert plan.migrations == ()
    assert not any(place.moved for place in placements)
    assert [place.resharded for place in placements] == [True, True, False, False]


@requires_torch
def test_reshard_layout_covers_moved_and_stationary_resharded_layers():
    _, placements = plan_migration((2, 2, 2), (2, 2, 2), (2, 1, 1))
    layout = reshard_layout(placements, new_owner=1)

    # Layer 4 arrives from stage 2; layer 3 stays on stage 1 but its degree dropped
    # 2 -> 1, so it must be re-chunked too. Selecting only arriving layers would
    # leave layer 3's state in the old two-way layout.
    assert {name.split(".")[1] for name in layout} == {"3", "4"}
    assert layout["layers.4.attn.q_proj.weight"] == 0  # column-parallel
    assert layout["layers.4.attn.out_proj.weight"] == 1  # row-parallel
    assert layout["layers.4.attn_norm.weight"] is None  # replicated
    # Embedding / LM head belong to the first / last stage, never to a layer.
    assert not any(name.startswith(("token_", "position_", "final_", "lm_head")) for name in layout)


@requires_torch
def test_reshard_layout_selects_stage_that_only_lost_a_rank():
    """No layer moves anywhere, yet stage 0's retained layers must still be listed."""
    _, placements = plan_migration((2, 2), (2, 2), (1, 2))
    assert {name.split(".")[1] for name in reshard_layout(placements, new_owner=0)} == {"0", "1"}
    # Stage 1 kept both its layers at the same degree, so it has no state work.
    assert reshard_layout(placements, new_owner=1) == {}


# --- distributed: 1F1B execution against the reference ----------------------------


def _compare_pp(rank, num_stages, device):
    """Run the reference and the pipeline on ``device``; return a per-rank summary."""
    import torch
    from torch.nn import functional as F

    import torch.distributed as dist

    from resihp.model import ReferenceTransformer
    from resihp.parallel.pp import PipelineRuntime, balanced_layers
    from resihp.parallel.reshard import shard_dims, shard_logical_state
    from resihp.parallel.tp import TensorParallelStage
    from resihp.reference import ADAM_BETAS, ADAM_EPS, LEARNING_RATE, WEIGHT_DECAY

    def adamw(params):
        return torch.optim.AdamW(
            params, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
        )

    torch.manual_seed(CONFIG.seed)
    reference = ReferenceTransformer(CONFIG, vocab_size=VOCAB, sequence_length=SEQLEN)
    # Clone the init before the reference's own step mutates it: the stage must start
    # from the same weights the reference starts from.
    source = {name: param.detach().clone() for name, param in reference.logical_state_dict().items()}

    groups = balanced_layers(CONFIG.num_layers, num_stages)
    # One TP rank per stage: every rank builds every solo group, in the same order,
    # because ``new_group`` is collective over the world.
    solo = [dist.new_group([peer]) for peer in range(num_stages)][rank]
    stage = TensorParallelStage(
        CONFIG,
        vocab_size=VOCAB,
        sequence_length=SEQLEN,
        layer_ids=groups[rank],
        is_first=rank == 0,
        is_last=rank == num_stages - 1,
        local_state=shard_logical_state(
            source, layout=shard_dims(groups[rank]), tp_rank=0, tp_size=1
        ),
        group=solo,
    ).to(device)
    stage.train()

    reference = reference.to(device).train()
    ref_opt = adamw(reference.parameters())

    generator = torch.Generator().manual_seed(99)
    tokens = torch.randint(0, VOCAB, (CONFIG.batch_size, SEQLEN), generator=generator).to(device)

    ref_logits = reference(tokens)
    ref_loss = F.cross_entropy(ref_logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_grads = {name: p.grad.detach().clone() for name, p in reference.logical_state_dict().items()}
    ref_opt.step()
    ref_updated = {name: p.detach().clone() for name, p in reference.logical_state_dict().items()}

    runtime = PipelineRuntime(stage, stage_ranks=range(num_stages), num_micro_batches=MICRO)
    loss = runtime.train_step(tokens)

    owned = stage.logical_state_dict()
    grad_close = step_close = True
    max_grad_diff = max_param_diff = 0.0
    for name, param in owned.items():
        grad_close &= torch.allclose(param.grad, ref_grads[name], rtol=RTOL, atol=ATOL)
        step_close &= torch.allclose(param.detach(), ref_updated[name], rtol=RTOL, atol=ATOL)
        max_grad_diff = max(max_grad_diff, (param.grad - ref_grads[name]).abs().max().item())
        max_param_diff = max(max_param_diff, (param.detach() - ref_updated[name]).abs().max().item())

    return {
        "owned": sorted(owned),
        "all_names": sorted(source),
        "schedule": runtime.schedule,
        "loss": loss,
        "reference_loss": float(ref_loss.detach()),
        "grad_close": bool(grad_close),
        "step_close": bool(step_close),
        "max_grad_diff": max_grad_diff,
        "max_param_diff": max_param_diff,
        "is_cuda": bool(next(stage.parameters()).is_cuda),
    }


def _diffs(result):
    """Compact diff view for the failure printout."""
    return {key: result[key] for key in ("loss", "reference_loss", "max_grad_diff", "max_param_diff")}


def _assert_matches_reference(results, num_stages, label):
    for rank, result in enumerate(results):
        # Surfaced so a failure shows the magnitude (reassociation vs real bug).
        print(f"{label} stage {rank}: {_diffs(result)}")
    for result in results:
        assert result["grad_close"], result
        assert result["step_close"], result

    last = results[-1]
    assert abs(last["loss"] - last["reference_loss"]) < 1e-4, last
    for result in results[:-1]:
        assert result["loss"] is None, result  # only the last stage produces the loss

    owned = [set(result["owned"]) for result in results]
    union = set().union(*owned)
    assert union == set(results[0]["all_names"])  # every parameter owned exactly once
    for i, left in enumerate(owned):
        for right in owned[i + 1 :]:
            assert not (left & right), (left, right)

    if num_stages > 1:
        for result in results:
            # Real pipelining: no stage holds the whole model.
            assert set(result["owned"]) < set(result["all_names"]), result
        assert any(n.startswith("token_embedding") for n in results[0]["owned"])
        assert any(n.startswith("lm_head") for n in results[-1]["owned"])
        for rank, result in enumerate(results):
            assert result["schedule"] == EXPECTED_SCHEDULE[rank], result["schedule"]


# --- distributed: layer state migration -------------------------------------------

_MIGRATION_DIM = 4
_MIGRATION_FIELDS = ("param", "grad", "exp_avg", "exp_avg_sq")


def _layer_shape(name, dim):
    suffix = name.split(".", 2)[2]
    if suffix.endswith(("norm.weight", "norm.bias")):
        return (dim,)
    if suffix == "mlp.fc1.weight":
        return (4 * dim, dim)
    if suffix == "mlp.fc2.weight":
        return (dim, 4 * dim)
    return (dim, dim)


def _anchor_state(layout, seed):
    """The layer's full logical state, as the pre-failure checkpoint holds it (CPU)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    anchor = {}
    for name in layout:
        base = torch.rand(_layer_shape(name, _MIGRATION_DIM), generator=generator)
        anchor[name] = {
            "param": base,
            "grad": base * 2.0,
            "exp_avg": base * 0.5,
            "exp_avg_sq": base.abs() + 1.0,
            "step": torch.tensor(5.0),
        }
    return anchor


def _shard_of(full, dim, index, size):
    import torch

    if dim is None:
        return full.clone()
    return torch.chunk(full, size, dim=dim)[index].contiguous().clone()


def _run_migration(rank, device):
    """Stage 1's whole state workload at TP2 -> TP1, over a real collective.

    Under ``(2,2,2)/TP(2,2,2) -> TP(2,1,1)`` stage 1 must reacquire two layers at
    once: layer 4 arrives from stage 2 *and* drops to degree 1, while layer 3 never
    leaves stage 1 yet still has to be re-chunked because the stage lost a TP rank.
    Both are handled by one reshard (they share the same 2 -> 1 degree pair). The two
    ranks are the current TP2 shards; afterwards rank 0 must hold the full logical
    tensors byte for byte equal to the checkpoint anchor, AdamW moments and step
    included, and the dropped rank must keep nothing.
    """
    import torch

    from resihp.parallel.pp import plan_migration, reshard_layout
    from resihp.parallel.reshard import reshard_tp_state

    _, placements = plan_migration((2, 2, 2), (2, 2, 2), (2, 1, 1))
    arriving = next(place for place in placements if place.layer == 4)
    staying = next(place for place in placements if place.layer == 3)
    layout = reshard_layout(placements, new_owner=1)
    # One reshard call handles one degree pair; these two layers share it.
    assert (arriving.old_degree, arriving.new_degree) == (staying.old_degree, staying.new_degree)
    anchor = _anchor_state(layout, seed=2024)

    local_state = {
        name: {
            "shard_index": rank,
            **{
                field: _shard_of(anchor[name][field], layout[name], rank, arriving.old_degree).to(device)
                for field in _MIGRATION_FIELDS
            },
            "step": anchor[name]["step"].to(device),
        }
        for name in layout
    }
    local_was_device = next(iter(local_state.values()))["param"].device == device

    new_local = reshard_tp_state(
        local_state,
        layout=layout,
        old_size=arriving.old_degree,
        new_size=arriving.new_degree,
        new_rank=0 if rank == 0 else None,
        checkpoint=anchor,
    )

    if rank != 0:
        return {"dropped": True, "empty": new_local == {}, "local_was_device": local_was_device}

    matched = True
    for name in layout:
        for field in _MIGRATION_FIELDS + ("step",):
            matched &= torch.equal(new_local[name][field], anchor[name][field])
    return {
        "dropped": False,
        "match": bool(matched),
        "layers": sorted({name.split(".")[1] for name in new_local}),
        "names": sorted(new_local),
        "arriving_moved": arriving.moved and arriving.resharded,
        "staying_resharded_only": not staying.moved and staying.resharded,
        "local_was_device": local_was_device,
    }


def _assert_migration(results):
    receiver, donor = results
    assert receiver["arriving_moved"], receiver  # layer 4 both moved and resharded
    assert receiver["staying_resharded_only"], receiver  # layer 3 stayed but reshards
    assert receiver["match"], receiver  # every field equals the checkpoint anchor
    assert receiver["layers"] == ["3", "4"], receiver
    assert len(receiver["names"]) == 20, receiver  # both layers whole, 10 tensors each
    assert donor["dropped"] and donor["empty"], donor


# --- backend wrappers -------------------------------------------------------------


def _init_env(rank, world_size, port):
    import os

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )


def _gloo_pp_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _compare_pp(rank, world_size, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_pp_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    assert torch.cuda.current_device() == rank
    result = _compare_pp(rank, world_size, torch.device(f"cuda:{rank}"))
    assert result["is_cuda"]
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _gloo_migration_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    result = _run_migration(rank, torch.device("cpu"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _nccl_migration_worker(rank, world_size, result_dir, port):
    _init_env(rank, world_size, port)
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    assert dist.get_backend() == "nccl"
    result = _run_migration(rank, torch.device(f"cuda:{rank}"))
    Path(result_dir, f"result_{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()


def _results(result_dir, count):
    return [json.loads(Path(result_dir, f"result_{rank}.json").read_text()) for rank in range(count)]


def _skip_if_few_gpus(count):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < count:
        pytest.skip(f"needs {count} GPU(s), found {torch.cuda.device_count()}")


# --- Gloo gates (always run where torch is installed) -----------------------------


@requires_torch
@pytest.mark.parametrize("num_stages", [1, 2])
def test_pp_matches_reference_gloo(tmp_path, num_stages):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_pp_worker, args=(num_stages, str(tmp_path), _free_port()), nprocs=num_stages, join=True)
    _assert_matches_reference(_results(tmp_path, num_stages), num_stages, f"PP{num_stages} Gloo")


@requires_torch
def test_pp_layer_migration_is_lossless_gloo(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_gloo_migration_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    _assert_migration(_results(tmp_path, 2))


# --- NCCL gates (real GPU tensors + real transfers; skip without enough GPUs) -----


@requires_torch
@pytest.mark.parametrize("num_stages", [1, 2])
def test_pp_cuda_nccl_matches_reference(tmp_path, num_stages):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(num_stages)
    mp.spawn(_nccl_pp_worker, args=(num_stages, str(tmp_path), _free_port()), nprocs=num_stages, join=True)
    results = _results(tmp_path, num_stages)
    for result in results:
        assert result["is_cuda"], result
    _assert_matches_reference(results, num_stages, f"PP{num_stages} NCCL")


@requires_torch
def test_pp_layer_migration_is_lossless_cuda_nccl(tmp_path):
    import torch.multiprocessing as mp

    _skip_if_few_gpus(2)
    mp.spawn(_nccl_migration_worker, args=(2, str(tmp_path), _free_port()), nprocs=2, join=True)
    results = _results(tmp_path, 2)
    assert results[0]["local_was_device"], results  # shards genuinely started on GPU
    _assert_migration(results)
