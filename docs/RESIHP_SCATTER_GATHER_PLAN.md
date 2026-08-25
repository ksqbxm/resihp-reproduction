# Plan: Reproduce ResiHP scatter/gather P2P optimization (3 commits)

## Context

**Why:** ResiHP (§ "P2P Communication Optimization", Fig 7) cuts cross-node InfiniBand traffic at a
pipeline stage boundary via Megatron-LM's scatter/gather. Adjacent stages replicate the boundary
activation inside each TP group; the paper's rule scatters the tensor into `N = max(TP_send, TP_recv)`
equal contiguous chunks, P2P-sends each chunk over IB to its corresponding receiver rank, and the
receiver reconstructs via a fast intra-node **all-gather**. Net cross-node volume is one copy, spread
across N parallel links.

**Current gap:** [resihp/parallel/pp.py](resihp/parallel/pp.py) moves one copy **leader → leader**
over a two-rank hop group, then **broadcasts** within the receiving TP group (`_replicate`). Correct
across heterogeneous TP degrees but not performance-equivalent: one leader link, not N; broadcast, not
all-gather.

**Outcome (user decision):** Full replacement, **no backward-compat** — remove the leader-pair /
`_replicate` logic entirely and keep scatter/gather as the single boundary path. Wire it through the
whole stack and cover it with its own test plus the integration suite. Ship as **3 commits** so each is
independently verifiable. Record the 3 phases in `docs/PROGRESS.md` (project's sole progress doc).

## Verified design facts

- Adjacent stages own **disjoint** rank sets ⇒ a rank is sender xor receiver.
- TP degrees are powers of two ⇒ `min|max` and `dim % N == 0`, so a flat activation of
  `numel = mb*seq*dim` splits into `N` equal chunks.
- **Routing (pure):** chunk `k`'s sender = `up_members[k*U//N]`, receiver = `down_members[k*D//N]`.
  Each `(sender,receiver)` pair carries exactly one chunk (`1/N`); exactly `N` distinct pairs; forward
  chunk `k` and backward-grad chunk `k` share the same rank pair (reversed), so the fused steady-state
  `batch_isend_irecv` still works.
- **Union group:** all chunk ops for a hop ride ONE group = sorted union of both stages' TP members, so
  NCCL coalesced P2P uses one communicator. TP1↔TP1 union == today's two-rank pair, so TP1 tests are
  unaffected by construction.

---

## Commit 1 — Control plane only (no runtime change)

**Scope:** boundary group topology changes from two-leader pairs to union groups; runtime untouched.

- `resihp/plan.py`: rewrite [`boundary_pairs`](resihp/plan.py:300) → `boundary_hops(plan)` returning,
  per distinct hop, the **sorted union tuple** of the two adjacent stages' full `executor_ranks`
  (de-dup on the union). Update docstring.
- `resihp/control.py`: [`build_training_groups`](resihp/control.py:177) iterates `boundary_hops`,
  builds one `new_group` per union tuple, keys `boundary_groups` by that tuple; update import +
  docstring. `destroy_training_groups` unchanged (iterates `.values()`).
- Tests: update [test_plan.py:468-486](tests/test_plan.py:468) to `boundary_hops`, asserting union
  tuples (e.g. `(0,1,2,3)`,`(4,5,6,7)` instead of `(0,2)`,`(4,6)`). test_control consumes
  `control.boundary_groups` and picks up the new groups automatically.

**验收:** `pytest tests/test_plan.py tests/test_control.py -v`

## Commit 2 — Scatter/gather runtime

**Scope:** rewrite the boundary transfer in the runtime; add the pure helper and its test.

- `resihp/parallel/pp.py`:
  - Add module-level pure helper (torch-free):
    `scatter_routing(up_members, down_members) -> [(sender,receiver) for k in range(N)]`.
  - Rewrite Send/Recv section of [`PipelineRuntime`](resihp/parallel/pp.py:151): remove `_hop`,
    `_replicate`, `_is_leader`; add `_hop_group(peer_members)` (union-tuple lookup),
    `_scatter_send_ops`, `_gather_recv` (slab + irecv P2POps), `_reconstruct` (all-gather over
    `stage.group`, skip when `tp_size==1`, concat in tp-rank order, reshape). Rewire the six methods
    (`_send_forward`/`_recv_forward`, `_send_backward`/`_recv_backward`, fused
    `_send_forward_recv_backward`/`_send_backward_recv_forward`) to one `batch_isend_irecv` per hop,
    keeping return values + `requires_grad_` semantics. Keep send-chunk clones alive until `wait()`.
    Update module + method docstrings (§ "Stage boundaries").
- Tests in [tests/test_parallel_pp.py](tests/test_parallel_pp.py):
  - Torch-free `scatter_routing` test over `(1,1),(2,2),(2,1),(1,2),(4,2)`: exactly `N` chunks, each
    pair distinct and once ⇒ one copy across N parallel links (the traffic-reduction proof).
  - New CPU/Gloo distributed gate: heterogeneous **TP2→TP1** (world=3) and **TP1→TP2** pipelines through
    `PipelineRuntime`, asserting grads + post-AdamW weights match the single-process reference; build
    the union hop group over all ranks.

**验收:** `pytest tests/test_parallel_pp.py -v`

## Commit 3 — Full-stack integration + NCCL + cleanup

**Scope:** update the combination harness, regress the fault/recovery/e2e gates, run on GPU, delete old logic.

- [tests/test_combinations.py](tests/test_combinations.py) `_run_tp_pp` (~line 279): replace the
  two-leader `{(0,2): hop over (0,2)}` with a union group over `(0,1,2,3)` keyed `(0,1,2,3)`.
  `tp_dp` (no hop) and `pp_dp`/TP1 hops need no change (unions of 1-rank stages == pairs).
- Regress test_end_to_end / test_fault_sequences / test_recovery (consume `control.boundary_groups`);
  confirm [test_fault_sequences.py:310](tests/test_fault_sequences.py:310) `len(control.boundary_groups)`
  still holds (hop count unchanged; only keys/contents change). `recovery.py`/`train.py` only forward
  the dict — no change.
- Run the GPU/NCCL real-machine gates (target server).
- **Cleanup:** ensure `_replicate` and `boundary_pairs` are fully gone.

**验收:**
```bash
pytest tests/test_combinations.py tests/test_end_to_end.py tests/test_fault_sequences.py tests/test_recovery.py -v
grep -rn "boundary_pairs\|_replicate" resihp/   # must be empty
```

---

## Docs

As part of Commit 1 (or a leading doc step), append a new `## P…` section to
[docs/PROGRESS.md](docs/PROGRESS.md) describing this optimization and the 3-commit breakdown above,
matching the file's existing Chinese section style (根因 / 修法 / 验收), and noting that the
distributed + NCCL gates must be re-run on the 8-GPU target machine (no torch on this dev box).

## Overall verification

The user runs GPU/distributed tests on the target server. Success:
1. Each commit's 验收 command passes on CPU/Gloo.
2. New heterogeneous TP2↔TP1 gates + all existing gates match the reference within the existing
   `allclose` band on Gloo, and on NCCL where GPUs are available.
3. Final `grep -rn "boundary_pairs\|_replicate" resihp/` is empty — old logic fully removed.
