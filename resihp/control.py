"""Control plane and fail-stop safe-point orchestration (T9).

Two process groups exist at all times:

* the **control group** -- the always-alive Gloo ``WORLD`` group. Every process
  (including ones whose rank has been marked failed) stays in it for the whole
  run, so control collectives -- the fail-stop broadcast and the plan-digest
  agreement -- never deadlock.
* the **training group** -- a subgroup over the currently live ranks, built from
  the current :class:`~resihp.plan.ExecutionPlan` (NCCL on GPU, Gloo on CPU). It
  is torn down and rebuilt, in unison across every process, on each fail-stop.

A fail-stop is *simulated* by exclusion: the process is not killed, it is dropped
from the training group and stops doing training work while remaining in the
control group. This keeps the deterministic failure schedule fully testable.

The nine-step safe point (plan 3.2) is driven by :meth:`ControlPlane.safe_point`.
Communication ranks, TP shards, and PP owners come only from the plan; nothing is
inferred from an older layout.
"""

from .config import TrainConfig
from .plan import ExecutionPlan, build_plan


class ControlError(RuntimeError):
    """Raised when ranks disagree on the plan/state digest at a safe point."""


def reconfigure(
    config: TrainConfig,
    previous: ExecutionPlan | None,
    failed_ranks,
    *,
    version: int,
    step: int,
) -> ExecutionPlan:
    """Pure TP->PP->DP replan into one new versioned plan (safe-point step 5).

    Deterministic in its inputs, so every rank that calls it with the same
    ``(config, previous, failed_ranks, version, step)`` produces a byte-identical
    plan and therefore an identical digest (``build_plan`` normalizes/sorts the
    failed ranks itself, so the order they arrive in does not matter).
    """
    return build_plan(
        config,
        step=step,
        version=version,
        failed_ranks=failed_ranks,
        previous=previous,
    )


class ControlPlane:
    """Owns the always-alive control group and the rebuildable training group."""

    def __init__(self, rank: int, world_size: int, control_group, training_backend: str):
        self.rank = rank
        self.world_size = world_size
        self.control_group = control_group
        self.training_backend = training_backend
        self.training_group = None
        # Real training state attached once the runtime holds it (T10+). While
        # unset the safe point behaves as the T9 skeleton: no checkpoint, empty
        # state digest.
        self.training_run = None
        self.checkpoint_path = None

    @classmethod
    def initialize(cls, *, training_backend: str | None = None) -> "ControlPlane":
        """Init the Gloo ``WORLD`` control group from the torchrun environment."""
        import torch
        import torch.distributed as dist

        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if training_backend is None:
            training_backend = "nccl" if torch.cuda.is_available() else "gloo"
        return cls(rank, world_size, dist.group.WORLD, training_backend)

    def is_training_rank(self, plan: ExecutionPlan) -> bool:
        """True when this rank is a member of ``plan``'s training group."""
        return self.rank in set(plan.live_ranks)

    def build_training_group(self, plan: ExecutionPlan):
        """Build the training subgroup over the live ranks (safe-point step 6b).

        ``new_group`` is collective over the control group, so every process --
        members and non-members alike -- must call it. Non-members get a sentinel
        handle, not a real group, so we keep ``None`` for them: they neither run
        training collectives nor destroy a group they never joined.
        """
        import torch.distributed as dist

        group = dist.new_group(ranks=list(plan.live_ranks), backend=self.training_backend)
        self.training_group = group if self.is_training_rank(plan) else None
        return self.training_group

    def destroy_training_group(self) -> None:
        """Release the current training group in unison (safe-point step 6a)."""
        import torch.distributed as dist

        if self.training_group is not None:
            dist.destroy_process_group(self.training_group)
            self.training_group = None

    def training_step(self, plan: ExecutionPlan):
        """One simplest-possible training iteration (stand-in for fwd/bwd).

        A failed rank is not in the training group and returns without touching
        any training collective -- it has permanently left the training path.
        """
        if not self.is_training_rank(plan):
            return None
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([float(self.rank)])
        dist.all_reduce(tensor, group=self.training_group)
        return tensor

    def _commit_checkpoint(self, version: int) -> None:
        """Safe-point step 2: atomic checkpoint of the just-committed iteration.

        Once real training state is attached (T10+) this writes it atomically via
        :func:`resihp.checkpoint.save_checkpoint`. With no state attached it is a
        no-op, exactly as the T9 skeleton (whose all-reduce carries nothing to
        save). Gathering a *sharded* run into a full logical checkpoint is the
        T11+ reshard path; here the call point consumes whatever logical run it holds.
        """
        if self.training_run is None or self.checkpoint_path is None:
            return
        from .checkpoint import save_checkpoint

        save_checkpoint(self.checkpoint_path, self.training_run, plan_version=version)

    def _state_digest(self) -> str:
        """Step 8's logical-state digest: empty until real state is attached (T10+)."""
        if self.training_run is None:
            return ""
        from .reference import logical_state_digest

        return logical_state_digest(self.training_run.model)

    def _recover_state(self, plan: ExecutionPlan) -> None:
        """Safe-point step 7: recover/migrate/reshard param + AdamW state.

        Call point only. Real gather-from-peer / checkpoint-fallback / reshard
        (following ``plan.state_routes``) lands in T11-T14; the skeleton moves no
        tensors.
        """

    def broadcast_failure(self, failed_rank: int) -> int:
        """Safe-point step 3: broadcast the fail-stop event over the control group.

        Source is rank 0, the coordinator, which this reproduction's deterministic
        failure schedule never targets and which stays alive in the control group.
        """
        import torch
        import torch.distributed as dist

        payload = torch.tensor([failed_rank], dtype=torch.long)
        dist.broadcast(payload, src=0, group=self.control_group)
        return int(payload.item())

    def agree_on_digest(self, plan: ExecutionPlan, state_digest: str = "") -> None:
        """Safe-point step 8: confirm every rank holds identical plan/state digests."""
        import torch.distributed as dist

        local = (plan.digest, state_digest)
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.control_group)
        for other in gathered:
            if other != local:
                raise ControlError(
                    f"rank {self.rank} plan/state digest disagreement: {other} != {local}"
                )

    def safe_point(
        self,
        config: TrainConfig,
        plan: ExecutionPlan,
        failed_ranks: tuple[int, ...],
        new_failed: int,
        *,
        next_step: int,
    ) -> tuple[ExecutionPlan, tuple[int, ...]]:
        """Run the strict nine-step fail-stop safe point and return the new plan.

        Step 1 (complete + commit the current iteration) is done by the caller
        before this returns control here. Steps 2-8 run in the mandated order;
        step 9 (resume) is the caller's next loop turn.
        """
        self._commit_checkpoint(plan.version)  # 2. atomic checkpoint save
        confirmed = self.broadcast_failure(new_failed)  # 3. broadcast fail-stop
        failed = tuple(sorted(set(failed_ranks) | {confirmed}))  # 4. mark failed
        new_plan = reconfigure(  # 5. TP->PP->DP replan (one new version)
            config, plan, failed, version=plan.version + 1, step=next_step
        )
        self.destroy_training_group()  # 6a. release old training group
        self.build_training_group(new_plan)  # 6b. build new training group
        self._recover_state(new_plan)  # 7. recover/migrate/reshard (call point)
        self.agree_on_digest(new_plan, self._state_digest())  # 8. verify digests agree
        return new_plan, failed  # 9. caller continues from next iteration

    def shutdown(self) -> None:
        """Release the training group, then tear down the control group.

        Leaves no residual process group behind.
        """
        import torch.distributed as dist

        self.destroy_training_group()
        dist.barrier(group=self.control_group)
        dist.destroy_process_group()
