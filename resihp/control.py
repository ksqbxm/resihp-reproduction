"""Control plane, fail-stop safe point, and consistent stop (T9 + T14).

Two process groups exist at all times:

* the **control group** -- the always-alive Gloo ``WORLD`` group. Every process
  (including ones whose rank has been marked failed) stays in it for the whole
  run, so control collectives -- the fail-stop broadcast, the stop agreement, and
  the recovery gather -- never deadlock.
* the **training groups** -- built from the current
  :class:`~resihp.plan.ExecutionPlan` (NCCL on GPU, Gloo on CPU): one TP subgroup per
  active stage, which a stage's sharded execution rides on; one two-rank subgroup per
  pipeline hop the assignment creates, which the 1F1B boundary transfers ride on; and
  one group over every rank the plan places, which the DP gradient combine rides on.
  All are torn down and rebuilt, in unison across every process, on each fail-stop.

A fail-stop is *simulated* by exclusion: the process is not killed, it is dropped
from the training group and stops doing training work while remaining in the
control group. This keeps the deterministic failure schedule fully testable.

The nine-step safe point (plan 3.2) is driven by :meth:`ControlPlane.safe_point`.
Communication ranks, TP shards, and PP owners come only from the plan; nothing is
inferred from an older layout.

**Consistent stop (plan 3.6).** Six conditions end the run instead of publishing a
plan nobody can execute. Whichever rank observes one, every rank raises the same
:class:`ConsistentStop`: agreement runs over the always-alive control group, the
reason is derived from the gathered list rather than the local view, and it is
reached *before* any group is built. So no rank continues alone, no half-completed
plan or group is left behind, the pre-failure checkpoint is untouched, and the
caller can exit normally with the root cause in hand.
"""

from dataclasses import dataclass

from .config import TrainConfig
from .plan import ExecutionPlan, InfeasiblePlan, boundary_pairs, build_plan


#: The six consistent-stop conditions of plan section 3.6. The first three are the
#: planner infeasibility codes, passed through unchanged so the message names the
#: real root cause; the last three are observed by the control plane itself.
STOP_CODES = (
    "no_feasible_tp",
    "no_executable_pp",
    "no_feasible_dp_target",
    "checkpoint_unusable",
    "plan_disagreement",
    "state_mismatch",
)

#: Which stop a disagreement on each agreed digest reports.
_DIGEST_STOP = {"plan": "plan_disagreement", "state": "state_mismatch"}


@dataclass(frozen=True)
class StopReason:
    code: str
    message: str


class ConsistentStop(RuntimeError):
    """Raised on every rank at once when a stop condition has been agreed."""

    def __init__(self, reason: StopReason):
        super().__init__(f"{reason.code}: {reason.message}")
        self.reason = reason


def reconfigure(
    config: TrainConfig,
    previous: ExecutionPlan | None,
    failed_ranks,
    *,
    version: int,
    step: int,
    memory_budget: int | None = None,
    vocab_size: int = 1,
    sequence_length: int = 1,
) -> ExecutionPlan:
    """Pure TP->PP->DP replan into one new versioned plan (safe-point step 5).

    Deterministic in its inputs, so every rank that calls it with the same
    ``(config, previous, failed_ranks, version, step)`` produces a byte-identical
    plan and therefore an identical digest (``build_plan`` normalizes/sorts the
    failed ranks itself, so the order they arrive in does not matter). With a
    ``memory_budget`` the replan is memory-gated, which is what makes the
    ``no_feasible_tp`` stop reachable: a degree drop can push a surviving rank's
    resident bytes past the budget.
    """
    return build_plan(
        config,
        step=step,
        version=version,
        failed_ranks=failed_ranks,
        previous=previous,
        memory_budget=memory_budget,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
    )


class ControlPlane:
    """Owns the always-alive control group and the rebuildable training groups."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        control_group,
        training_backend: str,
        *,
        vocab_size: int | None = None,
        sequence_length: int = 1,
        memory_budget: int | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.control_group = control_group
        self.training_backend = training_backend
        self.tp_group = None
        self.executor_group = None
        self.boundary_groups: dict[tuple[int, int], object] = {}
        # Run-wide planning inputs: identical on every rank, so every rank replans
        # to the same plan. Per-rank state is attached separately by ``attach_run``.
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.memory_budget = memory_budget
        # This rank's state under the current plan, attached by the runtime.
        self.training_run = None
        self.checkpoint_path = None
        self.device = None
        self.state_digest = ""

    @classmethod
    def initialize(
        cls,
        *,
        training_backend: str | None = None,
        vocab_size: int | None = None,
        sequence_length: int = 1,
        memory_budget: int | None = None,
    ) -> "ControlPlane":
        """Init the Gloo ``WORLD`` control group from the torchrun environment."""
        import torch
        import torch.distributed as dist

        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if training_backend is None:
            training_backend = "nccl" if torch.cuda.is_available() else "gloo"
        return cls(
            rank,
            world_size,
            dist.group.WORLD,
            training_backend,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            memory_budget=memory_budget,
        )

    def attach_run(self, run, *, checkpoint_path, device=None) -> None:
        """Attach this rank's real training state to the safe point.

        ``run`` is this rank's :class:`resihp.recovery.PlannedRun` (``None`` when the
        plan places it on no stage); ``checkpoint_path`` is the one canonical
        checkpoint every rank shares.
        """
        self.training_run = run
        self.checkpoint_path = checkpoint_path
        self.device = device

    def is_training_rank(self, plan: ExecutionPlan) -> bool:
        """True when the plan places this rank on an active stage."""
        return self.rank in set(plan.active_ranks)

    def build_training_groups(self, plan: ExecutionPlan) -> None:
        """Build this rank's training groups from the plan (safe-point step 6b).

        ``new_group`` is collective over the control group, so every process --
        members and non-members alike -- must call it for *every* group, in the same
        plan-derived order. Non-members get a sentinel handle rather than a real group,
        so only the groups this rank actually belongs to are kept: it neither runs
        training collectives elsewhere nor destroys a group it never joined.

        The per-hop groups hold exactly two ranks -- the two stage leaders a boundary
        connects -- because the 1F1B steady state issues a fused ``batch_isend_irecv``
        and NCCL runs batched P2P on the group's collective communicator, which every
        member would then have to issue in the same order.
        """
        import torch.distributed as dist

        tp_group = None
        for stage in sorted(plan.stages, key=lambda stage: (stage.replica_id, stage.stage_id)):
            group = dist.new_group(ranks=list(stage.tp_members), backend=self.training_backend)
            if self.rank in stage.tp_members:
                tp_group = group
        boundaries: dict[tuple[int, int], object] = {}
        for pair in boundary_pairs(plan):
            group = dist.new_group(ranks=list(pair), backend=self.training_backend)
            if self.rank in pair:
                boundaries[pair] = group
        executors = dist.new_group(ranks=list(plan.active_ranks), backend=self.training_backend)
        self.tp_group = tp_group
        self.boundary_groups = boundaries
        self.executor_group = executors if self.is_training_rank(plan) else None

    def destroy_training_groups(self) -> None:
        """Release the current training groups in unison (safe-point step 6a).

        ``boundary_groups`` is built in sorted-pair order, so its two members release it
        at the same point in the sequence.
        """
        import torch.distributed as dist

        for group in (self.tp_group, *self.boundary_groups.values(), self.executor_group):
            if group is not None:
                dist.destroy_process_group(group)
        self.tp_group = None
        self.boundary_groups = {}
        self.executor_group = None

    def training_step(self):
        """Advance one iteration through the plan's assignment-driven runtime.

        A rank the plan places on no stage holds no run and returns without touching
        any training collective -- it has permanently left the training path.
        """
        if self.training_run is None:
            return None
        return self.training_run.step()

    def _commit_checkpoint(self, plan: ExecutionPlan) -> None:
        """Safe-point step 2: gather every shard into the one atomic checkpoint.

        Every process joins the gather, contributing nothing when it holds no state,
        so the union of the plan's stages and replicas is the whole model. The rank
        about to be marked failed still contributes here, exactly as the plan intends:
        this is the *pre-failure* checkpoint.
        """
        from .recovery import commit_checkpoint

        commit_checkpoint(
            self.training_run,
            self.checkpoint_path,
            plan=plan,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            group=self.control_group,
            # The lowest active rank writes: every rank derives it from the same plan,
            # and unlike a fixed rank 0 it is always one that actually holds state, so
            # the file cannot record a cursor from a rank that has nothing to commit.
            writer=self.rank == plan.active_ranks[0],
        )

    def _recover_state(self, plan: ExecutionPlan, previous: ExecutionPlan) -> None:
        """Safe-point step 7: run the single recovery path and keep its state digest.

        The failed rank contributes nothing, so its shards genuinely have to be found
        on a healthy peer replica or in the checkpoint. Which layers this rank's stage
        owns, at which TP degree, and which micro-batches it executes are all read
        from ``plan``; ``previous`` says only what changed, so nothing that did not
        change is moved.
        """
        from .recovery import recover

        self.training_run, self.state_digest = recover(
            self.training_run,
            plan=plan,
            previous=previous,
            rank=self.rank,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            checkpoint_path=self.checkpoint_path,
            control_group=self.control_group,
            tp_group=self.tp_group,
            executor_group=self.executor_group,
            boundary_groups=self.boundary_groups,
            device=self.device,
        )

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

    def agree(self, reason: StopReason | None = None, **digests: str) -> None:
        """Reach a unanimous decision over the always-alive control group.

        Every rank contributes the stop reason it observed locally (or ``None``) plus
        the digests it wants checked. Any reason reported anywhere, and any digest the
        ranks do not all agree on, raises the *same* :class:`ConsistentStop` on every
        rank -- derived from the gathered list, never from the local view, so two
        ranks can never stop for two different reasons. A locally observed fault
        therefore never leaves one rank raising while the others block on a collective
        it has already left.
        """
        import torch.distributed as dist

        local = (
            None if reason is None else (reason.code, reason.message),
            tuple(sorted(digests.items())),
        )
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.control_group)

        for rank, (reported, _) in enumerate(gathered):
            if reported is not None:
                raise ConsistentStop(StopReason(reported[0], f"rank {rank}: {reported[1]}"))
        for label in sorted(digests):
            values = {dict(entry)[label] for _, entry in gathered}
            if len(values) > 1:
                raise ConsistentStop(
                    StopReason(
                        _DIGEST_STOP[label],
                        f"ranks disagree on the {label} digest: {sorted(values)}",
                    )
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
        step 9 (resume) is the caller's next loop turn. Raises
        :class:`ConsistentStop` -- on every rank, for the same reason -- instead of
        returning a plan when any stop condition is met.
        """
        self._commit_checkpoint(plan)  # 2. atomic checkpoint save
        confirmed = self.broadcast_failure(new_failed)  # 3. broadcast fail-stop
        failed = tuple(sorted(set(failed_ranks) | {confirmed}))  # 4. mark failed

        new_plan = None
        reason = None
        try:  # 5. TP->PP->DP replan (one new version)
            new_plan = reconfigure(
                config,
                plan,
                failed,
                version=plan.version + 1,
                step=next_step,
                memory_budget=self.memory_budget,
                vocab_size=self.vocab_size,
                sequence_length=self.sequence_length,
            )
        except InfeasiblePlan as error:
            reason = StopReason(error.reason.code, error.reason.message)
        # Agree before anything is published: a plan no rank can execute, or one the
        # ranks disagree on, must never reach group construction. Building groups
        # from disagreeing plans is itself a deadlock, so this check cannot wait for
        # step 8.
        self.agree(reason, plan="" if new_plan is None else new_plan.digest)

        try:
            self.destroy_training_groups()  # 6a. release the old training groups
            self.build_training_groups(new_plan)  # 6b. build the new training groups
            reason = None
            try:  # 7. recover/migrate/reshard along the single path
                self._recover_state(new_plan, plan)
            except Exception as error:
                reason = _recovery_stop(error)
            self.agree(reason)
            self.agree(plan=new_plan.digest, state=self.state_digest)  # 8. verify digests
        except ConsistentStop:
            self.destroy_training_groups()  # leave no group behind from a stopped plan
            raise
        return new_plan, failed  # 9. caller continues from next iteration

    def shutdown(self) -> None:
        """Release the training group, then tear down the control group.

        Leaves no residual process group behind.
        """
        import torch.distributed as dist

        self.destroy_training_groups()
        dist.barrier(group=self.control_group)
        dist.destroy_process_group()


def _recovery_stop(error: Exception) -> StopReason:
    """Classify a step-7 failure as its own stop condition, or let it escape.

    Only the two faults plan 3.6 names are stops: an unusable checkpoint, and a
    logical tensor that does not survive resharding intact. Anything else is a bug,
    not a resource condition, and must not be dressed up as a clean stop.
    """
    from .checkpoint import CheckpointError
    from .parallel.reshard import ReshardError

    if isinstance(error, CheckpointError):
        return StopReason("checkpoint_unusable", str(error))
    if isinstance(error, ReshardError):
        return StopReason("state_mismatch", str(error))
    raise error
