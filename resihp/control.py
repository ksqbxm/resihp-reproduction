"""Control plane, fail-stop detection, world re-formation, and consistent stop.

A fail-stop is **real**: the failing process is killed outright (``SIGKILL``), leaving
no cleanup, no farewell collective, and no way to take part in anything afterwards.
Three consequences shape this module.

* **Nothing that includes the dead rank can be used again.** Its ``WORLD`` process
  group is unusable from the moment it dies -- ``new_group`` is collective over it, so
  not one training group could ever be rebuilt on top of it. The survivors therefore
  tear the whole world down and re-form a new one over themselves after every
  fail-stop; :mod:`resihp.membership`'s store, hosted by the supervisor, is what they
  rendezvous on, because it is the only channel a dead rank cannot block.
* **Detection cannot be a collective.** The supervisor reaps the killed process and
  publishes the membership of each iteration boundary; :meth:`ControlPlane.observe`
  reads it. A rank killed at a safe point is missing from that membership before any
  rank issues the next iteration's collectives, which is what keeps NCCL from ever
  being handed a dead peer.
* **A rank has two rank numbers.** Its **plan rank** is the identity it was launched
  with -- stable for the whole run, and the only one the planner, the checkpoint and
  the ``ExecutionPlan`` ever speak. Its **torch rank** is its index in the current
  membership, and changes every time the world is re-formed. This module owns the
  translation; every ``ranks=`` list handed to ``new_group`` passes through it.

Two kinds of group exist between fail-stops:

* the **world group** -- the Gloo group over the current membership, re-formed once
  per fail-stop. Control collectives (the stop agreement, the checkpoint gather, the
  recovery gather) ride it.
* the **training groups** -- built from the current
  :class:`~resihp.plan.ExecutionPlan` (NCCL on GPU, Gloo on CPU): one TP subgroup per
  active stage, one union subgroup per pipeline hop the assignment creates, and one
  group over every rank the plan places, which the DP gradient combine rides on.

**Consistent stop (plan 3.6).** Six conditions end the run instead of publishing a
plan nobody can execute. Whichever rank observes one, every surviving rank raises the
same :class:`ConsistentStop`: agreement runs over the re-formed world group, the
reason is derived from the gathered list rather than the local view, and it is reached
*before* any training group is built. So no rank continues alone, no half-completed
plan or group is left behind, the pre-failure checkpoint is untouched, and the caller
can exit normally with the root cause in hand.
"""

from dataclasses import dataclass
from datetime import timedelta
import os

from . import membership
from .config import TrainConfig
from .plan import ExecutionPlan, InfeasiblePlan, boundary_hops, build_plan


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

#: Bound on every process group this module builds. Nothing in the failure model
#: reaches it -- a killed rank is detected out of band, before the next collective is
#: issued -- so hitting it means a rank is stuck, and a bounded wait turns that into a
#: raised error instead of a job that hangs forever.
GROUP_TIMEOUT = timedelta(seconds=300)


@dataclass(frozen=True)
class StopReason:
    code: str
    message: str


class ConsistentStop(RuntimeError):
    """Raised on every surviving rank at once when a stop condition has been agreed."""

    def __init__(self, reason: StopReason):
        super().__init__(f"{reason.code}: {reason.message}")
        self.reason = reason


class ControlPlane:
    """Owns this rank's place in the world, and rebuilds both after a real fail-stop."""

    def __init__(
        self,
        rank: int,
        members,
        store,
        training_backend: str,
        *,
        vocab_size: int | None = None,
        sequence_length: int = 1,
        memory_budget: int | None = None,
    ):
        self.rank = int(rank)
        self.members = tuple(sorted(members))
        self.store = store
        self.training_backend = training_backend
        self.epoch = 0
        #: How many boundary rendezvous this rank has taken part in.
        self.boundary = 0
        self.world_group = None
        self.tp_group = None
        self.executor_group = None
        self.boundary_groups: dict[tuple[int, ...], object] = {}
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
        """Join the run from the supervisor's environment and form the first world."""
        import torch

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        store = membership.connect(
            os.environ[membership.HOST_ENV], os.environ[membership.PORT_ENV]
        )
        if training_backend is None:
            training_backend = "nccl" if torch.cuda.is_available() else "gloo"
        control = cls(
            rank,
            range(world_size),
            store,
            training_backend,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            memory_budget=memory_budget,
        )
        control.form_world(control.members, epoch=0)
        return control

    # --- plan ranks and torch ranks ---------------------------------------------------

    @property
    def world_size(self) -> int:
        return len(self.members)

    def torch_rank(self, plan_rank: int) -> int:
        """This membership's torch rank for a plan rank.

        The membership is sorted, so the mapping is monotonic: a ``ranks=`` list built
        from ascending plan ranks stays ascending, and a subgroup's local rank ``i`` is
        still its ``i``-th plan rank -- which is what lets ``tp_members.index(rank)``
        keep naming the TP shard after the world has been re-formed.
        """
        return self.members.index(plan_rank)

    # --- detection and world re-formation ---------------------------------------------

    def observe(self) -> tuple[int, ...]:
        """Safe-point steps 3-4: rendezvous at the next boundary, report the dead.

        This is the fail-stop event. It is not broadcast by a rank -- the rank that
        would have broadcast it no longer exists -- but read from the supervisor, which
        noticed the process exit. Every survivor reads the same published membership,
        so they cannot disagree about who was lost, and none of them issues another
        training collective until they have.

        Boundaries are counted here rather than named by the caller: the supervisor
        publishes them in order, so the only thing that has to hold is that every live
        rank reaches the same number of them -- which is what a barrier means anyway,
        and what a caller-chosen tag could silently get wrong.
        """
        self.boundary += 1
        published = membership.boundary(self.store, self.boundary, timeout=GROUP_TIMEOUT)
        return tuple(rank for rank in self.members if rank not in published)

    def form_world(self, members, *, epoch: int) -> None:
        """Build the Gloo world group over ``members`` (safe-point step 6b).

        Rendezvous runs on the supervisor's store under this epoch's own prefix, never
        on a rank-hosted store: a rendezvous whose host can die is a second single
        point of failure, and the epoch prefix keeps a new world from reading a key the
        world it replaces left behind.
        """
        import torch.distributed as dist

        self.members = tuple(sorted(members))
        self.epoch = int(epoch)
        dist.init_process_group(
            backend="gloo",
            store=dist.PrefixStore(membership.PG_PREFIX.format(epoch=self.epoch), self.store),
            rank=self.torch_rank(self.rank),
            world_size=self.world_size,
            timeout=GROUP_TIMEOUT,
        )
        self.world_group = dist.group.WORLD

    def dissolve_world(self) -> None:
        """Release every group this rank holds, world included (safe-point step 6a).

        The training groups go first, in the plan-derived order every member built them
        in, and the world -- which the dead rank was still a member of -- goes last.
        Nothing collective runs here: the boundary rendezvous already proved no transfer
        is in flight, so the communicators are quiescent and each rank can drop its own.
        """
        import torch.distributed as dist

        self.destroy_training_groups()
        dist.destroy_process_group()
        self.world_group = None

    # --- the plan's training groups ---------------------------------------------------

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
        """Build this rank's training groups from the plan (safe-point step 6c).

        ``new_group`` is collective over the world group, so every process -- members
        and non-members alike -- must call it for *every* group, in the same
        plan-derived order. Non-members get a sentinel handle rather than a real group,
        so only the groups this rank actually belongs to are kept: it neither runs
        training collectives elsewhere nor destroys a group it never joined.

        The per-hop groups hold the sorted union of the two adjacent stages' ranks --
        because the scatter/gather boundary transfer P2P-sends its chunks between ranks
        of both stages and NCCL runs coalesced P2P on the group's collective
        communicator, which every member would then have to issue in the same order.
        """
        tp_group = None
        for stage in sorted(plan.stages, key=lambda stage: (stage.replica_id, stage.stage_id)):
            group = self._new_group(stage.tp_members)
            if self.rank in stage.tp_members:
                tp_group = group
        boundaries: dict[tuple[int, ...], object] = {}
        for hop in boundary_hops(plan):
            group = self._new_group(hop)
            if self.rank in hop:
                boundaries[hop] = group
        executors = self._new_group(plan.active_ranks)
        self.tp_group = tp_group
        self.boundary_groups = boundaries
        self.executor_group = executors if self.is_training_rank(plan) else None

    def _new_group(self, plan_ranks):
        """One training group, named by plan ranks and built from this epoch's torch ranks."""
        import torch.distributed as dist

        return dist.new_group(
            ranks=[self.torch_rank(rank) for rank in plan_ranks],
            backend=self.training_backend,
            timeout=GROUP_TIMEOUT,
        )

    def destroy_training_groups(self) -> None:
        """Release the current training groups in unison.

        ``boundary_groups`` is built in sorted-hop order, so every member releases it at
        the same point in the sequence.
        """
        import torch.distributed as dist

        for group in (self.tp_group, *self.boundary_groups.values(), self.executor_group):
            if group is not None:
                dist.destroy_process_group(group)
        self.tp_group = None
        self.boundary_groups = {}
        self.executor_group = None

    # --- the iteration ----------------------------------------------------------------

    def training_step(self):
        """Advance one iteration through the plan's assignment-driven runtime.

        A rank the plan places on no stage holds no run and returns without touching any
        training collective -- it has permanently left the training path.
        """
        if self.training_run is None:
            return None
        return self.training_run.step()

    def commit_checkpoint(self, plan: ExecutionPlan) -> None:
        """Safe-point step 2: gather every shard into the one atomic checkpoint.

        Run at the end of *every* iteration, because a killed rank cannot contribute
        afterwards: the checkpoint that has to hold the dead rank's shards is the one
        written while it was still alive. Every process in the world joins the gather,
        contributing nothing when it holds no state, so the union of the plan's stages
        and replicas is the whole model.
        """
        from .recovery import commit_checkpoint

        commit_checkpoint(
            self.training_run,
            self.checkpoint_path,
            plan=plan,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            group=self.world_group,
            # The lowest active rank writes: every rank derives it from the same plan,
            # and unlike a fixed rank 0 it is always one that actually holds state, so
            # the file cannot record a cursor from a rank that has nothing to commit.
            writer=self.rank == plan.active_ranks[0],
        )

    def _recover_state(self, plan: ExecutionPlan, previous: ExecutionPlan) -> None:
        """Safe-point step 7: run the single recovery path and keep its state digest.

        The dead rank contributes nothing, so its shards genuinely have to be found on a
        healthy peer replica or in the checkpoint. Which layers this rank's stage owns,
        at which TP degree, and which micro-batches it executes are all read from
        ``plan``; ``previous`` says only what changed, so nothing that did not change is
        moved.
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
            world_group=self.world_group,
            tp_group=self.tp_group,
            executor_group=self.executor_group,
            boundary_groups=self.boundary_groups,
            device=self.device,
        )

    def agree(self, reason: StopReason | None = None, **digests: str) -> None:
        """Reach a unanimous decision over the current world group.

        Every rank contributes the stop reason it observed locally (or ``None``) plus
        the digests it wants checked. Any reason reported anywhere, and any digest the
        ranks do not all agree on, raises the *same* :class:`ConsistentStop` on every
        rank -- derived from the gathered list, never from the local view, so two ranks
        can never stop for two different reasons. A locally observed fault therefore
        never leaves one rank raising while the others block on a collective it has
        already left.
        """
        import torch.distributed as dist

        local = (
            None if reason is None else (reason.code, reason.message),
            tuple(sorted(digests.items())),
        )
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.world_group)

        for index, (reported, _) in enumerate(gathered):
            if reported is not None:
                raise ConsistentStop(
                    StopReason(reported[0], f"rank {self.members[index]}: {reported[1]}")
                )
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
        lost: tuple[int, ...],
        *,
        next_step: int,
    ) -> tuple[ExecutionPlan, tuple[int, ...]]:
        """Run the strict fail-stop safe point and return the new plan.

        Steps 1-2 (complete the iteration, commit the checkpoint) are the caller's, done
        while the rank about to die was still alive; steps 3-4 are :meth:`observe`,
        which handed back ``lost``. Steps 5-8 run here in the mandated order, with one
        addition a real kill forces: the world group is dissolved and re-formed over the
        survivors (6a/6b) before anything is agreed, because the old world contains a
        dead process and no collective on it would ever return. Step 9 (resume) is the
        caller's next loop turn. Raises :class:`ConsistentStop` -- on every surviving
        rank, for the same reason -- instead of returning a plan when any stop condition
        is met.
        """
        failed = tuple(sorted(set(failed_ranks) | set(lost)))  # 4. mark failed
        survivors = tuple(rank for rank in self.members if rank not in set(lost))

        new_plan = None
        reason = None
        # 5. TP->PP->DP replan (one new version). ``build_plan`` is deterministic in its
        # inputs and normalizes the failed ranks itself, so every rank that reaches this
        # point produces a byte-identical plan and therefore an identical digest.
        try:
            new_plan = build_plan(
                config,
                step=next_step,
                version=plan.version + 1,
                failed_ranks=failed,
                previous=plan,
                memory_budget=self.memory_budget,
                vocab_size=self.vocab_size,
                sequence_length=self.sequence_length,
            )
        except InfeasiblePlan as error:
            reason = StopReason(error.reason.code, error.reason.message)

        self.dissolve_world()  # 6a. release every group, the dead rank's world included
        self.form_world(survivors, epoch=self.epoch + 1)  # 6b. re-form over the survivors
        # Agree before anything is published: a plan no rank can execute, or one the
        # ranks disagree on, must never reach group construction. Building groups from
        # disagreeing plans is itself a deadlock, so this check cannot wait for step 8.
        self.agree(reason, plan="" if new_plan is None else new_plan.digest)

        try:
            self.build_training_groups(new_plan)  # 6c. build the new training groups
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
        """Release the training groups, then tear down the world.

        Leaves no residual process group behind.
        """
        import torch.distributed as dist

        self.destroy_training_groups()
        dist.barrier(group=self.world_group)
        dist.destroy_process_group()
        self.world_group = None


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
