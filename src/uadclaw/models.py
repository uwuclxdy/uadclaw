"""Job, scratch-lease and stage-run tables.

The job substrate every later pipeline stage runs on top of. Kept deliberately thin: no
firmware-specific columns here, those land with tasks 3+.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from uadclaw.db import Base


class JobKind(enum.StrEnum):
    FIRMWARE_ANALYSIS = "firmware_analysis"


# Whether a job kind occupies the scratch lease. FIRMWARE_ANALYSIS unpacks firmware onto
# disk and needs the single-occupant lease for its whole life; classification (task 7) and
# corroboration (task 8) only touch the DB and the network, never scratch. The requirement
# lives on the kind, not on the worker loop, so the pool stays meaningful (able to run
# scratch-free kinds concurrently) the moment one of those kinds lands.
JOB_KIND_NEEDS_SCRATCH: dict[JobKind, bool] = {
    JobKind.FIRMWARE_ANALYSIS: True,
}


class JobState(enum.StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Terminal states a claim/reclaim sweep must never touch.
TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED})

# Ordered pipeline stages from docs/pipeline-design.md § Pipeline. A job's `stage` records
# the furthest point it reached; resuming restarts from that stage, never from the start.
PIPELINE_STAGES: tuple[str, ...] = (
    "acquire",
    "unpack",
    "extract_facts",
    "corpus_graph",
    "filter",
    "rule_ladder",
    "llm",
    "corroborate",
    "triage",
    "branch",
)

# Bound the log tail kept on the job row; older lines fall off rather than growing the row
# without limit over a long-running or oft-retried job.
LOG_TAIL_MAX_CHARS = 8_000


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[JobState] = mapped_column(
        SAEnum(JobState, name="job_state", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=JobState.QUEUED,
    )
    # Furthest pipeline stage reached; NULL means "not started yet". Resume restarts here.
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt: Mapped[int] = mapped_column(nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    log_tail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Refreshed by the running worker; a job stuck CLAIMED/RUNNING past the stale threshold
    # with no heartbeat is reclaimable (its dead worker's crash is the whole point).
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set once this job's scratch artifacts are gone from disk — on success (retention:
    # facts-only) or once the failure-ceiling sweep evicts it. NULL means artifacts, if any,
    # are still on disk.
    artifacts_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_jobs_state_created_at", "state", "created_at"),
        Index("ix_jobs_state_heartbeat_at", "state", "heartbeat_at"),
    )


class JobStageRun(Base):
    """One row per (job, stage) attempt. Exists because per-stage duration cannot be
    derived from the four whole-job timestamps on `Job` — those cover the job, not each
    stage inside it."""

    __tablename__ = "job_stage_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # succeeded | failed

    __table_args__ = (Index("ix_job_stage_runs_job_id", "job_id"),)


class ScratchLease(Base):
    """Single-row table: scratch is a singleton resource, so the lease is a singleton row
    (id fixed at 1) rather than a set of rows to query for "the" holder."""

    __tablename__ = "scratch_lease"

    id: Mapped[int] = mapped_column(primary_key=True)
    holder_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    holder_worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (CheckConstraint("id = 1", name="ck_scratch_lease_singleton"),)


class ScratchLeaseEvent(Base):
    """Append-only history the mutable `ScratchLease` row cannot answer questions from once
    overwritten: occupancy over time and per-job wait duration both need every past
    acquire/release/reclaim, not just the current holder."""

    __tablename__ = "scratch_lease_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # requested | acquired | released | reclaimed
    event: Mapped[str] = mapped_column(String(16), nullable=False)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_scratch_lease_events_job_id", "job_id"),
        Index("ix_scratch_lease_events_at", "at"),
    )
