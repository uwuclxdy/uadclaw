"""Job lifecycle: creation, race-free claiming, fenced stage progress, completion.

Claiming uses `SELECT ... FOR UPDATE SKIP LOCKED`: the established Postgres job-queue
pattern (used by e.g. Oban, `pg-boss`, and the pattern the `SKIP LOCKED` feature — added in
PG 9.5 specifically for queue workloads — exists for). Each worker locks and takes the
oldest queued row nobody else already has locked, rather than blocking behind it or racing
a bare `UPDATE ... WHERE state = 'queued' LIMIT 1` (which has no atomic "claim one row"
semantics and would double-claim under concurrency).

Every write past the claim is fenced on `(id, worker_id, attempt)`: a worker that got
reclaimed (falsely, mid-lease-wait, or genuinely after a crash) must stop touching the job
the instant a newer attempt exists, rather than racing that new owner. See `_fenced_job`.
"""

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.models import (
    JOB_KIND_NEEDS_SCRATCH,
    LOG_TAIL_MAX_CHARS,
    PIPELINE_STAGES,
    Job,
    JobKind,
    JobStageRun,
    JobState,
)

logger = logging.getLogger(__name__)


class JobValidationError(ValueError):
    """A bad job input reached the service boundary. Distinct from a bug in this module so
    callers (API handlers) can map it to 4xx instead of a 500."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def first_stage() -> str:
    return PIPELINE_STAGES[0]


def next_stage(current: str | None) -> str | None:
    """The stage after `current`, or the first stage if nothing has run yet, or None once
    the last stage has completed (the job is done)."""
    if current is None:
        return PIPELINE_STAGES[0]
    try:
        idx = PIPELINE_STAGES.index(current)
    except ValueError as exc:
        raise JobValidationError(
            f"next_stage: stage {current!r} is not one of {PIPELINE_STAGES}; "
            "fix the caller to pass a valid pipeline stage name"
        ) from exc
    if idx + 1 >= len(PIPELINE_STAGES):
        return None
    return PIPELINE_STAGES[idx + 1]


def needs_scratch(kind: str) -> bool:
    """Whether this job kind occupies the single-occupant scratch lease. Raises
    `JobValidationError` for an unrecognized kind rather than silently defaulting either
    way — a wrong answer here means either false lease contention or two jobs sharing disk."""
    try:
        return JOB_KIND_NEEDS_SCRATCH[JobKind(kind)]
    except ValueError as exc:
        valid = ", ".join(k.value for k in JobKind)
        raise JobValidationError(
            f"needs_scratch: kind={kind!r} is not a recognized job kind; use one of {valid}"
        ) from exc


async def create_job(session: AsyncSession, *, kind: str) -> Job:
    """Validate `kind` at the boundary and insert a new queued job. Raises
    `JobValidationError` (not a bug in this code) when `kind` isn't a known job kind."""
    try:
        JobKind(kind)
    except ValueError as exc:
        valid = ", ".join(k.value for k in JobKind)
        raise JobValidationError(
            f"create_job: kind={kind!r} is not a recognized job kind; use one of {valid}"
        ) from exc

    job = Job(
        id=uuid.uuid4(),
        kind=kind,
        state=JobState.QUEUED,
        stage=None,
        attempt=0,
        log_tail="",
        created_at=_utcnow(),
    )
    session.add(job)
    await session.flush()
    return job


async def claim_job(session: AsyncSession, *, worker_id: str) -> Job | None:
    """Atomically take the oldest queued job, or the oldest job just reclaimed back to
    QUEUED by `reclaim_stale_jobs`. Returns None when nothing is waiting. The returned
    job's `.attempt` (already incremented) is the identity every subsequent fenced write
    for this run must be made under.

    Caller must run this inside `async with session.begin(): ...` — the row lock held by
    `FOR UPDATE SKIP LOCKED` and the following state flip must commit as one unit, or a
    second worker's SKIP LOCKED could see the same row as "still queued" before this one's
    UPDATE lands.
    """
    stmt = (
        select(Job)
        .where(Job.state == JobState.QUEUED)
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = (await session.execute(stmt)).scalar_one_or_none()
    if job is None:
        return None

    now = _utcnow()
    job.state = JobState.CLAIMED
    job.worker_id = worker_id
    job.claimed_at = now
    job.heartbeat_at = now
    job.attempt += 1
    return job


async def _fenced_job(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int
) -> Job | None:
    """Load the job row FOR UPDATE, but only if it is still owned by the identity
    (`worker_id`, `attempt`) the caller claimed it under. None means ownership moved on —
    reclaimed out from under this worker, or claimed fresh by a newer attempt — and the
    caller must abort locally rather than write anything: a stale caller repeating this
    call after that point can never see the newer identity's writes, only ever a
    consistent "not mine anymore"."""
    stmt = (
        select(Job)
        .where(Job.id == job_id, Job.worker_id == worker_id, Job.attempt == attempt)
        .with_for_update()
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def mark_running(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int
) -> bool:
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return False
    now = _utcnow()
    job.state = JobState.RUNNING
    if job.started_at is None:
        job.started_at = now
    job.heartbeat_at = now
    return True


async def heartbeat_job(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int
) -> bool:
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return False
    job.heartbeat_at = _utcnow()
    return True


def append_log(job: Job, text: str) -> None:
    """Append to the bounded log tail, keeping only the most recent
    `LOG_TAIL_MAX_CHARS` characters."""
    combined = f"{job.log_tail}{text}\n" if job.log_tail else f"{text}\n"
    job.log_tail = combined[-LOG_TAIL_MAX_CHARS:]


async def record_stage_started(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int, stage: str
) -> int | None:
    """None (rather than a run id) when the job is no longer owned by this identity — the
    caller must not run the stage handler at all in that case, let alone record it."""
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return None
    run = JobStageRun(job_id=job_id, stage=stage, started_at=_utcnow())
    session.add(run)
    await session.flush()
    return run.id


async def record_stage_finished(session: AsyncSession, run_id: int, *, outcome: str) -> None:
    run = await session.get(JobStageRun, run_id)
    if run is not None:
        run.finished_at = _utcnow()
        run.outcome = outcome


async def advance_stage(
    session: AsyncSession,
    job_id: uuid.UUID,
    worker_id: str,
    attempt: int,
    completed_stage: str,
    *,
    log_line: str | None = None,
) -> bool:
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return False
    job.stage = completed_stage
    job.heartbeat_at = _utcnow()
    if log_line:
        append_log(job, log_line)
    return True


async def complete_job(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int
) -> Job | None:
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return None
    job.state = JobState.SUCCEEDED
    job.finished_at = _utcnow()
    return job


async def fail_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    worker_id: str,
    attempt: int,
    *,
    reason: str,
    log_line: str | None = None,
) -> Job | None:
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return None
    job.state = JobState.FAILED
    job.finished_at = _utcnow()
    job.failure_reason = reason
    if log_line:
        append_log(job, log_line)
    return job


async def mark_artifacts_deleted(
    session: AsyncSession, job_id: uuid.UUID, worker_id: str, attempt: int
) -> bool:
    """Informational only — the retention ceiling sweep no longer keys off this column,
    it scans `scratch_root` directly (see `scratch.enforce_retention_ceiling`), so a
    missed or fenced-out write here does not leave anything unbounded."""
    job = await _fenced_job(session, job_id, worker_id, attempt)
    if job is None:
        return False
    job.artifacts_deleted_at = _utcnow()
    return True


async def reclaim_stale_jobs(
    session: AsyncSession, *, stale_after: timedelta, worker_id: str, max_attempts: int
) -> Sequence[Job]:
    """Requeue jobs stuck CLAIMED/RUNNING whose heartbeat went stale — the worker holding
    them is presumed dead. `stage` is left untouched so the next claim resumes from the
    last completed stage rather than the beginning. Reclaiming is explicit and logged,
    never silent.

    A job already at `max_attempts` is parked FAILED instead of requeued: claim order is
    FIFO by `created_at`, so a job that reliably kills its worker would otherwise come back
    to the front of the queue on every reclaim, forever, ahead of every healthy job.
    """
    cutoff = _utcnow() - stale_after
    stmt = (
        select(Job)
        .where(
            Job.state.in_([JobState.CLAIMED, JobState.RUNNING]),
            Job.heartbeat_at < cutoff,
        )
        .with_for_update(skip_locked=True)
    )
    stale_jobs = list((await session.execute(stmt)).scalars())
    now = _utcnow()
    reclaimed: list[Job] = []
    for job in stale_jobs:
        staleness = now - job.heartbeat_at if job.heartbeat_at else None
        if job.attempt >= max_attempts:
            logger.warning(
                "parking exhausted job: job_id=%s previous_worker=%s previous_state=%s "
                "stage=%s attempt=%d max_attempts=%d stale_for=%s reclaiming_worker=%s",
                job.id,
                job.worker_id,
                job.state.value,
                job.stage,
                job.attempt,
                max_attempts,
                staleness,
                worker_id,
            )
            job.state = JobState.FAILED
            job.finished_at = now
            job.failure_reason = (
                f"exhausted after {job.attempt} attempt(s) (max {max_attempts}); last held "
                f"by worker={job.worker_id}, stalled at stage={job.stage}"
            )
        else:
            logger.warning(
                "reclaiming stale job: job_id=%s previous_worker=%s previous_state=%s "
                "stage=%s stale_for=%s reclaiming_worker=%s",
                job.id,
                job.worker_id,
                job.state.value,
                job.stage,
                staleness,
                worker_id,
            )
            job.state = JobState.QUEUED
        job.worker_id = None
        job.heartbeat_at = None
        reclaimed.append(job)
    return reclaimed
