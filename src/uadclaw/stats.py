"""Usage/utilization aggregates for tuning the worker pool size.

Derived from the timestamps already on `Job`/`JobStageRun` wherever possible. The one
exception is `ScratchLeaseEvent`: lease occupancy and per-job wait time need history the
mutable, overwritten-in-place `ScratchLease` singleton row cannot answer once a later
acquire/release has clobbered it, which is exactly why that table exists (see
`models.py`). No sampling table, no metrics framework: `queue_depth_now` is a live gauge —
"queue depth over time" is a dashboard polling this endpoint, not a second table.

The window always ends at "now" and looks back `lookback`, never at the last time
something happened to finish: a wedged system (nothing finishing, nothing releasing) is
exactly the case this endpoint exists to surface, and a window anchored on the last
`finished_at` reads 100% healthy while wedged, which is the one reading that must never
happen. Still-open intervals (a lease held right now, a job still waiting on it, a job
still running) are closed at "now" for the same reason.
"""

import uuid
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.models import Job, JobStageRun, JobState, ScratchLease, ScratchLeaseEvent

# Lease events that close an open "acquired" interval — a "released" is the normal case,
# a "reclaimed" is the previous holder's interval being force-closed by the new claimant;
# dropping the latter silently understates occupancy for exactly the crashed-holder case
# this endpoint most needs to surface.
_LEASE_CLOSING_EVENTS = frozenset({"released", "reclaimed"})


class StageDurationStats(BaseModel):
    stage: str
    completed_count: int
    avg_seconds: float
    max_seconds: float


class WorkerUtilization(BaseModel):
    worker_id: str
    jobs_succeeded: int
    jobs_failed: int
    busy_seconds: float


class LeaseStats(BaseModel):
    total_acquisitions: int
    total_occupied_seconds: float
    occupancy_pct: float
    avg_wait_seconds: float | None
    max_wait_seconds: float | None
    current_holder_job_id: uuid.UUID | None


class StatsResponse(BaseModel):
    generated_at: datetime
    queue_depth_now: int
    window_start: datetime
    window_end: datetime
    window_seconds: float
    stage_durations: list[StageDurationStats]
    worker_utilization: list[WorkerUtilization]
    lease: LeaseStats


def _epoch_seconds(delta_expr):
    return func.extract("epoch", delta_expr)


async def _stage_durations(
    session: AsyncSession, window_start: datetime
) -> list[StageDurationStats]:
    duration = _epoch_seconds(JobStageRun.finished_at - JobStageRun.started_at)
    stmt = (
        select(
            JobStageRun.stage,
            func.count().label("completed_count"),
            func.avg(duration).label("avg_seconds"),
            func.max(duration).label("max_seconds"),
        )
        # Only successful runs: a stage that failed after 1s of a normal 300s run would
        # otherwise drag the average down and read as "this stage got fast", not "this
        # stage is failing fast".
        .where(JobStageRun.outcome == "succeeded", JobStageRun.started_at >= window_start)
        .group_by(JobStageRun.stage)
        .order_by(JobStageRun.stage)
    )
    rows = (await session.execute(stmt)).all()
    return [
        StageDurationStats(
            stage=stage,
            completed_count=count,
            avg_seconds=float(avg_seconds),
            max_seconds=float(max_seconds),
        )
        for stage, count, avg_seconds, max_seconds in rows
    ]


async def _worker_utilization(
    session: AsyncSession, now: datetime, window_start: datetime
) -> list[WorkerUtilization]:
    # A still-RUNNING job's elapsed time counts as busy too (coalesce to `now`) — a worker
    # six hours into a job that hasn't finished is not idle, and excluding it is exactly
    # the kind of reading that hides a wedge.
    duration = _epoch_seconds(func.coalesce(Job.finished_at, now) - Job.started_at)
    stmt = (
        select(
            Job.worker_id,
            func.count().filter(Job.state == JobState.SUCCEEDED).label("jobs_succeeded"),
            func.count().filter(Job.state == JobState.FAILED).label("jobs_failed"),
            func.sum(duration).label("busy_seconds"),
        )
        .where(
            Job.worker_id.is_not(None),
            Job.started_at.is_not(None),
            Job.started_at >= window_start,
        )
        .group_by(Job.worker_id)
        .order_by(Job.worker_id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        WorkerUtilization(
            worker_id=worker_id,
            jobs_succeeded=jobs_succeeded,
            jobs_failed=jobs_failed,
            busy_seconds=float(busy_seconds or 0.0),
        )
        for worker_id, jobs_succeeded, jobs_failed, busy_seconds in rows
    ]


async def _lease_stats(
    session: AsyncSession, now: datetime, window_start: datetime, window_seconds: float
) -> LeaseStats:
    events = (
        await session.execute(
            select(ScratchLeaseEvent)
            .where(ScratchLeaseEvent.at >= window_start)
            .order_by(ScratchLeaseEvent.at)
        )
    ).scalars()

    requested_at: dict[uuid.UUID, datetime] = {}
    open_acquired_at: dict[uuid.UUID, datetime] = {}
    wait_seconds: list[float] = []
    occupied_seconds: list[float] = []
    total_acquisitions = 0

    for event in events:
        if event.job_id is None:
            continue
        if event.event == "requested":
            requested_at[event.job_id] = event.at
        elif event.event == "acquired":
            total_acquisitions += 1
            open_acquired_at[event.job_id] = event.at
            request_time = requested_at.pop(event.job_id, None)
            if request_time is not None:
                wait_seconds.append((event.at - request_time).total_seconds())
        elif event.event in _LEASE_CLOSING_EVENTS:
            acquire_time = open_acquired_at.pop(event.job_id, None)
            if acquire_time is not None:
                occupied_seconds.append((event.at - acquire_time).total_seconds())

    lease = (
        await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
    ).scalar_one_or_none()

    # A lease acquired BEFORE the window never produced an "acquired" event inside the
    # `at >= window_start` filter above, so the loop never opened an interval for it —
    # without this, a lease held continuously since before the window reports zero
    # occupancy while `current_holder_job_id` simultaneously names its holder, which is
    # exactly backwards for the one number that exists to surface a starved lease. Seed the
    # open interval at the later of when it was actually acquired or the window's own
    # start, so a lease acquired mid-window still uses its real acquire time.
    if (
        lease is not None
        and lease.holder_job_id is not None
        and lease.holder_job_id not in open_acquired_at
    ):
        seed_at = lease.acquired_at if lease.acquired_at is not None else window_start
        open_acquired_at[lease.holder_job_id] = max(seed_at, window_start)

    # Close whatever is still open at "now": a lease held right now, or a job still
    # waiting on it. Otherwise both are invisible until they close, which is precisely
    # the wedge this endpoint exists to catch.
    for acquire_time in open_acquired_at.values():
        occupied_seconds.append((now - acquire_time).total_seconds())
    for request_time in requested_at.values():
        wait_seconds.append((now - request_time).total_seconds())

    total_occupied = sum(occupied_seconds)

    return LeaseStats(
        total_acquisitions=total_acquisitions,
        total_occupied_seconds=total_occupied,
        occupancy_pct=(total_occupied / window_seconds * 100) if window_seconds > 0 else 0.0,
        avg_wait_seconds=(sum(wait_seconds) / len(wait_seconds)) if wait_seconds else None,
        max_wait_seconds=max(wait_seconds) if wait_seconds else None,
        current_holder_job_id=lease.holder_job_id if lease else None,
    )


async def compute_stats(session: AsyncSession, *, lookback_seconds: float) -> StatsResponse:
    now = datetime.now(UTC)
    window_start = now - timedelta(seconds=lookback_seconds)
    window_seconds = lookback_seconds

    queue_depth_now = (
        await session.execute(select(func.count()).where(Job.state == JobState.QUEUED))
    ).scalar_one()

    return StatsResponse(
        generated_at=now,
        queue_depth_now=queue_depth_now,
        window_start=window_start,
        window_end=now,
        window_seconds=window_seconds,
        stage_durations=await _stage_durations(session, window_start),
        worker_utilization=await _worker_utilization(session, now, window_start),
        lease=await _lease_stats(session, now, window_start, window_seconds),
    )
