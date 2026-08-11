"""Job creation, race-free claiming (SELECT FOR UPDATE SKIP LOCKED), fenced writes, and
stale-job reclaim (including the max-attempts park).

Needs a real Postgres: SKIP LOCKED semantics cannot be proven against a mock.
"""

import asyncio
from datetime import timedelta

import pytest

from conftest import FIRMWARE_TARGET
from uadclaw.jobs import (
    JobValidationError,
    advance_stage,
    claim_job,
    create_job,
    mark_running,
    needs_scratch,
    next_stage,
    reclaim_stale_jobs,
)
from uadclaw.models import PIPELINE_STAGES, JobState

MAX_ATTEMPTS = 5


async def test_create_job_rejects_unknown_kind(db_session_factory):
    async with db_session_factory() as session, session.begin():
        with pytest.raises(JobValidationError):
            await create_job(session, kind="not-a-real-kind")


async def test_create_job_starts_queued_with_no_stage(db_session_factory):
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)

    assert job.state == JobState.QUEUED
    assert job.stage is None
    assert job.attempt == 0


def test_next_stage_walks_the_pipeline_in_order():
    assert next_stage(None) == PIPELINE_STAGES[0]
    assert next_stage(PIPELINE_STAGES[0]) == PIPELINE_STAGES[1]
    assert next_stage(PIPELINE_STAGES[-1]) is None


def test_needs_scratch_known_kind():
    assert needs_scratch("firmware_analysis") is True


def test_needs_scratch_rejects_unknown_kind():
    with pytest.raises(JobValidationError):
        needs_scratch("not-a-real-kind")


async def test_claim_job_is_race_free_under_concurrent_workers(db_session_factory):
    """Ten jobs, ten concurrent claimers. SKIP LOCKED must hand out exactly one job per
    claimer with no duplicates and nothing left over — a bare `UPDATE ... WHERE state =
    'queued' LIMIT 1` without row-level locking would double-claim under this concurrency."""
    async with db_session_factory() as session, session.begin():
        for _ in range(10):
            await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)

    async def _claim_one(worker_id: str):
        async with db_session_factory() as session, session.begin():
            return await claim_job(session, worker_id=worker_id)

    results = await asyncio.gather(*[_claim_one(f"w{i}") for i in range(10)])
    claimed_ids = [job.id for job in results if job is not None]

    assert len(claimed_ids) == 10
    assert len(set(claimed_ids)) == 10  # no job claimed twice

    async with db_session_factory() as session, session.begin():
        extra = await claim_job(session, worker_id="w-extra")
    assert extra is None  # nothing left to claim


async def test_fenced_write_is_rejected_under_the_wrong_identity(db_session_factory):
    """The core of M4: a write claiming the wrong worker_id, or the wrong attempt, must be
    rejected (return falsy / None) rather than silently applied — that's what lets a
    fenced-out worker's stale writes no-op instead of corrupting the current owner's run."""
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        claimed = await claim_job(session, worker_id="worker-a")
        real_attempt = claimed.attempt

    async with db_session_factory() as session, session.begin():
        wrong_worker = await mark_running(session, job.id, "worker-b", real_attempt)
    assert wrong_worker is False

    async with db_session_factory() as session, session.begin():
        wrong_attempt = await mark_running(session, job.id, "worker-a", real_attempt + 1)
    assert wrong_attempt is False

    async with db_session_factory() as session, session.begin():
        correct = await mark_running(session, job.id, "worker-a", real_attempt)
    assert correct is True


async def test_reclaim_stale_jobs_requeues_and_preserves_stage(db_session_factory):
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        claimed = await claim_job(session, worker_id="dead-worker")
        assert claimed.id == job.id
        await mark_running(session, claimed.id, "dead-worker", claimed.attempt)
        await advance_stage(session, claimed.id, "dead-worker", claimed.attempt, "acquire")
        # Simulate a heartbeat from long ago: the worker holding this job is presumed dead.
        claimed.heartbeat_at = claimed.heartbeat_at - timedelta(hours=1)

    async with db_session_factory() as session, session.begin():
        reclaimed = await reclaim_stale_jobs(
            session,
            stale_after=timedelta(seconds=1),
            worker_id="new-worker",
            max_attempts=MAX_ATTEMPTS,
        )

    assert len(reclaimed) == 1
    assert reclaimed[0].id == job.id
    assert reclaimed[0].state == JobState.QUEUED
    assert reclaimed[0].worker_id is None
    # Resume point survives the reclaim: stage stays "acquire", not reset to None.
    assert reclaimed[0].stage == "acquire"


async def test_reclaim_stale_jobs_leaves_fresh_heartbeats_alone(db_session_factory):
    async with db_session_factory() as session, session.begin():
        await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        claimed = await claim_job(session, worker_id="alive-worker")
        await mark_running(session, claimed.id, "alive-worker", claimed.attempt)

    async with db_session_factory() as session, session.begin():
        reclaimed = await reclaim_stale_jobs(
            session,
            stale_after=timedelta(hours=1),
            worker_id="new-worker",
            max_attempts=MAX_ATTEMPTS,
        )

    assert reclaimed == []


async def test_reclaim_stale_jobs_parks_exhausted_attempts_as_failed(db_session_factory):
    """M11: a job already at the attempt ceiling must be parked FAILED, not requeued —
    otherwise a job that reliably kills its worker comes back to the front of the FIFO
    queue on every reclaim, forever, ahead of every healthy job."""
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        job_id = job.id

    # Drive it through max_attempts claim/stale cycles.
    for i in range(MAX_ATTEMPTS):
        async with db_session_factory() as session, session.begin():
            claimed = await claim_job(session, worker_id=f"dead-worker-{i}")
            assert claimed.id == job_id
            assert claimed.attempt == i + 1
            await mark_running(session, job_id, f"dead-worker-{i}", claimed.attempt)
            claimed.heartbeat_at = claimed.heartbeat_at - timedelta(hours=1)

        async with db_session_factory() as session, session.begin():
            reclaimed = await reclaim_stale_jobs(
                session,
                stale_after=timedelta(seconds=1),
                worker_id="sweep",
                max_attempts=MAX_ATTEMPTS,
            )
        assert len(reclaimed) == 1
        if i + 1 < MAX_ATTEMPTS:
            assert reclaimed[0].state == JobState.QUEUED
        else:
            assert reclaimed[0].state == JobState.FAILED
            assert "exhausted" in reclaimed[0].failure_reason
            assert str(MAX_ATTEMPTS) in reclaimed[0].failure_reason

    async with db_session_factory() as session:
        from sqlalchemy import select

        from uadclaw.models import Job

        final = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()
    assert final.state == JobState.FAILED
    # Parked, not merely reclaimed: a subsequent claim must not pick it back up.
    async with db_session_factory() as session, session.begin():
        nothing_to_claim = await claim_job(session, worker_id="whoever-is-next")
    assert nothing_to_claim is None
