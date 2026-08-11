"""Scratch lease: single occupant, heartbeat + stale-expiry reclaim.

Acquire/release are exercised directly against the DB — no worker loop involved — so "a
second job waits for the lease" is a property of `uadclaw.scratch` itself, not a side
effect of running the pool at size 1.
"""

import asyncio
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from uadclaw.jobs import create_job
from uadclaw.models import ScratchLease
from uadclaw.scratch import (
    ScratchLeaseTimeout,
    _try_acquire_once,
    acquire_scratch_lease,
    heartbeat_scratch_lease,
    release_scratch_lease,
)

STALE_AFTER = timedelta(seconds=30)


async def _make_job(db_session_factory) -> uuid.UUID:
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis")
        return job.id


@pytest.fixture
async def two_jobs(db_session_factory):
    return await _make_job(db_session_factory), await _make_job(db_session_factory)


async def test_second_acquire_waits_while_first_holds_the_lease(db_session_factory, two_jobs):
    job_a, job_b = two_jobs

    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=STALE_AFTER
    )

    b_acquired = asyncio.Event()

    async def _acquire_b():
        await acquire_scratch_lease(
            db_session_factory,
            job_id=job_b,
            worker_id="worker-b",
            stale_after=STALE_AFTER,
            poll_interval=0.05,
        )
        b_acquired.set()

    task = asyncio.create_task(_acquire_b())
    try:
        # B must still be waiting a good while after A already holds the lease — not "won
        # the race by chance", genuinely blocked on A.
        await asyncio.sleep(0.3)
        assert not b_acquired.is_set()

        await release_scratch_lease(db_session_factory, job_id=job_a, worker_id="worker-a")
        await asyncio.wait_for(b_acquired.wait(), timeout=2)
        assert b_acquired.is_set()
    finally:
        await task


async def test_acquire_times_out_rather_than_waiting_forever_when_asked(
    db_session_factory, two_jobs
):
    job_a, job_b = two_jobs
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=STALE_AFTER
    )

    with pytest.raises(ScratchLeaseTimeout):
        await acquire_scratch_lease(
            db_session_factory,
            job_id=job_b,
            worker_id="worker-b",
            stale_after=STALE_AFTER,
            poll_interval=0.05,
            timeout=0.2,
        )


async def test_stale_lease_is_reclaimed_and_logged(db_session_factory, two_jobs, caplog):
    job_a, job_b = two_jobs
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=STALE_AFTER
    )

    # Simulate a crashed worker: its heartbeat stopped, so the lease is now stale relative
    # to a much shorter threshold.
    async with db_session_factory() as session, session.begin():
        lease = (
            await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
        ).scalar_one()
        lease.heartbeat_at = lease.heartbeat_at - timedelta(seconds=10)

    with caplog.at_level("WARNING"):
        await acquire_scratch_lease(
            db_session_factory,
            job_id=job_b,
            worker_id="worker-b",
            stale_after=timedelta(seconds=1),
        )

    reclaim_logs = [r for r in caplog.records if "reclaiming stale scratch lease" in r.message]
    assert len(reclaim_logs) == 1
    assert str(job_a) in reclaim_logs[0].message
    assert "worker-a" in reclaim_logs[0].message


async def test_fresh_lease_is_not_reclaimed(db_session_factory, two_jobs):
    job_a, job_b = two_jobs
    async with db_session_factory() as session, session.begin():
        acquired = await _try_acquire_once(
            session, job_id=job_a, worker_id="worker-a", stale_after=STALE_AFTER
        )
    assert acquired is True

    async with db_session_factory() as session, session.begin():
        acquired_b = await _try_acquire_once(
            session, job_id=job_b, worker_id="worker-b", stale_after=STALE_AFTER
        )
    assert acquired_b is False


async def test_heartbeat_keeps_a_held_lease_from_going_stale(db_session_factory, two_jobs):
    job_a, job_b = two_jobs
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=timedelta(seconds=1)
    )
    async with db_session_factory() as session, session.begin():
        lease = (
            await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
        ).scalar_one()
        lease.heartbeat_at = lease.heartbeat_at - timedelta(milliseconds=900)

    await heartbeat_scratch_lease(db_session_factory, job_id=job_a, worker_id="worker-a")

    async with db_session_factory() as session, session.begin():
        acquired_b = await _try_acquire_once(
            session, job_id=job_b, worker_id="worker-b", stale_after=timedelta(seconds=1)
        )
    assert acquired_b is False  # heartbeat refreshed it, so it is not stale yet


async def test_release_by_non_holder_is_a_noop(db_session_factory, two_jobs):
    job_a, job_b = two_jobs
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=STALE_AFTER
    )

    await release_scratch_lease(db_session_factory, job_id=job_b, worker_id="worker-b")

    async with db_session_factory() as session, session.begin():
        acquired = await _try_acquire_once(
            session, job_id=job_b, worker_id="worker-b", stale_after=STALE_AFTER
        )
    assert acquired is False  # A still holds it; B's bogus release did nothing


async def test_release_by_same_job_id_wrong_worker_is_a_noop(db_session_factory, two_jobs):
    """M4: after a job is fenced out and re-claimed under a new worker id, a release call
    from the OLD (fenced-out) worker must not tear down the NEW worker's live hold — same
    job_id, different worker_id, so the release has to check both, not just the job_id."""
    job_a, _job_b = two_jobs
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a-attempt-1", stale_after=STALE_AFTER
    )
    # Simulate a job-level reclaim + re-claim under a new worker identity, same job_id,
    # WITHOUT the lease itself changing hands (the lease's own staleness logic is covered
    # elsewhere; this test is specifically about the release-time identity check).
    await release_scratch_lease(db_session_factory, job_id=job_a, worker_id="worker-a-attempt-1")
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a-attempt-2", stale_after=STALE_AFTER
    )

    # The stale attempt-1 identity tries to release what it thinks is still its lease.
    await release_scratch_lease(db_session_factory, job_id=job_a, worker_id="worker-a-attempt-1")

    async with db_session_factory() as session:
        lease = (
            await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
        ).scalar_one()
    assert lease.holder_job_id == job_a
    assert lease.holder_worker_id == "worker-a-attempt-2"  # untouched by the stale release


async def test_acquire_is_reentrant_for_the_current_holder(db_session_factory, two_jobs):
    """M7: a job that already holds the lease must be able to re-acquire it immediately —
    refreshing the heartbeat — rather than waiting out the full stale window for a lease
    it already owns, which would block every other job behind it for nothing.

    Deliberately uses a LONG (multi-second) stale window, not a tiny one: with a tiny
    window, a non-reentrant implementation can still "succeed" fast by falling through to
    the ordinary staleness check and reclaiming from itself once real test overhead alone
    exceeds the tiny threshold — which looks the same as instant re-entrancy from the
    outside (both finish quickly) and would let a broken re-entrancy check pass this test
    for the wrong reason. A window long enough that self-reclaim could not possibly fire
    within the measured budget only leaves one way to finish fast: genuine re-entrancy.
    """
    job_a, job_b = two_jobs
    long_stale_after = timedelta(seconds=30)
    await acquire_scratch_lease(
        db_session_factory, job_id=job_a, worker_id="worker-a", stale_after=long_stale_after
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(
        acquire_scratch_lease(
            db_session_factory,
            job_id=job_a,
            worker_id="worker-a",
            stale_after=long_stale_after,
            poll_interval=0.01,
        ),
        timeout=1.0,
    )
    elapsed = loop.time() - started
    # Nowhere near the 30s stale window — a non-reentrant path could only finish this fast
    # by self-reclaiming, which cannot happen before 30s have passed.
    assert elapsed < 1.0

    # And it must still be genuinely exclusive: B still cannot get in.
    async with db_session_factory() as session, session.begin():
        acquired_b = await _try_acquire_once(
            session, job_id=job_b, worker_id="worker-b", stale_after=long_stale_after
        )
    assert acquired_b is False
