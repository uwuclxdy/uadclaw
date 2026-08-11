"""Usage/utilization stats endpoint: real numbers out of real job runs, behind auth like
every other route.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from uadclaw.jobs import create_job
from uadclaw.models import Job, JobState, ScratchLease, ScratchLeaseEvent
from uadclaw.settings import get_settings
from uadclaw.worker import default_stage_handlers


async def test_stats_endpoint_reports_real_numbers_from_real_job_runs(
    db_env, db_session_factory, run_pool_until, client
):
    """Regression: this used to run three separate `run_pool_until` calls, each with its
    own handler set, on the assumption that a call watching only its own target job's
    terminal state would touch nothing else. It does not — `_worker_slot` keeps claiming
    whatever is next in the FIFO queue until `shutdown_event` is observed, and the harness
    only sets that after its next 0.05s poll tick. All three jobs are created QUEUED up
    front, so under enough scheduling delay (reliably reproducible under the suite's
    default `-n auto`, ~1 run in 15) the single slot claims and runs all three back to back
    inside the FIRST call, using its `ok` handlers throughout — the "failing" job never
    sees the handler meant to fail it, and ends up SUCCEEDED. Confirmed via a full job-row
    dump: `attempt=1` (never reclaimed/retried) and all three jobs sharing one worker_id,
    claimed within milliseconds of each other, ruled out a retry/reclaim explanation.

    Fixed by keying the failing behavior on the stable job id the handler is given, not on
    which pool invocation happens to be active — the one thing that stays true regardless
    of claim order, timing, or how many jobs one pool run eats through in a single pass.
    """
    settings = get_settings()

    job_ids = []
    async with db_session_factory() as session, session.begin():
        for _ in range(2):
            job = await create_job(session, kind="firmware_analysis")
            job_ids.append(job.id)
        failing_job = await create_job(session, kind="firmware_analysis")
        job_ids.append(failing_job.id)
    failing_job_id = job_ids[2]

    async def _slow_acquire(ctx):
        (ctx.scratch_dir / "firmware.zip").write_bytes(b"\0" * 64)
        await asyncio.sleep(0.05)

    async def _boom_only_for_the_failing_job(ctx):
        if ctx.job_id == failing_job_id:
            raise RuntimeError("synthetic failure for stats coverage")

    handlers = default_stage_handlers()
    handlers["acquire"] = _slow_acquire
    handlers["unpack"] = _boom_only_for_the_failing_job

    async def _all_terminal() -> bool:
        async with db_session_factory() as session:
            rows = (await session.execute(select(Job).where(Job.id.in_(job_ids)))).scalars()
            return all(j.state in (JobState.SUCCEEDED, JobState.FAILED) for j in rows)

    await run_pool_until(db_session_factory, settings, handlers, _all_terminal)

    async with db_session_factory() as session:
        rows = (await session.execute(select(Job).where(Job.id.in_(job_ids)))).scalars()
        states = {j.id: j.state for j in rows}
    assert states[job_ids[0]] == JobState.SUCCEEDED
    assert states[job_ids[1]] == JobState.SUCCEEDED
    assert states[job_ids[2]] == JobState.FAILED

    await client.post("/login", json={"password": "test-only-admin-password"})
    resp = await client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    assert body["queue_depth_now"] == 0

    stage_durations = {row["stage"]: row for row in body["stage_durations"]}
    assert "acquire" in stage_durations
    assert stage_durations["acquire"]["completed_count"] >= 3  # 2 ok + 1 failing job's acquire
    assert stage_durations["acquire"]["avg_seconds"] > 0

    assert body["worker_utilization"]
    assert sum(w["busy_seconds"] for w in body["worker_utilization"]) > 0
    # M9: split, not a single count that quietly folds failures in as if they were
    # equally "completed".
    assert sum(w["jobs_succeeded"] for w in body["worker_utilization"]) >= 2
    assert sum(w["jobs_failed"] for w in body["worker_utilization"]) >= 1

    lease = body["lease"]
    assert lease["total_acquisitions"] >= 3
    assert lease["total_occupied_seconds"] > 0
    assert lease["avg_wait_seconds"] is not None
    assert lease["current_holder_job_id"] is None  # everything released cleanly

    # M9: the window always ends at "now", never at the last time something finished.
    window_end = datetime.fromisoformat(body["window_end"])
    assert abs((datetime.now(UTC) - window_end).total_seconds()) < 5


async def test_stats_endpoint_requires_auth(db_env, client):
    resp = await client.get("/stats")
    assert resp.status_code == 401


async def test_stats_reflects_a_wedge_instead_of_reading_healthy(
    db_env, db_session_factory, client
):
    """M9's exact repro: one job has held the lease for hours with nothing released, one
    job has been waiting for hours with nothing acquired, and NOTHING has ever finished.
    The pre-fix endpoint anchored its window on the last `finished_at` (never happened) and
    only counted CLOSED intervals, so it read as if nothing were wrong. It must not."""
    now = datetime.now(UTC)
    six_hours_ago = now - timedelta(hours=6)
    five_hours_ago = now - timedelta(hours=5)

    async with db_session_factory() as session, session.begin():
        job_a = await create_job(session, kind="firmware_analysis")
        job_a.state = JobState.RUNNING
        job_a.started_at = six_hours_ago
        job_a.worker_id = "worker-a"
        job_a.attempt = 1
        job_a.heartbeat_at = now

        job_b = await create_job(session, kind="firmware_analysis")
        job_b.state = JobState.RUNNING
        job_b.started_at = five_hours_ago
        job_b.worker_id = "worker-b"
        job_b.attempt = 1
        job_b.heartbeat_at = now

        session.add(
            ScratchLease(
                id=1,
                holder_job_id=job_a.id,
                holder_worker_id="worker-a",
                acquired_at=six_hours_ago,
                heartbeat_at=now,
            )
        )
        session.add(
            ScratchLeaseEvent(
                job_id=job_a.id, worker_id="worker-a", event="requested", at=six_hours_ago
            )
        )
        session.add(
            ScratchLeaseEvent(
                job_id=job_a.id, worker_id="worker-a", event="acquired", at=six_hours_ago
            )
        )
        # No "released" — job A is still holding it right now.
        session.add(
            ScratchLeaseEvent(
                job_id=job_b.id, worker_id="worker-b", event="requested", at=five_hours_ago
            )
        )
        # No "acquired" — job B is still waiting right now.
        job_a_id = job_a.id

    await client.post("/login", json={"password": "test-only-admin-password"})
    resp = await client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()

    # Both jobs are RUNNING, not QUEUED — an empty queue is the correct reading here, the
    # wedge is entirely on the lease side.
    assert body["queue_depth_now"] == 0

    lease = body["lease"]
    assert lease["current_holder_job_id"] == str(job_a_id)
    # ~6 hours of real, ongoing occupancy — not 0 (the pre-fix bug: an unclosed interval
    # was invisible) and not silently capped.
    assert lease["total_occupied_seconds"] > 5 * 3600
    assert lease["occupancy_pct"] > 20
    # Job B has been waiting ~5 hours and counting — must show up as a real number.
    assert lease["max_wait_seconds"] > 4 * 3600

    window_end = datetime.fromisoformat(body["window_end"])
    assert abs((datetime.now(UTC) - window_end).total_seconds()) < 5


async def test_stats_reflects_a_lease_acquired_before_the_lookback_window(
    db_env, db_session_factory
):
    """Review round 2, finding 2: a lease acquired before window_start never produces an
    "acquired" event inside the `at >= window_start` filter, so the event loop never opens
    an interval for it. Without seeding that open interval from the holder itself, this
    reports total_occupied_seconds=0 / occupancy_pct=0.0 / total_acquisitions=0 while
    current_holder_job_id simultaneously names the holder — exactly backwards for the one
    number that exists to surface a starved lease. The default 24h lookback makes this
    unlikely in practice; a short operator-set lookback (used here) hits it on every
    request."""
    from uadclaw.stats import compute_stats

    acquired_long_ago = datetime.now(UTC) - timedelta(hours=1)

    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis")
        session.add(
            ScratchLease(
                id=1,
                holder_job_id=job.id,
                holder_worker_id="worker-a",
                acquired_at=acquired_long_ago,
                heartbeat_at=datetime.now(UTC),
            )
        )
        session.add(
            ScratchLeaseEvent(
                job_id=job.id, worker_id="worker-a", event="acquired", at=acquired_long_ago
            )
        )
        job_id = job.id

    async with db_session_factory() as session:
        # A 60s lookback: the acquire event from an hour ago falls well outside it, so the
        # event loop alone (pre-fix) would never see it.
        stats = await compute_stats(session, lookback_seconds=60)

    # Reported against the holder, not against zero.
    assert stats.lease.current_holder_job_id == job_id
    assert stats.lease.total_occupied_seconds > 55  # ~= the whole 60s window
    assert stats.lease.occupancy_pct > 90
