"""Full worker-loop integration: real jobs, real stage handlers, pool concurrency, and the
scratch lease staying exclusive even when the pool runs more than one job at once.

B1: `started_at` timestamp proximity is NOT proof of concurrency — it is stamped by
`mark_running` before the lease wait even begins, so a serial pool_size=1 run can produce
two `started_at` values a fraction of a second apart just by being fast, and did (measured
0.207s spread against a naive `< 0.5` threshold). The real proof below instead POLLS
`jobs.state` while the pool runs and asserts it observed two jobs RUNNING at the same
instant: with pool_size=1 a single slot processes jobs strictly one at a time, so no poll
of that table could ever return 2, no matter how fast the jobs complete — it is
structurally impossible, not just statistically unlikely. A negative-control test with the
pool forced back to 1 confirms the assertion actually discriminates.
"""

import asyncio

from sqlalchemy import func, select

from uadclaw.jobs import create_job
from uadclaw.models import Job, JobState, ScratchLeaseEvent
from uadclaw.settings import get_settings
from uadclaw.worker import default_stage_handlers


def _lease_intervals(events: list[ScratchLeaseEvent]) -> dict:
    open_acquire: dict = {}
    intervals: dict = {}
    for ev in events:
        if ev.event == "acquired":
            open_acquire[ev.job_id] = ev.at
        elif ev.event == "released" and ev.job_id in open_acquire:
            intervals[ev.job_id] = (open_acquire.pop(ev.job_id), ev.at)
    return intervals


def _overlaps(a: tuple, b: tuple) -> bool:
    (a_start, a_end), (b_start, b_end) = a, b
    return a_start < b_end and b_start < a_end


async def _run_two_scratch_jobs_sampling_running_count(
    db_session_factory, settings, run_pool_until
):
    """Two firmware_analysis jobs (they need_scratch, so the lease serializes their actual
    work) through a real pool run, while a background sampler polls how many of them are
    simultaneously in state RUNNING. Returns (samples, jobs_by_id, lease_events)."""

    async def _slow_scratch_write(ctx):
        (ctx.scratch_dir / "firmware.zip").write_bytes(b"\0" * 128)
        await asyncio.sleep(0.15)

    handlers = default_stage_handlers()
    handlers["acquire"] = _slow_scratch_write

    async with db_session_factory() as session, session.begin():
        job_a = await create_job(session, kind="firmware_analysis")
        job_b = await create_job(session, kind="firmware_analysis")
        ids = {job_a.id, job_b.id}

    samples: list[int] = []
    sampler_stop = asyncio.Event()

    async def _sample_loop():
        while not sampler_stop.is_set():
            async with db_session_factory() as session:
                count = (
                    await session.execute(
                        select(func.count()).where(Job.id.in_(ids), Job.state == JobState.RUNNING)
                    )
                ).scalar_one()
            samples.append(count)
            await asyncio.sleep(0.01)

    sampler_task = asyncio.create_task(_sample_loop())

    async def _both_terminal() -> bool:
        async with db_session_factory() as session:
            rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars()
            return all(j.state in (JobState.SUCCEEDED, JobState.FAILED) for j in rows)

    try:
        await run_pool_until(db_session_factory, settings, handlers, _both_terminal)
    finally:
        sampler_stop.set()
        await sampler_task

    async with db_session_factory() as session:
        jobs = {
            j.id: j for j in (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars()
        }
        events = list(
            (
                await session.execute(
                    select(ScratchLeaseEvent)
                    .where(ScratchLeaseEvent.job_id.in_(ids))
                    .order_by(ScratchLeaseEvent.at)
                )
            ).scalars()
        )
    return samples, jobs, events


async def test_pool_size_two_runs_two_jobs_concurrently_while_lease_stays_exclusive(
    monkeypatch, db_env, db_session_factory, run_pool_until
):
    monkeypatch.setenv("WORKER_POOL_SIZE", "2")
    settings = get_settings()

    samples, jobs, events = await _run_two_scratch_jobs_sampling_running_count(
        db_session_factory, settings, run_pool_until
    )

    for job in jobs.values():
        assert job.state == JobState.SUCCEEDED

    # The barrier substitute: at least one sample caught BOTH jobs RUNNING at once.
    assert max(samples, default=0) >= 2, (
        f"never observed 2 concurrently-RUNNING jobs; samples={samples}"
    )

    # Never both held the lease at once: the two jobs' [acquired, released) intervals
    # must not overlap.
    intervals = _lease_intervals(events)
    assert len(intervals) == 2
    (interval_a, interval_b) = intervals.values()
    assert not _overlaps(interval_a, interval_b)


async def test_pool_size_one_never_shows_two_jobs_running_concurrently(
    monkeypatch, db_env, db_session_factory, run_pool_until
):
    """Negative control: with the pool forced back to 1, the exact same sampling proof
    above must never see 2 — confirming that assertion is discriminating, not a tautology
    that would pass regardless of what the pool actually does."""
    monkeypatch.setenv("WORKER_POOL_SIZE", "1")
    settings = get_settings()

    samples, jobs, _events = await _run_two_scratch_jobs_sampling_running_count(
        db_session_factory, settings, run_pool_until
    )

    for job in jobs.values():
        assert job.state == JobState.SUCCEEDED
    assert max(samples, default=0) <= 1, (
        f"pool_size=1 showed concurrent RUNNING jobs; samples={samples}"
    )


async def test_job_waiting_on_a_contended_lease_is_not_falsely_reclaimed(
    monkeypatch, db_env, db_session_factory, run_pool_until
):
    """M3: the heartbeat must start BEFORE the lease wait, not after — a job blocked on a
    contended lease for longer than the stale window must not be reclaimed out from under
    itself just because it hasn't reached its own stage work yet. Job A holds the lease for
    1s; the stale window and sweep interval are both well under that, so if job B's
    heartbeat only started once ITS wait was over (the pre-fix bug), the periodic sweep
    would reclaim it mid-wait and it would come back as a second attempt."""
    monkeypatch.setenv("WORKER_POOL_SIZE", "2")
    monkeypatch.setenv("LEASE_STALE_AFTER_SECONDS", "0.3")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("SWEEP_INTERVAL_SECONDS", "0.2")
    settings = get_settings()

    async def _slow_holder(ctx):
        await asyncio.sleep(1.0)

    handlers = default_stage_handlers()
    handlers["acquire"] = _slow_holder

    async with db_session_factory() as session, session.begin():
        job_a = await create_job(session, kind="firmware_analysis")
        job_b = await create_job(session, kind="firmware_analysis")
        ids = {job_a.id, job_b.id}

    async def _both_terminal() -> bool:
        async with db_session_factory() as session:
            rows = (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars()
            return all(j.state in (JobState.SUCCEEDED, JobState.FAILED) for j in rows)

    await run_pool_until(db_session_factory, settings, handlers, _both_terminal, timeout=10)

    async with db_session_factory() as session:
        jobs = {
            j.id: j for j in (await session.execute(select(Job).where(Job.id.in_(ids)))).scalars()
        }
    for job in jobs.values():
        assert job.state == JobState.SUCCEEDED
        # attempt stays 1: a false reclaim mid-wait would have requeued and re-claimed the
        # waiting job, bumping this to 2+.
        assert job.attempt == 1, (
            f"job {job.id} attempt={job.attempt}, expected 1 (no false reclaim)"
        )


async def test_resumed_job_restarts_from_its_recorded_stage_not_the_beginning(
    db_env, db_session_factory, run_pool_until
):
    settings = get_settings()
    handlers = default_stage_handlers()
    stage_calls: list[str] = []

    def _make_recorder(stage_name):
        async def _handler(ctx):
            stage_calls.append(stage_name)

        return _handler

    for stage in handlers:
        handlers[stage] = _make_recorder(stage)

    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis")
        job.stage = "corpus_graph"  # pretend a previous run already got this far
        job_id = job.id

    async def _terminal() -> bool:
        async with db_session_factory() as session:
            db_job = await session.get(Job, job_id)
            return db_job is not None and db_job.state in (JobState.SUCCEEDED, JobState.FAILED)

    await run_pool_until(db_session_factory, settings, handlers, _terminal)

    # Resume must skip acquire/unpack/extract_facts/corpus_graph and start at "filter".
    assert stage_calls == ["filter", "rule_ladder", "llm", "corroborate", "triage", "branch"]

    async with db_session_factory() as session:
        db_job = await session.get(Job, job_id)
        assert db_job.state == JobState.SUCCEEDED
        assert db_job.stage == "branch"
