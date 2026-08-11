"""Crash reclaim: kill a worker mid-job with a real SIGKILL (not a graceful shutdown) and
confirm both the job and the scratch lease it held come back on the next worker start,
with the reclaim logged and naming the previous holder.
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from conftest import FIRMWARE_TARGET
from uadclaw.jobs import create_job
from uadclaw.models import Job, JobState, ScratchLease
from uadclaw.settings import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tests" / "worker_crash_harness.py"


async def _wait_until(predicate, *, timeout: float, poll_interval: float = 0.1) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(poll_interval)
    return False


@pytest.mark.timeout(45)  # spawns a real subprocess: give a loaded runner headroom above
# the worst case below (8s claim-wait + 1.5s + 8s resume-wait = 17.5s) without silently
# widening it — this used to budget 15s+1.5s+15s=31.5s against the file-wide default of
# 30s, passing today only because both waits finish in well under a second in practice.
async def test_worker_kill_mid_job_reclaims_lease_and_job_on_restart(
    monkeypatch, db_env, db_session_factory, run_pool_until, caplog
):
    monkeypatch.setenv("LEASE_STALE_AFTER_SECONDS", "1")
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "0.2")
    settings = get_settings()

    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        job_id = job.id

    env = os.environ.copy()
    env["UADCLAW_TEST_STAGE_SLEEP_SECONDS"] = "30"  # far longer than we'll let it run
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(HARNESS)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:

        async def _lease_acquired_by_job() -> bool:
            async with db_session_factory() as session:
                lease = (
                    await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
                ).scalar_one_or_none()
                return lease is not None and lease.holder_job_id == job_id

        claimed = await _wait_until(_lease_acquired_by_job, timeout=8)
        assert claimed, f"harness never claimed+acquired the lease for job {job_id}"

        async with db_session_factory() as session:
            lease = (
                await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
            ).scalar_one()
            dead_worker_id = lease.holder_worker_id
            assert dead_worker_id is not None

            db_job = await session.get(Job, job_id)
            assert db_job.state == JobState.RUNNING
    finally:
        # A real kill: SIGKILL can't be caught or handled, unlike SIGTERM. The process (and
        # any graceful-shutdown path it might otherwise have taken) has no say in this.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)

    # Let the dead worker's heartbeat go stale relative to the 1s threshold above.
    await asyncio.sleep(1.5)

    async def _job_terminal() -> bool:
        async with db_session_factory() as session:
            db_job = await session.get(Job, job_id)
            return db_job is not None and db_job.state in (JobState.SUCCEEDED, JobState.FAILED)

    with caplog.at_level("WARNING"):
        # A fresh worker start: its own startup reclaim sweep must free the job, and its
        # first lease-acquire attempt must free the lease — both logged, naming dead_worker_id.
        await run_pool_until(db_session_factory, settings, None, _job_terminal, timeout=8)

    job_reclaim_logs = [r for r in caplog.records if "reclaiming stale job" in r.message]
    lease_reclaim_logs = [
        r for r in caplog.records if "reclaiming stale scratch lease" in r.message
    ]
    assert len(job_reclaim_logs) == 1
    assert str(job_id) in job_reclaim_logs[0].message
    assert dead_worker_id in job_reclaim_logs[0].message

    assert len(lease_reclaim_logs) == 1
    assert dead_worker_id in lease_reclaim_logs[0].message

    async with db_session_factory() as session:
        db_job = await session.get(Job, job_id)
        assert db_job.state == JobState.SUCCEEDED  # resumed and ran to completion
        lease = (
            await session.execute(select(ScratchLease).where(ScratchLease.id == 1))
        ).scalar_one()
        assert lease.holder_job_id is None  # released cleanly at the end of the rerun
