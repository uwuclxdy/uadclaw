"""Retention: delete on success, byte-ceiling oldest-first eviction on failure — sized
against the actual scratch_root directory tree, not just FAILED-job rows (M10), so an
orphaned or superseded-attempt directory is bounded too.

Exercised directly against `uadclaw.scratch` (no worker loop) for precise byte-level
control over what's on disk when each assertion runs.
"""

import asyncio
import time
import uuid
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from uadclaw.jobs import claim_job, create_job, fail_job, mark_running, reclaim_stale_jobs
from uadclaw.models import Job
from uadclaw.scratch import (
    cleanup_scratch_dir,
    enforce_retention_ceiling,
    job_scratch_dir,
    prepare_scratch_dir,
)


def _write_fake_artifacts(scratch_dir: Path, num_bytes: int) -> None:
    (scratch_dir / "firmware.zip").write_bytes(b"\0" * (num_bytes // 2))
    apks = scratch_dir / "apks"
    apks.mkdir()
    (apks / "com.example.bloat.apk").write_bytes(b"\0" * (num_bytes - num_bytes // 2))


async def _claim_and_run(db_session_factory, worker_id: str) -> tuple[uuid.UUID, int]:
    """Create, claim and mark-running a job, returning (job_id, attempt) — the identity
    every fenced write needs."""
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis")
        claimed = await claim_job(session, worker_id=worker_id)
        assert claimed.id == job.id
        await mark_running(session, claimed.id, worker_id, claimed.attempt)
        return claimed.id, claimed.attempt


async def _fail(
    db_session_factory, job_id: uuid.UUID, worker_id: str, attempt: int, reason: str
) -> None:
    async with db_session_factory() as session, session.begin():
        await fail_job(session, job_id, worker_id, attempt, reason=reason)


async def test_successful_job_leaves_no_archive_or_apk_on_disk(db_session_factory, tmp_path):
    scratch_root = tmp_path / "scratch"
    job_id, attempt = await _claim_and_run(db_session_factory, "worker-a")

    scratch_dir = await prepare_scratch_dir(scratch_root, job_id, attempt)
    _write_fake_artifacts(scratch_dir, 4096)
    assert (scratch_dir / "firmware.zip").exists()
    assert list((scratch_dir / "apks").iterdir())

    await cleanup_scratch_dir(scratch_root, job_id, attempt)

    assert not scratch_dir.exists()


async def test_failure_ceiling_evicts_oldest_job_first(db_session_factory, tmp_path):
    """Three failed jobs at ~600 bytes each against a 1000-byte ceiling: the oldest must be
    evicted before the newest ever is, and the newest must still be standing at the end."""
    scratch_root = tmp_path / "scratch"
    identities: list[tuple[uuid.UUID, int]] = []
    for i in range(3):
        job_id, attempt = await _claim_and_run(db_session_factory, f"worker-{i}")
        scratch_dir = await prepare_scratch_dir(scratch_root, job_id, attempt)
        _write_fake_artifacts(scratch_dir, 600)
        await _fail(
            db_session_factory,
            job_id,
            f"worker-{i}",
            attempt,
            "synthetic failure for retention test",
        )
        identities.append((job_id, attempt))

    (oldest_id, oldest_attempt), (middle_id, middle_attempt), (newest_id, newest_attempt) = (
        identities
    )

    evicted = await enforce_retention_ceiling(db_session_factory, scratch_root, ceiling_bytes=1000)

    assert evicted == [
        job_scratch_dir(scratch_root, oldest_id, oldest_attempt).name,
        job_scratch_dir(scratch_root, middle_id, middle_attempt).name,
    ]
    assert not job_scratch_dir(scratch_root, oldest_id, oldest_attempt).exists()
    assert not job_scratch_dir(scratch_root, middle_id, middle_attempt).exists()
    assert job_scratch_dir(scratch_root, newest_id, newest_attempt).exists()
    assert (job_scratch_dir(scratch_root, newest_id, newest_attempt) / "firmware.zip").exists()


async def test_failure_retention_leaves_artifacts_when_under_ceiling(db_session_factory, tmp_path):
    scratch_root = tmp_path / "scratch"
    job_id, attempt = await _claim_and_run(db_session_factory, "worker-a")
    scratch_dir = await prepare_scratch_dir(scratch_root, job_id, attempt)
    _write_fake_artifacts(scratch_dir, 200)
    await _fail(
        db_session_factory, job_id, "worker-a", attempt, "synthetic failure, well under ceiling"
    )

    evicted = await enforce_retention_ceiling(
        db_session_factory, scratch_root, ceiling_bytes=1_000_000
    )

    assert evicted == []
    assert job_scratch_dir(scratch_root, job_id, attempt).exists()


async def test_ceiling_leaves_a_live_jobs_directory_alone_even_when_old(
    db_session_factory, tmp_path
):
    """M10: a directory belonging to a job's CURRENT attempt while that job is still
    QUEUED/CLAIMED/RUNNING must never be evicted, no matter how old its mtime looks or how
    far over the ceiling the total goes — it's still actively owned."""
    scratch_root = tmp_path / "scratch"
    job_id, attempt = await _claim_and_run(db_session_factory, "worker-a")  # left RUNNING
    scratch_dir = await prepare_scratch_dir(scratch_root, job_id, attempt)
    _write_fake_artifacts(scratch_dir, 5000)  # way over a tiny ceiling

    evicted = await enforce_retention_ceiling(db_session_factory, scratch_root, ceiling_bytes=10)

    assert evicted == []
    assert scratch_dir.exists()


async def test_ceiling_evicts_orphan_directory_with_no_matching_job_row(
    db_session_factory, tmp_path
):
    """M10: a directory with no matching job row at all (the row was deleted, or never
    existed) is real bytes on disk that a ceiling keyed only on `state == FAILED` would
    never see. It must still be bounded."""
    scratch_root = tmp_path / "scratch"
    orphan_id = uuid.uuid4()
    orphan_dir = job_scratch_dir(scratch_root, orphan_id, 1)
    orphan_dir.mkdir(parents=True)
    _write_fake_artifacts(orphan_dir, 5000)

    evicted = await enforce_retention_ceiling(db_session_factory, scratch_root, ceiling_bytes=10)

    assert evicted == [orphan_dir.name]
    assert not orphan_dir.exists()


async def test_ceiling_evicts_superseded_attempt_even_though_current_attempt_is_live(
    db_session_factory, tmp_path
):
    """M10 + M4: attempt 1's directory outlives attempt 1 itself once the job gets
    reclaimed and re-claimed as attempt 2 — it must be evictable even while attempt 2's
    directory (the job's CURRENT attempt, still RUNNING) is protected."""
    scratch_root = tmp_path / "scratch"
    job_id, attempt_1 = await _claim_and_run(db_session_factory, "dead-worker")
    stale_dir = await prepare_scratch_dir(scratch_root, job_id, attempt_1)
    _write_fake_artifacts(stale_dir, 5000)

    # Simulate a reclaim + re-claim: a second attempt for the SAME job, different worker.
    async with db_session_factory() as session, session.begin():
        claimed_again = await claim_job(session, worker_id="dead-worker")
        assert claimed_again is None  # still CLAIMED/RUNNING under attempt 1, not QUEUED

    async with db_session_factory() as session, session.begin():
        db_job = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()
        db_job.heartbeat_at = db_job.heartbeat_at - timedelta(hours=1)
    async with db_session_factory() as session, session.begin():
        await reclaim_stale_jobs(
            session, stale_after=timedelta(seconds=1), worker_id="sweep", max_attempts=5
        )
    async with db_session_factory() as session, session.begin():
        claimed_2 = await claim_job(session, worker_id="new-worker")
        assert claimed_2.id == job_id
        attempt_2 = claimed_2.attempt
        await mark_running(session, job_id, "new-worker", attempt_2)
    assert attempt_2 == attempt_1 + 1
    live_dir = await prepare_scratch_dir(scratch_root, job_id, attempt_2)
    _write_fake_artifacts(live_dir, 100)

    evicted = await enforce_retention_ceiling(db_session_factory, scratch_root, ceiling_bytes=10)

    assert evicted == [stale_dir.name]
    assert not stale_dir.exists()
    assert live_dir.exists()  # the current, live attempt is untouched


async def test_dir_size_bytes_ignores_symlink_targets(tmp_path):
    """M6: `stat()` follows a symlink but `rmtree` never does — billing a symlink's
    target bytes would count space eviction can never actually free."""
    from uadclaw.scratch import _dir_size_bytes

    real_dir = tmp_path / "scratch-dir"
    real_dir.mkdir()
    (real_dir / "kept.bin").write_bytes(b"\0" * 100)

    outside_large_file = tmp_path / "outside-large.bin"
    outside_large_file.write_bytes(b"\0" * 10_000)
    (real_dir / "link-to-outside").symlink_to(outside_large_file)

    size = await _dir_size_bytes(real_dir)

    assert size == 100  # not 10,100


async def test_dir_size_bytes_survives_a_file_vanishing_mid_walk(tmp_path, monkeypatch):
    """M6: `is_file()` then `.stat()` is a TOCTOU — a file can disappear between the two
    (firmware-unpack tooling deletes temp files as it goes) and an uncaught OSError there
    must not crash the whole retention sweep over one race."""
    from uadclaw.scratch import _dir_size_bytes

    real_dir = tmp_path / "scratch-dir"
    real_dir.mkdir()
    (real_dir / "kept.bin").write_bytes(b"\0" * 100)
    (real_dir / "vanishing.bin").write_bytes(b"\0" * 500)

    original_stat = Path.stat
    # `is_symlink()` uses `stat(follow_symlinks=False)` and `is_file()` uses a plain
    # `stat()` internally before our own code ever calls `.stat()` again — to simulate the
    # file vanishing strictly BETWEEN `is_file()` succeeding and our own `.stat()` call
    # (not "is_file() itself already sees it gone", a different, already-safe case), only
    # the SECOND plain, no-kwargs `.stat()` call on this filename fails.
    plain_stat_calls = {"vanishing.bin": 0}

    def _flaky_stat(self, *args, **kwargs):
        if self.name == "vanishing.bin" and not args and not kwargs:
            plain_stat_calls["vanishing.bin"] += 1
            if plain_stat_calls["vanishing.bin"] >= 2:
                raise OSError("simulated: vanished between is_file() and stat()")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _flaky_stat)

    size = await _dir_size_bytes(real_dir)

    assert size == 100  # vanished file skipped, not crashed on and not double-counted


async def test_retention_sweep_rechecks_state_immediately_before_evicting(
    monkeypatch, db_session_factory, tmp_path
):
    """Review round 2, finding 1: the snapshot the walk scores against is read once, before
    a filesystem walk that can run for seconds on a large tree. A slot can claim a job and
    start writing into it during that window. The eviction loop must re-read
    (state, attempt) immediately before deleting each directory, not trust the pre-walk
    snapshot — otherwise the directory actually in use gets deleted out from under the job
    writing into it, which is the dangerous direction; the other direction (a stale
    directory staying protected for one extra sweep cycle) is bounded and self-correcting,
    so this test is specifically about the first one.

    Two directories, both looking evictable when the snapshot is taken (neither job has
    been claimed yet). One (`racing_dir`) gets claimed and marked RUNNING for real while
    the walk is still in flight — simulated with a real concurrent task, not a mock, by
    making the walk block synchronously on that one directory long enough for the claim to
    land. The other (`stale_dir`) is untouched and must still go.
    """
    scratch_root = tmp_path / "scratch"

    async with db_session_factory() as session, session.begin():
        # Created first so claim_job's FIFO-by-created_at ordering picks this one.
        racing_job = await create_job(session, kind="firmware_analysis")
        stale_job = await create_job(session, kind="firmware_analysis")

    # Both directories look evictable at scan time: neither job has been claimed yet
    # (attempt 0), so a directory labeled attempt 1 for either one doesn't match the job's
    # current attempt and _is_live correctly says "not live" — for now.
    racing_dir = await prepare_scratch_dir(scratch_root, racing_job.id, 1)
    _write_fake_artifacts(racing_dir, 5000)
    stale_dir = await prepare_scratch_dir(scratch_root, stale_job.id, 1)
    _write_fake_artifacts(stale_dir, 5000)

    import uadclaw.scratch as scratch_module

    real_iter_file_sizes = scratch_module._iter_file_sizes

    def _slow_iter_file_sizes(root):
        # The scan calls this once per candidate directory; block synchronously (this runs
        # inside asyncio.to_thread's worker thread, so the event loop stays free) only for
        # racing_dir, long enough for the concurrent claim below to land for real before
        # the walk — and therefore the whole snapshot — is done being used.
        if root.name.startswith(f"{racing_job.id}-"):
            time.sleep(0.3)
        yield from real_iter_file_sizes(root)

    monkeypatch.setattr(scratch_module, "_iter_file_sizes", _slow_iter_file_sizes)

    async def _claim_racing_job_mid_walk() -> None:
        await asyncio.sleep(0.05)  # let the walk begin before mutating underneath it
        async with db_session_factory() as session, session.begin():
            claimed = await claim_job(session, worker_id="racing-worker")
            assert claimed is not None
            assert claimed.id == racing_job.id
            await mark_running(session, claimed.id, "racing-worker", claimed.attempt)

    evicted, _ = await asyncio.gather(
        enforce_retention_ceiling(db_session_factory, scratch_root, ceiling_bytes=10),
        _claim_racing_job_mid_walk(),
    )

    # WHICH one survived and which went, not merely "something was evicted": the stale one
    # (never touched) must go, the racing one (now genuinely live) must survive despite
    # having looked evictable in the snapshot the walk started with.
    assert evicted == [stale_dir.name]
    assert not stale_dir.exists()
    assert racing_dir.exists()
