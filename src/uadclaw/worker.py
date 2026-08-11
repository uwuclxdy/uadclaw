"""Worker: claims jobs (SKIP LOCKED), runs them stage by stage under the scratch lease (for
kinds that need it), and enforces retention around the outcome.

`stage_handlers` is the injection point: production runs `uadclaw.stages`, which implements
the stages that exist (acquire and unpack today) and leaves the rest no-ops, so a stage
landing later is one entry there and no change here. Tests inject synthetic handlers (sleep,
write scratch files, raise) to exercise concurrency, lease contention, crash reclaim and
retention without needing a real firmware pipeline.

Every job write is fenced on (job_id, worker_id, attempt) via `uadclaw.jobs`: a worker that
gets reclaimed — falsely, mid-lease-wait, or genuinely after a crash — stops touching the
job and its scratch directory the instant a newer attempt exists, rather than racing it.
"""

import asyncio
import contextlib
import enum
import logging
import os
import socket
import traceback
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from uadclaw import jobs as jobs_module
from uadclaw import scratch
from uadclaw.db import get_session_factory
from uadclaw.models import PIPELINE_STAGES, Job
from uadclaw.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StageContext:
    job_id: uuid.UUID
    attempt: int
    scratch_dir: Path | None  # None for a job kind that doesn't need_scratch
    session_factory: async_sessionmaker[AsyncSession]


StageHandler = Callable[[StageContext], Awaitable[None]]


class _StageOutcome(enum.Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Ownership moved on (reclaimed, or a newer attempt claimed the job) between our claim
    # and this stage's fenced write. Not a failure of the job — a failure of OUR right to
    # keep touching it. Abort locally without recording anything further.
    FENCED = "fenced"


async def _noop_stage(_ctx: StageContext) -> None:
    return None


def default_stage_handlers() -> dict[str, StageHandler]:
    return dict.fromkeys(PIPELINE_STAGES, _noop_stage)


def worker_identity(slot: int | str) -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{slot}"


async def _heartbeat_loop(
    job_id: uuid.UUID,
    worker_id: str,
    attempt: int,
    needs_lease: bool,
    session_factory: async_sessionmaker[AsyncSession],
    interval: float,
    stop_event: asyncio.Event,
) -> None:
    """Starts before the scratch lease is acquired (M3): a job can block on a contended
    lease for far longer than the stale window, and if the heartbeat only started once
    that wait was over, a perfectly healthy job waiting its turn would be reclaimed out
    from under itself. Refreshes the job's own heartbeat unconditionally, and the lease's
    heartbeat only while `needs_lease` — `heartbeat_scratch_lease` itself no-ops until this
    worker actually holds it, so calling it before acquisition is harmless.
    """
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return  # stop_event was set
        except TimeoutError:
            pass
        async with session_factory() as session, session.begin():
            ok = await jobs_module.heartbeat_job(session, job_id, worker_id, attempt)
        if not ok:
            logger.warning(
                "job %s attempt %d heartbeat found it fenced out; stopping heartbeat",
                job_id,
                attempt,
            )
            return
        if needs_lease:
            await scratch.heartbeat_scratch_lease(
                session_factory, job_id=job_id, worker_id=worker_id
            )


async def _run_one_stage(
    job_id: uuid.UUID,
    worker_id: str,
    attempt: int,
    stage: str,
    scratch_dir: Path | None,
    session_factory: async_sessionmaker[AsyncSession],
    stage_handlers: Mapping[str, StageHandler],
) -> _StageOutcome:
    ctx = StageContext(
        job_id=job_id, attempt=attempt, scratch_dir=scratch_dir, session_factory=session_factory
    )
    handler = stage_handlers.get(stage, _noop_stage)

    async with session_factory() as session, session.begin():
        run_id = await jobs_module.record_stage_started(session, job_id, worker_id, attempt, stage)
    if run_id is None:
        logger.warning(
            "job %s attempt %d fenced out before stage %s could start", job_id, attempt, stage
        )
        return _StageOutcome.FENCED

    try:
        await handler(ctx)
    except Exception as exc:
        # Full traceback logged before recording, never swallowed.
        logger.exception("job %s failed at stage %s", job_id, stage)
        async with session_factory() as session, session.begin():
            await jobs_module.record_stage_finished(session, run_id, outcome="failed")
            failed = await jobs_module.fail_job(
                session,
                job_id,
                worker_id,
                attempt,
                reason=f"stage={stage}: {exc}\n{traceback.format_exc()}",
                log_line=f"FAILED at stage {stage}: {exc}",
            )
        if failed is None:
            logger.warning(
                "job %s attempt %d fenced out while recording stage %s failure",
                job_id,
                attempt,
                stage,
            )
            return _StageOutcome.FENCED
        return _StageOutcome.FAILED

    async with session_factory() as session, session.begin():
        await jobs_module.record_stage_finished(session, run_id, outcome="succeeded")
        advanced = await jobs_module.advance_stage(
            session, job_id, worker_id, attempt, stage, log_line=f"completed stage {stage}"
        )
    if not advanced:
        logger.warning(
            "job %s attempt %d fenced out right after finishing stage %s", job_id, attempt, stage
        )
        return _StageOutcome.FENCED
    return _StageOutcome.SUCCEEDED


async def _run_job(
    job: Job,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    worker_id: str,
    stage_handlers: Mapping[str, StageHandler],
) -> None:
    job_id = job.id
    attempt = job.attempt  # already incremented by claim_job; the identity for every write below
    needs_lease = jobs_module.needs_scratch(job.kind)

    # Heartbeat task creation and every write below live inside this try/finally — not
    # before it — so ANY exception, including one from `mark_running` itself, still stops
    # the heartbeat task in `finally` rather than leaking it as an orphaned background task
    # that keeps refreshing a job's heartbeat forever with nothing else ever checking on it.
    heartbeat_task: asyncio.Task[None] | None = None
    stop_heartbeat = asyncio.Event()
    lease_acquired = False
    job_failed = False
    try:
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(
                job_id,
                worker_id,
                attempt,
                needs_lease,
                session_factory,
                settings.heartbeat_interval_seconds,
                stop_heartbeat,
            )
        )

        async with session_factory() as session, session.begin():
            started = await jobs_module.mark_running(session, job_id, worker_id, attempt)
        if not started:
            logger.warning(
                "job %s attempt %d fenced out before it could start running", job_id, attempt
            )
            return

        scratch_dir: Path | None = None
        if needs_lease:
            stale_after = timedelta(seconds=settings.lease_stale_after_seconds)
            await scratch.acquire_scratch_lease(
                session_factory,
                job_id=job_id,
                worker_id=worker_id,
                stale_after=stale_after,
                poll_interval=settings.lease_poll_interval_seconds,
            )
            lease_acquired = True
            scratch_dir = await scratch.prepare_scratch_dir(settings.scratch_root, job_id, attempt)

        stage = jobs_module.next_stage(job.stage)
        while stage is not None:
            outcome = await _run_one_stage(
                job_id, worker_id, attempt, stage, scratch_dir, session_factory, stage_handlers
            )
            if outcome is _StageOutcome.FENCED:
                return
            if outcome is _StageOutcome.FAILED:
                job_failed = True
                break
            stage = jobs_module.next_stage(stage)

        if not job_failed:
            # Commit SUCCEEDED first, clean up scratch second: a cleanup failure must never
            # roll a job that finished all ten stages back into looking FAILED.
            async with session_factory() as session, session.begin():
                completed = await jobs_module.complete_job(session, job_id, worker_id, attempt)
            if completed is None:
                logger.warning(
                    "job %s attempt %d finished every stage but was fenced out before the "
                    "SUCCEEDED write landed",
                    job_id,
                    attempt,
                )
                return
            if needs_lease:
                try:
                    await scratch.cleanup_scratch_dir(settings.scratch_root, job_id, attempt)
                except OSError:
                    logger.exception(
                        "job %s attempt %d succeeded but scratch cleanup raised; the "
                        "retention ceiling sweep will pick up any leftover bytes",
                        job_id,
                        attempt,
                    )
                else:
                    async with session_factory() as session, session.begin():
                        await jobs_module.mark_artifacts_deleted(
                            session, job_id, worker_id, attempt
                        )
    except Exception as exc:
        # Must never escape: one bad job would otherwise take the whole slot (and, without
        # a guard at the call site too, the whole pool) down with it.
        logger.exception("job %s attempt %d failed outside stage execution", job_id, attempt)
        async with session_factory() as session, session.begin():
            await jobs_module.fail_job(
                session, job_id, worker_id, attempt, reason=f"{exc}\n{traceback.format_exc()}"
            )
        job_failed = True
    finally:
        stop_heartbeat.set()
        if heartbeat_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        if lease_acquired:
            await scratch.release_scratch_lease(session_factory, job_id=job_id, worker_id=worker_id)
        if job_failed:
            try:
                await scratch.enforce_retention_ceiling(
                    session_factory,
                    settings.scratch_root,
                    ceiling_bytes=settings.failure_retention_bytes,
                )
            except Exception:
                logger.exception(
                    "retention ceiling sweep after job %s failure raised; the periodic "
                    "sweep will retry",
                    job_id,
                )


async def _worker_slot(
    slot: int,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    stage_handlers: Mapping[str, StageHandler],
    shutdown_event: asyncio.Event,
) -> None:
    worker_id = worker_identity(slot)
    logger.info("worker slot %s starting as %s", slot, worker_id)
    while not shutdown_event.is_set():
        async with session_factory() as session, session.begin():
            job = await jobs_module.claim_job(session, worker_id=worker_id)
        if job is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=settings.job_claim_poll_interval_seconds
                )
            continue
        logger.info(
            "worker %s claimed job %s kind=%s resume_stage=%s attempt=%d",
            worker_id,
            job.id,
            job.kind,
            job.stage,
            job.attempt,
        )
        try:
            await _run_job(
                job,
                session_factory=session_factory,
                settings=settings,
                worker_id=worker_id,
                stage_handlers=stage_handlers,
            )
        except Exception:
            # Belt and suspenders on top of _run_job's own internal guards: nothing a
            # single job does should ever take this slot (or, via asyncio.gather, the
            # whole pool) down with it.
            logger.exception(
                "worker %s: job %s raised out of _run_job unexpectedly; slot continues",
                worker_id,
                job.id,
            )


async def _periodic_sweep_loop(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    shutdown_event: asyncio.Event,
) -> None:
    """Reclaim + retention on an interval, not just once at startup: a job or lease
    orphaned by a slot that died mid-run (without killing the whole process) would
    otherwise never recover in a still-live process, since nothing else would ever call
    `reclaim_stale_jobs` again."""
    worker_id = worker_identity("sweep")
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=settings.sweep_interval_seconds)
            return
        except TimeoutError:
            pass
        try:
            async with session_factory() as session, session.begin():
                await jobs_module.reclaim_stale_jobs(
                    session,
                    stale_after=timedelta(seconds=settings.lease_stale_after_seconds),
                    worker_id=worker_id,
                    max_attempts=settings.max_job_attempts,
                )
        except Exception:
            logger.exception("periodic reclaim sweep raised; will retry next interval")
        try:
            await scratch.enforce_retention_ceiling(
                session_factory,
                settings.scratch_root,
                ceiling_bytes=settings.failure_retention_bytes,
            )
        except Exception:
            logger.exception("periodic retention sweep raised; will retry next interval")


async def run_worker_pool(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    shutdown_event: asyncio.Event,
    stage_handlers: Mapping[str, StageHandler] | None = None,
) -> None:
    """Reclaim anything a previous, now-dead worker left stuck, then run
    `settings.worker_pool_size` claim/execute loops plus one periodic sweep loop
    concurrently until `shutdown_event` is set."""
    handlers = dict(stage_handlers) if stage_handlers is not None else default_stage_handlers()
    startup_worker_id = worker_identity("startup")
    async with session_factory() as session, session.begin():
        await jobs_module.reclaim_stale_jobs(
            session,
            stale_after=timedelta(seconds=settings.lease_stale_after_seconds),
            worker_id=startup_worker_id,
            max_attempts=settings.max_job_attempts,
        )

    await asyncio.gather(
        _periodic_sweep_loop(session_factory, settings, shutdown_event),
        *[
            _worker_slot(
                slot,
                session_factory=session_factory,
                settings=settings,
                stage_handlers=handlers,
                shutdown_event=shutdown_event,
            )
            for slot in range(settings.worker_pool_size)
        ],
    )


async def run() -> None:
    # Imported here, not at module scope: `uadclaw.stages` imports StageContext from this
    # module, and a top-level import in both directions is a cycle.
    from uadclaw.stages import pipeline_stage_handlers

    settings = get_settings()
    session_factory = get_session_factory()
    logger.info("worker starting: pool_size=%d", settings.worker_pool_size)
    await run_worker_pool(
        session_factory=session_factory,
        settings=settings,
        shutdown_event=asyncio.Event(),  # never set: this process runs until killed
        stage_handlers=pipeline_stage_handlers(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
