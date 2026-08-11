"""M5: one job's exception must never take the whole pool down with it, whether it comes
from inside `_run_job`'s own guarded body or from something that raises before that body
even starts (e.g. a bug in a guard itself).
"""

import uadclaw.jobs as jobs_module
from conftest import FIRMWARE_TARGET
from uadclaw.jobs import create_job
from uadclaw.models import Job, JobState
from uadclaw.settings import get_settings


async def test_slot_survives_an_exception_raised_outside_run_jobs_own_try_block(
    monkeypatch, db_env, db_session_factory, run_pool_until
):
    """`_run_job`'s own try/except wraps almost its entire body (including `mark_running`,
    after the M3/M5 restructure that also fixed a heartbeat-task leak), so to actually
    exercise `_worker_slot`'s outer guard — the belt-and-suspenders M5 asks for — the
    injected failure has to be in one of the few lines that still run BEFORE that try
    block: `needs_scratch(job.kind)`. Confirm a second, healthy job claimed by the same
    pool still runs to completion — the slot logs and moves on rather than dying."""
    settings = get_settings()

    async with db_session_factory() as session, session.begin():
        poison_job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        healthy_job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        poison_id, healthy_id = poison_job.id, healthy_job.id

    real_needs_scratch = jobs_module.needs_scratch

    def _flaky_needs_scratch(kind):
        if _flaky_needs_scratch.calls == 0:
            _flaky_needs_scratch.calls += 1
            raise RuntimeError("simulated bug in a guard that runs before _run_job's try block")
        return real_needs_scratch(kind)

    _flaky_needs_scratch.calls = 0

    # `worker.py` does `from uadclaw import jobs as jobs_module` and calls
    # `jobs_module.needs_scratch(...)` — same module object, so patching the attribute
    # here is visible there too; attribute lookup happens at call time, not import time.
    monkeypatch.setattr(jobs_module, "needs_scratch", _flaky_needs_scratch)

    async def _healthy_terminal() -> bool:
        async with db_session_factory() as session:
            db_job = await session.get(Job, healthy_id)
            return db_job is not None and db_job.state in (JobState.SUCCEEDED, JobState.FAILED)

    # Bounded timeout: if the pool actually died, this would hang until run_pool_until's
    # own timeout fires and raises — a real failure signal, not a false pass.
    await run_pool_until(db_session_factory, settings, None, _healthy_terminal)

    async with db_session_factory() as session:
        healthy = await session.get(Job, healthy_id)
        poison = await session.get(Job, poison_id)

    assert healthy.state == JobState.SUCCEEDED
    # The poison job (claimed first, FIFO by created_at) blew up entirely outside
    # `_run_job`'s own try/except; only `_worker_slot`'s outer guard stood between that
    # and the whole pool dying. It stayed CLAIMED — nothing recorded the failure, because
    # nothing inside _run_job ever ran — but the pool kept going regardless.
    assert poison.attempt == 1
    assert poison.state == JobState.CLAIMED
