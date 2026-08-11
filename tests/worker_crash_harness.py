"""Standalone worker process for the crash-reclaim test (tests/test_worker_crash.py).

Runs one job through an "acquire" stage that sleeps long enough for the parent test to
SIGKILL this process mid-stage, so the scratch lease and the job are left holding a
heartbeat that goes stale — exactly the crash this task's reclaim logic exists for.

Not a pytest test module (no `test_` prefix, not under `testpaths` collection by name) —
invoked as a subprocess with `python tests/worker_crash_harness.py`. Config comes entirely
from the environment the parent test sets, same as the real `uadclaw.worker` entrypoint.
"""

import asyncio
import logging
import os

from uadclaw import worker
from uadclaw.db import get_session_factory
from uadclaw.settings import get_settings


async def _sleep_stage(_ctx) -> None:
    await asyncio.sleep(float(os.environ["UADCLAW_TEST_STAGE_SLEEP_SECONDS"]))


async def _main() -> None:
    settings = get_settings()
    session_factory = get_session_factory()
    handlers = worker.default_stage_handlers()
    handlers["acquire"] = _sleep_stage
    await worker.run_worker_pool(
        session_factory=session_factory,
        settings=settings,
        shutdown_event=asyncio.Event(),  # never set: this process is meant to be killed
        stage_handlers=handlers,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
