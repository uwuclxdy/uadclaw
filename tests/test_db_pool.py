"""Pin the warm-pool reconnect behavior the production engine config rests on.

`db.get_engine()` builds the engine with `pool_pre_ping=True` and a connect timeout in
`connect_args`. The timeout's failure side is measured elsewhere — a stalling socket raises
`TimeoutError`, which `web.DB_UNREACHABLE` catches, so a dashboard renders its error state
instead of hanging for asyncpg's 60-second default. This file pins the other half, which was
never observed: a backend that dies server-side while its connection sits in the warm pool
is replaced silently on the next checkout, and the replacement connection is created under
the same connect timeout.
"""

import asyncio

import asyncpg
from sqlalchemy import event, text

from uadclaw.db import get_engine
from uadclaw.settings import get_settings


async def test_warm_pool_reconnects_after_server_side_kill(db_env):
    """A backend killed server-side while its connection sits in the warm pool is
    transparently replaced: the next query succeeds and reports a new pid."""
    settings = get_settings()
    engine = get_engine()

    # The dialect's `do_connect` event fires with the actual connect params on every
    # connection the pool creates. `connect_args` is merged into the pool creator's
    # cparams at engine build time and stored on no engine/dialect attribute in
    # SQLAlchemy 2.0.51, so this event is the observable leg for the timeout — and it
    # observes the RECONNECT path specifically, which is the path this test kills.
    timeouts_seen = []

    def capture(dialect, conn_rec, cargs, cparams):
        timeouts_seen.append(cparams.get("timeout"))

    event.listen(engine.sync_engine, "do_connect", capture)
    try:
        async with engine.connect() as conn:
            pid_before = (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one()
        # The connection is back in the warm pool, its backend still alive.

        # A second, raw connection to the same test database does the killing — never the
        # pooled connection under test. The test role may terminate its own backends.
        killer = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
            database=settings.postgres_db,
            timeout=5,
        )
        try:
            killed = await killer.fetchval("SELECT pg_terminate_backend($1)", pid_before)
            assert killed, f"backend {pid_before} was not there to terminate"
            # `pg_terminate_backend` signals and returns before the target has exited.
            # Only once its pg_stat_activity row is gone is the socket actually closed —
            # the condition that makes the pre-ping deterministically fail rather than
            # race the dying backend.
            for _ in range(500):
                if not await killer.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid = $1", pid_before
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError(f"terminated backend {pid_before} never left pg_stat_activity")
        finally:
            await killer.close()

        async with engine.connect() as conn:
            value = (await conn.execute(text("SELECT 1"))).scalar_one()
            pid_after = (await conn.execute(text("SELECT pg_backend_pid()"))).scalar_one()
    finally:
        event.remove(engine.sync_engine, "do_connect", capture)

    assert value == 1
    # The dead connection was replaced, not reused: pre-ping discarded it and the pool
    # made a fresh one.
    assert pid_after != pid_before
    # Both connects this test forced — the warm-up and the replacement — carried the
    # settings timeout, so the connect timeout is configured on the reconnect path too.
    assert len(timeouts_seen) >= 2
    assert all(timeout == settings.postgres_connect_timeout_seconds for timeout in timeouts_seen)


async def test_production_engine_config(db_env):
    """The pool_pre_ping flag the reconnect behavior rests on, read off the production
    engine. The reconnect behavior itself is pinned by the kill test above; this is the
    config leg SQLAlchemy 2.0.51 stores no public accessor for."""
    assert get_engine().pool._pre_ping is True
