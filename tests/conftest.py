"""Test env: settings' required fields set to non-secret dummy values before app creation.

Tests run outside docker (no `postgres` host resolvable), so the DB health check is
expected to fail closed (`db: "error"`) unless a real Postgres is reachable at these
coordinates.

Job/lease/stats tests need a REAL Postgres: SKIP LOCKED semantics, lease contention and
crash reclaim cannot be proven against SQLite or a mock. `db_env`/`db_session_factory`
below point at one isolated per pytest-xdist worker (a dedicated database on the same
server, so parallel workers never see each other's rows) and fail loudly — not skip — when
nothing is reachable, since a skip would silently pass a suite that never actually proved
anything.
"""

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from uadclaw.db import get_engine, get_session_factory
from uadclaw.settings import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def _xdist_worker_id() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def _secrets_dir_password(name: str, fallback: str) -> str:
    path = REPO_ROOT / "secrets" / name
    return path.read_text().strip() if path.exists() else fallback


PG_HOST = os.environ.get("UADCLAW_TEST_PG_HOST", "localhost")
PG_PORT = int(os.environ.get("UADCLAW_TEST_PG_PORT", "55432"))
PG_USER = os.environ.get("UADCLAW_TEST_PG_USER", "uadclaw")
PG_PASSWORD = os.environ.get(
    "UADCLAW_TEST_PG_PASSWORD", _secrets_dir_password("postgres_password", "uadclaw")
)
PG_ADMIN_DB = os.environ.get("UADCLAW_TEST_PG_ADMIN_DB", "uadclaw")
PG_TEST_DB = f"uadclaw_test_{_xdist_worker_id()}"

_DB_TABLES_TRUNCATE_ORDER = "scratch_lease_events, job_stage_runs, jobs, scratch_lease"


def _pg_url(db_name: str) -> str:
    return f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{db_name}"


async def _ensure_test_database() -> None:
    try:
        conn = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_ADMIN_DB,
            timeout=5,
        )
    except (OSError, asyncpg.PostgresError) as exc:
        raise RuntimeError(
            "DB-backed tests need a real Postgres reachable at "
            f"{PG_HOST}:{PG_PORT} (db={PG_ADMIN_DB}, user={PG_USER}). Bring one up with "
            "`docker compose up -d --wait postgres` (loopback port 55432 by default) "
            f"before running these tests. Connection failed: {exc}"
        ) from exc
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", PG_TEST_DB)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{PG_TEST_DB}"')
    finally:
        await conn.close()


def _run_alembic_upgrade(db_name: str) -> None:
    """`alembic upgrade head` against this worker's isolated test database — not
    `Base.metadata.create_all`. `create_all` only adds tables/columns that don't exist yet
    (`checkfirst=True`); it can never detect that a model changed without a migration to
    match, so a real schema-drift bug would pass green on a fresh box and only surface
    later as a confusing `UndefinedColumn` on a box that happens to have run the suite
    before (these per-worker databases persist between runs). Running the actual migration
    here closes that gap and exercises the migration itself in the same step, in CI too —
    nothing else in this repo runs `alembic upgrade head` against the test database.
    """
    env = os.environ.copy()
    env["POSTGRES_HOST"] = PG_HOST
    env["POSTGRES_PORT"] = str(PG_PORT)
    env["POSTGRES_USER"] = PG_USER
    env["POSTGRES_PASSWORD"] = PG_PASSWORD
    env["POSTGRES_DB"] = db_name
    # get_settings() inside alembic/env.py constructs the whole Settings object, so the
    # other required fields need *a* value even though migrations never touch auth.
    env.setdefault("AUTH_PASSWORD", "test-only-admin-password")
    env.setdefault("SESSION_SECRET", "test-only-session-secret")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed for test database {db_name!r}:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


@pytest.fixture(scope="session")
def pg_test_db_ready() -> str:
    """Create (if needed) and migrate this xdist worker's isolated test database.
    Session-scoped: migrations only need running once per worker process."""
    asyncio.run(_ensure_test_database())
    _run_alembic_upgrade(PG_TEST_DB)
    return PG_TEST_DB


@pytest.fixture
def db_env(monkeypatch, test_env, tmp_path, pg_test_db_ready):
    """Point `get_settings()`/`get_engine()` at this worker's isolated real Postgres
    instead of the unreachable-by-default `postgres` host, and give each test its own
    scratch root. Must be requested before any fixture that calls `create_app()` /
    `get_settings()` for the first time in a given test (list it first in the test's
    parameters) or the env vars land too late to take effect."""
    monkeypatch.setenv("POSTGRES_HOST", PG_HOST)
    monkeypatch.setenv("POSTGRES_PORT", str(PG_PORT))
    monkeypatch.setenv("POSTGRES_USER", PG_USER)
    monkeypatch.setenv("POSTGRES_PASSWORD", PG_PASSWORD)
    monkeypatch.setenv("POSTGRES_DB", pg_test_db_ready)
    monkeypatch.setenv("SCRATCH_ROOT", str(tmp_path / "scratch"))
    return pg_test_db_ready


@pytest.fixture
async def db_session_factory(db_env):
    """A session factory bound to the isolated test database, tables truncated so this
    test starts from an empty slate regardless of what earlier tests on this worker left
    behind.

    Deliberately NOT `uadclaw.db.get_session_factory()`: that's an `lru_cache`d singleton
    keyed off `get_settings()`, and resolving it here — during fixture setup, before the
    test body runs — would pin worker-tuning env vars (pool size, lease staleness, ...) a
    test sets later at their *default* values forever, since a cache hit never re-reads the
    environment. A dedicated engine sidesteps that trap entirely; tests that want the app's
    own cached engine (e.g. hitting `/stats` through `client`) still get it fresh, because
    nothing here ever calls `get_settings()`/`get_engine()`.
    """
    engine = create_async_engine(_pg_url(db_env))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session, session.begin():
        await session.execute(
            text(f"TRUNCATE TABLE {_DB_TABLES_TRUNCATE_ORDER} RESTART IDENTITY CASCADE")
        )
    yield factory
    await engine.dispose()


# Every FIRMWARE_ANALYSIS job needs a target: `create_job` validates params against the kind's
# model at the boundary, so a job with no target is a 422 and never reaches a worker. The
# substrate tests care about claiming, leases and retention rather than about which build, so
# they all queue the same one.
FIRMWARE_TARGET = {"driver": "pixel", "device": "comet"}


def utcnow() -> datetime:
    return datetime.now(UTC)


@pytest.fixture(autouse=True)
async def _reset_caches():
    """`get_settings`/`get_engine`/`get_session_factory` are process-wide `lru_cache`s;
    without a reset, the first test to call any of them pins it for every test after it in
    this worker, silently ignoring any env vars a later test sets."""
    yield
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def test_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-only-password")
    monkeypatch.setenv("AUTH_PASSWORD", "test-only-admin-password")
    monkeypatch.setenv("SESSION_SECRET", "test-only-session-secret")


@pytest.fixture
def app(test_env):
    from uadclaw.app import create_app

    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def run_pool_until():
    """Drive `uadclaw.worker.run_worker_pool` in the background until `predicate()` (an
    async callable returning bool) is true, then request shutdown and wait for the pool to
    actually stop. Lets tests observe real job execution (claim, stages, lease, retention)
    without running the worker forever."""

    async def _run(
        session_factory, settings, stage_handlers, predicate, *, poll_interval=0.05, timeout=15
    ):
        from uadclaw.worker import run_worker_pool

        shutdown_event = asyncio.Event()
        task = asyncio.create_task(
            run_worker_pool(
                session_factory=session_factory,
                settings=settings,
                shutdown_event=shutdown_event,
                stage_handlers=stage_handlers,
            )
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        satisfied = False
        try:
            while loop.time() < deadline:
                if await predicate():
                    satisfied = True
                    break
                await asyncio.sleep(poll_interval)
        finally:
            shutdown_event.set()
            await asyncio.wait_for(task, timeout=5)
        if not satisfied:
            raise AssertionError("run_pool_until: predicate never became true within timeout")

    return _run
