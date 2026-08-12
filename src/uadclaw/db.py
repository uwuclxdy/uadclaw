"""Async SQLAlchemy engine, keyed off settings so tests can override the DSN via env vars."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from uadclaw.settings import get_settings


class Base(DeclarativeBase):
    """Shared declarative base. `src/uadclaw/models.py` holds the mapped tables; alembic's
    `--autogenerate` diffs against this via `target_metadata` in `alembic/env.py`."""


@lru_cache
def get_engine() -> AsyncEngine:
    # `connect_args` reaches `asyncpg.connect(timeout=...)`. Without it asyncpg waits its own
    # 60-second default, and the failure mode that needs a bound is not the refused connection
    # or the unresolvable name — both of those raise straight away — but a host that is
    # ROUTABLE AND DEAD, which accepts nothing and answers nothing. Every dashboard screen
    # hangs for the full minute rather than rendering the error state each of them has.
    # The timeout surfaces as `TimeoutError`, which is an `OSError`, so `web.DB_UNREACHABLE`
    # already catches it.
    return create_async_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"timeout": get_settings().postgres_connect_timeout_seconds},
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: job/lease objects get read after the `async with
    # session.begin()` block that committed them closes the session — the worker loop and
    # stage handlers pass job.id/.kind/.stage around across separate short transactions
    # rather than holding one session open for a job's whole lifetime.
    return async_sessionmaker(get_engine(), expire_on_commit=False)
