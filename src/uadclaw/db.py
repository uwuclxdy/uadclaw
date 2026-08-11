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
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: job/lease objects get read after the `async with
    # session.begin()` block that committed them closes the session — the worker loop and
    # stage handlers pass job.id/.kind/.stage around across separate short transactions
    # rather than holding one session open for a job's whole lifetime.
    return async_sessionmaker(get_engine(), expire_on_commit=False)
