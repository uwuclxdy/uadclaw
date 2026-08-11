"""Async SQLAlchemy engine, keyed off settings so tests can override the DSN via env vars."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from uadclaw.settings import get_settings


class Base(DeclarativeBase):
    """Shared declarative base. No models yet — task 2 adds the first mapped table, and
    alembic's `--autogenerate` needs this wired to `target_metadata` now (not then) or it
    silently diffs against an empty schema and emits nothing."""


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)
