"""Test env: settings' required fields set to non-secret dummy values before app creation.

Tests run outside docker (no `postgres` host resolvable), so the DB health check is
expected to fail closed (`db: "error"`) unless a real Postgres is reachable at these
coordinates.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from uadclaw.db import get_engine
from uadclaw.settings import get_settings


@pytest.fixture(autouse=True)
async def _reset_caches():
    """`get_settings`/`get_engine` are process-wide `lru_cache`s; without a reset, the
    first test to call either pins its Settings/Engine for every test after it in this
    worker, silently ignoring any env vars a later test sets."""
    yield
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
    get_engine.cache_clear()
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
