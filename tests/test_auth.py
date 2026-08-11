"""Every route except health and login sits behind the session cookie, default-deny."""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect


async def test_protected_route_rejects_without_session(client):
    resp = await client.get("/")
    assert resp.status_code == 401


async def test_docs_routes_require_session(client):
    """FastAPI registers these on the app instance itself, not through any router — a
    router-scoped `Depends` can never see them. Only app-wide middleware can."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        resp = await client.get(path)
        assert resp.status_code == 401, path


async def test_unexempted_route_defaults_to_protected(app, client):
    """A brand new router mounted without remembering an auth dependency must still be
    rejected — the failure mode that let /docs etc. through unauthenticated."""

    @app.get("/some-future-router-nobody-exempted")
    async def _throwaway() -> dict[str, bool]:
        return {"reached": True}

    resp = await client.get("/some-future-router-nobody-exempted")
    assert resp.status_code == 401


def test_unauthenticated_websocket_is_closed(app):
    """The `scope["type"] != "http"` early-out used to wave every websocket through
    unconditionally (only `lifespan` should get that treatment) — a task-2 job-progress
    socket would have shipped open. Assert the specific close code, not merely that the
    connection didn't stay up: a server-side exception during the handshake also drops the
    client without accepting, which would look like "rejected" for the wrong reason."""

    @app.websocket("/ws-throwaway")
    async def _throwaway_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("LEAKED: reached without a session")

    with (
        TestClient(app) as test_client,
        pytest.raises(WebSocketDisconnect) as exc_info,
        test_client.websocket_connect("/ws-throwaway") as ws,
    ):
        ws.receive_text()

    assert exc_info.value.code == 1008


async def test_login_rejects_wrong_password(client):
    resp = await client.post("/login", json={"password": "wrong"})
    assert resp.status_code == 401
    assert not resp.cookies


async def test_login_then_protected_route_succeeds(client):
    resp = await client.post("/login", json={"password": "test-only-admin-password"})
    assert resp.status_code == 204

    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"app": "uadclaw"}


async def test_logout_revokes_session(client):
    await client.post("/login", json={"password": "test-only-admin-password"})
    resp = await client.get("/")
    assert resp.status_code == 200

    await client.post("/logout")
    resp = await client.get("/")
    assert resp.status_code == 401
