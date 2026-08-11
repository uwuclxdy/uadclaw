"""Health endpoint: process and DB liveness must be reported as separate fields."""


async def test_health_reports_web_and_db_separately(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"web", "db"}
    # Process is up (this handler ran); DB is unreachable outside docker, so it must degrade
    # independently rather than collapsing into one shared boolean.
    assert body["web"] == "ok"
    assert body["db"] == "error"


async def test_health_is_exempt_from_auth(client):
    resp = await client.get("/health")
    assert resp.status_code != 401
