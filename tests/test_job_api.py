"""B2: the job create/read API the dashboard (and every "start two jobs" verify step) can
actually drive, instead of only pytest/psql. Auth-protected automatically, like every
route — no PUBLIC_PATHS entry.
"""

import uuid


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": "test-only-admin-password"})
    assert resp.status_code == 204


async def test_create_job_route_requires_auth(client):
    resp = await client.post("/jobs", json={"kind": "firmware_analysis"})
    assert resp.status_code == 401


async def test_get_job_route_requires_auth(client):
    resp = await client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 401


async def test_create_job_route_returns_the_new_job(db_env, client):
    await _login(client)
    resp = await client.post("/jobs", json={"kind": "firmware_analysis"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["kind"] == "firmware_analysis"
    assert body["state"] == "queued"
    assert body["stage"] is None
    assert body["attempt"] == 0
    assert body["worker_id"] is None
    assert body["log_tail"] == ""
    uuid.UUID(body["id"])  # a real id, parses cleanly


async def test_create_job_route_rejects_unknown_kind(db_env, client):
    await _login(client)
    resp = await client.post("/jobs", json={"kind": "not-a-real-kind"})
    assert resp.status_code == 422


async def test_get_job_route_returns_state_stage_and_log_tail(db_env, client):
    await _login(client)
    created = (await client.post("/jobs", json={"kind": "firmware_analysis"})).json()

    resp = await client.get(f"/jobs/{created['id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == created["id"]
    assert body["state"] == "queued"
    assert body["stage"] is None
    assert body["failure_reason"] is None
    assert "created_at" in body


async def test_get_job_route_404s_for_an_unknown_id(db_env, client):
    await _login(client)
    resp = await client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_created_job_is_pollable_after_the_worker_runs_it(
    db_env, db_session_factory, run_pool_until, client
):
    """The actual end-to-end path: create via the API, run it through the real worker
    pool, and confirm the dashboard-facing read route reflects the outcome — this is what
    "the worker claims jobs and reports progress the dashboard can poll" (task 2 todo)
    means in practice."""
    from uadclaw.models import Job, JobState
    from uadclaw.settings import get_settings
    from uadclaw.worker import default_stage_handlers

    await _login(client)
    created = (await client.post("/jobs", json={"kind": "firmware_analysis"})).json()
    job_id = created["id"]
    job_uuid = uuid.UUID(job_id)

    settings = get_settings()

    async def _terminal() -> bool:
        async with db_session_factory() as session:
            db_job = await session.get(Job, job_uuid)
            return db_job is not None and db_job.state in (JobState.SUCCEEDED, JobState.FAILED)

    await run_pool_until(db_session_factory, settings, default_stage_handlers(), _terminal)

    resp = await client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "succeeded"
    assert body["stage"] == "branch"
    assert "completed stage branch" in body["log_tail"]
    assert body["finished_at"] is not None
