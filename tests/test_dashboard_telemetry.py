"""The telemetry screen: `stats.compute_stats`, rendered to answer "what should
`worker_pool_size` be". Never reimplements an aggregate — every assertion here checks that
the numbers `compute_stats` already produces reach the page, not that the arithmetic is right
(that belongs to `test_stats.py`).
"""

from datetime import UTC, datetime, timedelta

from conftest import FIRMWARE_TARGET
from uadclaw.jobs import create_job
from uadclaw.models import JobStageRun, JobState
from uadclaw.views.telemetry import TELEMETRY_REFRESH_INTERVAL_SECONDS

PASSWORD = "test-only-admin-password"


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


# --- auth -----------------------------------------------------------------------------------


async def test_telemetry_requires_auth(db_env, client):
    resp = await client.get("/telemetry")
    assert resp.status_code == 401


# --- empty vs error, the same distinction the corpus screen has to make ----------------------


async def test_a_fresh_database_reads_as_no_runs_yet_not_broken(db_env, db_session_factory, client):
    await _login(client)
    resp = await client.get("/telemetry", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "no job runs yet" in resp.text
    assert "could not load" not in resp.text


async def test_a_db_failure_reads_as_an_error_never_as_no_runs_yet(client):
    """No `db_env`: `POSTGRES_HOST` defaults to `postgres`, unreachable outside docker."""
    await _login(client)
    resp = await client.get("/telemetry", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "could not load" in resp.text
    assert "no job runs yet" not in resp.text


# --- success: real numbers reach the page -----------------------------------------------------


async def test_real_stage_and_worker_numbers_reach_the_page(db_env, db_session_factory, client):
    now = datetime.now(UTC)
    async with db_session_factory() as session, session.begin():
        job = await create_job(session, kind="firmware_analysis", params=FIRMWARE_TARGET)
        job.state = JobState.SUCCEEDED
        job.worker_id = "worker-telemetry-test"
        job.started_at = now - timedelta(seconds=10)
        job.finished_at = now - timedelta(seconds=2)
        job.attempt = 1
        session.add(
            JobStageRun(
                job_id=job.id,
                stage="acquire",
                started_at=now - timedelta(seconds=9),
                finished_at=now - timedelta(seconds=7),
                outcome="succeeded",
            )
        )

    await _login(client)
    resp = await client.get("/telemetry", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "no job runs yet" not in resp.text
    assert "acquire" in resp.text
    assert "worker-telemetry-test" in resp.text


# --- the htmx-fragment / boosted-navigation distinction ---------------------------------------


async def test_a_bare_hx_get_gets_the_fragment_without_the_shell(
    db_env, db_session_factory, client
):
    """The periodic auto-refresh issues a bare (non-boosted) `hx-get`. It must get just the
    content div's contents, or the poll would nest a whole extra page shell inside itself on
    every refresh."""
    await _login(client)
    resp = await client.get("/telemetry", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "<nav" not in resp.text
    assert 'id="telemetry-content"' not in resp.text


async def test_a_boosted_nav_click_still_gets_the_whole_page(db_env, db_session_factory, client):
    """`HX-Request: true` alone does not mean "wants a fragment" — hx-boost sets it too on an
    ordinary nav click. Without checking `HX-Boosted`, clicking into telemetry from another
    page via the boosted navbar would swap in a bare fragment missing the whole shell."""
    await _login(client)
    resp = await client.get("/telemetry", headers={"HX-Request": "true", "HX-Boosted": "true"})
    assert resp.status_code == 200
    assert "<nav" in resp.text
    assert 'id="telemetry-content"' in resp.text


async def test_a_plain_browser_navigation_gets_the_whole_page(db_env, db_session_factory, client):
    await _login(client)
    resp = await client.get("/telemetry", headers={"accept": "text/html"})
    assert "<nav" in resp.text


# --- the refresh interval is a named constant, not a literal sprinkled into the template ------


async def test_the_refresh_interval_in_the_page_matches_the_named_constant(
    db_env, db_session_factory, client
):
    await _login(client)
    resp = await client.get("/telemetry", headers={"accept": "text/html"})
    assert f'hx-trigger="every {TELEMETRY_REFRESH_INTERVAL_SECONDS}s"' in resp.text, (
        "the template must read the interval from code, not hardcode its own number"
    )
