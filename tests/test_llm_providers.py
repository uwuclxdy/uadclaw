"""The provider generalization: creation-boundary validation plus one classification job run
end to end against an OpenAI-format mock.

The end-to-end test here is the verify line for the provider table (decisions 2026-08-24):
one classification job queued through the real creation seam and run through the real
worker walks the whole kind — `llm` then `corroborate` — with every transport mocked, and
the same validator, floor and rejection guarantees the stage tests assert have to hold
through the new provider resolution. The mock speaks the OpenAI wire format the client
sends; nothing here touches the network.

The creation tests pin the other ruling: a job's `provider` param names a provider id,
validated against the settings table at creation — never at classify time — and the
RESOLVED id is what the row stores.
"""

import json

import httpx
import pytest
from sqlalchemy import select

from test_classification_stage import GOOD, UPSTREAM_JSON, WIDE_CORPUS, envelope, seed
from test_corroboration_stage import default_brave, verdict_response
from uadclaw import jobs as jobs_module
from uadclaw import stages as stages_module
from uadclaw.brave import BraveClient, PageFetcher
from uadclaw.jobs import JobValidationError
from uadclaw.llm import LlmClient
from uadclaw.models import (
    Job,
    JobKind,
    JobState,
    PackageClassification,
    PackageCorroboration,
)
from uadclaw.settings import get_settings
from uadclaw.stages import StageInputError, llm_stage
from uadclaw.worker import StageContext

PROVIDER_TABLE = {
    "deepseek": {
        "base_url": "https://api.deepseek.test",
        "model": "deepseek-v4-flash",
        "thinking": True,
        "max_tokens": 4096,
        "max_concurrency": 4,
    }
}


@pytest.fixture
def provider_env(monkeypatch, tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text(UPSTREAM_JSON, encoding="utf-8")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", str(path))
    monkeypatch.setenv("LLM_PROVIDERS", json.dumps(PROVIDER_TABLE))
    monkeypatch.setenv("LLM_DEEPSEEK_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0.001")
    monkeypatch.setenv("BRAVE_KEY", "BSA-test-not-a-real-token")
    monkeypatch.setenv("BRAVE_SEARCH_URL", "https://api.search.brave.test/res/v1/web/search")
    return path


@pytest.fixture
def fake_provider(monkeypatch):
    """Mock the TRANSPORT every client is built with, never the client itself: the llm
    validator, the judge's citation gate, the retry loops and the semaphores all still run.

    The same handler answers both halves of the job, because both stages build their client
    from the SAME provider id — dispatch on the prompt shape, exactly as the model reads it.
    """
    state = {"llm": [], "judgements": []}

    def llm_handler(request: httpx.Request) -> httpx.Response:
        state["llm"].append(request)
        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        if '"proposed_description"' in user:
            state["judgements"].append(user)
            return verdict_response("corroborated")(user)
        # The same floor guarantee the stage tests assert: `com.example.launcher` sits in
        # priv-app (floor Advanced) and the mock keeps answering below it, so the run must
        # park it, never clamp it.
        package = json.loads(user.split("EVIDENCE:\n", 1)[1])["package"]
        payload = dict(GOOD, removal="Recommended") if package == "com.example.launcher" else GOOD
        return httpx.Response(200, json=envelope(payload))

    original_llm = LlmClient.from_settings
    original_brave = BraveClient.from_settings
    original_pages = PageFetcher.from_settings

    monkeypatch.setattr(
        LlmClient,
        "from_settings",
        lambda settings, *, provider_id, transport=None: original_llm(
            settings, provider_id=provider_id, transport=httpx.MockTransport(llm_handler)
        ),
    )
    monkeypatch.setattr(
        BraveClient,
        "from_settings",
        lambda settings, *, transport=None: original_brave(
            settings,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=default_brave(request.url.params["q"]))
            ),
        ),
    )
    monkeypatch.setattr(
        PageFetcher,
        "from_settings",
        lambda settings, *, transport=None: original_pages(
            settings,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    content=b"<html><body><p>This page describes the package.</p></body></html>",
                )
            ),
        ),
    )
    return state


# --- creation: the provider id is validated at the boundary ----------------------------------


async def test_an_unknown_provider_is_refused_at_creation_naming_the_valid_ids(
    db_env, provider_env, db_session_factory
):
    async with db_session_factory() as session, session.begin():
        with pytest.raises(JobValidationError) as caught:
            await jobs_module.create_job(
                session, kind=JobKind.CLASSIFICATION.value, params={"provider": "openai"}
            )
    assert "provider 'openai'" in str(caught.value)
    assert "deepseek" in str(caught.value)


async def test_an_empty_provider_table_refuses_a_classification_job_at_creation(
    db_env, provider_env, monkeypatch, db_session_factory
):
    """The DEEPSEEK_* migration: a box that has not configured the table cannot queue the
    paid stage at all, and the refusal names the fix rather than a worker discovering it."""
    monkeypatch.setenv("LLM_PROVIDERS", "{}")
    get_settings.cache_clear()
    async with db_session_factory() as session, session.begin():
        with pytest.raises(JobValidationError) as caught:
            await jobs_module.create_job(session, kind=JobKind.CLASSIFICATION.value)
    assert "'deepseek'" in str(caught.value)
    assert "LLM_PROVIDERS" in str(caught.value)


async def test_the_default_provider_is_resolved_and_stored_on_the_row(
    db_env, provider_env, db_session_factory
):
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind=JobKind.CLASSIFICATION.value)
    assert job.params["provider"] == "deepseek"


async def test_a_named_provider_is_stored_on_the_row(
    db_env, provider_env, monkeypatch, db_session_factory
):
    table = dict(PROVIDER_TABLE)
    # thinking must be true on a non-deepseek id: the loader refuses false there, because
    # it would send DeepSeek's off-switch wire field to a provider that may not know it.
    table["openai"] = {
        "base_url": "https://api.openai.test",
        "model": "gpt-test",
        "thinking": True,
        "max_tokens": 2048,
        "max_concurrency": 2,
    }
    monkeypatch.setenv("LLM_PROVIDERS", json.dumps(table))
    get_settings.cache_clear()
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.CLASSIFICATION.value, params={"provider": "openai"}
        )
    assert job.params["provider"] == "openai"


async def test_a_provider_that_vanished_after_creation_fails_the_stage_naming_job_and_ids(
    db_env, provider_env, monkeypatch, db_session_factory, fake_provider
):
    """Creation validated the id, but the table can change between creation and run — the
    stage re-resolves and refuses before a single request rather than dying on a KeyError."""
    job_id = await seed(db_session_factory)
    monkeypatch.setenv("LLM_PROVIDERS", "{}")
    get_settings.cache_clear()

    with pytest.raises(StageInputError) as caught:
        await llm_stage(
            StageContext(
                job_id=job_id, attempt=1, scratch_dir=None, session_factory=db_session_factory
            )
        )

    assert f"job {job_id}" in str(caught.value)
    assert "'deepseek'" in str(caught.value)
    assert fake_provider["llm"] == []


# --- end to end: one job through the worker against the OpenAI-format mock ---------------------


async def test_one_classification_job_runs_end_to_end_against_an_openai_format_mock(
    db_env, provider_env, fake_provider, db_session_factory, run_pool_until
):
    """The verify line: the real creation seam, the real worker, the real stage walk, and a
    mock speaking the OpenAI wire format. The guarantees the stage tests assert — the floor
    rejection, the validator, the provenance, the corroboration gate — have to hold with the
    provider resolved from the table at every step."""
    job_id = await seed(db_session_factory, WIDE_CORPUS)
    settings = get_settings()

    async def _job_terminal() -> bool:
        async with db_session_factory() as session:
            job = await session.get(Job, job_id)
            return job is not None and job.state in {JobState.SUCCEEDED, JobState.FAILED}

    await run_pool_until(
        db_session_factory,
        settings,
        stages_module.pipeline_stage_handlers(),
        _job_terminal,
    )

    async with db_session_factory() as session:
        job = await session.get(Job, job_id)
        classifications = {
            row.package: row
            for row in (await session.execute(select(PackageClassification))).scalars()
        }
        verdicts = {
            row.package: row
            for row in (await session.execute(select(PackageCorroboration))).scalars()
        }

    assert job is not None and job.state is JobState.SUCCEEDED, job and job.failure_reason
    assert job.params["provider"] == "deepseek", "the creation seam stores the resolved id"
    # The floor guarantee, end to end: launcher is priv-app (floor Advanced), the mock keeps
    # answering Recommended, and the run must PARK it — a clamped row would read Advanced
    # here and nothing would catch it.
    parked = classifications["com.example.launcher"]
    assert parked.parked is True
    assert parked.removal is None, "a rejected proposal must not be stored, raised or otherwise"
    assert "removal" in parked.parked_reason
    assert "Advanced" in parked.parked_reason
    # The validator guarantee on the sibling the mock answered properly.
    notes = classifications["com.example.notes"]
    assert notes.parked is False
    assert notes.description == GOOD["description"]
    assert notes.provenance["description"].startswith("llm:")
    assert len(notes.bundle_sha256) == 64
    # The corroborate half ran through the SAME provider's client and stored a judged row.
    assert verdicts["com.example.notes"].status == "corroborated"
    assert verdicts["com.example.notes"].model == "deepseek-v4-flash"
    # The table drove the run: every request went to the configured base_url carrying the
    # configured model — not a hardcoded client.
    assert fake_provider["llm"], "the mock never saw a request"
    for request in fake_provider["llm"]:
        assert str(request.url).startswith("https://api.deepseek.test/chat/completions")
        assert json.loads(request.content)["model"] == "deepseek-v4-flash"
    assert fake_provider["judgements"], "the judge was never asked"
