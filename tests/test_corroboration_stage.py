"""The `corroborate` stage against a real database, with both APIs mocked at the transport.

The default suite must never touch the network and never need a key, so Brave, the page
fetcher and the judge all get an `httpx.MockTransport`. What only a database can prove is the
wiring, and four shapes of it are easy to get wrong and silent when they are:

- the FOUR statuses have to be storable and distinguishable — "we looked and found nothing",
  "we could not look" and "we could not judge" are three different facts with three different
  retry costs, and collapsing any two tells triage a package lacks support when nobody asked;
- a search failure on one package must leave the others' verdicts standing, because the stage
  is per-package resumable and one flaky request must not discard completed work;
- search rows are cached on the package NAME, so a `judge_failed` re-run must spend ZERO
  search quota while a row past its TTL must re-search;
- a corroboration run must leave `package_classification` byte-identical, because that table's
  contract is that it holds what the model proposed.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import select

from uadclaw import jobs as jobs_module
from uadclaw import stages as stages_module
from uadclaw.brave import BraveClient, PageFetcher
from uadclaw.classify import UNKNOWN, Classification, Confidence, UadList
from uadclaw.classifystore import park_package, store_classification
from uadclaw.corroborate import (
    RULE_PROVENANCE,
    CorroborationStatus,
    code_verdict,
    description_digest,
)
from uadclaw.corroboratestore import CorroborationStoreError, record_failure, store_verdict
from uadclaw.deepseek import DeepSeekBalanceError, DeepSeekClient
from uadclaw.ladder import Removal
from uadclaw.models import PackageClassification, PackageCorroboration, PackageSearchResult
from uadclaw.stages import corroborate_stage
from uadclaw.worker import StageContext

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)

NOTES = "com.example.notes"
LAUNCHER = "com.example.launcher"
NOTES_DESCRIPTION = "Vendor notes application. Removing it loses locally stored notes."
LAUNCHER_DESCRIPTION = "Vendor home screen. Removing it leaves the device with no launcher."


# --- the fake APIs ------------------------------------------------------------------------


def brave_body(package, *, web=None, discussions=None, mixed=None):
    body = {"query": {"original": package}}
    if web is not None:
        body["web"] = {"results": web}
    if discussions is not None:
        body["discussions"] = {"results": discussions}
    if mixed is not None:
        body["mixed"] = {"main": mixed}
    return body


def hit(url, *, title="A page", description="a summary"):
    return {"url": url, "title": title, "description": description}


def default_brave(package):
    """Two results per package, one per block, on hosts derived from the package name so a
    test can tell which package's search produced which row."""
    slug = package.rsplit(".", 1)[-1]
    return brave_body(
        package,
        web=[hit(f"https://docs.test/{slug}", title=f"{slug} docs", description="doc summary")],
        discussions=[
            hit(f"https://forum.test/{slug}", title=f"{slug} thread", description="forum summary")
        ],
    )


def judge_envelope(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": "chat-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9000,
            "completion_tokens": 90,
            "total_tokens": 9090,
            "prompt_cache_hit_tokens": 8192,
            "prompt_cache_miss_tokens": 808,
        },
    }


@pytest.fixture
def fake_apis(monkeypatch):
    """Replace the TRANSPORT each client is built with, never the client itself: the redirect
    refusal, the byte ceiling, the semaphores, the judge's retry loop and the request bodies
    all still run."""
    state = {
        # (package -> body) or a callable(request) -> httpx.Response
        "search": default_brave,
        "pages": lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><body><p>This page describes the package in some detail.</p>"
            b"</body></html>",
        ),
        "judge": None,
        "searches": [],
        "fetches": [],
        "judgements": [],
    }

    def search_handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        state["searches"].append(query)
        answer = state["search"]
        if callable(answer) and not isinstance(answer, dict):
            answer = answer(query)
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(200, json=answer)

    def page_handler(request: httpx.Request) -> httpx.Response:
        state["fetches"].append(str(request.url))
        return state["pages"](request)

    def judge_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["judgements"].append(body["messages"][1]["content"])
        answer = state["judge"]
        if callable(answer):
            return answer(body["messages"][1]["content"])
        return answer

    original_brave = BraveClient.from_settings
    original_pages = PageFetcher.from_settings
    original_judge = DeepSeekClient.from_settings

    monkeypatch.setattr(
        BraveClient,
        "from_settings",
        lambda settings, *, transport=None: original_brave(
            settings, transport=httpx.MockTransport(search_handler)
        ),
    )
    monkeypatch.setattr(
        PageFetcher,
        "from_settings",
        lambda settings, *, transport=None: original_pages(
            settings, transport=httpx.MockTransport(page_handler)
        ),
    )
    monkeypatch.setattr(
        DeepSeekClient,
        "from_settings",
        lambda settings, *, transport=None: original_judge(
            settings, transport=httpx.MockTransport(judge_handler)
        ),
    )
    return state


def prompt_urls(user_prompt: str) -> list[str]:
    """The urls the judge was actually handed, read out of the prompt the way the judge reads
    it. The prompt is one json object, so this parses rather than scraping lines."""
    return [source["url"] for source in json.loads(user_prompt)["sources"]]


def verdict_response(status="corroborated", *, sources=None, reasoning="It says so."):
    def answer(user_prompt: str) -> httpx.Response:
        cited = sources
        if cited is None:
            handed = prompt_urls(user_prompt)
            cited = handed[:1] if status == "corroborated" else []
        return httpx.Response(
            200,
            json=judge_envelope({"status": status, "sources": list(cited), "reasoning": reasoning}),
        )

    return answer


@pytest.fixture
def corroboration_env(monkeypatch):
    monkeypatch.setenv("BRAVE_KEY", "BSA-test-not-a-real-token")
    monkeypatch.setenv("DEEPSEEK_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("DEEPSEEK_RETRY_BACKOFF_SECONDS", "0.001")
    monkeypatch.setenv("BRAVE_SEARCH_URL", "https://api.search.brave.test/res/v1/web/search")


# --- seeding ------------------------------------------------------------------------------


def classification(package: str, description: str) -> Classification:
    return Classification(
        package=package,
        bundle_sha256="a" * 64,
        description=description,
        list=UadList.MISC,
        removal=Removal.ADVANCED,
        confidence=Confidence.MEDIUM,
        unknown_fields=(),
        reasoning_brief="",
        provenance={"description": "llm:deepseek-v4-flash"},
    )


async def seed(session_factory, proposals: dict[str, str]) -> uuid.UUID:
    """Classification rows written through their PRODUCTION writer, so this fixture cannot
    drift from what `llm_stage` actually stores."""
    async with session_factory() as session, session.begin():
        for package, description in proposals.items():
            await store_classification(
                session,
                classification(package, description),
                model="deepseek-v4-flash",
                thinking=True,
                usage={},
                attempts=1,
                at=NOW,
            )
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind="classification")
        return job.id


def context(job_id, session_factory) -> StageContext:
    return StageContext(job_id=job_id, attempt=1, scratch_dir=None, session_factory=session_factory)


async def rows(session_factory) -> dict[str, PackageCorroboration]:
    async with session_factory() as session:
        result = await session.execute(select(PackageCorroboration))
        return {row.package: row for row in result.scalars()}


async def search_rows(session_factory, package: str) -> list[PackageSearchResult]:
    async with session_factory() as session:
        result = await session.execute(
            select(PackageSearchResult)
            .where(PackageSearchResult.package == package)
            .order_by(PackageSearchResult.position)
        )
        return list(result.scalars())


# --- the four statuses ----------------------------------------------------------------------


async def test_a_corroborated_verdict_stores_its_sources_with_llm_provenance(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.CORROBORATED)
    assert row.sources == [{"url": "https://docs.test/notes", "title": "notes docs"}]
    assert row.provenance["status"] == "llm:deepseek-v4-flash"
    assert row.provenance["sources"] == "search:brave"
    assert row.model == "deepseek-v4-flash"
    assert row.attempts == 1
    assert row.failure_reason is None
    assert row.description_sha256 == description_digest(NOTES, NOTES_DESCRIPTION)


async def test_an_uncorroborated_verdict_is_stored_distinctly_and_names_no_source(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("uncorroborated", reasoning="Nothing identifies it.")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.UNCORROBORATED)
    assert row.sources == []
    assert row.reasoning == "Nothing identifies it."


async def test_a_search_failure_is_its_own_status_and_never_the_uncorroborated_one(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """ "We could not look" is not "we looked and found nothing": one is retryable and the
    other is an answer triage acts on."""
    fake_apis["search"] = lambda package: httpx.Response(503, text="overloaded")
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.SEARCH_FAILED)
    assert "503" in row.failure_reason
    assert row.model is None, "no model was asked, so no model is credited"
    assert row.provenance["status"] == RULE_PROVENANCE
    assert fake_apis["judgements"] == [], "nothing to judge, so nothing was billed"


async def test_a_judge_that_cannot_answer_is_its_own_status(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    assert "503" in row.failure_reason
    assert row.model == "deepseek-v4-flash", "the judge WAS asked, and was billed"
    assert row.attempts == 3
    assert len(await search_rows(db_session_factory, NOTES)) == 2, (
        "the search rows are committed before the judge runs, so a retry is free"
    )


async def test_a_search_returning_nothing_is_uncorroborated_without_asking_a_model(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """With zero sources, "no independent source supports this" is arithmetic rather than a
    judgement, so the row carries `rule:` provenance and no call was billed."""
    fake_apis["search"] = lambda package: brave_body(package, web=[])
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.UNCORROBORATED)
    assert row.provenance["status"] == RULE_PROVENANCE
    assert row.model is None
    assert fake_apis["judgements"] == []


# --- per-package resumability -----------------------------------------------------------------


async def test_a_search_failure_on_one_package_leaves_every_other_verdict_standing(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """One flaky request must not discard the run's completed work."""
    packages = {
        f"com.example.p{index}": f"Package {index} does a thing worth describing."
        for index in range(1, 5)
    }

    def search(package):
        if package == "com.example.p3":
            raise httpx.ConnectError("connection refused")
        return default_brave(package)

    fake_apis["search"] = search
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, packages)

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert sorted(stored) == sorted(packages)
    assert stored["com.example.p3"].status == str(CorroborationStatus.SEARCH_FAILED)
    for package in ("com.example.p1", "com.example.p2", "com.example.p4"):
        assert stored[package].status == str(CorroborationStatus.CORROBORATED), package


async def test_one_packages_fetch_failure_never_cancels_a_sibling_or_loses_its_paid_search(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The blocker, at the stage level and in both halves.

    `fetch_all` used `asyncio.gather` with no `return_exceptions` and `_corroborate_and_store`
    ran inside a `TaskGroup`, so an exception class nothing caught — `httpx.InvalidURL` is not
    an `httpx.HTTPError` — cancelled every sibling package including judge calls already paid
    for. And the search rows were written only AFTER the fetch phase, so the poisoned package
    cached nothing, recorded nothing, and burned a fresh Brave query on every future run.

    Asserted on the specific outcome rather than on "nothing raised": the siblings hold real
    verdicts, and the poisoned package holds BOTH a status and its stored search rows.
    """
    packages = {
        f"com.example.p{index}": f"Package {index} does a thing worth describing."
        for index in range(1, 6)
    }
    poisoned = "com.example.p3"

    def search(package):
        if package == poisoned:
            return brave_body(package, web=[hit("https://ok.test/a"), hit("https://bad.test/b")])
        return default_brave(package)

    def pages(request):
        if request.url.host == "bad.test":
            raise httpx.InvalidURL("Invalid IDNA hostname: 'ａｂｃ.com'")
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<p>a real page body</p>"
        )

    fake_apis["search"] = search
    fake_apis["pages"] = pages
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, packages)

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert sorted(stored) == sorted(packages), "every package reached a row"
    for package in packages:
        assert stored[package].status == str(CorroborationStatus.CORROBORATED), package
    # The paid query landed: a re-run reads these rows and spends nothing.
    cached = {row.url: row for row in await search_rows(db_session_factory, poisoned)}
    assert sorted(cached) == ["https://bad.test/b", "https://ok.test/a"]
    assert cached["https://ok.test/a"].page_text == "a real page body"
    assert "InvalidURL" in cached["https://bad.test/b"].fetch_error


async def test_a_nul_byte_in_a_page_body_is_one_packages_problem_and_not_the_jobs(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The second trigger for the same abort. A NUL survives `decode(errors="replace")` and
    `str.split()`, so a `text/plain` body carrying one reached asyncpg and raised
    `CharacterNotInRepertoireError` at the store — killing the whole job with queries already
    spent. Stripped at extraction now, so the row is written and the text is clean."""
    packages = {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}

    def pages(request):
        if "notes" in str(request.url):
            return httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"notes\x00 app body"
            )
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<p>the launcher body</p>"
        )

    fake_apis["pages"] = pages
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, packages)

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert stored[NOTES].status == str(CorroborationStatus.CORROBORATED)
    assert stored[LAUNCHER].status == str(CorroborationStatus.CORROBORATED)
    texts = [row.page_text for row in await search_rows(db_session_factory, NOTES)]
    assert texts == ["notes app body", "notes app body"]


async def test_an_unexpected_failure_after_the_search_is_recorded_as_this_packages_own(
    db_env, corroboration_env, fake_apis, db_session_factory, monkeypatch
):
    """The general containment, with the phase it lands in. The sources were already stored,
    so the retry is free and the row says `judge_failed` rather than `search_failed`."""
    packages = {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}
    real_judge = stages_module._judge_one

    async def judge_one(client, *, package, **kwargs):
        if package == NOTES:
            raise RuntimeError("something nobody predicted")
        return await real_judge(client, package=package, **kwargs)

    monkeypatch.setattr(stages_module, "_judge_one", judge_one)
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, packages)

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert stored[LAUNCHER].status == str(CorroborationStatus.CORROBORATED), "sibling survived"
    assert stored[NOTES].status == str(CorroborationStatus.JUDGE_FAILED)
    assert "RuntimeError" in stored[NOTES].failure_reason
    assert stored[NOTES].sources == []
    assert await search_rows(db_session_factory, NOTES), "the paid search is cached for the retry"


async def test_an_account_level_deepseek_failure_still_aborts_the_whole_job(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The control for the containment above. `DeepSeekBalanceError` is about the ACCOUNT, not
    about this package, so burning 499 more packages' budgets against it is not a diagnosis —
    a boundary that swallowed everything would turn this into 500 quiet `judge_failed` rows."""
    fake_apis["judge"] = lambda _prompt: httpx.Response(402, text="Insufficient Balance")
    job_id = await seed(
        db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        await corroborate_stage(context(job_id, db_session_factory))

    assert caught.group_contains(DeepSeekBalanceError)
    assert await rows(db_session_factory) == {}, "no package was recorded as its own failure"


async def test_a_failed_recovery_write_costs_its_own_package_and_not_its_siblings(
    db_env, corroboration_env, fake_apis, db_session_factory, monkeypatch
):
    """The same hole as `llm_stage`'s, in the same shape, closed the same way.

    `record_failure` is the call that turns any per-package failure into that package's own
    row, and it ran with nothing around it inside a `TaskGroup`: a recovery write that raised
    cancelled every sibling, judge calls already paid for included, to report a failure that
    belonged to one package. It stays job-level and it stops being group-level.
    """
    packages = {
        f"com.example.p{index}": f"Package {index} does a thing worth describing."
        for index in range(1, 5)
    }
    poisoned = "com.example.p2"
    real_record = stages_module.record_failure

    async def judge_one(client, *, package, **kwargs):
        if package == poisoned:
            raise RuntimeError("the judge phase blew up in a way nobody predicted")
        return await real_judge(client, package=package, **kwargs)

    async def record(session, package, **kwargs):
        if package == poisoned:
            raise RuntimeError("and the recovery write failed too")
        await real_record(session, package, **kwargs)

    real_judge = stages_module._judge_one
    monkeypatch.setattr(stages_module, "_judge_one", judge_one)
    monkeypatch.setattr(stages_module, "record_failure", record)
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, packages)

    with pytest.raises(stages_module.StageRecordError, match=poisoned):
        await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert poisoned not in stored, "nothing could be written for it, which is why the job fails"
    for package in packages:
        if package == poisoned:
            continue
        assert stored[package].status == str(CorroborationStatus.CORROBORATED), package


async def test_a_paid_search_reaches_the_database_whatever_the_fetch_phase_does(
    db_env, corroboration_env, fake_apis, db_session_factory, monkeypatch
):
    """`fetch_all` no longer raises; this is the second half of the guarantee at the site that
    spent the query. A search that has been PAID FOR must never be lost to the phase after it,
    or the package re-poisons itself and burns a fresh query on every future run."""

    async def exploding_fetch_all(self, sources):
        raise RuntimeError("the fetch phase blew up in a way nobody predicted")

    monkeypatch.setattr(PageFetcher, "fetch_all", exploding_fetch_all)
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    cached = await search_rows(db_session_factory, NOTES)
    assert [row.url for row in cached] == ["https://docs.test/notes", "https://forum.test/notes"]
    assert all(row.page_text is None for row in cached)
    assert all("the fetch phase failed" in row.fetch_error for row in cached)
    assert len(fake_apis["searches"]) == 1


# --- the fabricated-citation gate, end to end ---------------------------------------------------


async def test_a_verdict_citing_a_url_that_was_never_fetched_is_refused_and_parks(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """`docs/todo.md` §8's verify line, at the stage level: a judge that answers corroborated
    with an invented link must not produce a corroborated row."""
    fake_apis["judge"] = verdict_response(
        "corroborated", sources=["https://invented.test/com.example.notes"]
    )
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    assert row.sources == [], "nothing invented reaches the row"
    assert "fabricated citation" in row.failure_reason
    assert "invented.test" in row.failure_reason
    assert len(fake_apis["judgements"]) == 3, "re-prompted within the budget, then recorded"


async def test_a_judge_claiming_a_pipeline_failure_status_is_refused(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("search_failed", sources=[])
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    assert "status" in row.failure_reason
    assert "PIPELINE" in row.failure_reason


# --- what the judge is shown ---------------------------------------------------------------------


async def test_discussion_results_reach_the_judge_beside_the_web_ones(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """Forum threads are where real packages corroborate, and they only arrive in the
    `discussions` block."""
    fake_apis["judge"] = verdict_response("uncorroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    prompt = fake_apis["judgements"][0]
    assert "https://forum.test/notes" in prompt
    assert "https://docs.test/notes" in prompt
    blocks = {row.url: row.block for row in await search_rows(db_session_factory, NOTES)}
    assert blocks["https://forum.test/notes"] == "discussions"


async def test_a_source_whose_page_failed_reaches_the_judge_as_its_snippet(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    def pages(request):
        if "forum" in str(request.url):
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><body><p>The documentation body.</p></body></html>",
        )

    fake_apis["pages"] = pages
    fake_apis["judge"] = verdict_response("uncorroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    handed = {source["url"]: source for source in json.loads(fake_apis["judgements"][0])["sources"]}
    forum = handed["https://forum.test/notes"]
    assert forum["text"] == "forum summary", "the snippet stands in rather than the result dropped"
    assert forum["evidence_kind"] == "search_snippet"
    assert handed["https://docs.test/notes"]["evidence_kind"] == "page_text"
    stored = {row.url: row for row in await search_rows(db_session_factory, NOTES)}
    assert stored["https://forum.test/notes"].page_text is None
    assert "ReadTimeout" in stored["https://forum.test/notes"].fetch_error
    assert stored["https://docs.test/notes"].page_text == "The documentation body."


async def test_the_stored_search_row_keeps_text_rather_than_raw_html(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("uncorroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await search_rows(db_session_factory, NOTES)
    assert all("<html>" not in (row.page_text or "") for row in stored)
    assert stored[0].page_text == "This page describes the package in some detail."


# --- the cache and its TTL --------------------------------------------------------------------


async def test_a_judge_failed_rerun_spends_zero_search_quota(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """Search rows are keyed on the package NAME precisely so this is free. If it were not,
    every prompt-change re-judge would re-buy the same search."""
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    assert (await rows(db_session_factory))[NOTES].status == str(CorroborationStatus.JUDGE_FAILED)
    searches = len(fake_apis["searches"])
    fetches = len(fake_apis["fetches"])

    fake_apis["judge"] = verdict_response("corroborated")
    await corroborate_stage(context(job_id, db_session_factory))

    assert len(fake_apis["searches"]) == searches, "no second search"
    assert len(fake_apis["fetches"]) == fetches, "no second page fetch"
    assert (await rows(db_session_factory))[NOTES].status == str(CorroborationStatus.CORROBORATED)


async def test_a_rewritten_description_re_judges_against_the_cached_search(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("uncorroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    searches = len(fake_apis["searches"])

    rewritten = "Vendor notes application. Removing it loses notes and their attachments."
    async with db_session_factory() as session, session.begin():
        await store_classification(
            session,
            classification(NOTES, rewritten),
            model="deepseek-v4-flash",
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )
    fake_apis["judge"] = verdict_response("corroborated")
    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.CORROBORATED)
    assert row.description_sha256 == description_digest(NOTES, rewritten)
    assert len(fake_apis["searches"]) == searches, (
        "the query did not change, so neither did the rows"
    )


async def test_a_search_row_past_its_ttl_is_re_searched(
    db_env, corroboration_env, fake_apis, monkeypatch, db_session_factory
):
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    searches = len(fake_apis["searches"])

    # Backdate the rows past the window rather than waiting one out.
    stale = datetime.now(UTC) - timedelta(days=31)
    async with db_session_factory() as session, session.begin():
        for row in await search_rows(db_session_factory, NOTES):
            fetched = await session.get(PackageSearchResult, row.id)
            fetched.fetched_at = stale

    await corroborate_stage(context(job_id, db_session_factory))

    assert len(fake_apis["searches"]) == searches + 1, "outside the TTL, the package re-searches"


async def test_a_re_search_that_finds_nothing_removes_the_rows_it_replaces(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The empty case of the replacement, which the non-empty one cannot reach: a search whose
    answer is now zero results must leave zero rows, or the package keeps a source set the web
    no longer returns and the next judge weighs evidence nothing found."""
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    assert len(await search_rows(db_session_factory, NOTES)) == 2

    stale = datetime.now(UTC) - timedelta(days=31)
    async with db_session_factory() as session, session.begin():
        for row in await search_rows(db_session_factory, NOTES):
            fetched = await session.get(PackageSearchResult, row.id)
            fetched.fetched_at = stale
    fake_apis["search"] = lambda package: brave_body(package, web=[])
    await corroborate_stage(context(job_id, db_session_factory))

    assert await search_rows(db_session_factory, NOTES) == []
    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.UNCORROBORATED)
    assert row.provenance["status"] == RULE_PROVENANCE


async def test_the_two_retry_layers_do_not_multiply_and_the_row_reports_true_spend(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The judge's budget is counted in REQUESTS: a `503, 503, rejected-verdict` cycle costs
    three requests inside ONE re-prompt, so charging one per `complete_json` would let the
    package spend three times its ceiling while the row reported a third of it.

    This shape does not exercise the HAND-OFF, because the first `complete_json` consumes the
    whole budget and there is no second call to hand a remainder to. The test below is the one
    that separates `max_calls - calls` from `max_calls`.
    """
    cycle = [
        httpx.Response(503, text="busy"),
        httpx.Response(503, text="busy"),
        httpx.Response(
            200,
            json=judge_envelope(
                {"status": "corroborated", "sources": ["https://invented.test/x"], "reasoning": ""}
            ),
        ),
    ]
    state = {"index": 0}

    def judge(_prompt):
        response = cycle[state["index"] % len(cycle)]
        state["index"] += 1
        return response

    fake_apis["judge"] = judge
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    # The row's own count IS the transport's count, and both are the single named budget.
    assert row.attempts == len(fake_apis["judgements"]) == 3


async def test_the_second_call_is_handed_only_what_the_first_one_left(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The budget hand-off itself: `complete_json(max_calls=max_calls - calls)`.

    A `503, rejected, 503, 503, rejected` cycle against a ceiling of 3 is the shape that
    separates it. Handing the remainder, the first call spends two requests (a 503 and a
    parsable-but-rejected answer) and the second is allowed exactly one; handing the FULL
    ceiling again, the second call retries its own way through three more, so the package makes
    five wire requests against a documented ceiling of three and the row still reports three.
    """
    cycle = [
        httpx.Response(503, text="busy"),
        httpx.Response(
            200,
            json=judge_envelope(
                {"status": "corroborated", "sources": ["https://invented.test/x"], "reasoning": ""}
            ),
        ),
        httpx.Response(503, text="busy"),
        httpx.Response(503, text="busy"),
        httpx.Response(
            200,
            json=judge_envelope(
                {"status": "corroborated", "sources": ["https://invented.test/y"], "reasoning": ""}
            ),
        ),
    ]
    state = {"index": 0}

    def judge(_prompt):
        response = cycle[min(state["index"], len(cycle) - 1)]
        state["index"] += 1
        return response

    fake_apis["judge"] = judge
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    await corroborate_stage(context(job_id, db_session_factory))

    assert len(fake_apis["judgements"]) == 3, "five requests means the remainder was not handed on"
    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    assert row.attempts == 3


async def test_a_re_search_replaces_the_previous_rows_rather_than_accumulating(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """A result the new search no longer returns has to stop being on the package: leaving it
    there dated from a previous run is indistinguishable from this search having found it."""
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))

    stale = datetime.now(UTC) - timedelta(days=31)
    async with db_session_factory() as session, session.begin():
        for row in await search_rows(db_session_factory, NOTES):
            fetched = await session.get(PackageSearchResult, row.id)
            fetched.fetched_at = stale
    fake_apis["search"] = lambda package: brave_body(
        package, web=[hit("https://fresh.test/only", title="fresh")]
    )
    await corroborate_stage(context(job_id, db_session_factory))

    stored = await search_rows(db_session_factory, NOTES)
    assert [row.url for row in stored] == ["https://fresh.test/only"]


# --- idempotence and selection -------------------------------------------------------------------


async def test_a_second_run_over_unchanged_descriptions_calls_nothing(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(
        db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}
    )
    await corroborate_stage(context(job_id, db_session_factory))
    before = (len(fake_apis["searches"]), len(fake_apis["fetches"]), len(fake_apis["judgements"]))

    await corroborate_stage(context(job_id, db_session_factory))

    assert (
        len(fake_apis["searches"]),
        len(fake_apis["fetches"]),
        len(fake_apis["judgements"]),
    ) == before


async def test_an_uncorroborated_verdict_is_not_re_asked_on_a_re_run(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """It is an ANSWER. Re-asking it every run would spend a query and a judge call per
    package forever, which is what separates it from the two failure statuses."""
    fake_apis["judge"] = verdict_response("uncorroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    before = len(fake_apis["judgements"])

    await corroborate_stage(context(job_id, db_session_factory))

    assert len(fake_apis["judgements"]) == before


async def test_a_limit_is_a_deterministic_name_ordered_prefix(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION})
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind="classification", params={"limit": 1})
        job_id = job.id

    await corroborate_stage(context(job_id, db_session_factory))

    assert list(await rows(db_session_factory)) == [LAUNCHER], "name-ordered prefix"


async def test_named_packages_replace_the_queue(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION})
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind="classification", params={"packages": [NOTES]}
        )
        job_id = job.id

    await corroborate_stage(context(job_id, db_session_factory))

    assert list(await rows(db_session_factory)) == [NOTES]


async def test_the_settings_ceiling_cannot_be_raised_by_a_jobs_own_params(
    db_env, corroboration_env, fake_apis, monkeypatch, db_session_factory
):
    monkeypatch.setenv("CORROBORATION_MAX_PACKAGES", "1")
    fake_apis["judge"] = verdict_response("corroborated")
    await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION})
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind="classification", params={"limit": 500})
        job_id = job.id

    await corroborate_stage(context(job_id, db_session_factory))

    assert len(await rows(db_session_factory)) == 1


async def test_the_query_budget_bounds_the_search_spend_and_records_the_rest(
    db_env, corroboration_env, fake_apis, monkeypatch, db_session_factory
):
    """A spent budget is a `search_failed` row rather than a silent skip: the package has to
    stay visible and retryable."""
    monkeypatch.setenv("CORROBORATION_MAX_QUERIES_PER_JOB", "1")
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(
        db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}
    )

    await corroborate_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert len(fake_apis["searches"]) == 1
    statuses = sorted(row.status for row in stored.values())
    assert statuses == ["corroborated", "search_failed"]
    spent = next(row for row in stored.values() if row.status == "search_failed")
    assert "budget" in spent.failure_reason


# --- what is and is not a candidate ---------------------------------------------------------------


async def test_a_parked_classification_is_never_a_candidate(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    async with db_session_factory() as session, session.begin():
        await park_package(
            session,
            LAUNCHER,
            bundle_sha256="b" * 64,
            model="deepseek-v4-flash",
            thinking=True,
            reason="removal: below the floor",
            usage={},
            attempts=3,
            at=NOW,
        )

    await corroborate_stage(context(job_id, db_session_factory))

    assert list(await rows(db_session_factory)) == [NOTES]


async def test_a_description_of_unknown_is_never_a_candidate(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """There is no claim in it to corroborate. Asking a judge whether the internet supports the
    word "unknown" spends a query and a call to learn nothing."""
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: UNKNOWN})

    await corroborate_stage(context(job_id, db_session_factory))

    assert list(await rows(db_session_factory)) == [NOTES]
    assert fake_apis["searches"] == [NOTES]


async def test_an_empty_classification_table_is_refused_rather_than_silently_succeeding(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """Zero rows means the llm stage never ran, not "nothing worth corroborating"."""
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind="classification")
        job_id = job.id

    with pytest.raises(CorroborationStoreError, match="never run"):
        await corroborate_stage(context(job_id, db_session_factory))

    assert fake_apis["searches"] == []


async def test_a_corpus_that_parked_everything_is_a_no_op_rather_than_a_failure(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """Distinct from the above: the llm stage DID run and legitimately produced nothing to
    check, which is a corpus state rather than a pipeline-order error."""
    async with db_session_factory() as session, session.begin():
        await park_package(
            session,
            NOTES,
            bundle_sha256="b" * 64,
            model="deepseek-v4-flash",
            thinking=True,
            reason="removal: below the floor",
            usage={},
            attempts=3,
            at=NOW,
        )
        job = await jobs_module.create_job(session, kind="classification")
        job_id = job.id

    await corroborate_stage(context(job_id, db_session_factory))

    assert await rows(db_session_factory) == {}
    assert fake_apis["searches"] == []


# --- the failure write never leaves a stale claim -------------------------------------------------


async def test_a_failure_against_a_new_description_clears_the_previous_sources(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    """The `classifystore.park_package` shape, in this table. A row whose
    `description_sha256` names a claim while its `sources` answered a different one asserts
    that those sources support something nobody checked them against."""
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})
    await corroborate_stage(context(job_id, db_session_factory))
    first = (await rows(db_session_factory))[NOTES]
    assert first.sources

    rewritten = "Vendor notes application. It also uploads every note to the vendor's cloud."
    async with db_session_factory() as session, session.begin():
        await store_classification(
            session,
            classification(NOTES, rewritten),
            model="deepseek-v4-flash",
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )
    fake_apis["judge"] = lambda _prompt: httpx.Response(503, text="overloaded")
    await corroborate_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))[NOTES]
    assert row.status == str(CorroborationStatus.JUDGE_FAILED)
    assert row.description_sha256 == description_digest(NOTES, rewritten)
    assert row.sources == [], "a verdict that answered a different claim does not survive"
    assert row.reasoning is None
    assert row.created_at == first.created_at, "created_at keeps meaning 'first corroborated'"


# --- nothing model-owned reaches package_classification ------------------------------------


async def test_a_corroboration_run_leaves_package_classification_untouched(
    db_env, corroboration_env, fake_apis, db_session_factory
):
    fake_apis["judge"] = verdict_response("corroborated")
    job_id = await seed(
        db_session_factory, {NOTES: NOTES_DESCRIPTION, LAUNCHER: LAUNCHER_DESCRIPTION}
    )

    async def snapshot():
        async with db_session_factory() as session:
            result = await session.execute(
                select(PackageClassification).order_by(PackageClassification.package)
            )
            return [
                (row.package, row.description, row.removal, row.updated_at, row.provenance)
                for row in result.scalars()
            ]

    before = await snapshot()
    await corroborate_stage(context(job_id, db_session_factory))
    assert await snapshot() == before


# --- the key is checked before anything is spent ------------------------------------------------


async def test_a_missing_brave_key_fails_before_any_request_is_made(
    db_env, corroboration_env, fake_apis, monkeypatch, db_session_factory
):
    monkeypatch.setenv("BRAVE_KEY", "")
    job_id = await seed(db_session_factory, {NOTES: NOTES_DESCRIPTION})

    with pytest.raises(Exception, match="BRAVE_KEY is empty"):
        await corroborate_stage(context(job_id, db_session_factory))

    assert fake_apis["searches"] == []
    assert fake_apis["judgements"] == []
    assert await rows(db_session_factory) == {}


# --- the two writers refuse each other's rows ---------------------------------------------------


@pytest.mark.parametrize(
    "status", [CorroborationStatus.CORROBORATED, CorroborationStatus.UNCORROBORATED]
)
async def test_record_failure_refuses_to_write_a_verdict_as_a_failure(
    db_env, db_session_factory, status
):
    """The guard survived a mutation across the whole suite because nothing called it with a
    verdict. `record_failure` always clears the sources, so a verdict written through it would
    lose them while the row still claimed to be an answer."""
    async with db_session_factory() as session, session.begin():
        with pytest.raises(CorroborationStoreError, match="is a verdict, not a failure"):
            await record_failure(
                session,
                NOTES,
                description=NOTES_DESCRIPTION,
                status=status,
                reason="whatever",
                at=NOW,
            )
    assert await rows(db_session_factory) == {}


@pytest.mark.parametrize(
    "status", [CorroborationStatus.SEARCH_FAILED, CorroborationStatus.JUDGE_FAILED]
)
async def test_store_verdict_refuses_to_write_a_failure_as_a_verdict(
    db_env, db_session_factory, status
):
    """The missing mirror. Without it a `search_failed` went in through `store_verdict` WITH
    sources and `failure_reason=None` — the exact row shape the pair of guards exists to
    prevent, arriving through the door nobody had locked."""
    failure = code_verdict(NOTES, NOTES_DESCRIPTION, status=status, reasoning="")
    async with db_session_factory() as session, session.begin():
        with pytest.raises(CorroborationStoreError, match="is a pipeline failure, not a verdict"):
            await store_verdict(
                session,
                failure,
                model="deepseek-v4-flash",
                thinking=True,
                sources=[{"url": "https://docs.test/notes", "title": "notes docs"}],
                usage={},
                attempts=1,
                at=NOW,
            )
    assert await rows(db_session_factory) == {}


async def test_the_control_both_writers_still_write_the_rows_they_own(db_env, db_session_factory):
    """Without this, "raise on every call" would satisfy both assertions above."""
    async with db_session_factory() as session, session.begin():
        await store_verdict(
            session,
            code_verdict(NOTES, NOTES_DESCRIPTION, status=CorroborationStatus.UNCORROBORATED),
            model=None,
            thinking=None,
            sources=[],
            usage={},
            attempts=0,
            at=NOW,
        )
        await record_failure(
            session,
            LAUNCHER,
            description=LAUNCHER_DESCRIPTION,
            status=CorroborationStatus.SEARCH_FAILED,
            reason="the search never happened",
            at=NOW,
        )
    stored = await rows(db_session_factory)
    assert stored[NOTES].status == str(CorroborationStatus.UNCORROBORATED)
    assert stored[LAUNCHER].status == str(CorroborationStatus.SEARCH_FAILED)


async def test_the_corroborate_handler_is_registered_for_the_classification_kind_only():
    handlers = stages_module.pipeline_stage_handlers()
    assert handlers["corroborate"] is corroborate_stage
    assert "corroborate" not in jobs_module.stages_for("firmware_analysis")
    assert jobs_module.stages_for("classification") == ("llm", "corroborate")
