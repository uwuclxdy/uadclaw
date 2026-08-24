"""The real Brave API and a real DeepSeek judge, opt-in and billed.

    UADCLAW_LIVE_API_TESTS=1 uv run pytest -n0 -s tests/test_corroborate_live.py

Excluded from the default suite, mirroring `test_deepseek_live.py` and `test_oppo_live.py`.
Everything the clients do is already pinned against a mocked transport in `test_brave.py` and
`test_corroboration_stage.py`; what only the live API can prove is that the response SHAPE this
repo parses is the shape Brave actually sends. A mock proves the code does what the mock was
written to expect, which is the wrong question for a wire format somebody else owns.

Deliberately small: a handful of searches, a handful of page fetches, two judge calls. The
full-corpus run is a separate, deliberate spend.

The verdict assertions are recorded rather than demanded. Whether an obscure OEM package name
corroborates at all is an open hypothesis, and a test that failed on an uncorroborated answer
would be asserting the answer to the question the stage exists to ask.
The ONE thing this file does demand is the fabricated-citation gate: an invented package name
must not come back with a made-up link.
"""

import json
import os

import pytest

from uadclaw.brave import BraveClient, PageFetcher, source_links
from uadclaw.corroborate import (
    SYSTEM_PROMPT,
    CorroborationRejected,
    CorroborationStatus,
    user_prompt,
    validate_verdict,
)
from uadclaw.deepseek import DeepSeekClient
from uadclaw.settings import Settings

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("UADCLAW_LIVE_API_TESTS") != "1",
        reason="opt-in: set UADCLAW_LIVE_API_TESTS=1 (spends real Brave quota and DeepSeek money)",
    ),
]

# A package upstream carries and the web talks about, and one that cannot exist.
REAL_PACKAGE = "com.android.cellbroadcastreceiver"
REAL_DESCRIPTION = (
    "Receives and displays government emergency alerts such as AMBER and presidential "
    "warnings. Removing it can be illegal in some countries."
)
INVENTED_PACKAGE = "com.zzznotarealpackage.qqqwwweee.fake"
INVENTED_DESCRIPTION = (
    "Manages the device's holographic projector calibration and its firmware updates."
)


@pytest.fixture
def settings(test_env) -> Settings:
    """`test_env` first, and not for the database: a `Settings()` that fails validation renders
    every field it had already collected into the ValidationError, the real credentials among
    them, and that traceback ends up in pytest output."""
    loaded = Settings()
    if not loaded.brave_key.get_secret_value().strip():
        pytest.skip("BRAVE_KEY is not configured on this box")
    if not loaded.deepseek_key.get_secret_value().strip():
        pytest.skip("DEEPSEEK_KEY is not configured on this box")
    return loaded


async def evidence(settings: Settings, package: str):
    async with (
        BraveClient.from_settings(settings) as brave,
        PageFetcher.from_settings(settings) as pages,
    ):
        hits = await brave.search(package, limit=settings.corroboration_sources_per_package)
        return await pages.fetch_all(hits)


async def test_the_live_api_answers_the_shape_this_client_parses(settings):
    """The whole point of this file. `description` rather than `snippet`, the `discussions`
    block beside `web`, and `mixed.main` as the display order."""
    sources = await evidence(settings, REAL_PACKAGE)

    assert sources, "the live API returned nothing to merge"
    print(
        "live brave probe:",
        json.dumps(
            {
                "package": REAL_PACKAGE,
                "sources": len(sources),
                "blocks": sorted({source.block for source in sources}),
                "page_text_used": sum(1 for source in sources if source.uses_page_text),
                "snippet_fallbacks": [
                    {
                        "url": source.url,
                        "error": source.fetch_error,
                        "extracted_chars": len(source.text or ""),
                    }
                    for source in sources
                    if not source.uses_page_text
                ],
                "text_chars": [len(source.text or "") for source in sources],
            },
            indent=2,
        ),
    )
    assert all(source.url.startswith(("http://", "https://")) for source in sources)
    assert [source.position for source in sources] == list(range(1, len(sources) + 1))
    assert any(source.uses_page_text for source in sources), (
        "not one of the top results yielded a page body beating its own snippet; the judge "
        "would be reading snippets alone, which is the fallback rather than the design"
    )


async def test_an_invented_package_still_returns_results_so_the_judge_carries_the_verdict(
    settings,
):
    """Measured 2026-08-12 and re-asserted here because it is the reason the four statuses
    exist: "no results" can never be the uncorroborated signal."""
    sources = await evidence(settings, INVENTED_PACKAGE)
    print("live invented-name probe:", len(sources), "result(s)")
    assert sources, "an invented package name returning zero results would change the design"


async def test_a_real_package_is_judged_against_real_sources(settings):
    """Recorded rather than demanded: whether obscure OEM package names corroborate is the
    open question, so a rejection here is a finding and not a test failure."""
    sources = await evidence(settings, REAL_PACKAGE)
    async with DeepSeekClient.from_settings(settings) as judge:
        result = await judge.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt(REAL_PACKAGE, REAL_DESCRIPTION, sources),
        )
    corroboration = validate_verdict(
        result.json_object(),
        package=REAL_PACKAGE,
        description=REAL_DESCRIPTION,
        sources=sources,
        model=result.model,
    )
    print(
        "live judge probe:",
        json.dumps(
            {
                "package": REAL_PACKAGE,
                "status": str(corroboration.status),
                "sources": source_links(sources, corroboration.sources),
                "reasoning": corroboration.reasoning,
                "usage": result.usage,
            },
            indent=2,
        ),
    )
    assert corroboration.status in {
        CorroborationStatus.CORROBORATED,
        CorroborationStatus.UNCORROBORATED,
    }
    handed = {source.url for source in sources}
    assert set(corroboration.sources) <= handed


async def test_an_invented_package_never_produces_a_fabricated_citation(settings):
    """The one live assertion this file demands, and the thing this stage owes upstream's
    maintainer: a judge that answers corroborated for a package that does not exist must cite
    a url it was actually given, or the validator refuses the whole response."""
    sources = await evidence(settings, INVENTED_PACKAGE)
    async with DeepSeekClient.from_settings(settings) as judge:
        result = await judge.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt(INVENTED_PACKAGE, INVENTED_DESCRIPTION, sources),
        )
    payload = result.json_object()
    print("live fabrication probe:", json.dumps(payload))
    handed = {source.url for source in sources}
    try:
        corroboration = validate_verdict(
            payload,
            package=INVENTED_PACKAGE,
            description=INVENTED_DESCRIPTION,
            sources=sources,
            model=result.model,
        )
    except CorroborationRejected as rejected:
        # A refusal is a pass: the row this produces is `judge_failed`, never a corroborated
        # entry carrying an invented link into a PR body.
        assert rejected.field in {"status", "sources", "reasoning"}
        return
    assert set(corroboration.sources) <= handed
    if corroboration.status is CorroborationStatus.CORROBORATED:
        pytest.fail(
            "the judge corroborated a package that does not exist, citing "
            f"{corroboration.sources}: {corroboration.reasoning}. That is a prompt finding "
            "rather than a validator bug — every url it cited was one it was handed."
        )
