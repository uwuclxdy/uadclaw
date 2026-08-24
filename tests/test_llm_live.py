"""The real DeepSeek API, opt-in and billed, through the generic client.

    UADCLAW_LIVE_API_TESTS=1 uv run pytest -n0 tests/test_llm_live.py

Excluded from the default suite, mirroring `UADCLAW_HEAVY_TESTS=1`. Everything the client
does is already pinned against a mocked transport in `test_llm.py`; what only the live
API can prove is that the request shape this repo ships is one it accepts, and that the
`usage` fields the whole cost measurement depends on are actually returned. A mock proves the
code does what the mock was written to expect, which is the wrong question for a wire format
somebody else owns.

Deliberately small: one classification-shaped call per behaviour, a handful of calls per run.
The full-corpus run is a separate, deliberate spend.

The key is read through the settings loader and is never written to a fixture, a log or an
assertion message.
"""

import json
import os

import pytest

from uadclaw.bundle import PackageIdentity, build_bundle
from uadclaw.classify import SYSTEM_PROMPT, derive_list, user_prompt, validate_response
from uadclaw.corpus import CorpusPackage
from uadclaw.ladder import Removal, compute_floors
from uadclaw.llm import CACHE_HIT_TOKENS, CACHE_MISS_TOKENS, LlmBudgetError, LlmClient
from uadclaw.settings import Settings

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("UADCLAW_LIVE_API_TESTS") != "1",
        reason="opt-in: set UADCLAW_LIVE_API_TESTS=1 (spends real money on the DeepSeek key)",
    ),
]

CORPUS = [
    CorpusPackage(
        package="com.android.settings",
        core_app=True,
        priv_app=True,
        partitions=("system",),
        devices=("pixel:oriole",),
    ),
    CorpusPackage(
        package="com.example.vendor.notes",
        partitions=("product",),
        devices=("pixel:oriole",),
    ),
]
FLOORS = compute_floors(CORPUS)


@pytest.fixture
def settings(test_env) -> Settings:
    """`test_env` first, and not for the database: a `Settings()` that fails validation
    renders EVERY field it had already collected into the ValidationError, the real provider
    key among them, and that traceback ends up in pytest output and in a job's `log_tail`.
    Satisfying the three required credentials keeps this construction from being the thing
    that prints one.

    The provider table is now part of what the live run needs: the deepseek entry has to be
    in `LLM_PROVIDERS` (the old `DEEPSEEK_*` settings are gone) and its key beside it.
    """
    loaded = Settings()
    if "deepseek" not in loaded.llm_providers:
        pytest.skip("the deepseek provider is not in LLM_PROVIDERS on this box")
    if not loaded.provider_key("deepseek"):
        pytest.skip(
            "the deepseek provider key (LLM_DEEPSEEK_KEY / secrets/llm_deepseek_key) "
            "is not configured on this box"
        )
    return loaded


def bundle(index: int):
    item = CORPUS[index]
    return build_bundle(
        item,
        floor=FLOORS[item.package],
        identity=PackageIdentity(label="Settings", cert_issuer="Organization: Google Inc."),
    )


async def test_the_real_api_accepts_the_shipped_request_shape(settings):
    """The whole point of this file: json_object mode, the shipped system prompt, a real
    evidence bundle, and a response the shipped validator accepts."""
    evidence = bundle(1)
    async with LlmClient.from_settings(settings, provider_id="deepseek") as client:
        result = await client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(evidence))

    provider = settings.llm_providers["deepseek"]
    assert result.finish_reason == "stop", (
        f"finish_reason={result.finish_reason!r} with max_tokens={provider.max_tokens}; "
        f"reasoning spent {result.reasoning_tokens}"
    )
    payload = result.json_object()
    classification = validate_response(
        payload,
        bundle=evidence,
        floor=FLOORS["com.example.vendor.notes"],
        derivation=derive_list("com.example.vendor.notes"),
        model=result.model,
    )
    assert classification.description
    assert classification.provenance["description"].startswith("llm:")


async def test_the_usage_envelope_carries_the_cache_fields_the_cost_measurement_reads(settings):
    """The reason this repo calls the OpenAI-format endpoint rather than /anthropic. If these
    two fields ever stop being returned, the cost measurement silently reads zero."""
    async with LlmClient.from_settings(settings, provider_id="deepseek") as client:
        result = await client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(bundle(1)))

    assert CACHE_HIT_TOKENS in result.usage, sorted(result.usage)
    assert CACHE_MISS_TOKENS in result.usage, sorted(result.usage)
    assert result.usage["prompt_tokens"] > 0
    assert result.usage["completion_tokens"] > 0


async def test_a_repeated_shared_prefix_eventually_reports_a_cache_hit(settings):
    """DeepSeek's own worked example shows the first two calls missing and the third hitting,
    because a common prefix only becomes cacheable once the system has DETECTED it. So this
    asserts the mechanism exists at all rather than a hit rate."""
    evidence = bundle(0)
    hits = []
    async with LlmClient.from_settings(settings, provider_id="deepseek") as client:
        for _ in range(3):
            result = await client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(evidence))
            hits.append(result.cache_hit_tokens)

    # Printed unconditionally, not only on failure: the warm-up shape is the input task 8's
    # cost measurement needs, and a number that only appears when a test fails is a number
    # nobody has.
    print("live cache probe, prompt_cache_hit_tokens per call:", hits)
    assert any(value > 0 for value in hits), (
        f"no cache hit across three identical calls: {hits}. Caching is automatic and "
        "best-effort, so a run of zeroes is worth investigating rather than asserting away."
    )


async def test_a_too_small_budget_reports_a_budget_failure_rather_than_a_malformed_one(settings):
    """The measured trap: reasoning is charged against `max_tokens` and thinking is on by
    default, so a small ceiling returns EMPTY content that looks exactly like DeepSeek's
    documented empty-content bug and has the opposite fix."""
    async with LlmClient.from_settings(settings, provider_id="deepseek") as client:
        client.max_tokens = 32
        client.max_attempts = 1
        with pytest.raises(LlmBudgetError) as caught:
            await client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(bundle(1)))
    assert "max_tokens in LLM_PROVIDERS" in str(caught.value)


async def test_the_model_respects_the_floor_on_a_package_the_ladder_pins_at_unsafe(settings):
    """Not a guarantee — the model may still answer below the floor and the validator exists
    for exactly that — so this records what actually came back rather than demanding
    compliance. A rejection here is a prompt finding, not a test failure."""
    evidence = bundle(0)
    async with LlmClient.from_settings(settings, provider_id="deepseek") as client:
        result = await client.complete_json(system=SYSTEM_PROMPT, user=user_prompt(evidence))
    payload = result.json_object()
    assert payload["removal"] in list(Removal), payload
    print(
        "live floor probe:",
        json.dumps(
            {
                "package": evidence.package,
                "floor": str(FLOORS[evidence.package].floor),
                "answered": payload["removal"],
                "usage": result.usage,
            }
        ),
    )
