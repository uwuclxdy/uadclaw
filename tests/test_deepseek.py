"""The DeepSeek client, against a mocked transport.

The default suite never touches the network and never needs a key: every response here is
synthesized by an `httpx.MockTransport`, so a box with no DeepSeek account runs these tests
identically. `tests/test_deepseek_live.py` is the opt-in half that proves the real API
accepts the real request shape.

The two failure modes worth naming, because they are the ones the envelope cannot tell apart
on its own and they have opposite fixes: an empty body with `finish_reason="length"` means
`max_tokens` was too small for reasoning plus the JSON, and an empty body with
`finish_reason="stop"` is DeepSeek's own documented, unresolved empty-content bug. Retrying
the first burns the whole cap on a request that cannot succeed at that ceiling.
"""

import json

import httpx
import pytest

from uadclaw.deepseek import (
    ChatResult,
    DeepSeekAuthError,
    DeepSeekBalanceError,
    DeepSeekBudgetError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekError,
    DeepSeekMalformedError,
    DeepSeekUnavailableError,
    require_api_key,
    require_json_prompt,
)
from uadclaw.settings import Settings

SYSTEM = 'Answer with one json object like {"ok": true}.'
USER = "Classify this."
SECRET = "sk-test-not-a-real-key-0000"


def envelope(content="{}", *, finish_reason="stop", reasoning=None, usage=None):
    body = {
        "id": "chat-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 1200,
            "completion_tokens": 110,
            "total_tokens": 1310,
            "prompt_cache_hit_tokens": 1024,
            "prompt_cache_miss_tokens": 176,
        },
    }
    if reasoning is not None:
        body["usage"].setdefault("completion_tokens_details", {})["reasoning_tokens"] = reasoning
    return body


def client(responses, **overrides):
    """A client whose transport replays `responses` in order, recording every request."""
    sent: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    kwargs = {
        "api_key": SECRET,
        "base_url": "https://api.deepseek.test",
        "model": "deepseek-v4-flash",
        "max_tokens": 4096,
        "max_concurrency": 2,
        "max_attempts": 3,
        "retry_backoff_seconds": 0.001,
        "request_timeout_seconds": 5.0,
        "thinking": True,
    }
    kwargs.update(overrides)
    return DeepSeekClient(transport=httpx.MockTransport(handler), **kwargs), sent


# --- the request shape ----------------------------------------------------------------------


async def test_the_request_is_the_openai_format_json_mode_call():
    """The OpenAI endpoint rather than /anthropic, because only this envelope carries
    prompt_cache_hit_tokens, and json_object rather than the beta strict tool mode."""
    api, sent = client([httpx.Response(200, json=envelope('{"ok": true}'))])
    async with api:
        await api.complete_json(system=SYSTEM, user=USER)
    body = json.loads(sent[0].content)
    assert sent[0].url.path == "/chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 4096
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "tools" not in body


async def test_thinking_off_sends_the_real_off_switch():
    """Measured: `reasoning_effort: low` only halves the reasoning spend (104 -> 52 tokens),
    `thinking: disabled` removes it (5 completion tokens, no reasoning field)."""
    api, sent = client([httpx.Response(200, json=envelope('{"ok": true}'))], thinking=False)
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert json.loads(sent[0].content)["thinking"] == {"type": "disabled"}
    assert result.thinking is False


async def test_thinking_on_sends_no_override_and_records_the_mode():
    api, sent = client([httpx.Response(200, json=envelope('{"ok": true}'))])
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert "thinking" not in json.loads(sent[0].content)
    assert result.thinking is True


async def test_the_system_message_is_byte_identical_across_calls():
    """DeepSeek's cache persists a DETECTED common prefix, and a hit is 50x cheaper than a
    miss, so a system message that varied per call would quietly cost 50x on that span."""
    api, sent = client([httpx.Response(200, json=envelope('{"ok": true}'))])
    async with api:
        await api.complete_json(system=SYSTEM, user="one")
        await api.complete_json(system=SYSTEM, user="two")
    systems = {json.loads(request.content)["messages"][0]["content"] for request in sent}
    assert len(systems) == 1


# --- the JSON-mode Notice, enforced in code ---------------------------------------------------


def test_a_prompt_without_the_word_json_is_refused_before_it_is_sent():
    with pytest.raises(DeepSeekConfigError, match="must contain the word 'json'"):
        require_json_prompt("Describe this package.", "{}")


def test_a_prompt_without_an_example_object_is_refused():
    with pytest.raises(DeepSeekConfigError, match="example of the desired JSON"):
        require_json_prompt("Answer in json.", "no example here")


async def test_the_guard_runs_before_any_request_is_made():
    api, sent = client([httpx.Response(200, json=envelope())])
    async with api:
        with pytest.raises(DeepSeekConfigError):
            await api.complete_json(system="describe it", user="please")
    assert sent == []


# --- budget vs the documented empty-content bug -------------------------------------------------


async def test_finish_reason_length_is_a_budget_error_and_is_not_retried():
    api, sent = client(
        [httpx.Response(200, json=envelope("", finish_reason="length", reasoning=32))]
    )
    async with api:
        with pytest.raises(DeepSeekBudgetError) as caught:
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 1, "a budget failure must not consume the retry cap"
    assert "DEEPSEEK_MAX_TOKENS" in str(caught.value)
    assert "reasoning_tokens=32" in str(caught.value)


async def test_a_truncated_body_is_a_budget_error_rather_than_a_success():
    """`finish_reason="length"` with a half-written object parses as broken JSON at best. It
    is the ceiling, not the model."""
    api, _ = client(
        [httpx.Response(200, json=envelope('{"description": "half a sen', finish_reason="length"))]
    )
    async with api:
        with pytest.raises(DeepSeekBudgetError):
            await api.complete_json(system=SYSTEM, user=USER)


async def test_empty_content_with_reasoning_at_the_ceiling_is_a_budget_error():
    """The measured shape: thinking consumed the whole budget and left nothing for the body,
    while `finish_reason` did not say `length`."""
    api, sent = client([httpx.Response(200, json=envelope("", reasoning=120))], max_tokens=128)
    async with api:
        with pytest.raises(DeepSeekBudgetError):
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 1


async def test_empty_content_with_no_token_pressure_is_the_documented_bug_and_retries():
    api, sent = client(
        [
            httpx.Response(200, json=envelope("", reasoning=12)),
            httpx.Response(200, json=envelope('{"ok": true}', reasoning=12)),
        ]
    )
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 2
    assert result.attempts == 2
    assert result.json_object() == {"ok": True}


async def test_the_empty_content_bug_exhausts_the_cap_and_raises_the_last_failure():
    api, sent = client([httpx.Response(200, json=envelope(""))])
    async with api:
        with pytest.raises(DeepSeekMalformedError, match="empty-content bug"):
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 3


# --- status handling ------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_retryable_status_is_retried_then_succeeds(status):
    api, sent = client(
        [httpx.Response(status, text="busy"), httpx.Response(200, json=envelope('{"ok": true}'))]
    )
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 2
    assert result.attempts == 2


async def test_the_retry_cap_is_honoured():
    api, sent = client([httpx.Response(503, text="overloaded")])
    async with api:
        with pytest.raises(DeepSeekUnavailableError):
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 3


async def test_402_is_not_retried():
    api, sent = client([httpx.Response(402, text="Insufficient Balance")])
    async with api:
        with pytest.raises(DeepSeekBalanceError, match="insufficient balance"):
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_credential_is_not_retried(status):
    api, sent = client([httpx.Response(status, text="Authentication Fails")])
    async with api:
        with pytest.raises(DeepSeekAuthError):
            await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 1


async def test_an_undocumented_status_is_not_retried():
    api, sent = client([httpx.Response(418, text="teapot")])
    async with api:
        with pytest.raises(DeepSeekError) as caught:
            await api.complete_json(system=SYSTEM, user=USER)
    assert type(caught.value) is DeepSeekError
    assert len(sent) == 1


async def test_a_transport_failure_is_retryable():
    api, sent = client(
        [httpx.ConnectError("connection refused"), httpx.Response(200, json=envelope('{"ok":1}'))]
    )
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 2
    assert result.attempts == 2


# --- the body -----------------------------------------------------------------------------------


async def test_a_non_json_body_is_malformed_and_retried():
    api, sent = client(
        [httpx.Response(200, text="<html>gateway</html>"), httpx.Response(200, json=envelope("{}"))]
    )
    async with api:
        await api.complete_json(system=SYSTEM, user=USER)
    assert len(sent) == 2


async def test_a_response_with_no_choices_is_malformed():
    api, _ = client([httpx.Response(200, json={"id": "x", "usage": {}})])
    async with api:
        with pytest.raises(DeepSeekMalformedError, match="no choices"):
            await api.complete_json(system=SYSTEM, user=USER)


def test_content_that_is_not_an_object_is_malformed():
    result = ChatResult(
        content="[1, 2]", model="m", finish_reason="stop", usage={}, attempts=1, thinking=True
    )
    with pytest.raises(DeepSeekMalformedError, match="JSON list, not an object"):
        result.json_object()


def test_content_that_is_not_json_names_what_came_back():
    result = ChatResult(
        content="Sure! Here you go:",
        model="m",
        finish_reason="stop",
        usage={},
        attempts=1,
        thinking=True,
    )
    with pytest.raises(DeepSeekMalformedError, match="Sure! Here you go"):
        result.json_object()


# --- usage is returned raw --------------------------------------------------------------------


async def test_the_raw_usage_envelope_reaches_the_caller():
    """It is persisted per row: the cost and cache measurement reads it off real traffic, so
    a field nobody thought to break out today still has to survive."""
    api, _ = client([httpx.Response(200, json=envelope('{"ok": true}', reasoning=104))])
    async with api:
        result = await api.complete_json(system=SYSTEM, user=USER)
    assert result.usage["prompt_cache_hit_tokens"] == 1024
    assert result.usage["prompt_cache_miss_tokens"] == 176
    assert result.usage["completion_tokens_details"]["reasoning_tokens"] == 104
    assert result.cache_hit_tokens == 1024
    assert result.reasoning_tokens == 104


# --- the key never leaks ---------------------------------------------------------------------


async def test_the_key_is_sent_as_a_bearer_header_and_appears_nowhere_else():
    api, sent = client([httpx.Response(500, text="server error")])
    async with api:
        with pytest.raises(DeepSeekUnavailableError) as caught:
            await api.complete_json(system=SYSTEM, user=USER)
    assert sent[0].headers["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(api)
    assert SECRET not in sent[0].content.decode()


async def test_an_auth_failure_message_does_not_quote_the_key():
    api, _ = client([httpx.Response(401, text="Authentication Fails, Your api key is invalid")])
    async with api:
        with pytest.raises(DeepSeekAuthError) as caught:
            await api.complete_json(system=SYSTEM, user=USER)
    assert SECRET not in str(caught.value)


# --- the key is checked at the point of use ------------------------------------------------------


def test_a_blank_key_is_refused_at_the_point_of_use_not_at_settings_load(monkeypatch):
    """The whole deterministic pipeline must boot on a box with no DeepSeek account, so
    `Settings()` accepts a blank key and the call site is what fails."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("AUTH_PASSWORD", "y")
    monkeypatch.setenv("SESSION_SECRET", "z")
    monkeypatch.setenv("DEEPSEEK_KEY", "")
    settings = Settings()
    assert settings.deepseek_key.get_secret_value() == ""
    with pytest.raises(DeepSeekConfigError, match="DEEPSEEK_KEY is empty"):
        require_api_key(settings)


def test_a_whitespace_only_key_is_refused_too(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("AUTH_PASSWORD", "y")
    monkeypatch.setenv("SESSION_SECRET", "z")
    monkeypatch.setenv("DEEPSEEK_KEY", "   ")
    with pytest.raises(DeepSeekConfigError):
        require_api_key(Settings())


def test_the_error_names_both_places_the_key_can_live(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("AUTH_PASSWORD", "y")
    monkeypatch.setenv("SESSION_SECRET", "z")
    monkeypatch.setenv("DEEPSEEK_KEY", "")
    with pytest.raises(DeepSeekConfigError) as caught:
        require_api_key(Settings())
    # An empty mounted secret file SHADOWS a set env var (file secrets win in this repo's
    # source order), which is a confusing failure the message has to name.
    assert "secrets/deepseek_key" in str(caught.value)
    assert "WINS over the environment" in str(caught.value)


def test_the_client_refuses_to_be_built_with_an_empty_key():
    with pytest.raises(DeepSeekConfigError, match="require_api_key"):
        DeepSeekClient(
            api_key="",
            base_url="https://api.deepseek.test",
            model="deepseek-v4-flash",
            max_tokens=16,
            max_concurrency=1,
            max_attempts=1,
            retry_backoff_seconds=1.0,
            request_timeout_seconds=1.0,
            thinking=True,
        )
