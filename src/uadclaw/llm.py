"""The OpenAI-format chat-completions client every LLM provider runs behind.

One client, any provider: `Settings.llm_providers` is the operator's provider table and this
module builds the client from the entry a classification job named, so a new provider is a
row in `LLM_PROVIDERS` plus a key file — never a second client.

Deliberate choices, each with a reason that is not obvious from the code:

- **The OpenAI-format endpoint (`/chat/completions`), never `/anthropic`.** Both wire formats
  are first-class and DeepSeek's caching is automatic either way, but only the OpenAI envelope
  returns `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, and those two fields
  are the entire cost-and-cache measurement this project owes itself (§3, §9).
- **`response_format={"type":"json_object"}`, never the beta `strict` tool mode.** DeepSeek
  has no `json_schema` structured output; the closest thing is `strict: true` on a forced
  function call, which its own reference labels Beta with no published failure rate. The
  hand-written validator in `classify.py` is the real gate whichever mode is used, so the
  beta buys nothing and costs an undocumented failure mode.
- **Every clause of the JSON-mode Notice is honoured in code rather than in a comment.** The
  prompt is refused unless it contains the word "json" AND a `{`-shaped example, because
  without an explicit instruction the model "may generate an unending stream of whitespace
  until the generation reaches the token limit"; `max_tokens` is always sent; and
  `finish_reason == "length"` is a failure rather than a truncated success.
- **Retry and backoff live here and nowhere else.** The repo rule is one retry layer per
  concern: a second one in the stage would multiply the caps and turn a documented 3 attempts
  into 9 silent ones against a paid API.

Three failure classes are kept apart because they have three different fixes, and the
envelope alone does not distinguish two of them:

- `LlmBudgetError` — `finish_reason == "length"`, or empty content with the reasoning
  spend at the ceiling. Measured 2026-08-11 against the live API: a call with
  `max_tokens: 32` returned `finish_reason="length"`, EMPTY content and
  `usage.completion_tokens_details.reasoning_tokens: 32`, i.e. thinking consumed the whole
  budget and left nothing for the body; the identical request at 512 answered normally. That
  is indistinguishable from the documented empty-content bug *from the response*, and the fix
  is the opposite one — raise the provider's `max_tokens`, do not re-prompt — so retrying it
  as malformed burns the whole cap and parks a package that would have answered. Not
  retryable.
- `DeepSeekMalformedError` — empty content with `finish_reason == "stop"` (DeepSeek's own
  documented, unresolved bug) or a body that is not a JSON object. Retryable.
- `DeepSeekUnavailableError` — 429/500/503. The docs prescribe no backoff and send no
  `Retry-After`, so the schedule is ours. Retryable.

`402 insufficient balance` is never retried: it will not resolve inside a retry window and a
loop against it just delays a stop that has to happen anyway.

**The key is never logged, never in an exception message and never in a repr.** The client
does not hold the `Settings` object at all, because pydantic renders every field of one on
`repr()` and a `Settings` in a traceback frame would put the key in a log.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from uadclaw.settings import LlmProviderConfig, Settings

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"

# Response fields the cost measurement (task 8) reads off real rows. Named here so a rename
# upstream shows up as a missing key in one place rather than as a silently absent metric.
CACHE_HIT_TOKENS = "prompt_cache_hit_tokens"
CACHE_MISS_TOKENS = "prompt_cache_miss_tokens"

# How close the reasoning spend has to get to the ceiling before an empty body is read as a
# budget failure rather than as the documented empty-content bug. Not 1.0: the sampler stops
# on a token boundary, so a budget-exhausted call can report a few tokens under the ceiling.
_REASONING_PRESSURE_RATIO = 0.9


class DeepSeekError(RuntimeError):
    """Base for every failure of the model call. Distinct from a bug in this module.

    `attempts` is how many requests actually reached the wire before this was raised, set by
    `complete_json` on the way out. Without it a caller holding a request BUDGET cannot
    decrement it on the failure path — it would charge one for a call that burned three, and
    its ceiling would be three times looser than the number it prints. It is also what makes
    a parked row's `attempts` true when the package parked because the wire gave up.
    """

    attempts: int = 0


class DeepSeekConfigError(DeepSeekError):
    """The client cannot be built from the current configuration. Operator input."""


class DeepSeekAuthError(DeepSeekError):
    """401/403. The key is missing, wrong, or revoked. Never retried."""


class DeepSeekBalanceError(DeepSeekError):
    """402 insufficient balance. Never retried: it cannot resolve inside a retry window."""


class LlmBudgetError(DeepSeekError):
    """`max_tokens` was too small for reasoning plus the JSON body. Never retried: the same
    request will fail the same way, and the fix is a bigger budget rather than a re-prompt."""


class DeepSeekMalformedError(DeepSeekError):
    """The response body is not a usable JSON object. Retryable."""


class DeepSeekUnavailableError(DeepSeekError):
    """429/500/503, or a transport failure. Retryable."""


@dataclass(frozen=True, slots=True)
class ChatResult:
    """One completed call. `usage` is the raw envelope, unreshaped: it is persisted per row
    so the cost and cache measurement is read off real traffic rather than a throwaway
    script, and a field we did not think to name today is still there tomorrow."""

    content: str
    model: str
    finish_reason: str
    usage: dict[str, Any]
    attempts: int
    thinking: bool

    @property
    def cache_hit_tokens(self) -> int:
        value = self.usage.get(CACHE_HIT_TOKENS)
        return value if isinstance(value, int) else 0

    @property
    def reasoning_tokens(self) -> int:
        details = self.usage.get("completion_tokens_details")
        value = details.get("reasoning_tokens") if isinstance(details, dict) else None
        return value if isinstance(value, int) else 0

    def json_object(self) -> dict[str, Any]:
        """The content parsed as a JSON object, or `DeepSeekMalformedError`."""
        try:
            parsed = json.loads(self.content)
        except json.JSONDecodeError as exc:
            raise DeepSeekMalformedError(
                f"llm: model {self.model} answered with something that is not JSON "
                f"({exc}); first 200 characters: {self.content[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise DeepSeekMalformedError(
                f"llm: model {self.model} answered with a JSON "
                f"{type(parsed).__name__}, not an object. The prompt asks for one object per "
                "package."
            )
        return parsed


def resolve_provider(settings: Settings, provider_id: str) -> LlmProviderConfig:
    """The table entry for one provider id, or a fail-fast error naming the valid ids.

    The creation seam already refuses an unknown id, so reaching this with one means the
    table changed between creation and run — the error names both rather than surfacing as a
    KeyError on a job somebody queued earlier.
    """
    try:
        return settings.llm_providers[provider_id]
    except KeyError as exc:
        valid = ", ".join(sorted(settings.llm_providers)) or "(none configured)"
        raise DeepSeekConfigError(
            f"llm: provider {provider_id!r} is not in the provider table (configured ids: "
            f"{valid}). Add it to LLM_PROVIDERS (or mount secrets/llm_providers), or re-create "
            "the job with a provider param naming one of them."
        ) from exc


def require_api_key(settings: Settings, provider_id: str) -> str:
    """The configured key for one provider, or a fail-fast error naming where to put one.

    Checked here rather than by a `Settings` validator on purpose: the three other
    credentials are refused at load because the app is unsafe without them, but the whole
    deterministic pipeline (acquire through rule_ladder, milestone M2) is independently
    useful and must boot on a box with no LLM account at all.
    """
    key = settings.provider_key(provider_id).strip()
    if not key:
        raise DeepSeekConfigError(
            f"llm: provider {provider_id!r} has no API key, so its calls have nothing to "
            f"authenticate with. Put the key in secrets/llm_{provider_id}_key (docker mounts "
            "it at /run/secrets/llm_<id>_key; a non-blank FILE wins over the environment "
            f"variable, a blank one falls back to it) or set LLM_{provider_id.upper()}_KEY "
            "for a local run. Every other stage runs without it."
        )
    return key


def require_json_prompt(*parts: str) -> None:
    """Refuse a prompt that would trip either documented JSON-mode failure mode.

    Both clauses are the API reference's, verbatim in intent: without an explicit instruction
    to produce JSON "the model may generate an unending stream of whitespace until the
    generation reaches the token limit", and the guide separately requires "an example of the
    desired JSON format". Enforced in code because a prompt is edited far more often than
    this module is, and a comment does not fail a build.
    """
    joined = "\n".join(parts)
    if "json" not in joined.lower():
        raise DeepSeekConfigError(
            "llm: the prompt must contain the word 'json'. Without it the model may "
            "emit an unbounded whitespace stream until it hits max_tokens, which reads as a "
            "stuck request rather than as a failure."
        )
    if "{" not in joined or "}" not in joined:
        raise DeepSeekConfigError(
            "llm: the prompt must include an example of the desired JSON object. The "
            "JSON Output guide requires one, and json_object mode constrains the syntax "
            "only, never the shape."
        )


class LlmClient:
    """One bounded, retrying connection to a provider's OpenAI-format API.

    Concurrency is capped by a semaphore because the provider gates on in-flight requests
    alone — no documented RPM or TPM — and counts them **account-wide across every API key**
    (measured on DeepSeek), so the ceiling is shared with whatever else the account is doing
    and a client that paced itself against the published ceiling would be pacing against the
    wrong number.

    `provider_id` is held for messages only: an error that names the provider is one an
    operator can fix without tracing which job built which client.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        max_concurrency: int,
        max_attempts: int,
        retry_backoff_seconds: float,
        request_timeout_seconds: float,
        thinking: bool,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise DeepSeekConfigError(
                "LlmClient: refusing to build with an empty api_key; call "
                "`require_api_key(settings, provider_id)` so the failure names the setting "
                "to fix"
            )
        self.provider_id = provider_id
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.thinking = thinking
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout_seconds),
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        provider_id: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "LlmClient":
        provider = resolve_provider(settings, provider_id)
        return cls(
            provider_id=provider_id,
            api_key=require_api_key(settings, provider_id),
            base_url=provider.base_url,
            model=provider.model,
            max_tokens=provider.max_tokens,
            max_concurrency=provider.max_concurrency,
            max_attempts=settings.llm_max_attempts,
            retry_backoff_seconds=settings.llm_retry_backoff_seconds,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
            thinking=provider.thinking,
            transport=transport,
        )

    def __repr__(self) -> str:
        # Explicit, not the default: the default would be harmless today and would start
        # printing the key the moment somebody makes this a dataclass.
        return (
            f"LlmClient(provider_id={self.provider_id!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, max_tokens={self.max_tokens}, thinking={self.thinking})"
        )

    async def __aenter__(self) -> "LlmClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                # System first and byte-identical every call: DeepSeek's cache persists a
                # detected common prefix as its own unit, so the shared rubric is what turns
                # into 50x-cheaper hit tokens once the run is warm.
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            # Always sent: the guide's own third clause, and the only thing standing between
            # a whitespace runaway and a ten-minute stuck request.
            "max_tokens": self.max_tokens,
        }
        if not self.thinking:
            # The real off switch. `reasoning_effort: "low"` only halves the reasoning spend
            # (measured 104 -> 52 tokens on one prompt); `disabled` removes it entirely
            # (measured 5 completion tokens, no reasoning field at all).
            body["thinking"] = {"type": "disabled"}
        return body

    async def complete_json(
        self, *, system: str, user: str, max_calls: int | None = None
    ) -> ChatResult:
        """One JSON-mode completion, retried on the retryable failures only.

        `max_calls` lets a caller holding a per-package request budget cap THIS call's share
        of it, so a client whose own retry cap is larger than the budget's remainder cannot
        overshoot it. None means "use the client's own cap".

        Raises the LAST failure once the attempt cap is spent, rather than a summary
        exception: the caller parks a package with a reason, and "3 attempts failed" is not a
        reason anybody can act on.
        """
        require_json_prompt(system, user)
        body = self._body(system, user)
        allowance = self.max_attempts if max_calls is None else min(self.max_attempts, max_calls)
        if allowance < 1:
            raise DeepSeekConfigError(
                f"complete_json: max_calls={max_calls} leaves no requests to make. A caller "
                "tracking a budget must stop before it reaches zero rather than asking for a "
                "call it cannot pay for."
            )
        last: DeepSeekError | None = None
        for attempt in range(1, allowance + 1):
            try:
                async with self._semaphore:
                    return await self._attempt(body, attempt)
            except DeepSeekError as exc:
                # Every failure carries what it actually spent, retryable or not: the caller's
                # budget is in REQUESTS, so a 402 on the second attempt has to decrement two.
                exc.attempts = attempt
                if not isinstance(exc, DeepSeekUnavailableError | DeepSeekMalformedError):
                    raise
                last = exc
                if attempt >= allowance:
                    break
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "llm provider=%s attempt %d/%d failed (%s); retrying in %.1fs",
                    self.provider_id,
                    attempt,
                    allowance,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last is not None  # noqa: S101 - the loop cannot exit without raising or returning
        raise last

    async def _attempt(self, body: dict[str, Any], attempt: int) -> ChatResult:
        try:
            response = await self._client.post(
                CHAT_COMPLETIONS_PATH,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            # `exc` carries the request URL, never the headers, so this cannot leak the key.
            raise DeepSeekUnavailableError(
                f"llm: provider {self.provider_id} POST "
                f"{self.base_url}{CHAT_COMPLETIONS_PATH} failed at the transport "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        _raise_for_status(response, self.base_url, provider_id=self.provider_id)
        return self._result(response, attempt)

    def _result(self, response: httpx.Response, attempt: int) -> ChatResult:
        try:
            envelope = response.json()
        except ValueError as exc:
            raise DeepSeekMalformedError(
                f"llm: HTTP {response.status_code} body is not JSON at all "
                f"({exc}); first 200 characters: {response.text[:200]!r}"
            ) from exc
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekMalformedError(
                f"llm: response carries no choices (keys: {sorted(envelope)}), so there "
                "is no completion to validate"
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        finish_reason = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        result = ChatResult(
            content=content if isinstance(content, str) else "",
            model=str(envelope.get("model") or self.model),
            finish_reason=finish_reason,
            usage=usage,
            attempts=attempt,
            thinking=self.thinking,
        )
        self._check_budget(result)
        if not result.content.strip():
            raise DeepSeekMalformedError(
                "llm: the model returned empty content with "
                f"finish_reason={finish_reason!r} and no token pressure "
                f"(reasoning_tokens={result.reasoning_tokens}, max_tokens={self.max_tokens}). "
                "This is the empty-content bug DeepSeek documents as unresolved in its JSON "
                "Output guide; retrying, and a prompt change is the documented mitigation if "
                "it persists."
            )
        return result

    def _check_budget(self, result: ChatResult) -> None:
        """Separate a too-small `max_tokens` from the documented empty-content bug.

        Both present as an unusable body, and treating a budget failure as malformed spends
        the whole retry cap on a request that cannot succeed at this ceiling.
        """
        reasoning = result.reasoning_tokens
        pressure = reasoning >= self.max_tokens * _REASONING_PRESSURE_RATIO
        if result.finish_reason != "length" and not (pressure and not result.content.strip()):
            return
        raise LlmBudgetError(
            f"llm: provider {self.provider_id} model {result.model} hit the output ceiling "
            f"(finish_reason={result.finish_reason!r}, max_tokens={self.max_tokens}, "
            f"reasoning_tokens={reasoning}, content {len(result.content)} chars). Reasoning "
            "is charged against max_tokens and thinking is ON by default on this model, so "
            "the JSON body can be truncated or missing entirely with nothing else wrong. "
            "Raise this provider's max_tokens in LLM_PROVIDERS, or set its thinking to false "
            "to stop paying the reasoning budget out of the same ceiling. Not retried: the "
            "same request fails the same way."
        )


def _raise_for_status(response: httpx.Response, base_url: str, *, provider_id: str) -> None:
    status = response.status_code
    if status < 400:
        return
    # `response.text` is the API's own error body. It echoes the request's error, never its
    # Authorization header, so it is safe to put in a message.
    detail = response.text[:400]
    if status in (401, 403):
        raise DeepSeekAuthError(
            f"llm: provider {provider_id} at {base_url} rejected the credential (HTTP "
            f"{status}). Check secrets/llm_{provider_id}_key or "
            f"LLM_{provider_id.upper()}_KEY. Not retried. Body: {detail}"
        )
    if status == 402:
        raise DeepSeekBalanceError(
            f"llm: provider {provider_id} answered HTTP 402 insufficient balance on the "
            "account behind its key. Top it up before re-running the classification job; "
            f"not retried, because a balance does not refill inside a retry window. Body: {detail}"
        )
    if status in (429, 500, 503):
        raise DeepSeekUnavailableError(
            f"llm: HTTP {status} from {base_url}. The provider gates on account-wide "
            "concurrency and prescribes no backoff of its own, so this is retried on our "
            f"own schedule; lower provider {provider_id}'s max_concurrency in LLM_PROVIDERS "
            f"if it persists. Body: {detail}"
        )
    raise DeepSeekError(
        f"llm: unexpected HTTP {status} from {base_url}{CHAT_COMPLETIONS_PATH}. Not "
        f"retried, because nothing documents this status as transient. Body: {detail}"
    )
