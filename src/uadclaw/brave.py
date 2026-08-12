"""The Brave Search client and the page fetcher behind the corroboration stage.

Two clients, and keeping them apart is a security property rather than tidiness:

- **The API client carries the subscription token and never follows a redirect.**
  `X-Subscription-Token` is a CUSTOM header, and httpx strips only `Authorization` when a
  redirect crosses origins — a custom header rides along to wherever the 3xx points. So this
  client is built `follow_redirects=False` and a 3xx from the API is an error naming the
  location it refused, never a hop taken with the key attached.
- **The page fetcher carries no credential of any kind.** It fetches attacker-influenceable
  URLs — anything that can rank for a package name — which nothing else in this repo does. It
  refuses a non-`http`/`https` scheme before any request, refuses a redirect that would
  downgrade `https` to `http`, bounds the hop count, gates on `Content-Type`, and enforces a
  hard byte ceiling WHILE STREAMING so a multi-gigabyte page never reaches memory.

**No retry layer lives here.** The repo rule is one retry layer per concern, and the
corroboration stage's answer to a failed search is a `search_failed` row rather than a loop:
the row is per package, cheap to re-run, and says which of "we found nothing" and "we could
not look" happened. The judge's retries belong to `DeepSeekClient`, which already owns them.

**This is the one index in this repo whose empty answer is a legitimate result.** Everywhere
else an extraction or index fetch that yields nothing is an error, because a terms-walled
index answers 200 with zero links. Here it is not: `docs/todo.md` §8 measured an invented
package name returning 5 unrelated results, so the judge carries the whole verdict and a
genuinely empty `web.results` is an answer. What is still refused is an envelope carrying
NEITHER result block, because that is a shape change wearing zero results' clothes — and the
two failures are not symmetric. A shape change read as "zero results" marks every package
uncorroborated in silence; a zero-result answer read as a failure marks one package
`search_failed`, which is retryable and visible.
"""

import asyncio
import html
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
import lxml.html
from lxml.etree import LxmlError

from uadclaw.corroborate import SEARCH_BLOCKS, SourceEvidence
from uadclaw.settings import Settings

logger = logging.getLogger(__name__)

# The schemes a result URL may carry. Checked before any request is made: a `javascript:` or
# `file:` URL out of a search result must never reach a transport at all.
ALLOWED_SCHEMES = frozenset({"http", "https"})

# What the page fetcher will extract. Anything else (a PDF, an image, an octet-stream) falls
# back to the Brave snippet: lxml would either fail or produce binary noise, and a judge
# reading noise is worse than a judge reading a one-line snippet.
EXTRACTABLE_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})

# Elements whose text is site furniture rather than content. Dropped before `text_content()`
# so the byte budget handed to the judge is spent on the page instead of on its navigation.
_STRIPPED_ELEMENTS = "//script|//style|//nav|//footer|//noscript|//svg|//template|//iframe"

_CHUNK_BYTES = 64 * 1024

# Sent by the page fetcher. A real UA string rather than httpx's default: a bare `python-httpx`
# is refused or served a challenge page by a good share of the web, and a challenge page is
# evidence of nothing while still costing the byte budget.
PAGE_USER_AGENT = "Mozilla/5.0 (compatible; uadclaw/0.1; +https://github.com/uwuclxdy/uadclaw)"


class BraveError(RuntimeError):
    """Base for every failure of the search half. Distinct from a bug in this module."""


class BraveConfigError(BraveError):
    """The client cannot be built from the current configuration. Operator input."""


class BraveAuthError(BraveError):
    """401/403. The subscription token is missing, wrong, or revoked."""


class BraveRedirectError(BraveError):
    """The API answered a redirect. Refused rather than followed: the subscription token is a
    custom header, so httpx would carry it to the new host."""


class BraveUnavailableError(BraveError):
    """429/5xx, or a transport failure."""


class BraveMalformedError(BraveError):
    """The response body is not a search envelope this client can read."""


def require_brave_key(settings: Settings) -> str:
    """The configured token, or a fail-fast error naming where to put one.

    Checked here rather than by a `Settings` validator, exactly as `deepseek.require_api_key`
    is and for the same reason: the whole deterministic pipeline (acquire through rule_ladder,
    milestone M2) is independently useful and must boot on a box with no search account.
    """
    key = settings.brave_key.get_secret_value().strip()
    if not key:
        raise BraveConfigError(
            "brave: BRAVE_KEY is empty, so the corroborate stage has nothing to authenticate "
            "with. Put the token in secrets/brave_key (docker mounts it at "
            "/run/secrets/brave_key, which WINS over the environment variable — an empty file "
            "there shadows a set BRAVE_KEY) or set BRAVE_KEY for a local run. Every stage "
            "except corroborate runs without it, and a classification job now ends at "
            "corroborate."
        )
    return key


def is_fetchable_url(url: str) -> bool:
    """Whether a URL out of a search result may become a request at all."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme.lower() in ALLOWED_SCHEMES and bool(parsed.netloc)


def _results(block: Any) -> list[Mapping[str, Any]]:
    if not isinstance(block, Mapping):
        return []
    items = block.get("results")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _hit(item: Mapping[str, Any], *, block: str, position: int) -> SourceEvidence | None:
    url = item.get("url")
    if not isinstance(url, str) or not is_fetchable_url(url):
        return None
    title = item.get("title")
    # The snippet field is called `description`, NOT `snippet`, and it arrives HTML-escaped
    # (`&amp;`). Measured 2026-08-12 against the live API.
    snippet = item.get("description")
    return SourceEvidence(
        url=url,
        title=html.unescape(title) if isinstance(title, str) else "",
        snippet=html.unescape(snippet) if isinstance(snippet, str) else "",
        block=block,
        position=position,
    )


def merge_results(payload: Mapping[str, Any], *, limit: int) -> list[SourceEvidence]:
    """The `web` and `discussions` blocks as one ranked list.

    Both blocks, because they are different source classes and only one of them is where real
    packages corroborate: `docs/todo.md` §8 measured 10 `web` results beside 13 `discussions`
    for one probe, and forum threads (Reddit, Stack Exchange, Google support) land in the
    latter. A client reading only `web` drops the exact class this stage exists to find.

    Order comes from `mixed.main`, which is the interleaved order Brave itself would display,
    and anything `mixed.main` does not mention is appended in block order rather than dropped —
    a display hint is not an allowlist. Deduplicated by URL, because one page can appear in
    both blocks.
    """
    blocks = {name: _results(payload.get(name)) for name in SEARCH_BLOCKS}
    if not any(name in payload for name in SEARCH_BLOCKS):
        raise BraveMalformedError(
            "brave: the response carries neither a `web` nor a `discussions` block (keys: "
            f"{sorted(payload)}), so this is an envelope shape this client cannot read rather "
            "than a search with no results. Refused deliberately: reading a shape change as "
            "'nothing corroborates' would mark every package uncorroborated in silence."
        )

    ordered: list[SourceEvidence] = []
    seen_urls: set[str] = set()
    taken: set[tuple[str, int]] = set()

    def take(name: str, index: int) -> None:
        items = blocks.get(name, [])
        if not 0 <= index < len(items) or (name, index) in taken:
            return
        taken.add((name, index))
        hit = _hit(items[index], block=name, position=len(ordered) + 1)
        if hit is None or hit.url in seen_urls:
            return
        seen_urls.add(hit.url)
        ordered.append(hit)

    mixed = payload.get("mixed")
    main = mixed.get("main") if isinstance(mixed, Mapping) else None
    for entry in main if isinstance(main, list) else []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("type")
        if name not in blocks:
            continue
        index = entry.get("index")
        if entry.get("all") or not isinstance(index, int):
            for position in range(len(blocks[name])):
                take(name, position)
        else:
            take(name, index)
    for name in SEARCH_BLOCKS:
        for position in range(len(blocks[name])):
            take(name, position)
    return ordered[:limit]


class BraveClient:
    """One connection to the Brave web-search API.

    Holds the token as a plain string rather than the `Settings` object, for the reason
    `DeepSeekClient` does: pydantic renders every field of a `Settings` on `repr()`, so a
    `Settings` in a traceback frame would put the credential in a job's `log_tail`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        search_url: str,
        result_count: int,
        request_timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise BraveConfigError(
                "BraveClient: refusing to build with an empty api_key; call "
                "`require_brave_key(settings)` so the failure names the setting to fix"
            )
        self._api_key = api_key
        self.search_url = search_url
        self.result_count = result_count
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_seconds),
            transport=transport,
            # Never True. The token is a custom header, so httpx would carry it across an
            # origin change; a 3xx here is an error rather than a hop.
            follow_redirects=False,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> "BraveClient":
        return cls(
            api_key=require_brave_key(settings),
            search_url=settings.brave_search_url,
            result_count=settings.brave_result_count,
            request_timeout_seconds=settings.brave_request_timeout_seconds,
            transport=transport,
        )

    def __repr__(self) -> str:
        # Explicit, not the default: the default is harmless today and starts printing the
        # token the moment somebody makes this a dataclass.
        return f"BraveClient(search_url={self.search_url!r}, result_count={self.result_count})"

    async def __aenter__(self) -> "BraveClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, *, limit: int) -> list[SourceEvidence]:
        """One search, merged and ranked. Raises rather than returning a partial answer: the
        caller records `search_failed`, which is a different fact from "found nothing"."""
        try:
            response = await self._client.get(
                self.search_url,
                params={"q": query, "count": self.result_count},
                headers={
                    "X-Subscription-Token": self._api_key,
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            # `exc` carries the request URL, never the headers, so this cannot leak the token.
            raise BraveUnavailableError(
                f"brave: GET {self.search_url} failed at the transport "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        _raise_for_status(response, self.search_url)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BraveMalformedError(
                f"brave: HTTP {response.status_code} body is not JSON at all ({exc}); first "
                f"200 characters: {response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise BraveMalformedError(
                f"brave: the response body is a JSON {type(payload).__name__}, not an object"
            )
        return merge_results(payload, limit=limit)


def _raise_for_status(response: httpx.Response, search_url: str) -> None:
    status = response.status_code
    if response.has_redirect_location:
        location = response.headers.get("location", "<no location header>")
        raise BraveRedirectError(
            f"brave: {search_url} answered HTTP {status} redirecting to {location!r}. Refused "
            "rather than followed: X-Subscription-Token is a custom header and httpx strips "
            "only Authorization across an origin change, so following this would hand the "
            "search token to whatever host the redirect names."
        )
    if status < 400:
        return
    # `response.text` is the API's own error body; it echoes the request's error, never the
    # request headers, so it is safe to quote.
    detail = response.text[:400]
    if status in (401, 403):
        raise BraveAuthError(
            f"brave: {search_url} rejected the credential (HTTP {status}). Check "
            f"secrets/brave_key. Body: {detail}"
        )
    if status == 429 or status >= 500:
        raise BraveUnavailableError(
            f"brave: HTTP {status} from {search_url}. Recorded as a search failure for this "
            "package rather than retried here, so a re-run retries only the packages that "
            f"actually failed. Body: {detail}"
        )
    raise BraveError(f"brave: unexpected HTTP {status} from {search_url}. Body: {detail}")


def extract_text(body: bytes, *, content_type: str, max_chars: int) -> str:
    """One page's bytes as collapsed plain text, capped.

    Blocking and CPU-bound over up to `page_fetch_max_bytes` of hostile HTML — run it through
    `asyncio.to_thread`, which is what `PageFetcher` does.

    `no_network=True` on the parser is explicit rather than inherited: this is the only place
    in the repo that parses a document off the open web, and a parser that can be talked into
    a fetch turns a page body into an outbound request.
    """
    if content_type.startswith("text/plain"):
        return " ".join(body.decode("utf-8", errors="replace").split())[:max_chars]
    parser = lxml.html.HTMLParser(no_network=True, remove_comments=True)
    document = lxml.html.fromstring(body, parser=parser)
    for element in document.xpath(_STRIPPED_ELEMENTS):
        element.drop_tree()
    return " ".join(document.text_content().split())[:max_chars]


class PageFetcher:
    """The bodies behind the search results, bounded on every axis.

    Carries no credential: it is pointed at whatever ranked for a package name, so anything it
    held would be handed to a host chosen by an attacker who can rank.
    """

    def __init__(
        self,
        *,
        request_timeout_seconds: float,
        max_bytes: int,
        max_text_chars: int,
        max_concurrency: int,
        max_redirects: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_text_chars = max_text_chars
        self.max_redirects = max_redirects
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_seconds),
            transport=transport,
            # Redirects are walked by hand below so each hop's scheme can be checked. httpx
            # would follow a 302 from https to http without complaint.
            follow_redirects=False,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> "PageFetcher":
        return cls(
            request_timeout_seconds=settings.page_fetch_timeout_seconds,
            max_bytes=settings.page_fetch_max_bytes,
            max_text_chars=settings.page_text_max_chars,
            max_concurrency=settings.page_fetch_max_concurrency,
            max_redirects=settings.page_fetch_max_redirects,
            transport=transport,
        )

    async def __aenter__(self) -> "PageFetcher":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_all(self, sources: Iterable[SourceEvidence]) -> list[SourceEvidence]:
        """Every source, concurrency-bounded, in the order it was given.

        A failure is never raised: it becomes `fetch_error` on that one source and the judge
        reads its Brave snippet instead. One dead host must not cost the other nine results,
        and it must not cost the package its verdict.
        """
        return list(await asyncio.gather(*(self.fetch(source) for source in sources)))

    async def fetch(self, source: SourceEvidence) -> SourceEvidence:
        if not is_fetchable_url(source.url):
            return source.with_error(
                f"refused before any request: {source.url!r} is not an http(s) url"
            )
        async with self._semaphore:
            return await self._walk(source)

    async def _walk(self, source: SourceEvidence) -> SourceEvidence:
        url = source.url
        for _hop in range(self.max_redirects + 1):
            try:
                async with self._client.stream(
                    "GET", url, headers={"User-Agent": PAGE_USER_AGENT}
                ) as response:
                    if response.has_redirect_location:
                        following, refusal = self._redirect_target(url, response)
                        if following is None:
                            return source.with_error(refusal)
                        url = following
                        continue
                    if response.status_code != httpx.codes.OK:
                        return source.with_error(f"HTTP {response.status_code} from {url}")
                    content_type = (response.headers.get("content-type") or "").split(";")[0]
                    if content_type.strip().lower() not in EXTRACTABLE_TYPES:
                        return source.with_error(
                            f"content-type {content_type.strip()!r} is not text; the search "
                            "snippet stands in for this source"
                        )
                    body = await self._read_bounded(response)
            except httpx.HTTPError as exc:
                return source.with_error(f"{type(exc).__name__}: {exc}")
            try:
                text = await asyncio.to_thread(
                    extract_text,
                    body,
                    content_type=content_type,
                    max_chars=self.max_text_chars,
                )
            except (LxmlError, ValueError) as exc:
                return source.with_error(f"could not be read as text ({type(exc).__name__}: {exc})")
            if not text:
                return source.with_error("the page yielded no text")
            return source.with_text(text)
        return source.with_error(f"more than {self.max_redirects} redirect(s) from {source.url}")

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        """The body, up to the ceiling, never past it.

        Truncated rather than refused: the head of a long page is still evidence, and the
        point of the ceiling is that a 5 GB response never reaches memory. `aiter_bytes` is
        what enforces it — `await response.aread()` would have the whole body in the heap
        before any check could run.
        """
        chunks = bytearray()
        async for chunk in response.aiter_bytes(_CHUNK_BYTES):
            chunks += chunk
            if len(chunks) >= self.max_bytes:
                del chunks[self.max_bytes :]
                logger.info(
                    "page body truncated at the %d-byte ceiling: %s",
                    self.max_bytes,
                    response.request.url,
                )
                break
        return bytes(chunks)

    def _redirect_target(self, current: str, response: httpx.Response) -> tuple[str | None, str]:
        """`(next_url, "")`, or `(None, why it was refused)`.

        `response.next_request` is httpx's own redirect request, so relative locations and the
        303 method change are resolved the way the library would resolve them; what changes is
        that the scheme is checked before it is sent.
        """
        following = response.next_request
        if following is None:
            return None, f"HTTP {response.status_code} with no usable location from {current}"
        target = str(following.url)
        if following.url.scheme not in ALLOWED_SCHEMES:
            return None, f"refused a redirect to a non-http(s) url: {target!r}"
        if urlsplit(current).scheme == "https" and following.url.scheme == "http":
            return None, f"refused a redirect downgrading https to http: {current!r} -> {target!r}"
        return target, ""


def source_links(sources: Sequence[SourceEvidence], urls: Sequence[str]) -> list[dict[str, str]]:
    """The judge's cited urls as `{url, title}` pairs for the triage card.

    The title comes off the search result rather than out of the model, so a citation cannot
    be labelled with a description of a page nobody fetched.
    """
    by_url = {source.url: source for source in sources}
    return [{"url": url, "title": by_url[url].title if url in by_url else ""} for url in urls]
