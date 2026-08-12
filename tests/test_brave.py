"""The Brave client and the page fetcher, against a mocked transport.

The default suite never touches the network and never needs a token: every response here is
synthesized by an `httpx.MockTransport`, so a box with no search account runs these identically.
`tests/test_corroborate_live.py` is the opt-in half that proves the real API answers the shape
this file assumes.

Most of what is asserted here is a refusal rather than a capability, because this is the only
place in the repo that fetches attacker-influenceable URLs: a URL that ranks for a package name
is chosen by whoever can rank for it.
"""

import httpx
import pytest

from uadclaw.brave import (
    BraveAuthError,
    BraveClient,
    BraveConfigError,
    BraveError,
    BraveMalformedError,
    BraveRedirectError,
    BraveUnavailableError,
    PageFetcher,
    extract_text,
    merge_results,
    require_brave_key,
    source_links,
)
from uadclaw.settings import Settings

TOKEN = "BSA-test-not-a-real-token-0000"
SEARCH_URL = "https://api.search.brave.test/res/v1/web/search"


def result(url, *, title="T", description="D"):
    return {"url": url, "title": title, "description": description}


def envelope(*, web=None, discussions=None, mixed=None):
    body = {}
    if web is not None:
        body["web"] = {"results": web}
    if discussions is not None:
        body["discussions"] = {"results": discussions}
    if mixed is not None:
        body["mixed"] = {"main": mixed}
    return body


def brave(responses):
    sent: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    client = BraveClient(
        api_key=TOKEN,
        search_url=SEARCH_URL,
        result_count=10,
        request_timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )
    return client, sent


# --- the request shape --------------------------------------------------------------------


async def test_the_token_rides_in_the_subscription_header_and_nowhere_else():
    api, sent = brave([httpx.Response(200, json=envelope(web=[result("https://a.test/1")]))])
    async with api:
        await api.search("com.example.app", limit=10)
    assert sent[0].headers["x-subscription-token"] == TOKEN
    assert sent[0].headers["accept"] == "application/json"
    assert sent[0].url.params["q"] == "com.example.app"
    assert sent[0].url.params["count"] == "10"
    assert TOKEN not in repr(api)


# --- the redirect refusal -----------------------------------------------------------------


async def test_a_redirect_from_the_api_is_refused_rather_than_followed():
    """`X-Subscription-Token` is a CUSTOM header, and httpx strips only `Authorization` across
    an origin change — so following a 3xx here would hand the search token to whatever host
    the redirect names."""
    api, sent = brave([httpx.Response(302, headers={"Location": "https://attacker.test/collect"})])
    async with api:
        with pytest.raises(BraveRedirectError) as caught:
            await api.search("com.example.app", limit=10)
    assert "attacker.test/collect" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert len(sent) == 1, "refused, not retried at the new location"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_every_redirect_status_from_the_api_is_refused(status):
    api, _ = brave([httpx.Response(status, headers={"Location": "http://elsewhere.test/"})])
    async with api:
        with pytest.raises(BraveRedirectError):
            await api.search("com.example.app", limit=10)


# --- status handling ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_token_names_the_secret_file_and_not_the_token(status):
    api, sent = brave([httpx.Response(status, text="Unauthorized")])
    async with api:
        with pytest.raises(BraveAuthError) as caught:
            await api.search("com.example.app", limit=10)
    assert "secrets/brave_key" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert len(sent) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_transient_status_is_a_search_failure_and_is_not_retried_here(status):
    """No retry layer lives in this client: a failed search becomes a `search_failed` row,
    which is per package and retried by re-running the job."""
    api, sent = brave([httpx.Response(status, text="slow down")])
    async with api:
        with pytest.raises(BraveUnavailableError):
            await api.search("com.example.app", limit=10)
    assert len(sent) == 1


async def test_an_undocumented_status_is_its_own_error():
    api, _ = brave([httpx.Response(418, text="teapot")])
    async with api:
        with pytest.raises(BraveError) as caught:
            await api.search("com.example.app", limit=10)
    assert type(caught.value) is BraveError


async def test_a_transport_failure_is_a_search_failure():
    api, _ = brave([httpx.ConnectError("connection refused")])
    async with api:
        with pytest.raises(BraveUnavailableError, match="transport"):
            await api.search("com.example.app", limit=10)


async def test_a_body_that_is_not_json_is_malformed():
    api, _ = brave([httpx.Response(200, text="<html>gateway</html>")])
    async with api:
        with pytest.raises(BraveMalformedError, match="not JSON"):
            await api.search("com.example.app", limit=10)


# --- merging web and discussions ------------------------------------------------------------


def test_discussions_results_reach_the_merged_list():
    """Measured 2026-08-12: one probe answered 10 `web` results beside 13 `discussions`, and
    forum threads — where real packages actually corroborate — land in the second block. A
    client reading only `web` drops the exact source class this stage exists to find."""
    merged = merge_results(
        envelope(
            web=[result("https://vendor.test/docs")],
            discussions=[result("https://reddit.test/r/android/1")],
        ),
        limit=10,
    )
    assert [source.url for source in merged] == [
        "https://vendor.test/docs",
        "https://reddit.test/r/android/1",
    ]
    assert [source.block for source in merged] == ["web", "discussions"]


def test_mixed_main_decides_the_display_order():
    merged = merge_results(
        envelope(
            web=[result("https://w0.test"), result("https://w1.test")],
            discussions=[result("https://d0.test")],
            mixed=[
                {"type": "discussions", "index": 0, "all": False},
                {"type": "web", "index": 1, "all": False},
                {"type": "web", "index": 0, "all": False},
            ],
        ),
        limit=10,
    )
    assert [source.url for source in merged] == [
        "https://d0.test",
        "https://w1.test",
        "https://w0.test",
    ]
    assert [source.position for source in merged] == [1, 2, 3]


def test_a_result_mixed_main_never_mentions_is_appended_rather_than_dropped():
    """A display hint is not an allowlist: a block Brave chose not to interleave still holds
    results, and dropping them would silently narrow the evidence."""
    merged = merge_results(
        envelope(
            web=[result("https://w0.test"), result("https://w1.test")],
            discussions=[result("https://d0.test")],
            mixed=[{"type": "web", "index": 1, "all": False}],
        ),
        limit=10,
    )
    assert [source.url for source in merged] == [
        "https://w1.test",
        "https://w0.test",
        "https://d0.test",
    ]


def test_an_all_entry_takes_the_whole_block_in_order():
    merged = merge_results(
        envelope(
            web=[result("https://w0.test"), result("https://w1.test")],
            mixed=[{"type": "web", "all": True}],
        ),
        limit=10,
    )
    assert [source.url for source in merged] == ["https://w0.test", "https://w1.test"]


def test_one_page_in_both_blocks_is_taken_once():
    merged = merge_results(
        envelope(
            web=[result("https://both.test/x")],
            discussions=[result("https://both.test/x")],
        ),
        limit=10,
    )
    assert [source.url for source in merged] == ["https://both.test/x"]


def test_the_snippet_comes_off_description_and_is_html_unescaped():
    """The field is called `description`, not `snippet`, and it arrives HTML-escaped. A client
    reading `snippet` gets an empty string for every result and never notices."""
    merged = merge_results(
        envelope(web=[result("https://a.test", title="A &amp; B", description="x &amp; y")]),
        limit=10,
    )
    assert merged[0].snippet == "x & y"
    assert merged[0].title == "A & B"


def test_a_result_whose_scheme_is_not_http_never_becomes_a_source():
    merged = merge_results(
        envelope(
            web=[
                result("javascript:alert(1)"),
                result("file:///etc/passwd"),
                result("https://ok.test/x"),
            ]
        ),
        limit=10,
    )
    assert [source.url for source in merged] == ["https://ok.test/x"]


def test_the_merged_list_is_truncated_to_the_limit():
    merged = merge_results(
        envelope(web=[result(f"https://w{index}.test") for index in range(20)]), limit=10
    )
    assert len(merged) == 10
    assert merged[-1].url == "https://w9.test"


def test_zero_results_is_a_legitimate_answer_rather_than_an_error():
    """The one index in this repo whose empty answer is a result. An invented package name
    returns 5 unrelated results (measured 2026-08-12), so the judge carries the whole verdict
    and "no results" is never the uncorroborated signal on its own."""
    assert merge_results(envelope(web=[]), limit=10) == []


def test_an_envelope_carrying_neither_block_is_refused():
    """A shape change wearing zero results' clothes. Read as "nothing corroborates" it would
    mark every package uncorroborated in silence; read as a failure it marks one package
    `search_failed`, which is retryable and visible."""
    with pytest.raises(BraveMalformedError) as caught:
        merge_results({"query": {"original": "com.example.app"}}, limit=10)
    assert "neither" in str(caught.value)
    assert "query" in str(caught.value), "the message names the keys that WERE present"


# --- the token is checked at the point of use ------------------------------------------------


def test_a_blank_token_is_refused_at_the_point_of_use_not_at_settings_load(monkeypatch):
    """The deterministic pipeline must boot on a box with no search account, so `Settings()`
    accepts a blank token and the call site is what fails."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setenv("AUTH_PASSWORD", "y")
    monkeypatch.setenv("SESSION_SECRET", "z")
    monkeypatch.setenv("BRAVE_KEY", "")
    settings = Settings()
    assert settings.brave_key.get_secret_value() == ""
    with pytest.raises(BraveConfigError) as caught:
        require_brave_key(settings)
    assert "BRAVE_KEY is empty" in str(caught.value)
    assert "secrets/brave_key" in str(caught.value)
    assert "WINS over the environment" in str(caught.value)


def test_the_client_refuses_to_be_built_with_an_empty_token():
    with pytest.raises(BraveConfigError, match="require_brave_key"):
        BraveClient(api_key="", search_url=SEARCH_URL, result_count=1, request_timeout_seconds=1.0)


# --- the page fetcher ---------------------------------------------------------------------


def source(url="https://page.test/a", *, snippet="the search index summary"):
    return merge_results(
        envelope(web=[result(url, description=snippet)]),
        limit=1,
    )[0]


class CountingStream(httpx.AsyncByteStream):
    """A body that records how many bytes were actually pulled out of it.

    The byte ceiling has to be asserted on bytes READ, not on a log line: `await
    response.aread()` would have the whole body in the heap before any check could run, and a
    test that only asserts the stored text length cannot tell the two apart.
    """

    def __init__(self, chunk: bytes, count: int) -> None:
        self.chunk = chunk
        self.count = count
        self.yielded = 0

    async def __aiter__(self):
        for _ in range(self.count):
            self.yielded += len(self.chunk)
            yield self.chunk


def fetcher(handler, **overrides):
    kwargs = {
        "request_timeout_seconds": 5.0,
        "max_bytes": 64 * 1024,
        "max_text_chars": 4000,
        "max_concurrency": 4,
        "max_redirects": 3,
    }
    kwargs.update(overrides)
    return PageFetcher(transport=httpx.MockTransport(handler), **kwargs)


def html_response(body: str, *, content_type="text/html; charset=utf-8", status=200):
    return httpx.Response(status, headers={"content-type": content_type}, content=body.encode())


async def test_a_fetched_page_becomes_collapsed_text_with_the_chrome_stripped():
    def handler(request):
        return html_response(
            "<html><head><style>a{}</style></head><body><nav>Menu Home</nav>"
            "<p>com.example.app is  the   settings provider.</p>"
            "<script>track()</script><footer>Copyright</footer></body></html>"
        )

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source())
    assert fetched.text == "com.example.app is the settings provider."
    assert fetched.fetch_error is None


async def test_the_fetcher_sends_no_credential_of_any_kind():
    """It is pointed at whatever ranked for a package name, so anything it held would be
    handed to a host an attacker chose."""
    sent: list[httpx.Request] = []

    def handler(request):
        sent.append(request)
        return html_response("<html><body><p>hello there friend</p></body></html>")

    async with fetcher(handler) as pages:
        await pages.fetch(source())
    header_names = {name.lower() for name in sent[0].headers}
    assert "authorization" not in header_names
    assert "x-subscription-token" not in header_names
    assert "cookie" not in header_names


async def test_a_non_http_url_is_refused_before_any_request_is_made():
    sent: list[httpx.Request] = []

    def handler(request):
        sent.append(request)
        return html_response("<html><body>should never happen</body></html>")

    from uadclaw.corroborate import SourceEvidence

    hostile = SourceEvidence(
        url="file:///etc/passwd", title="", snippet="s", block="web", position=1
    )
    async with fetcher(handler) as pages:
        fetched = await pages.fetch(hostile)
    assert sent == [], "the transport was never reached"
    assert fetched.text is None
    assert "not an http(s) url" in fetched.fetch_error


@pytest.mark.parametrize(
    "content_type", ["application/pdf", "image/png", "application/octet-stream"]
)
async def test_a_non_text_content_type_falls_back_to_the_snippet(content_type):
    def handler(request):
        return httpx.Response(200, headers={"content-type": content_type}, content=b"%PDF-1.7")

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(snippet="a summary from the index"))
    assert fetched.text is None
    assert content_type in fetched.fetch_error
    assert fetched.judged_text == "a summary from the index"


async def test_text_plain_is_extracted_without_going_through_the_html_parser():
    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"com.example.app  does\n things"
        )

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source())
    assert fetched.text == "com.example.app does things"


async def test_a_body_past_the_ceiling_stops_being_read_at_the_ceiling():
    """Asserted on bytes actually pulled out of the stream. A 5 GB page must not reach memory,
    and a check that ran after `aread()` would already have lost."""
    stream = CountingStream(b"<p>" + b"x" * 4093 + b"</p>", 4096)  # ~16 MB offered
    ceiling = 64 * 1024

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, stream=stream)

    async with fetcher(handler, max_bytes=ceiling) as pages:
        fetched = await pages.fetch(source())
    assert stream.yielded <= ceiling + len(stream.chunk), (
        f"read {stream.yielded} bytes of an offered {stream.count * len(stream.chunk)}"
    )
    assert stream.yielded < stream.count * len(stream.chunk)
    assert fetched.text, "the head of a long page is still evidence"


async def test_the_control_a_body_under_the_ceiling_is_read_whole():
    """The positive leg. Without it, "read fewer bytes than offered" is satisfied by a fetcher
    that reads nothing at all, and the ceiling assertion above would pin nothing."""
    stream = CountingStream(b"<p>" + b"x" * 4093 + b"</p>", 4)

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, stream=stream)

    async with fetcher(handler, max_bytes=64 * 1024) as pages:
        fetched = await pages.fetch(source())
    assert stream.yielded == 4 * len(stream.chunk)
    assert fetched.text


async def test_a_page_that_yields_no_text_is_recorded_as_such_rather_than_as_empty_evidence():
    def handler(request):
        return html_response("<html><body><script>only()</script></body></html>")

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(snippet="index summary"))
    assert fetched.text is None
    assert "no text" in fetched.fetch_error
    assert fetched.judged_text == "index summary"


async def test_a_non_200_status_falls_back_to_the_snippet():
    def handler(request):
        return html_response("<html>nope</html>", status=404)

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(snippet="index summary"))
    assert fetched.text is None
    assert "HTTP 404" in fetched.fetch_error
    assert fetched.judged_text == "index summary"


async def test_a_timeout_falls_back_to_the_snippet_rather_than_dropping_the_result():
    def handler(request):
        raise httpx.ReadTimeout("timed out")

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(snippet="index summary"))
    assert "ReadTimeout" in fetched.fetch_error
    assert fetched.judged_text == "index summary"


# --- redirects ------------------------------------------------------------------------------


async def test_a_same_scheme_redirect_is_followed_within_the_hop_budget():
    hops = []

    def handler(request):
        hops.append(str(request.url))
        if len(hops) < 3:
            return httpx.Response(302, headers={"Location": f"https://page.test/hop{len(hops)}"})
        return html_response("<html><body><p>the real page body</p></body></html>")

    async with fetcher(handler, max_redirects=3) as pages:
        fetched = await pages.fetch(source())
    assert fetched.text == "the real page body"
    assert len(hops) == 3


async def test_a_redirect_downgrading_https_to_http_is_refused():
    """§13's class, and this client is scheme-safe regardless of what the shared driver seam
    ends up being: httpx would follow a 302 from https to http without complaint."""

    def handler(request):
        return httpx.Response(302, headers={"Location": "http://plain.test/page"})

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source())
    assert fetched.text is None
    assert "downgrading https to http" in fetched.fetch_error
    assert "http://plain.test/page" in fetched.fetch_error


async def test_a_redirect_to_a_non_http_scheme_is_refused():
    def handler(request):
        return httpx.Response(302, headers={"Location": "ftp://files.test/x"})

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source())
    assert "non-http(s) url" in fetched.fetch_error


async def test_a_redirect_loop_stops_at_the_hop_budget():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://page.test/loop"})

    async with fetcher(handler, max_redirects=2) as pages:
        fetched = await pages.fetch(source())
    assert len(seen) == 3, "the first request plus two hops"
    assert "more than 2 redirect(s)" in fetched.fetch_error


async def test_a_plain_http_page_is_still_fetched():
    """The downgrade rule is about a redirect CHANGING the scheme, not about refusing http
    outright: a forum thread served over plain http is still evidence, and the fetcher carries
    no secret to lose."""

    def handler(request):
        return html_response("<html><body><p>an old forum thread</p></body></html>")

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(url="http://oldforum.test/t/1"))
    assert fetched.text == "an old forum thread"


async def test_an_http_page_may_redirect_to_https():
    def handler(request):
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"Location": "https://oldforum.test/t/1"})
        return html_response("<html><body><p>upgraded</p></body></html>")

    async with fetcher(handler) as pages:
        fetched = await pages.fetch(source(url="http://oldforum.test/t/1"))
    assert fetched.text == "upgraded"


# --- fetch_all --------------------------------------------------------------------------------


async def test_one_dead_host_costs_only_its_own_result():
    def handler(request):
        if "dead" in str(request.url):
            raise httpx.ConnectError("refused")
        return html_response("<html><body><p>a live page body</p></body></html>")

    merged = merge_results(
        envelope(
            web=[
                result("https://live1.test/a", description="s1"),
                result("https://dead.test/b", description="s2"),
                result("https://live2.test/c", description="s3"),
            ]
        ),
        limit=10,
    )
    async with fetcher(handler) as pages:
        fetched = await pages.fetch_all(merged)
    assert [source.url for source in fetched] == [source.url for source in merged], "order kept"
    assert [bool(source.text) for source in fetched] == [True, False, True]
    assert fetched[1].judged_text == "s2"


async def test_the_fetcher_never_exceeds_its_concurrency_bound():
    import asyncio

    state = {"live": 0, "peak": 0}

    async def handler(request):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        await asyncio.sleep(0.01)
        state["live"] -= 1
        return html_response("<html><body><p>a page body here</p></body></html>")

    merged = merge_results(
        envelope(web=[result(f"https://p{index}.test") for index in range(10)]), limit=10
    )
    async with fetcher(handler, max_concurrency=2) as pages:
        await pages.fetch_all(merged)
    # Exactly 2, not "at most 2": a fetcher that serialised everything would also satisfy an
    # upper bound while proving nothing about the semaphore.
    assert state["peak"] == 2, state["peak"]


# --- text extraction ---------------------------------------------------------------------------


def test_extraction_caps_the_text_at_the_configured_ceiling():
    body = ("<html><body><p>" + "word " * 5000 + "</p></body></html>").encode()
    assert len(extract_text(body, content_type="text/html", max_chars=100)) == 100


def test_extraction_survives_a_body_truncated_mid_tag():
    """The ceiling truncates at a byte boundary, so the parser is always handed broken HTML."""
    assert "readable" in extract_text(
        b'<html><body><p>readable</p><div class="x', content_type="text/html", max_chars=100
    )


# --- the citation titles come off the search result ----------------------------------------------


def test_source_links_take_the_title_from_the_search_result_rather_than_the_model():
    merged = merge_results(envelope(web=[result("https://a.test/1", title="Real Title")]), limit=1)
    assert source_links(merged, ["https://a.test/1"]) == [
        {"url": "https://a.test/1", "title": "Real Title"}
    ]
