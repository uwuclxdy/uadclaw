# Corroboration

**A second model pass checks whether an independent public source supports the description the classification stage proposed, and a fabricated citation refuses the whole answer.**

This implements the maintainer's stated review bar directly: `@AnonymousWP` on PR #1180 asked that an AI-suggested package function be checked "by an external source (by searching the internet)." Corroboration appends to the CLASSIFICATION job kind's stage walk (`JOB_KIND_STAGES[JobKind.CLASSIFICATION] = ("llm", "corroborate")`), because it has no input without a classification to check. See [Classification](Classification) for the stage before it.

## Two Brave clients, split for a security reason

`brave.py` holds a search client and a page fetcher, kept apart deliberately rather than as one HTTP wrapper.

- **`BraveClient` carries the subscription token and never follows a redirect.** `X-Subscription-Token` is a CUSTOM header. httpx strips only `Authorization` across an origin change on a redirect; a custom header rides along to wherever the 3xx points. `BraveClient` is built with `follow_redirects=False`, and a 3xx from the search API is `BraveRedirectError` naming the refused location rather than a hop taken with the key attached.
- **`PageFetcher` carries no credential of any kind.** It requests URLs a third party chose by ranking for a package name, which nothing else in this repository does, so it is the one client with no secret to leak and the one client with the SSRF gate.

## The SSRF gate

`PageFetcher` fetches whatever URL landed in the search results, so every axis of that request is bounded:

- **Scheme refusal.** `is_fetchable_url` refuses anything but `http`/`https` before a request is built, and refuses hostnames that pass `urlsplit` but fail httpx's IDNA encoder.
- **Streaming byte ceiling.** `_read_bounded` reads via `response.aiter_bytes`, truncating at `page_fetch_max_bytes` (4 MiB) so a multi-gigabyte response never reaches memory; `await response.aread()` would defeat this by buffering the whole body first.
- **Content-type gate.** Only `text/html`, `application/xhtml+xml` and `text/plain` are parsed; anything else falls back to the Brave snippet.
- **Redirect bound.** Each hop is walked by hand (`follow_redirects=False` again) so the scheme is checked on every hop, not just the first; `page_fetch_max_redirects` (3) bounds the count, and a hop downgrading `https` to `http` is refused outright.
- **Wall-clock deadline.** `page_fetch_deadline_seconds` (30.0) bounds the whole fetch. httpx's own timeout is per-operation, so a server writing one byte every three seconds would otherwise stay alive far past any single-operation timeout.
- **`address_refusal` runs on the initial URL and on every redirect hop.** A loopback, unspecified, link-local, multicast, private or reserved literal address is refused by name (`_blocked_address`), and a hostname that resolves to one of those is refused the same way after a bounded DNS lookup.

### The `2002::/16` refuse-by-name rule

Two IPv6 transition blocks (6to4 `2002::/16`, Teredo `2001::/32`) are refused by name rather than left to `ipaddress.is_private`. Both wrap an IPv4 address inside an IPv6 literal, and `ipaddress`'s classification of them moved between CPython 3.12.3 and 3.12.13: on 3.12.3 the whole of `2002::/16` reads `is_private=False`, `is_global=True`, no label at all, cloud metadata and loopback addresses included; on 3.12.13 every one of them reads `is_private=True`. This project's floor is `requires-python = ">=3.12.13"` for that reason. The two blocks are deliberately NOT normalized to their embedded IPv4 the way `::ffff:0:0/96` is: normalizing `2002:0808:0808::` would turn a 6to4 address the shipping interpreter refuses into one that fetches, because the embedded `8.8.8.8` carries no label of its own. Refusing the block whole is the narrower answer, and it holds regardless of which interpreter classification table is current. See [Configuration](Configuration) for the Python floor.

`address_refusal` states its own bound honestly: a name that resolves clean at check time can resolve differently a moment later when httpx actually connects (DNS rebinding), and this gate cannot close that without owning the socket.

## The judge and its four statuses

`corroborate.py` is pure: no database, no network, no clock. The judge is `DeepSeekClient` again, reusing its one retry layer, but the prompt asks a different question: given the package name, the proposed description, and the search results, does any source support the claim?

| Status | Set by | Meaning | Retryable |
|---|---|---|---|
| `corroborated` | model | a source identifies the package and agrees with what the description says it does | no, terminal until the description changes |
| `uncorroborated` | model | nothing found supports it | no, terminal, but the candidate still reaches triage flagged |
| `search_failed` | code | the search itself could not be completed | yes, a retry spends a Brave query |
| `judge_failed` | code | search succeeded but the judge could not answer | yes, a retry spends zero Brave quota, the search rows are cached |

`JudgeVerdict.status` is validated as a plain string rather than the enum directly, so a model answering `search_failed` or `judge_failed` is refused by NAME as a status it is not allowed to reach for, rather than as an opaque enum-membership error. Both are facts about the pipeline, not judgements a model can make.

## The fabricated-citation gate

`_check_sources` is the safety property of this module. A judge that cites a URL outside the set of sources it was handed is refused WHOLE, never repaired by dropping the bad URL: a judge that cited something it never read was not reading the others either, and the fabricated-citation gate exists because a fabricated link reaches the upstream maintainer looking exactly like the diligence they asked for.

A source handed over carrying no usable text at all (its page fetch failed and Brave returned no snippet either) is refused the same way if cited, one step short of outright fabrication: the judge saw a bare URL and nothing about it, so citing it asserts support from evidence that was never in the prompt.

The judge's evidence is serialized as one JSON object (`user_prompt` uses `json.dumps`) rather than delimited between markers. `json.dumps` escapes every quote, backslash and control character by construction, so a hostile page cannot terminate its own quoted region by any spelling of the attempt; a prior delimiter-based guard was walked through by 9 of 17 review payloads using em-dashes, minus signs, zero-width spaces and a Cyrillic homoglyph.

## `corroboratestore.py`: two caches with two different keys

- **Search rows are keyed on the package NAME with a TTL, never on a bundle or description hash.** `cached_sources` returns a package's stored `PackageSearchResult` rows with no HTTP call at all when the newest row is inside `corroboration_search_ttl_days` (30.0 by default). Re-judging after a prompt change, or re-running a `judge_failed` package, spends zero Brave quota, because the query is the package name and nothing else.
- **The verdict is keyed on `description_digest(package, description)`**, a canonical-JSON sha256 over both fields. A re-classification that changes the description opens a new question; one that leaves it unchanged does not, and `select_candidates` skips a package whose stored digest matches.

`store_verdict` and `record_failure` each refuse to write the other's shape: `store_verdict` raises if handed a `RETRYABLE` status, `record_failure` raises if handed a status that is not `RETRYABLE`. A failure write always clears `sources`, because a row whose `description_sha256` names one claim and whose `sources` answers a stale one would assert support for a claim nobody actually checked those sources against.

## Measured corroboration rate

Run over the real merged Pixel-plus-emulator corpus (393 packages, 48 queued survivors, 44 judged): **6 of 44 corroborated, 13.6%**. The design's own hypothesis (obscure OEM package names would corroborate poorly) is confirmed rather than refuted.

| Name family | Corroborated |
|---|---|
| `com.google.*` | 6 of 32 |
| `com.android.*` | 0 of 12 |

`com.android.*` corroborating at 0 of 12 is the expected reading rather than a broken search: AOSP-internal components have no independent public footprint to find. The remaining 4 of the 48 queued packages were never judged, because the model answered `unknown` for the description and there is no claim in the literal string `unknown` for a source to support; those reach triage with no corroboration row at all rather than a false negative. The human gate absorbs the majority of the review load by design, not because the search stage is failing. See [Triage-and-Emission](Triage-and-Emission) for what happens next.
