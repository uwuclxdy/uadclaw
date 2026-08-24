"""The corpus screen: `package_facts` joined to `package_analysis`, read-only.

Real Postgres throughout — the tri-state NULL/false/true distinction, the floor's
danger-rank ordering, and ILIKE substring search all need real query behavior, not a mock.
`db_session_factory` truncates every table per test, so a test's seeded rows are the entire
corpus that test sees; nothing here depends on what the app's own `uadclaw` database holds.
"""

import re

import pytest

from conftest import utcnow
from uadclaw.models import (
    BranchEmission,
    BranchEmissionPackage,
    PackageAnalysis,
    PackageClassification,
    PackageFact,
    PackageTriageDecision,
)
from uadclaw.views import corpus as corpus_view

PASSWORD = "test-only-admin-password"


def _verdict_badge(text: str, package: str) -> str:
    """The list row's verdict-cell tag text for `package`, walked from the row's own `<td>`
    cells rather than by proximity to the package name — the floor column's badge sits
    closer, and `<option value="true"...>already upstream</option>` in the filter select
    would satisfy a bare `resp.text` substring check regardless of what the row's badge says.
    Column order is package, devices, floor, verdict, conflict, so index 3."""
    idx = text.index(package)
    row_start = text.rindex("<tr>", 0, idx)
    row_end = text.index("</tr>", idx)
    row = text[row_start:row_end]
    cells = row.split("<td")[1:]
    verdict_cell = cells[3]
    match = re.search(r">([^<]*)</span>", verdict_cell)
    assert match, f"no tag span in the verdict cell for {package}: {verdict_cell!r}"
    return match.group(1).strip()


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


def _fact(
    package: str,
    *,
    device_count: int = 1,
    devices: list[str] | None = None,
    label: str | None = None,
    has_conflict: bool = False,
    conflicts: list | None = None,
) -> PackageFact:
    now = utcnow()
    return PackageFact(
        package=package,
        device_count=device_count,
        devices=devices or ["pixel:oriole"],
        first_seen_at=now,
        last_seen_at=now,
        label=label or package,
        has_conflict=has_conflict,
        conflicts=conflicts or [],
    )


def _analysis(
    package: str,
    *,
    floor: str | None = None,
    floor_rule: str | None = None,
    floor_reasons: list | None = None,
    queued: bool | None = None,
    upstream_present: bool | None = None,
    filter_verdict: str | None = None,
    edges: list | None = None,
    dependencies: list | None = None,
    needed_by: list | None = None,
    evidence: dict | None = None,
    privapp_allowlisted: bool | None = None,
    privapp_permission_count: int | None = None,
) -> PackageAnalysis:
    return PackageAnalysis(
        package=package,
        updated_at=utcnow(),
        floor=floor,
        floor_rule=floor_rule,
        floor_reasons=floor_reasons or [],
        queued=queued,
        upstream_present=upstream_present,
        filter_verdict=filter_verdict,
        edges=edges or [],
        dependencies=dependencies or [],
        needed_by=needed_by or [],
        evidence=evidence or {},
        privapp_allowlisted=privapp_allowlisted,
        privapp_permission_count=privapp_permission_count,
    )


async def _seed(session_factory, *rows) -> None:
    async with session_factory() as session, session.begin():
        session.add_all(rows)


def _classification(package: str) -> PackageClassification:
    now = utcnow()
    return PackageClassification(
        package=package,
        created_at=now,
        updated_at=now,
        bundle_sha256="b" * 64,
        model="deepseek-v4-flash",
        thinking=True,
        description=f"Vendor application {package}. Removing it loses its local data.",
        uad_list="Misc",
        removal="Advanced",
        confidence="medium",
        unknown_fields=[],
        reasoning_brief="No privileged surface.",
        provenance={"description": "llm:deepseek-v4-flash"},
        usage={},
        attempts=1,
        parked=False,
    )


def _approve(package: str) -> PackageTriageDecision:
    return PackageTriageDecision(
        package=package,
        bundle_sha256="b" * 64,
        action="approve",
        reason=None,
        edited_fields={},
        decided_at=utcnow(),
    )


async def _seed_emission(session_factory, package: str, *, branch: str) -> None:
    now = utcnow()
    async with session_factory() as session, session.begin():
        row = BranchEmission(
            vendor="pixel",
            branch=branch,
            repo_path="/srv/uadclaw/upstream",
            list_path="resources/assets/uad_lists.json",
            base_commit="a" * 40,
            list_sha256="b" * 64,
            pipeline_version="0.1.0",
            pipeline_commit_sha="c" * 40,
            package_count=1,
            pr_body="## pixel: 1 package addition(s)\n",
            created_at=now,
            commit_oid="e" * 40,
            committed_at=now,
            reconciled=False,
        )
        session.add(row)
        await session.flush()
        session.add(
            BranchEmissionPackage(
                emission_id=row.id,
                package=package,
                bundle_sha256="b" * 64,
                uad_list="Misc",
                removal="Advanced",
                floor="Advanced",
            )
        )


# --- auth ---------------------------------------------------------------------------------


async def test_corpus_list_requires_auth(db_env, client):
    resp = await client.get("/corpus")
    assert resp.status_code == 401


async def test_corpus_detail_requires_auth(db_env, client):
    resp = await client.get("/corpus/com.example.app")
    assert resp.status_code == 401


# --- empty state ----------------------------------------------------------------------------


async def test_a_fresh_database_reads_as_empty_not_broken(db_env, db_session_factory, client):
    await _login(client)
    resp = await client.get("/corpus", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "no packages yet" in resp.text
    # The empty state must not carry the error wording, or a reviewer cannot tell "nothing
    # scanned yet" from "the query broke" by reading the page.
    assert "could not load" not in resp.text


async def test_filtering_to_nothing_offers_a_way_back_rather_than_a_bare_no_results(
    db_env, db_session_factory, client
):
    await _seed(db_session_factory, _fact("com.example.only"), _analysis("com.example.only"))
    await _login(client)
    resp = await client.get("/corpus?q=zzz-does-not-exist")
    assert resp.status_code == 200
    assert "no packages match these filters" in resp.text
    assert "clear filters" in resp.text


@pytest.mark.parametrize(
    ("query", "value"),
    [
        ("floor", "Extremely Unsafe"),
        ("verdict", "maybe"),
    ],
)
async def test_a_filter_value_nobody_defined_says_so_instead_of_going_quiet(
    db_env, db_session_factory, client, query, value
):
    """An unrecognised value fell through with no filter applied while the context still
    carried it, so the operator got the WHOLE corpus, a "clear filters" button, and a select
    showing nothing selected — three things that disagree about whether a filter is on.

    The jobs screen already answers this shape with a warning; this is the same answer, out
    of the same helper. The value is dropped as well as reported, so `filters_active` and the
    select agree with the rows.
    """
    await _seed(db_session_factory, _fact("com.example.only"), _analysis("com.example.only"))
    await _login(client)

    resp = await client.get(f"/corpus?{query}={value}")

    assert resp.status_code == 200
    assert "ignored the unknown filter value" in resp.text
    assert "com.example.only" in resp.text, "the whole corpus is what an ignored filter shows"
    assert ">clear<" not in resp.text, "…so nothing claims a filter is active"


async def test_a_filter_value_that_is_defined_still_narrows_and_warns_about_nothing(
    db_env, db_session_factory, client
):
    """The positive leg. Without it the test above passes for a screen that ignores every
    filter it is given."""
    await _seed(
        db_session_factory,
        _fact("com.example.queued"),
        _analysis("com.example.queued", queued=True, filter_verdict="queued"),
        _fact("com.example.dropped"),
        _analysis("com.example.dropped", queued=False, filter_verdict="already_upstream"),
    )
    await _login(client)

    resp = await client.get("/corpus?verdict=queued")

    assert "ignored the unknown filter value" not in resp.text
    assert "com.example.queued" in resp.text
    assert "com.example.dropped" not in resp.text


async def test_a_stored_floor_outside_the_enum_renders_a_tag_rather_than_a_500(
    db_env, db_session_factory, client
):
    """The tag map was indexed directly on two screens and read with a default on a third, so
    a stored floor outside the four tiers rendered a 500 on the first two under
    `StrictUndefined` and a plain tag on the third. That value is representable — the store's
    own floor gate has a test seeding it — so the three have to agree, and the reading that
    keeps the screen up is the one that shows the value beside a neutral tag."""
    await _seed(
        db_session_factory,
        _fact("com.example.corrupt"),
        _analysis("com.example.corrupt", floor="Extremely Unsafe", floor_rule="core_app"),
    )
    await _login(client)

    listing = await client.get("/corpus")
    detail = await client.get("/corpus/com.example.corrupt")

    assert listing.status_code == 200
    assert detail.status_code == 200
    assert "Extremely Unsafe" in listing.text
    assert "Extremely Unsafe" in detail.text
    assert "tag-default" in detail.text


# --- error state, distinct from empty --------------------------------------------------------


async def test_a_db_failure_reads_as_an_error_never_as_an_empty_corpus(client):
    """No `db_env`: `POSTGRES_HOST` defaults to `postgres`, unreachable outside docker (see
    `test_health.py`), so the query fails for real rather than being mocked."""
    await _login(client)
    resp = await client.get("/corpus", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "could not load" in resp.text
    assert "no packages yet" not in resp.text


async def test_a_db_failure_on_the_detail_page_also_reads_as_an_error(client):
    await _login(client)
    resp = await client.get("/corpus/com.example.app", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "could not load" in resp.text
    assert "no package named" not in resp.text


# --- list: default sort, filters, search -----------------------------------------------------


async def test_default_sort_is_device_count_descending(db_env, db_session_factory, client):
    await _seed(
        db_session_factory,
        _fact("com.example.low", device_count=1),
        _analysis("com.example.low"),
        _fact("com.example.high", device_count=9),
        _analysis("com.example.high"),
        _fact("com.example.mid", device_count=5),
        _analysis("com.example.mid"),
    )
    await _login(client)
    resp = await client.get("/corpus")
    text = resp.text
    assert (
        text.index("com.example.high")
        < text.index("com.example.mid")
        < text.index("com.example.low")
    )


async def test_floor_filter_narrows_to_the_selected_tier(db_env, db_session_factory, client):
    await _seed(
        db_session_factory,
        _fact("com.example.unsafe"),
        _analysis("com.example.unsafe", floor="Unsafe", floor_rule="core_app"),
        _fact("com.example.rec"),
        _analysis("com.example.rec", floor="Recommended", floor_rule="default"),
    )
    await _login(client)
    resp = await client.get("/corpus?floor=Unsafe")
    assert "com.example.unsafe" in resp.text
    assert "com.example.rec" not in resp.text


async def test_search_matches_a_package_name_substring(db_env, db_session_factory, client):
    await _seed(
        db_session_factory,
        _fact("com.example.needle.thing"),
        _analysis("com.example.needle.thing"),
        _fact("com.other.hay"),
        _analysis("com.other.hay"),
    )
    await _login(client)
    resp = await client.get("/corpus?q=needle")
    assert "com.example.needle.thing" in resp.text
    assert "com.other.hay" not in resp.text


# --- NULL vs false, the tri-state contract ----------------------------------------------------


async def test_the_verdict_badge_distinguishes_pending_not_queued_and_queued(
    db_env, db_session_factory, client
):
    """The list row used to carry two separate badges (`queued`, `upstream`) derived from the
    same `filter_verdict` enum — redundant, and together with `floor`/`conflict` it put four
    tag-styled badges in one row against SPEC §5's "at most three" cap. They are merged into
    one `verdict` badge here, so this is the tri-state test that badge now has to pass."""
    await _seed(
        db_session_factory,
        _fact("com.example.pending"),
        # No PackageAnalysis row at all: the filter stage has not touched this package.
        _fact("com.example.rejected"),
        _analysis("com.example.rejected", queued=False, filter_verdict="already_upstream"),
        _fact("com.example.queued"),
        _analysis("com.example.queued", queued=True, filter_verdict="queued"),
    )
    await _login(client)
    resp = await client.get("/corpus")
    text = resp.text

    assert _verdict_badge(text, "com.example.pending") == "filter pending"
    assert _verdict_badge(text, "com.example.rejected") == "already upstream"
    assert _verdict_badge(text, "com.example.queued") == "queued"


async def test_a_row_never_shows_more_than_three_badges_at_once(db_env, db_session_factory, client):
    """SPEC §5: at most three badges in any one place. Before this round the row could show
    four (floor, queued, upstream, conflict) — queued/upstream merged into one verdict badge
    since both were derived from the same `filter_verdict` enum. floor + verdict + conflict is
    the ceiling case; this pins that a row carrying all three never exceeds it."""
    await _seed(
        db_session_factory,
        _fact(
            "com.example.busy",
            has_conflict=True,
            conflicts=[
                {
                    "field": "cert_issuer",
                    "values": [
                        {"value": "a", "devices": ["d1"]},
                        {"value": "b", "devices": ["d2"]},
                    ],
                }
            ],
        ),
        _analysis(
            "com.example.busy",
            floor="Unsafe",
            floor_rule="core_app",
            queued=True,
            filter_verdict="queued",
        ),
    )
    await _login(client)

    resp = await client.get("/corpus")
    text = resp.text
    idx = text.index("com.example.busy")
    row_start = text.rindex("<tr>", 0, idx)
    row_end = text.index("</tr>", idx)
    row = text[row_start:row_end]

    badge_count = row.count('class="tag ')
    assert badge_count == 3, f"expected floor + verdict + conflict, got {badge_count}: {row!r}"


async def test_the_list_page_carries_a_keyboard_reachable_badge_legend(
    db_env, db_session_factory, client
):
    """Badges on a 50-row list stay bare text — a per-row explanation would add up to 100 tab
    stops to a table meant to be scanned. One shared `<details>` legend explains every
    verdict/pending/conflict value instead, and `<summary>` is natively focusable so a
    keyboard user reaches it without a pointer (SPEC §5). Seeded with ZERO packages
    deliberately: the empty-corpus state still renders the legend (it sits above the
    `total == 0` branch), so nothing here can be satisfied by a row's own badge markup —
    only the legend itself can produce this text, which is what makes this a standalone pin
    rather than one that happens to red alongside a row-badge mutation."""
    await _login(client)

    resp = await client.get("/corpus")

    assert "<summary" in resp.text
    assert "what the badges mean" in resp.text
    assert "reaches the additions queue" in resp.text
    assert "dropped from the queue" in resp.text
    assert "review signal, not a verdict" in resp.text


async def test_filter_verdict_never_renders_its_raw_enum_spelling(
    db_env, db_session_factory, client
):
    """The bug: `package_analysis.filter_verdict` rendered as its raw `already_upstream` spelling
    on the corpus detail screen. `FILTER_VERDICT_LABEL` maps every defined member to display
    text; this pins that the raw spelling never reaches either screen, list or detail."""
    await _seed(
        db_session_factory,
        _fact("com.example.mapped"),
        _analysis(
            "com.example.mapped",
            queued=False,
            upstream_present=True,
            filter_verdict="already_upstream",
        ),
    )
    await _login(client)

    listing = await client.get("/corpus")
    detail = await client.get("/corpus/com.example.mapped")

    # `>already_upstream<`, not a bare substring: the verdict filter's own
    # `<option value="already_upstream">already upstream</option>` legitimately carries the
    # raw spelling as a wire VALUE (SPEC §4 — form values don't change), so a blanket
    # `"already_upstream" not in listing.text` is a false positive against that option now
    # that M4's filter exists. What must never appear is the raw spelling as visible text.
    assert ">already_upstream<" not in listing.text
    assert "already_upstream" not in detail.text
    # Row-scoped, not a bare `resp.text` membership check: `<option value="true"...>already
    # upstream</option>` in the upstream-style filter vocabulary would satisfy a substring
    # assertion whether or not the row's own badge is mapped at all.
    assert _verdict_badge(listing.text, "com.example.mapped") == "already upstream"
    assert "already upstream" in detail.text


async def test_an_unmapped_filter_verdict_falls_through_raw_rather_than_prettified(
    db_env, db_session_factory, client
):
    """A value nobody enumerated has to read as visibly unhandled, never as plausible prose a
    blanket `.replace('_', ' ')` would produce silently — that is the exact failure mode
    `FILTER_VERDICT_LABEL` exists to avoid (SPEC §4)."""
    await _seed(
        db_session_factory,
        _fact("com.example.futureverdict"),
        _analysis("com.example.futureverdict", filter_verdict="some_future_verdict"),
    )
    await _login(client)

    resp = await client.get("/corpus/com.example.futureverdict")

    assert "some_future_verdict" in resp.text
    assert "some future verdict" not in resp.text


async def test_verdict_filter_null_returns_only_the_unfiltered_packages(
    db_env, db_session_factory, client
):
    await _seed(
        db_session_factory,
        _fact("com.example.pending"),
        _fact("com.example.rejected"),
        _analysis("com.example.rejected", filter_verdict="already_upstream"),
        _fact("com.example.queued"),
        _analysis("com.example.queued", filter_verdict="queued"),
    )
    await _login(client)
    resp = await client.get("/corpus?verdict=null")
    assert "com.example.pending" in resp.text
    assert "com.example.rejected" not in resp.text
    assert "com.example.queued" not in resp.text


async def test_verdict_filter_reaches_a_value_the_old_queued_upstream_controls_could_not(
    db_env, db_session_factory, client
):
    """The reach gap a reviewer named: `queued`/`upstream_present` are two booleans derived
    from `filter_verdict`, so the old two-select filter could narrow to "queued" or "already
    upstream" but never to `generated overlay` or `emulator only` specifically — both read as
    "not queued and not upstream" and were bucketed together. One select over the verdict
    column itself reaches all four exactly."""
    await _seed(
        db_session_factory,
        _fact("com.example.overlay"),
        _analysis("com.example.overlay", filter_verdict="auto_generated_rro"),
        _fact("com.example.emu"),
        _analysis("com.example.emu", filter_verdict="emulator_only"),
    )
    await _login(client)
    resp = await client.get("/corpus?verdict=auto_generated_rro")
    assert "com.example.overlay" in resp.text
    assert "com.example.emu" not in resp.text


# --- floor sort must use danger_rank, never string comparison --------------------------------


async def test_floor_sort_ascending_orders_by_danger_rank_not_by_string(
    db_env, db_session_factory, client
):
    """`danger_rank(Recommended) == 0 < danger_rank(Expert) == 2`, but the strings sort the
    other way (`"Expert" < "Recommended"` lexicographically). A sort that compares the raw
    column would put Expert first under `dir=asc`; a correct one puts Recommended first."""
    await _seed(
        db_session_factory,
        _fact("com.example.expert"),
        _analysis("com.example.expert", floor="Expert", floor_rule="persistent"),
        _fact("com.example.recommended"),
        _analysis("com.example.recommended", floor="Recommended", floor_rule="default"),
    )
    await _login(client)
    resp = await client.get("/corpus?sort=floor&dir=asc")
    text = resp.text
    assert text.index("com.example.recommended") < text.index("com.example.expert"), (
        "floor=asc must rank Recommended before Expert; a raw string sort would invert this"
    )


# --- pagination -------------------------------------------------------------------------------


async def test_pagination_is_bounded_and_clamps_an_out_of_range_page(
    db_env, db_session_factory, client, monkeypatch
):
    monkeypatch.setattr(corpus_view, "PAGE_SIZE", 2)
    await _seed(
        db_session_factory,
        *[
            row
            for i in range(5)
            for row in (_fact(f"com.example.p{i}", device_count=i), _analysis(f"com.example.p{i}"))
        ],
    )
    await _login(client)

    resp = await client.get("/corpus")
    assert "page 1 of 3" in resp.text  # ceil(5/2)

    resp_out = await client.get("/corpus?page=999")
    assert "page 3 of 3" in resp_out.text
    assert "showing 5-5 of 5" in resp_out.text


# --- detail view --------------------------------------------------------------------------


async def test_detail_renders_facts_floor_edges_and_evidence(db_env, db_session_factory, client):
    await _seed(
        db_session_factory,
        _fact("com.example.detailed", device_count=2),
        _analysis(
            "com.example.detailed",
            floor="Unsafe",
            floor_rule="core_app",
            floor_reasons=[
                {
                    "rule": "core_app",
                    "floor": "Unsafe",
                    "detail": 'coreApp="true": AOSP puts it in the minimalist boot environment',
                }
            ],
            queued=False,
            upstream_present=True,
            filter_verdict="already_upstream",
            edges=[
                {
                    "kind": "overlay",
                    "dependent": "com.example.detailed",
                    "provider": "com.example.target",
                    "detail": "com.example.target",
                }
            ],
            dependencies=["com.example.target"],
            evidence={
                "queries_packages_in_corpus": ["com.example.target"],
                "content_uri_authorities_in_corpus": ["com.example.target"],
                "content_uri_authorities_absent": ["com.gone.provider"],
            },
        ),
    )
    await _login(client)
    resp = await client.get("/corpus/com.example.detailed", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "Unsafe" in resp.text
    assert "core_app" in resp.text
    assert "com.example.target" in resp.text
    assert "already upstream" in resp.text
    assert "content uri refs (in corpus)" in resp.text
    assert "content uri refs (absent)" in resp.text
    assert "com.gone.provider" in resp.text


async def test_detail_for_a_package_with_no_analysis_row_shows_the_facts_and_says_pending(
    db_env, db_session_factory, client
):
    await _seed(db_session_factory, _fact("com.example.unanalyzed"))
    await _login(client)
    resp = await client.get("/corpus/com.example.unanalyzed")
    assert resp.status_code == 200
    assert "has not analyzed this package yet" in resp.text


async def test_detail_for_a_missing_package_reads_as_not_found_not_as_an_error(
    db_env, db_session_factory, client
):
    await _login(client)
    resp = await client.get("/corpus/does.not.exist")
    # 200, not 404: htmx 2.0.10 does not swap a non-2xx response by default, so a boosted
    # link into a missing package must still render visible content.
    assert resp.status_code == 200
    assert "no package named" in resp.text
    assert "could not load" not in resp.text


# --- triage standing: the decision and the shipped state ------------------------------------


async def test_detail_renders_a_shipped_package_with_the_branch_it_went_out_on(
    db_env, db_session_factory, client
):
    """§19's corpus half: this screen shows a package's full standing, and a package whose
    approval went out on a branch has to say so, with the branch."""
    await _seed(
        db_session_factory,
        _fact("com.example.shipped"),
        _analysis("com.example.shipped", floor="Advanced", floor_rule="default"),
        _classification("com.example.shipped"),
        _approve("com.example.shipped"),
    )
    # Two shipments, the older first: the detail must show the NEWEST branch, not the first.
    await _seed_emission(
        db_session_factory, "com.example.shipped", branch="uadclaw/pixel-000000000001"
    )
    await _seed_emission(
        db_session_factory, "com.example.shipped", branch="uadclaw/pixel-000000000002"
    )
    await _login(client)

    resp = await client.get("/corpus/com.example.shipped", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "triage standing" in resp.text
    assert "shipped → uadclaw/pixel-000000000002" in resp.text
    assert "shipped → uadclaw/pixel-000000000001" not in resp.text


async def test_detail_renders_an_approval_still_awaiting_emission(
    db_env, db_session_factory, client
):
    """An approval with no emission row is not shipped, and the corpus screen must not call
    it one: it shows the decision and that it is still awaiting the next batch."""
    await _seed(
        db_session_factory,
        _fact("com.example.waiting"),
        _analysis("com.example.waiting", floor="Advanced", floor_rule="default"),
        _classification("com.example.waiting"),
        _approve("com.example.waiting"),
    )
    await _login(client)

    resp = await client.get("/corpus/com.example.waiting", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "triage standing" in resp.text
    assert "approved" in resp.text
    assert "awaiting the next emission" in resp.text
    assert "shipped →" not in resp.text


async def test_detail_renders_a_rejection_even_when_the_package_once_shipped(
    db_env, db_session_factory, client
):
    """A package that shipped, then was reopened and rejected, is a rejection: the current
    decision wins on this screen too, and the old emission must not read as 'shipped'."""
    await _seed(
        db_session_factory,
        _fact("com.example.rejected"),
        _analysis("com.example.rejected", floor="Advanced", floor_rule="default"),
        _classification("com.example.rejected"),
        _approve("com.example.rejected"),
    )
    await _seed_emission(
        db_session_factory, "com.example.rejected", branch="uadclaw/pixel-000000000001"
    )
    await _seed(
        db_session_factory,
        PackageTriageDecision(
            package="com.example.rejected",
            bundle_sha256="b" * 64,
            action="reject",
            reason="changed my mind",
            edited_fields={},
            decided_at=utcnow(),
        ),
    )
    await _login(client)

    resp = await client.get("/corpus/com.example.rejected", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "triage standing" in resp.text
    assert "rejected" in resp.text
    assert "shipped →" not in resp.text


async def test_a_zero_library_edge_count_is_not_rendered_as_an_error(
    db_env, db_session_factory, client
):
    """Measured on the real corpus: the library edge class yields zero edges, and that is
    the data, not a broken lookup (see `corpus.py`'s module docstring). The empty section
    must read as neutral, never as a warning or a danger callout."""
    await _seed(
        db_session_factory,
        _fact("com.example.noedges"),
        _analysis("com.example.noedges", floor="Recommended", floor_rule="default", edges=[]),
    )
    await _login(client)
    resp = await client.get("/corpus/com.example.noedges")
    assert "no edges recorded" in resp.text
    assert "callout-danger" not in resp.text
    assert "callout-warning" not in resp.text


# --- has_conflict is rendered, never filtered out by default ---------------------------------


async def test_conflicts_render_with_both_values_and_are_not_hidden_by_default(
    db_env, db_session_factory, client
):
    await _seed(
        db_session_factory,
        _fact(
            "com.example.conflicted",
            has_conflict=True,
            conflicts=[
                {
                    "field": "cert_issuer",
                    "values": [
                        {"value": "issuer-a", "devices": ["pixel:oriole"]},
                        {"value": "issuer-b", "devices": ["google:emulator-a16"]},
                    ],
                }
            ],
        ),
        _analysis("com.example.conflicted"),
    )
    await _login(client)

    list_resp = await client.get("/corpus")
    assert "com.example.conflicted" in list_resp.text
    assert "conflict" in list_resp.text

    detail_resp = await client.get("/corpus/com.example.conflicted")
    assert "cert_issuer" in detail_resp.text
    assert "issuer-a" in detail_resp.text
    assert "issuer-b" in detail_resp.text
    assert "pixel:oriole" in detail_resp.text
    assert "google:emulator-a16" in detail_resp.text


# --- icons: has_icon / the monogram fallback -------------------------------------------------


async def test_a_package_with_no_icon_renders_the_monogram_fallback(
    db_env, db_session_factory, client
):
    """`icon_bytes`/`icon_mime` land on `package_facts` in a sibling worktree of this same
    round and are not on this branch's `PackageFact` yet, so `_has_icon`'s `getattr` default
    always reads False here — this pins the fallback path (the only one this worktree can
    exercise) rather than the `<img>` path, which needs that migration to render for real.

    Letters and colour come from the shared `monogram_for`/`pkg_icon` macro
    (`src/uadclaw/monogram.py`, `templates/partials/pkg_icon.html`), not from this screen: the
    chip is `aria-hidden="true"` because every call site here renders the package name as
    text right beside it, and the letters are lowercase (`monogram_for`'s own derivation)."""
    await _seed(db_session_factory, _fact("com.example.noicon"), _analysis("com.example.noicon"))
    await _login(client)

    listing = await client.get("/corpus")
    detail = await client.get("/corpus/com.example.noicon")

    # The list is the 32px rail variant, the detail the 48px card variant (icon contract,
    # SPEC §6) — pinned here so a later edit cannot silently put the rail size on the card.
    assert 'class="monogram monogram-c' in listing.text
    assert ' monogram-sm"' in listing.text
    assert 'aria-hidden="true"' in listing.text
    assert ">no<" in listing.text  # last dotted component "noicon" -> "no"
    assert "/icons/com.example.noicon" not in listing.text
    assert 'class="monogram monogram-c' in detail.text
    assert ' monogram-sm"' not in detail.text
    assert ">no<" in detail.text


# --- untrusted bytes: package names are firmware data, never |safe -----------------------------


async def test_a_hostile_package_name_is_escaped_not_executed(db_env, db_session_factory, client):
    hostile = "com.example.<script>alert(1)</script>"
    await _seed(db_session_factory, _fact(hostile), _analysis(hostile))
    await _login(client)
    resp = await client.get("/corpus")
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


async def test_a_slash_in_a_package_name_is_encoded_in_every_link(
    db_env, db_session_factory, client
):
    """`_package_href`/`_package_links` go through `web.url_segment`, never the `|urlencode`
    filter: jinja2's `do_urlencode` calls `url_quote` with `safe=b"/"`, so a package name
    carrying a `/` comes back unescaped and the link points at a different path than the
    record. A package name is bytes out of a downloaded manifest, so nothing validates it as
    a Java identifier — this covers both `CorpusRow.href` (the list row) and
    `_package_links` (the detail page's dependency/needed-by tags)."""
    hostile = "com.example/evil"
    target = "com.example.target"
    await _seed(
        db_session_factory,
        _fact(hostile),
        _analysis(hostile, filter_verdict="queued", dependencies=[target]),
        _fact(target),
        _analysis(target, filter_verdict="queued", needed_by=[hostile]),
    )
    await _login(client)

    listing = await client.get("/corpus")
    assert 'href="/corpus/com.example%2Fevil"' in listing.text
    assert 'href="/corpus/com.example/evil"' not in listing.text

    target_detail = await client.get("/corpus/com.example.target")
    assert 'href="/corpus/com.example%2Fevil"' in target_detail.text
