"""The triage screen, driven the way a reviewer drives it.

The task's verify line is "drive the full queue for one device by keyboard alone; confirm a
rejected candidate persists with its reason and does not reappear". The keybinds click the
buttons the card renders, so driving those forms IS driving the keyboard — and the same forms
are what a browser with no JavaScript submits, which is why the flow is asserted through them
rather than through a JSON endpoint no control points at.

The five screen states each get a test. Four of them are the ones that ship broken.
"""

import json
import re
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from uadclaw import triagestore, web
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import park_package, store_classification
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import (
    PackageAnalysis,
    PackageClassification,
    PackageCorroboration,
    PackageFact,
    PackageTriageDecision,
)
from uadclaw.monogram import MONOGRAM_COLOURS, monogram_for
from uadclaw.views import triage as triage_view

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
MODEL = "deepseek-v4-flash"
BUNDLE = "b" * 64
PASSWORD = "test-only-admin-password"

UPSTREAM_JSON = json.dumps(
    {
        "com.example.sibling": {
            "list": "Oem",
            "removal": "Advanced",
            "description": "A sibling entry upstream already carries.",
        }
    }
)


@pytest.fixture
def triage_env(monkeypatch, tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text(UPSTREAM_JSON, encoding="utf-8")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", str(path))
    return path


@pytest.fixture
async def triage_db(db_session_factory):
    """`conftest`'s truncation list predates `package_triage_decision`; see the store suite."""
    async with db_session_factory() as session, session.begin():
        await session.execute(text("TRUNCATE TABLE package_triage_decision RESTART IDENTITY"))
    return db_session_factory


async def login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


def make_facts(package: str, **overrides) -> ApkFacts:
    values: dict[str, object] = {
        "package": package,
        "label": "Example",
        "label_unresolved": False,
        "version_code": 1,
        "partition": "product",
        "device_path": f"/product/app/{package}/{package}.apk",
        "priv_app": False,
        "sha256": "0" * 64,
        "cert_issuer": "Organization: Example Corp",
        "cert_subject": "Organization: Example Corp",
        "core_app": False,
        "shared_user_id": None,
        "persistent": False,
        "has_code": True,
        "overlay_target": None,
        "overlay_static": False,
        "overlay_priority": None,
    }
    values.update(overrides)
    return ApkFacts(**values)  # type: ignore[arg-type]


def proposal(package: str, *, removal: Removal = Removal.ADVANCED) -> Classification:
    return Classification(
        package=package,
        bundle_sha256=BUNDLE,
        description=f"Vendor application {package}. Removing it loses its local data.",
        list=UadList.MISC,
        removal=removal,
        confidence=Confidence.MEDIUM,
        unknown_fields=(),
        reasoning_brief="No privileged surface.",
        provenance={"description": f"llm:{MODEL}", "removal": f"llm:{MODEL}"},
    )


async def seed(session_factory, packages: dict[str, int], *, floor: str = "Advanced") -> None:
    async with session_factory() as session, session.begin():
        for package, devices in packages.items():
            for index in range(devices):
                await store_device_facts(
                    session,
                    device_key=f"pixel:device{index}",
                    build="bp1a.260505.001",
                    facts=[make_facts(package)],
                    observed_at=NOW,
                )
            session.add(
                PackageAnalysis(
                    package=package,
                    updated_at=NOW,
                    queued=True,
                    filter_verdict="queued",
                    floor=floor,
                    floor_rule="privileged",
                    floor_reasons=[
                        {
                            "rule": "privileged",
                            "floor": floor,
                            "detail": "privileged (priv-app)",
                        }
                    ],
                )
            )
    async with session_factory() as session, session.begin():
        for package in packages:
            await store_classification(
                session,
                proposal(package),
                model=MODEL,
                thinking=True,
                usage={},
                attempts=1,
                at=NOW,
            )


async def screen(client, query: str = "") -> str:
    resp = await client.get(f"/triage{query}", headers={"accept": "text/html"})
    assert resp.status_code == 200, resp.text[:400]
    return resp.text


async def act(client, **form) -> str:
    """One action, through the form a keybind clicks and a no-JS browser submits."""
    resp = await client.post(
        "/triage/decide", data=form, headers={"accept": "text/html", "HX-Request": "true"}
    )
    assert resp.status_code == 200, resp.text[:400]
    return resp.text


# --- the queue ------------------------------------------------------------------------------


async def test_the_queue_renders_in_device_count_order(db_env, triage_env, triage_db, client):
    await login(client)
    await seed(triage_db, {"com.example.one": 1, "com.example.three": 3, "com.example.two": 2})

    body = await screen(client)

    names = ("com.example.three", "com.example.two", "com.example.one")
    order = [body.index(name) for name in names]
    assert order == sorted(order)


async def test_the_card_shows_the_floor_and_the_rule_that_set_it(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert "minimum rating" in body
    assert ">Advanced</strong>, set by <code>privileged</code>" in body
    assert "privileged (priv-app)" in body


async def test_the_card_shows_the_evidence_and_the_nearest_upstream_entry(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert "firmware facts" in body
    assert "similar upstream entries" in body
    assert "Organization: Example Corp" in body
    assert "com.example.sibling" in body


async def test_no_two_controls_answer_to_the_same_key(db_env, triage_env, triage_db, client):
    """Found in a browser, not here: the buttons used to take their key from the action's
    first letter, so `reopen` claimed `r`, the script binds whichever control comes first in
    the document, and pressing `r` to write a rejection reopened the package instead. Two
    controls sharing a key is silent — the wrong one just answers.
    """
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    keys = re.findall(r'data-key="([^"]+)"', body)
    assert keys, "no control declares a key at all"
    assert len(keys) == len(set(keys)), f"two controls share a key: {sorted(keys)}"
    assert set(keys) <= {key for key, _ in triage_view.KEYBINDS}


async def test_every_advertised_keybind_has_a_control_or_moves_the_queue(
    db_env, triage_env, triage_db, client
):
    """The other half. A key in the help list with nothing bound to it is a key that does
    nothing, and the help is the only place a reviewer learns them from."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    bound = set(re.findall(r'data-key="([^"]+)"', body))
    navigation = {"j", "k"}
    for key, _label in triage_view.KEYBINDS:
        assert key in bound or key in navigation, f"{key} is advertised and bound to nothing"


async def test_the_keybinds_are_shown_on_the_card(db_env, triage_env, triage_db, client):
    """A keyboard-driven screen that does not say what the keys are is a screen only its
    author can drive."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    for key in ("<kbd>a</kbd>", "<kbd>r</kbd>", "<kbd>d</kbd>", "<kbd>e</kbd>"):
        assert key in body


# --- corroboration --------------------------------------------------------------------------


async def test_a_cited_source_is_rendered_as_a_working_link(db_env, triage_env, triage_db, client):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        session.add(
            PackageCorroboration(
                package="com.example.one",
                created_at=NOW,
                updated_at=NOW,
                description_sha256="e" * 64,
                status="corroborated",
                model=MODEL,
                thinking=True,
                sources=[{"url": "https://example.test/page", "title": "A page about it"}],
                reasoning="the page describes the same feature",
                provenance={},
                usage={},
                attempts=1,
            )
        )

    body = await screen(client)

    assert '<a href="https://example.test/page"' in body
    assert 'rel="noopener noreferrer nofollow"' in body


async def test_a_source_that_is_not_a_web_url_is_never_turned_into_a_link(
    db_env, triage_env, triage_db, client
):
    """A stored url came off a search result a third party chose. Escaping does nothing about
    a `javascript:` href, because the value is not markup — the scheme is."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        session.add(
            PackageCorroboration(
                package="com.example.one",
                created_at=NOW,
                updated_at=NOW,
                description_sha256="e" * 64,
                status="corroborated",
                model=MODEL,
                thinking=True,
                sources=[{"url": "javascript:alert(1)", "title": "hostile"}],
                reasoning="",
                provenance={},
                usage={},
                attempts=1,
            )
        )

    body = await screen(client)

    assert 'href="javascript:' not in body
    assert "not a web link" in body


async def test_an_uncorroborated_candidate_reaches_the_screen_flagged(
    db_env, triage_env, triage_db, client
):
    """13.6% corroborate overall and 0 of 12 for `com.android.*`. A screen that suppressed
    the rest would hide 86% of the queue."""
    await login(client)
    await seed(triage_db, {"com.android.example": 1})

    body = await screen(client)

    assert "com.android.example" in body
    assert ">not searched</span>" in body
    assert "not searched yet." in body


# --- the five states ---------------------------------------------------------------------------


async def test_the_empty_state_names_the_reason_and_offers_the_next_step(
    db_env, triage_env, triage_db, client
):
    await login(client)

    body = await screen(client)

    assert "nothing to triage yet" in body
    assert 'href="/jobs"' in body


async def test_a_cleared_queue_reads_as_cleared_rather_than_as_empty(
    db_env, triage_env, triage_db, client
):
    """ "Nothing has ever been classified" and "you finished the queue" are different facts,
    and the second one is an achievement rather than a void."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await act(client, package="com.example.one", action="approve", view="queue")

    body = await screen(client)

    assert "queue is clear" in body
    assert "nothing to triage yet" not in body


async def test_the_partial_state_names_what_is_missing(db_env, triage_env, triage_db, client):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert '<span class="callout-title">missing</span>' in body
    assert "sources. common rather than broken." in body


async def test_an_unknown_package_is_an_error_beside_the_queue_rather_than_a_500(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client, "?package=com.example.absent")

    assert "is not in the queue list" in body
    # …and the screen still works: the top of the queue is shown.
    assert "com.example.one" in body


async def test_an_unknown_view_falls_back_to_the_queue_and_says_so(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client, "?view=everything")

    assert "is not a list" in body
    assert "com.example.one" in body


async def test_every_action_control_disables_itself_for_the_round_trip(
    db_env, triage_env, triage_db, client
):
    """The loading treatment. These round-trips land inside the 1s band where a spinner reads
    as a glitch, so the control answers the click by disabling — which also drops the second
    press rather than sending the action twice."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert body.count('hx-disabled-elt="this"') >= 4


# --- driving the queue ----------------------------------------------------------------------


async def test_approving_removes_the_package_from_the_queue_and_offers_an_undo(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 2})

    body = await act(client, package="com.example.two", action="approve", view="queue")

    assert "approved com.example.two." in body
    assert "undo" in body
    assert "com.example.one" in body
    assert await screen(client) and "com.example.two" not in await screen(client)


async def test_undo_puts_a_decided_package_back_in_the_queue(db_env, triage_env, triage_db, client):
    """Append-only means a mistaken verdict cannot be deleted, so there has to be a way to
    supersede one. Without it the first wrong keypress is permanent."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await act(client, package="com.example.one", action="approve", view="queue")

    await act(client, package="com.example.one", action="reopen", view="queue")

    assert "com.example.one" in await screen(client)
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert [row.action for row in result.scalars()] == ["approve", "reopen"]


async def test_a_rejection_without_a_reason_is_refused_at_the_screen(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await act(client, package="com.example.one", action="reject", reason="  ", view="queue")

    assert "a rejection needs a reason" in body
    # The store's own message names the function and the invariant, which is a log's business.
    assert "record_decision" not in body
    assert "com.example.one" in body
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert result.scalars().all() == []


async def test_a_rejected_candidate_persists_with_its_reason_and_does_not_reappear(
    db_env, triage_env, triage_db, client
):
    """The task's verify line, end to end through the screen."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 2})

    await act(
        client,
        package="com.example.two",
        action="reject",
        reason="description names a vendor the evidence never did",
        view="queue",
    )

    assert "com.example.two" not in await screen(client)
    decided = await screen(client, "?view=decided")
    assert "com.example.two" in decided
    assert "description names a vendor the evidence never did" in decided


async def test_deferring_leaves_the_queue_and_stays_behind_the_filter(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    await act(client, package="com.example.one", action="defer", view="queue")

    assert "com.example.one" not in await screen(client)
    assert "com.example.one" in await screen(client, "?view=deferred")


async def test_a_parked_package_is_reachable_behind_its_own_filter(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        session.add(PackageAnalysis(package="com.example.parked", updated_at=NOW, floor="Unsafe"))
    async with triage_db() as session, session.begin():
        await park_package(
            session,
            "com.example.parked",
            bundle_sha256=BUNDLE,
            model=MODEL,
            thinking=True,
            reason="removal: answered below the floor three times",
            usage={},
            attempts=3,
            at=NOW,
        )

    assert "com.example.parked" not in await screen(client)
    parked = await screen(client, "?view=parked")
    assert "deepseek gave no answer" in parked


# --- the edit path ----------------------------------------------------------------------------


async def test_an_edit_lands_with_human_provenance(db_env, triage_env, triage_db, client):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    resp = await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": "Notes application. Removing it loses locally stored notes.",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )
    assert resp.status_code == 200

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == "Notes application. Removing it loses locally stored notes."
    assert row.provenance["description"] == "human:triage"


async def test_an_edit_below_the_floor_is_refused_and_says_why(
    db_env, triage_env, triage_db, client
):
    """The safety gate reaching the screen. It has to read as a refused edit rather than as a
    broken app, and the proposal it refused has to survive."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")

    resp = await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "removal": "Recommended", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200
    assert "computed floor" in resp.text
    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None and row.removal == "Advanced"


async def test_an_edit_keeps_the_package_in_front_of_the_reviewer(
    db_env, triage_env, triage_db, client
):
    """An edit is a revision, not a verdict. Being thrown to a different package after
    editing one is how an edit gets left unapproved."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 2})

    resp = await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": "Notes application. Removing it loses locally stored notes.",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert "edited description on com.example.one." in resp.text
    assert "com.example.one" in resp.text


# --- the shell contract -------------------------------------------------------------------------


async def test_an_htmx_request_gets_the_board_and_not_the_whole_page(
    db_env, triage_env, triage_db, client
):
    """A fragment carrying `<html>` would nest a whole page inside the element htmx swaps."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    resp = await client.get("/triage", headers={"accept": "text/html", "HX-Request": "true"})

    assert resp.status_code == 200
    assert "<html" not in resp.text
    assert 'id="triage-board"' in resp.text


async def test_a_package_name_out_of_firmware_is_escaped(db_env, triage_env, triage_db, client):
    """A package name is a string out of a downloaded manifest, not this pipeline's bytes.
    Nothing on this screen is `|safe`, and this is the assertion that says so."""
    await login(client)
    hostile = "com.example.<script>alert(1)</script>"
    await seed(triage_db, {hostile: 1})

    body = await screen(client)

    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


async def test_one_render_of_the_screen_reads_the_queue_once(
    db_env, triage_env, triage_db, client, monkeypatch
):
    """Pinned at the CALL SITE, not on the helper. `triagestore.load_board` having one read in
    it says nothing about the screen using it: swapping the view back to `load_counts` then
    `load_queue` left a helper-only test green and the screen computing its counts and its
    rows from two snapshots again."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    reads = []
    real = triagestore.load_rows

    async def counting(session):
        reads.append(1)
        return await real(session)

    monkeypatch.setattr(triagestore, "load_rows", counting)

    await screen(client)

    assert len(reads) == 1, f"the screen read the queue {len(reads)} times"


async def test_a_package_name_in_a_queue_link_is_url_encoded(db_env, triage_env, triage_db, client):
    """A package name is bytes out of downloaded firmware, and the record's identity is not
    the record's path — the rule the corpus screen already follows with `| urlencode`.

    Autoescape is not the same defence: it turns a literal `&` into `&amp;`, which the browser
    decodes straight back into a query-parameter separator, so the link selects a different
    package than the row it sits on.
    """
    await login(client)
    hostile = "com.example.notes&view=parked"
    await seed(triage_db, {hostile: 1})

    body = await screen(client)

    assert "package=com.example.notes%26view%3Dparked" in body
    assert "package=com.example.notes&amp;view=parked" not in body


@pytest.mark.parametrize("endpoint", ["/triage/decide", "/triage/edit"])
async def test_a_view_a_form_made_up_is_an_error_beside_the_queue_rather_than_a_500(
    db_env, triage_env, triage_db, client, endpoint
):
    """`view` is validated on the GET and was taken on trust on both POSTs, where it reaches
    `load_queue` and raises out of a handler that catches only its own input errors. It also
    reaches an `href` in the rendered board, which is the second reason it is parsed at the
    boundary rather than escaped at the end."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    resp = await client.post(
        endpoint,
        data={"package": "com.example.one", "action": "approve", "view": "everything"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200, resp.text[:400]
    assert "is not a list" in resp.text
    assert "com.example.one" in resp.text


async def test_a_database_that_refuses_the_password_is_an_error_state_not_a_500(
    monkeypatch, test_env, triage_env
):
    """Measured, not imagined: `asyncpg.InvalidPasswordError` is a `PostgresError` and not a
    `SQLAlchemyError`, so it escaped the handler and 500'd the screen. That is what a
    wrongly-mounted secret produces, and a 500 hides both the cause and the rest of the page.

    The client is built here rather than taken from the `client` fixture so the environment
    is wrong BEFORE the app resolves its settings.
    """
    from httpx import ASGITransport, AsyncClient

    from conftest import PG_HOST, PG_PORT
    from uadclaw.app import create_app

    monkeypatch.setenv("POSTGRES_HOST", PG_HOST)
    monkeypatch.setenv("POSTGRES_PORT", str(PG_PORT))
    monkeypatch.setenv("POSTGRES_PASSWORD", "not-the-password")

    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client)
        resp = await client.get("/triage", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "the queue could not be read" in resp.text
    assert "try again" in resp.text
    # The rest of the page survived its own failure.
    assert 'class="navbar-link active"' in resp.text


async def test_the_screen_is_behind_auth_like_every_other_route(db_env, triage_env, client):
    resp = await client.post("/triage/decide", data={"package": "x", "action": "approve"})
    assert resp.status_code == 401


# --- the three columns, the badges and the wording ---------------------------------------


def columns(body: str) -> tuple[str, str]:
    """The decide column and the evidence column, split where the second one starts.

    Asserted against the rendered halves rather than against the whole body, because every
    string this round moved is a string that already existed somewhere on the page: a bare
    `in body` cannot tell "the evidence is in its own column" from "the evidence is where it
    always was".
    """
    marker = '<div class="triage-evidence">'
    assert marker in body, "the evidence column was not rendered"
    before, evidence = body.split(marker, 1)
    # Split at BOTH ends. Splitting only at the evidence marker made `decide` mean "everything
    # that is not the evidence column", which is the page header and the whole queue rail as
    # well — so `badge_rows(decide)` read the RAIL's badges, and a card that rendered no badges
    # at all satisfied the liveness guard on this helper's own output.
    assert 'class="triage-decide"' in before, "the decide column was not rendered"
    return before.split('class="triage-decide"', 1)[1], evidence


def badge_rows(fragment: str) -> list[list[str]]:
    """Every `.badge-row` in a fragment, as the list of badge labels inside it."""
    return [
        re.findall(r'<span class="tag [^"]*"[^>]*>([^<]*)</span>', row)
        for row in re.findall(r'class="badge-row"[^>]*>(.*?)</div>', fragment, re.S)
    ]


async def corroborate(session_factory, package: str, *, status: str, sources: list) -> None:
    async with session_factory() as session, session.begin():
        session.add(
            PackageCorroboration(
                package=package,
                created_at=NOW,
                updated_at=NOW,
                description_sha256="e" * 64,
                status=status,
                model=MODEL,
                thinking=True,
                sources=sources,
                reasoning="",
                provenance={},
                usage={},
                attempts=1,
            )
        )


async def test_the_decision_and_the_evidence_are_separate_columns(
    db_env, triage_env, triage_db, client
):
    """The clutter this round was asked to fix: seven sections stacked in one column on one
    scroll. What the reviewer acts on and what backs it up are different columns now, so a
    long evidence section cannot push the verdict buttons off the screen."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)
    decide, evidence = columns(body)

    assert 'class="triage-board"' in body
    assert 'class="triage-rail"' in body
    for acted_on in ('name="action" value="approve"', "deepseek's answer", "minimum rating"):
        assert acted_on in decide, acted_on
    for backing in (">firmware facts</div>", ">similar upstream entries</summary>"):
        assert backing in evidence, backing
    assert ">firmware facts</div>" not in decide


async def test_the_lower_value_evidence_is_collapsed_rather_than_scrolled_past(
    db_env, triage_env, triage_db, client
):
    """`sources` and `firmware facts` are what a rating is argued from and stay open. The two
    reference sections were most of the scroll and are read on the packages where something
    looks wrong, so they open on request."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await act(client, package="com.example.one", action="defer", view="queue")

    _, evidence = columns(await screen(client, "?view=deferred"))

    assert (
        '<summary class="label" style="cursor:pointer">similar upstream entries</summary>'
        in evidence
    )
    assert '<summary class="label" style="cursor:pointer">past decisions (1)</summary>' in evidence
    assert (
        '<div class="label" style="margin-bottom:var(--space-3)">firmware facts</div>' in evidence
    )


async def flag_everything(session_factory, package: str) -> None:
    """One package carrying every flag at once: a rating, a device disagreement, no answer.

    The worst case is what the ceiling is measured against — a ceiling checked on a row that
    renders no badges at all passes for the wrong reason. The park reuses the bundle the
    proposal was stored under on purpose, because `park_package` only clears the answer when
    the bundle moved, and a park that wiped the rating would take the third badge with it.
    """
    async with session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key="pixel:rotated",
            build="bp1a.260505.001",
            facts=[make_facts(package, cert_issuer="Organization: Rotated Key")],
            observed_at=NOW,
        )
    async with session_factory() as session, session.begin():
        await park_package(
            session,
            package,
            bundle_sha256=BUNDLE,
            model=MODEL,
            thinking=True,
            reason="removal: answered below the floor three times",
            usage={},
            attempts=3,
            at=NOW,
        )


async def test_the_card_shows_at_most_three_badges_and_the_rail_at_most_two(
    db_env, triage_env, triage_db, client
):
    """The user's complaint, measured. The card header rendered up to seven at once and the
    rail up to five; a row carrying a rating, a confidence, a corroboration verdict, a
    conflict and a park is a row nobody reads any of."""
    await login(client)
    await seed(triage_db, {"com.example.flagged": 1})
    await corroborate(triage_db, "com.example.flagged", status="corroborated", sources=[])
    await flag_everything(triage_db, "com.example.flagged")

    body = await screen(client, "?view=parked")
    decide, _ = columns(body)
    rail = body.split('<aside class="triage-rail">', 1)[1].split("</aside>", 1)[0]

    card_badges = badge_rows(decide)
    rail_badges = badge_rows(rail)
    assert card_badges and card_badges[0], "the card rendered no badges, so no ceiling was read"
    assert rail_badges and rail_badges[0], "the rail rendered no badges, so no ceiling was read"
    for row in card_badges:
        assert len(row) <= 3, f"the card header carries {len(row)} badges: {row}"
    for row in rail_badges:
        assert len(row) <= 2, f"a queue row carries {len(row)} badges: {row}"
    # The SET, not just the ceiling: this fixture carries every flag at once, so the card is
    # at its ceiling and each of the three is one that earns its slot. `no answer` is also
    # spelled by the badge legend and by the parked callout, both inside this column, so a
    # bare `in decide` would stay green with the badge itself deleted.
    assert card_badges == [["removal advanced", "no answer", "conflict"]]


async def test_the_rail_drops_the_badges_that_do_not_change_a_decision(
    db_env, triage_env, triage_db, client
):
    """Only `no answer` and `conflict` earn a slot in the rail: both mean this package needs a
    DIFFERENT decision. A rating and a corroboration verdict are read on the card, on the one
    package in front of the reviewer.

    Driven on a row that DOES render badges. On a row with none, "the rating is not in the
    rail" is true because nothing is, and adding the rating back changes nothing the test can
    see — measured: the mutation that puts it back survived this test until the fixture grew
    a row with a badge row on it.
    """
    await login(client)
    await seed(triage_db, {"com.example.flagged": 1})
    await corroborate(triage_db, "com.example.flagged", status="corroborated", sources=[])
    await flag_everything(triage_db, "com.example.flagged")

    body = await screen(client, "?view=parked")
    rail = body.split('<aside class="triage-rail">', 1)[1].split("</aside>", 1)[0]

    assert badge_rows(rail) == [["no answer", "conflict"]]
    for dropped in ("advanced", "corroborated", "medium"):
        assert dropped not in rail.lower(), dropped


async def test_every_badge_is_explained_where_a_keyboard_can_reach_it(
    db_env, triage_env, triage_db, client
):
    """A `title` on a non-focusable `<span>` is a pointer affordance, and this screen exists to
    be driven by keyboard. One `<summary>` is one tab stop for the whole set."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert '<summary class="label" style="cursor:pointer">what the badges mean</summary>' in body
    legend = body.split('id="triage-badge-help"', 1)[1].split("</details>", 1)[0]
    # Spelled out rather than only looped over the constant: a test that reads its expected
    # values off the same table the template renders proves the table is rendered and never
    # that the table says anything.
    assert "how many sources back the description" in legend
    assert "deepseek could not answer" in legend
    for badge, meaning in triage_view.BADGE_MEANINGS:
        assert f">{badge}</span>" in legend, badge
        assert meaning in legend, meaning


async def test_the_words_this_project_invented_are_gone_from_the_screen(
    db_env, triage_env, triage_db, client
):
    """`corroborated` was the one the user named. The rest went with it: every string in the
    round's wording table is a display string, and the value under it is untouched — the view
    is still `deferred`, the action is still `defer`, the status column still says
    `corroborated`."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await corroborate(
        triage_db,
        "com.example.one",
        status="corroborated",
        sources=[{"url": "https://example.test/a", "title": "A page"}],
    )

    body = await screen(client)

    for retired in (
        "not corroborated",
        "deciding without",
        "the model's proposal",
        "the removal floor",
        "what the firmware declared",
        "nearest entries upstream",
        "decisions on this package",
        "the model could not answer this one",
        "pinned at",
    ):
        assert retired not in body, retired
    assert ">1 source</span>" in body
    # …and the values under the labels did not move.
    assert 'href="/triage?view=deferred"' in body
    assert 'value="defer"' in body


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("uncorroborated", "no sources"),
        ("search_failed", "search failed"),
        ("judge_failed", "check failed"),
    ],
)
async def test_a_corroboration_status_reads_as_its_label_and_never_as_its_value(
    db_env, triage_env, triage_db, client, status, label
):
    """A map rather than `.replace("_", " ")`: a substitution turns a status nobody wrote a
    label for into a sentence that looks authored, and it cannot spell `corroborated` as the
    count the reviewer actually decides on."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await corroborate(triage_db, "com.example.one", status=status, sources=[])

    body = await screen(client)

    assert f">{label}</span>" in body
    assert status not in body


# --- icons ---------------------------------------------------------------------------------


@pytest.fixture
def has_icons(monkeypatch):
    """Every row and card answers "this package has an icon".

    `package_facts.icon_mime` is another lane's column and does not exist in this checkout, so
    the store cannot produce a true `has_icon` here. What this fixture pins is the half that
    is this screen's: given the flag, the markup. The flag's own derivation is pinned in the
    store suite against the column being absent.
    """
    real_board = triagestore.load_board
    real_candidate = triagestore.load_candidate

    async def board(session, *, view):
        rows, counts = await real_board(session, view=view)
        return tuple(replace(row, has_icon=True) for row in rows), counts

    async def candidate(session, package, *, upstream=None):
        return replace(await real_candidate(session, package, upstream=upstream), has_icon=True)

    monkeypatch.setattr(triagestore, "load_board", board)
    monkeypatch.setattr(triagestore, "load_candidate", candidate)


def test_the_stylesheet_covers_every_class_the_icon_macro_can_emit():
    """The macro picks the class names and this lane owns the stylesheet, so the two halves
    are checked against each other here rather than by looking at a screen.

    A colour index with no class renders an UNSTYLED chip instead of raising, which ships
    invisibly: raising `MONOGRAM_COLOURS` past what `app.css` covers is the shape that does
    it, and it fails here instead.
    """
    css = (web.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    for name in (".pkg-icon", ".pkg-icon-sm", ".monogram", ".monogram-sm"):
        assert re.search(rf"\{name}\b[^{{]*{{", css), f"{name} is not defined in app.css"
    colours = {int(index) for index in re.findall(r"\.monogram-c(\d+)\s*{", css)}
    assert colours == set(range(MONOGRAM_COLOURS))


async def test_a_package_with_no_icon_gets_a_monogram_and_never_a_request(
    db_env, triage_env, triage_db, client
):
    """~62% of packages declare no icon at all, so the fallback is the ordinary case. A chip
    that fetched anything would be 62% of the corpus fetching a 404."""
    await login(client)
    await seed(triage_db, {"com.example.notes": 1})

    body = await screen(client)

    letters, colour = monogram_for("com.example.notes")
    assert "/icons/" not in body
    assert f'class="monogram monogram-c{colour} monogram-sm"' in body
    assert f'class="monogram monogram-c{colour}"' in body
    assert f">{letters}</span>" in body


async def test_a_package_with_an_icon_renders_the_image_at_both_sizes(
    db_env, triage_env, triage_db, client, has_icons
):
    await login(client)
    await seed(triage_db, {"com.example.notes": 1})

    body = await screen(client)

    assert 'class="pkg-icon pkg-icon-sm"' in body
    assert 'class="pkg-icon"\n         src="/icons/com.example.notes"' in body
    assert body.count('src="/icons/com.example.notes"') == 2
    assert "monogram" not in body


async def test_an_icon_url_encodes_the_package_name_rather_than_escaping_it(
    db_env, triage_env, triage_db, client, has_icons
):
    """The icon src is a PATH SEGMENT, which is a stricter rule than the queue link beside it.

    Two encodings are wrong here and they fail differently. Autoescape turns a literal `&` into
    `&amp;`, which the browser decodes straight back into a separator, so the card shows another
    package's icon and reads as a correct answer. Jinja's `urlencode` fixes that and keeps `/`
    safe, which is right for the `?package=` link and wrong here: a slash spells a segment
    boundary the record never had, and a browser resolves `..` before the request is sent, so
    the route is asked for a path nobody named. `web.url_segment` is the one that holds.
    """
    await login(client)
    await seed(triage_db, {"com.example.notes&x=1": 1, "../../secret": 1})

    body = await screen(client)

    assert "/icons/..%2F..%2Fsecret" in body
    assert "/icons/../.." not in body
    assert "/icons/com.example.notes%26x%3D1" in body
    assert "/icons/com.example.notes&amp;x=1" not in body


# --- focus after a swap ----------------------------------------------------------------------


async def test_exactly_one_element_asks_for_focus_after_a_swap(
    db_env, triage_env, triage_db, client
):
    """A swap replaces the board and focus falls back to `<body>`, so the next Tab restarts at
    the top of the document. Two candidates for it is the same bug wearing an answer: whichever
    the script finds first wins, and which one that is depends on document order."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await act(client, package="com.example.one", action="defer", view="queue")

    assert body.count("data-swap-focus") == 1
    assert '<div class="triage-decide" tabindex="-1" data-swap-focus>' in body


async def test_a_rejection_with_no_reason_puts_focus_on_the_field_it_is_about(
    db_env, triage_env, triage_db, client
):
    """The message names the field; focus lands there. Anywhere else and the reviewer answers a
    refusal by hunting for the input it refused."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await act(client, package="com.example.one", action="reject", reason=" ", view="queue")

    assert body.count("data-swap-focus") == 1
    assert 'id="triage-reject-reason"' in body
    reason_field = body.split('id="triage-reject-reason"', 1)[1].split(">", 1)[0]
    assert "data-swap-focus" in reason_field


async def test_a_refused_edit_puts_focus_back_in_the_form_that_still_holds_the_typing(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")

    resp = await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "removal": "Recommended", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.text.count("data-swap-focus") == 1
    textarea = resp.text.split('id="edit-description"', 1)[1].split(">", 1)[0]
    assert "data-swap-focus" in textarea
    assert '<details id="triage-edit" open' in resp.text


# --- the description a reviewer cannot write --------------------------------------------------


async def test_the_form_can_declare_a_description_unknown(db_env, triage_env, triage_db, client):
    """`_check_description` refuses a short description and tells the writer to declare it in
    `unknown_fields`. That message is shared with the model path deliberately, so the fix is
    the control it names rather than a second spelling of the rule for humans."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    resp = await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": "",
            "unknown_description": "1",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200, resp.text[:400]
    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == "unknown"
    assert row.unknown_fields == ["description"]
    assert row.provenance["description"] == "human:triage"


async def test_the_box_and_the_text_disagreeing_is_refused_rather_than_resolved(
    db_env, triage_env, triage_db, client
):
    """The card renders the box PRE-TICKED on exactly the rows a reviewer opens in order to
    replace the model's `unknown` with real words, so "the box wins" silently threw the
    sentence they had just typed away and answered `nothing changed` — true about the row and
    false about what they did. Refusing is the only resolution that cannot lose an answer.
    """
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    typed = "Notes application. Removing it loses locally stored notes."

    resp = await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": typed,
            "unknown_description": "1",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200, resp.text[:400]
    assert "pick one" in resp.text
    # Their words survive the refusal, in the box, with the tick dropped so saving again
    # keeps them rather than refusing again.
    assert typed in resp.text
    checkbox = resp.text.split('id="edit-unknown"', 1)[1].split(">", 1)[0]
    assert "checked" not in checkbox
    assert '<details id="triage-edit" open' in resp.text
    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description is not None and row.description.startswith("Vendor application")
    assert row.unknown_fields == []


async def test_a_declared_unknown_description_comes_back_with_the_control_ticked(
    db_env, triage_env, triage_db, client
):
    """Otherwise the next reviewer reads a row whose description says `unknown` with a box that
    says it does not, and un-ticking is the only way back."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "unknown_description": "1", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    body = await screen(client)

    checkbox = body.split('id="edit-unknown"', 1)[1].split(">", 1)[0]
    assert "checked" in checkbox


async def test_writing_a_real_description_takes_the_unknown_declaration_back_off(
    db_env, triage_env, triage_db, client
):
    """A row carrying a written description while still declaring it unknown asserts two
    contradictory things."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "unknown_description": "1", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": "Notes application. Removing it loses locally stored notes.",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.unknown_fields == []
    assert row.description.startswith("Notes application")


def test_every_list_the_screen_offers_has_a_display_label():
    """The rail renders `view_labels[name]` for every name in `VIEWS`, under `StrictUndefined`.
    A view added to one table and not the other is not a missing label, it is a 500 on the
    whole screen."""
    assert set(triage_view.VIEW_LABELS) == set(triagestore.VIEWS)


def key_owner(body: str, key: str) -> tuple[str, str]:
    """The (package, action) the key handler would submit for `key`.

    The handler takes the FIRST `[data-key]` matching, so what a key does is decided by the
    form it happens to sit in — which is a property of the whole document and cannot be read
    off any one control. Asserting a control merely EXISTS is what let `u` reach the wrong
    package: the undo button existed, and the key never reached it.
    """
    controls = re.findall(rf'data-key="{re.escape(key)}"', body)
    assert len(controls) == 1, f"{len(controls)} controls answer to {key!r}"
    form = next(
        block
        for block in re.findall(r"<form\b.*?</form>", body, re.S)
        if f'data-key="{key}"' in block
    )
    package = re.search(r'name="package" value="([^"]*)"', form)
    action = re.search(r'name="action" value="([^"]*)"', form)
    assert package and action, "the control answering that key is not in a decide form"
    return package.group(1), action.group(1)


async def test_the_undo_key_undoes_the_verdict_and_never_touches_the_next_package(
    db_env, triage_env, triage_db, client
):
    """Driven in a browser first: approve, then press `u` — the key the `keys` legend
    advertises as reopen and the obvious one to reach for with an `undo` banner on screen.

    It reopened the package the board had just ADVANCED to, writing a `human:` decision
    against a package nobody had read, and spent the undo doing it. The undo button carried no
    key, so the only `[data-key="u"]` in the document was the card's own reopen button, which
    by then belonged to a different package. Asserting the undo button renders would not have
    caught it, so this asserts what the KEY resolves to.
    """
    await login(client)
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 2})

    body = await act(client, package="com.example.two", action="approve", view="queue")

    assert "com.example.one" in body, "the board should have advanced to the next package"
    assert key_owner(body, "u") == ("com.example.two", "reopen")


async def test_the_undo_key_goes_back_to_the_card_once_the_banner_is_gone(
    db_env, triage_env, triage_db, client
):
    """The other half of the same rule: one control per key, and with no undo on screen the
    card's own reopen button is the only thing `u` can sensibly mean."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert key_owner(body, "u") == ("com.example.one", "reopen")


async def test_a_refusal_is_reachable_from_the_field_it_is_about(
    db_env, triage_env, triage_db, client
):
    """Focus alone puts a screen-reader user IN the field with no statement of the problem,
    and a reader who tabs back to it later never had one. WCAG 3.3.1 and 4.1.2 want the
    description reachable from the control, which costs two attributes and no tab stop."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await act(client, package="com.example.one", action="reject", reason=" ", view="queue")

    assert 'id="triage-error"' in body
    field = body.split('id="triage-reject-reason"', 1)[1].split(">", 1)[0]
    assert 'aria-describedby="triage-error"' in field
    assert 'aria-invalid="true"' in field


async def test_a_refused_edit_is_reachable_from_the_box_it_is_about(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")

    resp = await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "removal": "Recommended", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert 'id="triage-error"' in resp.text
    field = resp.text.split('id="edit-description"', 1)[1].split(">", 1)[0]
    assert 'aria-describedby="triage-error"' in field
    assert 'aria-invalid="true"' in field


def test_the_icon_mime_fallback_is_still_needed():
    """`triagestore._ICON_MIME` falls back to a SQL NULL because `package_facts.icon_mime`
    does not exist in this checkout. `getattr` with a default never fails, so a column that
    lands under another name would leave every icon silently unrendered with the suite green.

    This goes red the moment the fallback becomes wrong, which is the only moment anyone wants
    to hear about it: delete the fallback, the `getattr`, and this test together.
    """
    assert not hasattr(PackageFact, "icon_mime"), (
        "package_facts.icon_mime exists now — drop triagestore._ICON_MIME's null() fallback"
    )


async def test_the_tab_strip_reads_in_the_round_s_words_and_not_the_database_s(
    db_env, triage_env, triage_db, client
):
    """The rendered strip, not the dict behind it. `VIEW_LABELS` having the right KEYS is a
    different property — it guards a `StrictUndefined` 500 — and it stays green while every
    value reverts to the column spelling it labels.

    `skipped` and `no answer` are two of the wording table's rows and they are what a reviewer
    reads; `deferred` and `parked` remain the values in the url, in `VIEWS` and in the
    database, which the assertions under this one pin.
    """
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    tabs = [text.strip() for text in re.findall(r'class="tab-box[^"]*"[^>]*>([^<]*)<', body)]
    assert [tab.rsplit(" ", 1)[0] for tab in tabs] == ["queue", "skipped", "decided", "no answer"]
    assert 'href="/triage?view=deferred"' in body
    assert 'href="/triage?view=parked"' in body


# --- the chip has to be readable, and that is arithmetic rather than an opinion -------------


def _srgb_luminance(colour: str) -> float:
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    first, second = _srgb_luminance(foreground), _srgb_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _mix(colour: str, over: str, fraction: float) -> str:
    """`color-mix(in srgb, colour fraction%, transparent)` composited on an opaque surface.

    A partly transparent fill has no contrast of its own: what the letters actually sit on is
    this blend, which is why measuring the token alone says nothing.
    """
    pairs = [(int(colour[i : i + 2], 16), int(over[i : i + 2], 16)) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c * fraction + o * (1 - fraction)):02X}" for c, o in pairs)


def _tokens(css: str, theme: str) -> dict[str, str]:
    """Every colour token one theme resolves to, read block by block.

    Not a split on the light selector: `ui.css` defines the palette in several blocks and the
    categorical ramp comes AFTER the light surfaces, so one split files the dark ramp under
    light and the dark theme reads as having no tints at all.
    """
    tokens: dict[str, str] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if "--" not in body:
            continue
        if ('data-theme="light"' in selector) != (theme == "light"):
            continue
        tokens.update(re.findall(r"(--[a-z-]+):\s*(#[0-9A-Fa-f]{6})", body))
    return tokens


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_monogram_tint_clears_the_contrast_floor(theme):
    """WCAG 1.4.3 wants 4.5:1, and 12px and 16px at weight 550 are both normal-size text.

    Recomputed from the two stylesheets rather than hard-coded, so it follows the fill
    percentage and the colour token if either moves. It is the assertion that was missing when
    this shipped putting the LETTERS on the categorical tint: composited over the real surface
    that measured 2.08:1 for `--cat-pink` on light, against a floor of 4.5.
    """
    ui = (web.STATIC_DIR / "ui.css").read_text(encoding="utf-8")
    app = (web.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    tokens = _tokens(ui, theme)

    if theme == "light":
        block = app.split(':root[data-theme="light"] .monogram', 1)[1]
        ink = tokens[re.search(r"color:\s*var\((--[a-z-]+)\)", block).group(1)]
    else:
        block = app.split(".monogram,\n.monogram-sm {", 1)[1]
        ink = None  # the tint itself is the ink on dark
    fraction = int(re.search(r"tint, var\(--accent\)\) (\d+)%, transparent", block).group(1)) / 100

    tints = [
        tokens[f"--cat-{name}"]
        for name in re.findall(
            r"\.monogram-c\d+ \{ --monogram-tint: var\((?:--cat-)?([a-z]+)\)", app
        )
    ]
    assert len(tints) == 8, f"read {len(tints)} tints out of app.css"
    for surface in (tokens["--bg-raised"], tokens["--bg"], tokens["--bg-sunken"]):
        for tint in tints:
            fill = _mix(tint, surface, fraction)
            ratio = _contrast(ink or tint, fill)
            assert ratio >= 4.5, f"{theme}: {tint} on {surface} reads {ratio:.2f}:1"


async def test_a_refused_edit_gives_every_answer_back(db_env, triage_env, triage_db, client):
    """ux-patterns §7.1: keep every answer the user gave, passing and failing both.

    The expensive case is the one that is not about the field that failed — three careful
    sentences typed ALONGSIDE a below-floor rating. The refusal re-read the candidate from the
    store, so the box came back holding the STORED description, which is worse than an empty
    one: it looks legitimately filled in and the loss is only noticed after saving.
    """
    await login(client)
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")
    typed = "Notes application. Removing it loses every note stored only on the device."

    resp = await client.post(
        "/triage/edit",
        data={
            "package": "com.example.one",
            "description": typed,
            "removal": "Recommended",
            "confidence": "high",
            "view": "queue",
        },
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200, resp.text[:400]
    assert "computed floor" in resp.text
    assert typed in resp.text, "the description they typed was thrown away by the refusal"
    removal = resp.text.split('id="edit-removal"', 1)[1].split("</select>", 1)[0]
    assert '<option value="Recommended" selected>' in removal
    confidence = resp.text.split('id="edit-confidence"', 1)[1].split("</select>", 1)[0]
    assert '<option value="high" selected>' in confidence


async def test_a_too_short_description_names_the_control_the_form_actually_has(
    db_env, triage_env, triage_db, client
):
    """`classify`'s wording is shared with the model path deliberately, so it says to declare
    the field in `unknown_fields` — which is not a field on this form, while the control that
    performs it sits right under the box. Re-worded for the screen the way `ReasonRequired`
    already is; the rule itself stays where both writers meet it."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    resp = await client.post(
        "/triage/edit",
        data={"package": "com.example.one", "description": "short", "view": "queue"},
        headers={"accept": "text/html", "HX-Request": "true"},
    )

    assert resp.status_code == 200, resp.text[:400]
    # The whole sentence, not just the control's name: dropping "write one" left the message
    # ungrammatical and every looser assertion green, measured.
    assert (
        "a description needs 20 characters. write one, or tick &#34;can&#39;t describe it&#34;."
        in resp.text
    )
    assert "unknown_fields" not in resp.text
    assert "measured against the live upstream list" not in resp.text
    assert "short" in resp.text, "their text has to survive the refusal too"


async def test_a_cleared_view_points_at_a_list_that_still_has_work(
    db_env, triage_env, triage_db, client
):
    """An empty state offers the action that resolves it. `back to the queue` rendered inside
    the queue view pointed at the screen the reviewer was already on."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})
    await act(client, package="com.example.one", action="defer", view="queue")

    body = await screen(client)

    assert "queue is clear" in body
    assert 'href="/triage?view=deferred"' in body
    assert "go to skipped" in body
    assert "back to the queue" not in body


async def test_a_transport_failure_says_the_verdict_did_not_land(
    db_env, triage_env, triage_db, client
):
    """Every `_render` in the view sits outside its own try blocks, htmx swaps neither a 5xx
    nor a dropped connection, and the reviewer experiences both as a key that stopped working.
    The message lives OUTSIDE the swapped board, because the swap that failed is the swap that
    would have rendered it."""
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    shell = body.split('id="triage-board"', 1)[0]
    assert 'id="triage-transport-error"' in shell
    assert 'role="alert"' in shell.split('id="triage-transport-error"', 1)[1].split(">", 1)[0]
    for handler in ("htmx:responseError", "htmx:sendError"):
        assert handler in body, handler
