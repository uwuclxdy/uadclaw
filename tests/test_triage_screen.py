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
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from uadclaw import triagestore
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import park_package, store_classification
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import (
    PackageAnalysis,
    PackageClassification,
    PackageCorroboration,
    PackageTriageDecision,
)
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

    assert "privileged (priv-app)" in body
    assert "floor advanced" in body


async def test_the_card_shows_the_evidence_and_the_nearest_upstream_entry(
    db_env, triage_env, triage_db, client
):
    await login(client)
    await seed(triage_db, {"com.example.one": 1})

    body = await screen(client)

    assert "what the firmware declared" in body
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
    assert "not corroborated" in body


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

    assert "deciding without" in body
    assert "corroboration" in body


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
    assert "the model could not answer this one" in parked


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
