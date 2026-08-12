"""The triage queue, the append-only decision log, and the human edit path.

Three properties here are the task's own verify line rather than incidental:

- the queue is ranked by how many devices shipped the package, because that is what
  `package_facts.device_count` is stored for;
- a rejection persists with its reason and does not come back in the queue;
- a human edit lands as `human:` provenance, which is the machinery that makes it survive a
  re-classification. Asserting the edited VALUE alone would pass with the provenance missing,
  and the next model run would then quietly overwrite it.

The classification rows come from `classifystore.store_classification` and the facts from
`factstore.store_device_facts`, so no fixture here can drift from what production writes —
`device_count` in particular is computed by the merge rather than typed in.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from uadclaw import triagestore
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import BelowFloorError, store_classification
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import PackageAnalysis, PackageClassification, PackageTriageDecision
from uadclaw.upstream import UpstreamEntry, UpstreamList

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
MODEL = "deepseek-v4-flash"
BUNDLE = "b" * 64


@pytest.fixture
async def triage_db(db_session_factory):
    """`conftest`'s truncation list predates this table, so the log is cleared here.

    Without it a decision written by one test decides a package for the next one on the same
    xdist worker, and the queue test that reads as green is the one that was handed an empty
    queue for the wrong reason.
    """
    async with db_session_factory() as session, session.begin():
        await session.execute(text("TRUNCATE TABLE package_triage_decision RESTART IDENTITY"))
    return db_session_factory


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


def proposal(package: str, *, removal: Removal = Removal.ADVANCED, bundle: str = BUNDLE):
    return Classification(
        package=package,
        bundle_sha256=bundle,
        description=f"Vendor application {package}. Removing it loses its local data.",
        list=UadList.MISC,
        removal=removal,
        confidence=Confidence.MEDIUM,
        unknown_fields=(),
        reasoning_brief="No privileged surface.",
        provenance={
            "description": f"llm:{MODEL}",
            "list": f"llm:{MODEL}",
            "removal": f"llm:{MODEL}",
            "confidence": f"llm:{MODEL}",
        },
    )


async def seed(session_factory, packages: dict[str, int], *, floor: str = "Recommended") -> None:
    """One classification per package, on as many devices as the value says."""
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
                    floor_rule="default",
                    floor_reasons=[{"rule": "default", "floor": floor, "detail": "no rule fired"}],
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


async def decide(session_factory, package, action, *, reason=None, at=NOW, bundle=BUNDLE) -> None:
    async with session_factory() as session, session.begin():
        await triagestore.record_decision(
            session,
            package=package,
            bundle_sha256=bundle,
            action=action,
            reason=reason,
            at=at,
        )


async def queue(session_factory, view: str = "queue") -> list[str]:
    async with session_factory() as session:
        return [row.package for row in await triagestore.load_queue(session, view=view)]


# --- ranking --------------------------------------------------------------------------------


async def test_the_queue_ranks_by_device_count_descending(db_env, triage_db):
    """`package_facts.device_count` is stored precisely to be this ranking signal. The
    packages are seeded in an order that is neither the answer nor its reverse, so a queue
    that simply preserved insertion order would fail this."""
    await seed(triage_db, {"com.example.two": 2, "com.example.three": 3, "com.example.one": 1})

    assert await queue(triage_db) == [
        "com.example.three",
        "com.example.two",
        "com.example.one",
    ]


async def test_the_queue_breaks_a_device_count_tie_by_package_name(db_env, triage_db):
    """Without the tie-break the order is whatever Postgres returns, so a reviewer walking
    the queue twice walks it differently and cannot tell where they were."""
    await seed(triage_db, {"com.example.zeta": 2, "com.example.alpha": 2})

    assert await queue(triage_db) == ["com.example.alpha", "com.example.zeta"]


async def test_a_package_with_no_facts_row_sorts_last_rather_than_first(db_env, triage_db):
    """A NULL device_count sorted descending lands FIRST by default in Postgres, which puts
    the least evidence at the top of a queue ranked by evidence."""
    await seed(triage_db, {"com.example.seen": 1})
    async with triage_db() as session, session.begin():
        session.add(PackageAnalysis(package="com.example.unseen", updated_at=NOW, floor="Advanced"))
        await store_classification(
            session,
            proposal("com.example.unseen"),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    assert await queue(triage_db) == ["com.example.seen", "com.example.unseen"]


# --- what reaches the queue -------------------------------------------------------------------


async def test_an_uncorroborated_candidate_reaches_the_queue(db_env, triage_db):
    """Corroboration lands at 13.6% over the real corpus and 0 of 12 for `com.android.*`, so
    a queue that only carried corroborated candidates would hide most of it."""
    await seed(triage_db, {"com.android.example": 1})

    assert await queue(triage_db) == ["com.android.example"]


async def test_a_conflicted_package_reaches_the_queue(db_env, triage_db):
    """134 of 147 shared packages carry `has_conflict`, all of it measured as ordinary key
    rotation. The flag is rendered, never filtered on."""
    await seed(triage_db, {"com.example.rotated": 1})
    async with triage_db() as session, session.begin():
        # A second device with a different signing certificate: the merge flags it.
        await store_device_facts(
            session,
            device_key="pixel:other",
            build="bp1a.260505.001",
            facts=[make_facts("com.example.rotated", cert_issuer="Organization: Other Corp")],
            observed_at=NOW,
        )

    rows = {row.package: row for row in await _rows(triage_db)}
    assert rows["com.example.rotated"].has_conflict is True
    assert await queue(triage_db) == ["com.example.rotated"]


async def test_a_parked_package_is_not_in_the_queue_but_is_reachable(db_env, triage_db):
    """A parked row carries no proposal at all, so there is nothing to approve — and it is
    still a fact about the corpus a reviewer has to be able to reach."""
    await seed(triage_db, {"com.example.answered": 1})
    async with triage_db() as session, session.begin():
        session.add(PackageAnalysis(package="com.example.parked", updated_at=NOW, floor="Advanced"))
    async with triage_db() as session, session.begin():
        from uadclaw.classifystore import park_package

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

    assert await queue(triage_db) == ["com.example.answered"]
    assert await queue(triage_db, "parked") == ["com.example.parked"]


# --- the decision log ---------------------------------------------------------------------------


async def test_an_approved_package_leaves_the_queue(db_env, triage_db):
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 2})

    await decide(triage_db, "com.example.two", "approve")

    assert await queue(triage_db) == ["com.example.one"]
    assert await queue(triage_db, "decided") == ["com.example.two"]


async def test_a_rejection_persists_with_its_reason_and_does_not_come_back(db_env, triage_db):
    """The task's own verify line. Both halves matter: a rejection that vanishes from the
    queue but keeps no reason is exactly the row prompt work cannot be done from."""
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "reject", reason="description names a vendor")

    assert await queue(triage_db) == []
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        rows = result.scalars().all()
    assert [(row.action, row.reason) for row in rows] == [("reject", "description names a vendor")]


@pytest.mark.parametrize("reason", [None, "", "   "])
async def test_a_rejection_without_a_reason_is_refused(db_env, triage_db, reason):
    await seed(triage_db, {"com.example.one": 1})

    with pytest.raises(triagestore.TriageError, match="reason"):
        await decide(triage_db, "com.example.one", "reject", reason=reason)

    assert await queue(triage_db) == ["com.example.one"]


async def test_the_database_refuses_a_reasonless_rejection_too(db_env, triage_db):
    """The code path above is the one a form hits; this is the backstop under it. A future
    writer that reaches the table directly meets the same rule."""
    await seed(triage_db, {"com.example.one": 1})

    with pytest.raises(Exception, match="ck_package_triage_decision_reject_reason"):
        async with triage_db() as session, session.begin():
            session.add(
                PackageTriageDecision(
                    package="com.example.one",
                    bundle_sha256=BUNDLE,
                    action="reject",
                    reason=None,
                    decided_at=NOW,
                )
            )


async def test_a_deferred_package_leaves_the_queue_and_stays_reachable(db_env, triage_db):
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "defer")

    assert await queue(triage_db) == []
    assert await queue(triage_db, "deferred") == ["com.example.one"]


async def test_the_log_is_append_only_and_the_latest_decision_is_the_current_one(db_env, triage_db):
    """Deciding twice supersedes by being later, never by overwriting. The history is the
    only measurement of whether the funnel is improving, so it has to survive a re-decision.
    """
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "reject", reason="too vague", at=NOW)
    await decide(triage_db, "com.example.one", "reopen", at=NOW + timedelta(minutes=1))
    await decide(triage_db, "com.example.one", "approve", at=NOW + timedelta(minutes=2))

    async with triage_db() as session:
        result = await session.execute(
            select(PackageTriageDecision).order_by(PackageTriageDecision.decided_at)
        )
        actions = [row.action for row in result.scalars()]
    assert actions == ["reject", "reopen", "approve"]
    assert await queue(triage_db, "decided") == ["com.example.one"]


async def test_a_reopened_package_is_back_in_the_queue(db_env, triage_db):
    """The way out of a mistaken verdict. Without it a wrong reject is permanent, since the
    log is append-only and nothing else returns a package to the queue."""
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "reject", reason="wrong list", at=NOW)
    await decide(triage_db, "com.example.one", "reopen", at=NOW + timedelta(minutes=1))

    assert await queue(triage_db) == ["com.example.one"]


async def test_a_re_classification_returns_a_decided_package_to_the_queue(db_env, triage_db):
    """A decision is about ONE proposal. New evidence means a new bundle hash, so the old
    verdict stops being current on its own and the history survives untouched."""
    await seed(triage_db, {"com.example.one": 1})
    await decide(triage_db, "com.example.one", "approve")
    assert await queue(triage_db) == []

    async with triage_db() as session, session.begin():
        await store_classification(
            session,
            proposal("com.example.one", bundle="c" * 64),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    assert await queue(triage_db) == ["com.example.one"]
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert len(result.scalars().all()) == 1


async def test_an_unknown_action_is_refused(db_env, triage_db):
    await seed(triage_db, {"com.example.one": 1})

    with pytest.raises(triagestore.TriageError, match="delete"):
        await decide(triage_db, "com.example.one", "delete")


# --- the human edit ------------------------------------------------------------------------------


async def test_an_edit_writes_the_value_and_its_human_provenance(db_env, triage_db):
    """Both halves. The provenance is what `classifystore._preserved` reads to keep the edit
    across a re-run, so a value written without it is an edit the next model run erases."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session, session.begin():
        changed = await triagestore.apply_edit(
            session,
            package="com.example.one",
            edits={"description": "Notes application. Removing it loses locally stored notes."},
            at=NOW,
        )

    assert set(changed) == {"description"}
    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == "Notes application. Removing it loses locally stored notes."
    assert row.provenance["description"] == "human:triage"
    assert row.provenance["list"] == f"llm:{MODEL}"


async def test_an_edit_that_changes_nothing_writes_no_decision(db_env, triage_db):
    """An edit form submitted unchanged is a no-op, not a log entry. A log full of empty
    edits is a log nobody reads."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session, session.begin():
        changed = await triagestore.apply_edit(
            session,
            package="com.example.one",
            edits={"description": proposal("com.example.one").description},
            at=NOW,
        )

    assert changed == {}
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert result.scalars().all() == []


async def test_an_edit_records_what_it_replaced(db_env, triage_db):
    """The row carries the value that won; the log has to carry the one it replaced, or the
    only measurement of what the model got wrong is gone."""
    await seed(triage_db, {"com.example.one": 1})
    before = proposal("com.example.one").description

    async with triage_db() as session, session.begin():
        await triagestore.apply_edit(
            session,
            package="com.example.one",
            edits={"description": "Notes application. Removing it loses locally stored notes."},
            at=NOW,
        )

    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        row = result.scalars().one()
    assert row.action == "edit"
    assert row.edited_fields["description"]["from"] == before


async def test_an_edited_package_stays_in_the_queue(db_env, triage_db):
    """An edit is a revision, not a verdict: the reviewer still has to approve it."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session, session.begin():
        await triagestore.apply_edit(
            session,
            package="com.example.one",
            edits={"removal": "Expert"},
            at=NOW,
        )

    assert await queue(triage_db) == ["com.example.one"]


async def test_an_edit_below_the_floor_is_refused_by_the_store(db_env, triage_db):
    """The reason part A landed first. This path writes `removal` with `human:` provenance
    and has no `validate_response` anywhere in it."""
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")

    with pytest.raises(BelowFloorError):
        async with triage_db() as session, session.begin():
            await triagestore.apply_edit(
                session, package="com.example.one", edits={"removal": "Recommended"}, at=NOW
            )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.removal == "Advanced"
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert result.scalars().all() == [], "a refused edit must log nothing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("removal", "Extremely Unsafe"),
        ("list", "Nonsense"),
        ("confidence", "certain"),
        ("description", "short"),
        ("description", "   leading space is trimmed by upstream nobody   "),
        ("dependencies", "com.example.other"),
    ],
)
async def test_an_edit_refuses_a_value_the_pipeline_could_not_have_emitted(
    db_env, triage_db, field, value
):
    """A human edit ships to the same file a model answer does, so it meets the same shape
    rules — and `dependencies` is never either writer's to set."""
    await seed(triage_db, {"com.example.one": 1})

    with pytest.raises((triagestore.TriageError, ValueError)):
        async with triage_db() as session, session.begin():
            await triagestore.apply_edit(
                session, package="com.example.one", edits={field: value}, at=NOW
            )


async def test_an_edit_to_a_package_with_no_proposal_is_refused(db_env, triage_db):
    with pytest.raises(triagestore.TriageError, match="com.example.absent"):
        async with triage_db() as session, session.begin():
            await triagestore.apply_edit(
                session,
                package="com.example.absent",
                edits={"removal": "Expert"},
                at=NOW,
            )


# --- the card ---------------------------------------------------------------------------------


UPSTREAM = UpstreamList(
    path="/data/uad_lists.json",
    sha256="d" * 64,
    obtained_at=NOW,
    loaded_at=NOW,
    entry_count=2,
    packages=frozenset({"com.example.sibling", "com.other.stranger"}),
    entries={
        "com.example.sibling": UpstreamEntry(
            package="com.example.sibling",
            list="Oem",
            removal="Advanced",
            description="A sibling entry upstream already carries.",
        ),
        "com.other.stranger": UpstreamEntry(
            package="com.other.stranger",
            list="Oem",
            removal="Advanced",
            description="An entry sharing no namespace with anything here.",
        ),
    },
)


async def test_the_card_carries_the_floor_and_the_rule_that_set_it(db_env, triage_db):
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")

    async with triage_db() as session:
        card = await triagestore.load_candidate(session, "com.example.one", upstream=UPSTREAM)

    assert card.floor == "Advanced"
    assert card.floor_rule == "default"
    assert card.floor_reasons[0][0] == "default"


async def test_the_card_carries_the_nearest_upstream_entries_and_not_the_strangers(
    db_env, triage_db
):
    """Nearest is the longest shared dotted prefix, which is `bundle.nearest_entries` — the
    same rule the model was given its style anchors under, so the reviewer compares against
    what the model compared against."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session:
        card = await triagestore.load_candidate(session, "com.example.one", upstream=UPSTREAM)

    assert [entry.package for entry in card.anchors] == ["com.example.sibling"]


async def test_the_card_names_the_evidence_that_is_missing(db_env, triage_db):
    """The partial state. A candidate with no corroboration is the ordinary case rather than
    a broken one, so it is named rather than hidden."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session:
        card = await triagestore.load_candidate(session, "com.example.one", upstream=UPSTREAM)

    assert "corroboration" in card.missing


async def test_the_card_refuses_a_package_with_no_proposal(db_env, triage_db):
    with pytest.raises(triagestore.TriageError, match="com.example.absent"):
        async with triage_db() as session:
            await triagestore.load_candidate(session, "com.example.absent", upstream=UPSTREAM)


async def _rows(session_factory):
    async with session_factory() as session:
        return await triagestore.load_queue(session, view="queue")
