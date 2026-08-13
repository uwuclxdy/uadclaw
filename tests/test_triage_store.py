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
from uadclaw.classify import UNKNOWN, Classification, Confidence, UadList
from uadclaw.classifystore import (
    BelowFloorError,
    park_package,
    store_classification,
    store_human_edit,
)
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import (
    BranchEmission,
    BranchEmissionPackage,
    PackageAnalysis,
    PackageClassification,
    PackageTriageDecision,
)
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


async def _seed_emission(
    session_factory,
    packages: list[str],
    *,
    branch: str = "uadclaw/pixel-000000000001",
    committed: bool = True,
) -> None:
    """A completed emission carrying `packages`, seeded straight into the run tables — the
    state the shipped view has to read. `committed=False` seeds a pending one (intent written,
    commit not landed), which must NOT read as shipped. The git half that normally produces
    these rows is exercised by the emission suite, not here."""
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
            package_count=len(packages),
            pr_body="## pixel: 1 package addition(s)\n",
            created_at=NOW,
            commit_oid="e" * 40 if committed else None,
            committed_at=NOW if committed else None,
            reconciled=False,
        )
        session.add(row)
        await session.flush()
        for package in packages:
            session.add(
                BranchEmissionPackage(
                    emission_id=row.id,
                    package=package,
                    bundle_sha256=BUNDLE,
                    uad_list="Misc",
                    removal="Advanced",
                    floor="Recommended",
                )
            )


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
    the queue twice walks it differently and cannot tell where they were.

    Five packages, inserted in exactly reverse order, so the answer is not the order they
    were written in. The width is measured rather than chosen: with two rows, dropping the
    tie-break from the query still passed, because Postgres handed those two back in package
    order for its own reasons. Five separates them.
    """
    await seed(triage_db, {f"com.example.{name}": 2 for name in "edcba"})

    assert await queue(triage_db) == [f"com.example.{name}" for name in "abcde"]


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


async def test_an_approval_with_no_emission_row_stays_decided_and_offerable(db_env, triage_db):
    """The pre-§19 shape must keep working: an approval that has not shipped yet is still a
    decided package, still offerable to the next emission."""
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "approve")

    assert await queue(triage_db, "decided") == ["com.example.one"]
    assert await queue(triage_db, "shipped") == []


async def test_a_shipped_package_leaves_decided_and_enters_shipped(db_env, triage_db):
    """§19's verify line at the store: an `approve` with a `branch_emission_package` row is
    its own state, not a decided one — the board must stop offering something already gone."""
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "approve")
    await _seed_emission(triage_db, ["com.example.one"], branch="uadclaw/pixel-000000000007")

    assert await queue(triage_db, "decided") == []
    assert await queue(triage_db, "shipped") == ["com.example.one"]


async def test_a_rejected_package_with_an_emission_row_stays_decided(db_env, triage_db):
    """`shipped` requires the CURRENT verdict to be `approve`. A package that shipped, was
    reopened and then rejected is a rejection: the emission row must not drag it into the
    shipped view, and it must stay visible in decided."""
    await seed(triage_db, {"com.example.one": 1})

    await decide(triage_db, "com.example.one", "approve", at=NOW)
    await _seed_emission(triage_db, ["com.example.one"])
    await decide(triage_db, "com.example.one", "reopen", at=NOW + timedelta(minutes=1))
    await decide(
        triage_db,
        "com.example.one",
        "reject",
        reason="changed my mind",
        at=NOW + timedelta(minutes=2),
    )

    assert await queue(triage_db, "shipped") == []
    assert await queue(triage_db, "decided") == ["com.example.one"]


async def test_a_pending_emission_is_not_shipped(db_env, triage_db):
    """The emission row is written BEFORE the git commit, so a `commit_oid IS NULL` row is an
    intent whose outcome is unknown and whose branch may never have been cut. It must not read
    as shipped — the package stays decided until the commit actually lands."""
    await seed(triage_db, {"com.example.one": 1})
    await decide(triage_db, "com.example.one", "approve")
    await _seed_emission(triage_db, ["com.example.one"], committed=False)

    assert await queue(triage_db, "shipped") == []
    assert await queue(triage_db, "decided") == ["com.example.one"]


async def test_a_package_that_shipped_twice_shows_the_latest_branch(db_env, triage_db):
    """An approval can ship more than once (a re-approval after re-classification), and the
    branch shown is the latest emission's. Both readers are asserted — the row derives it via
    the bulk query, the standing via the single-package one — because the two are separate
    queries and either could independently pick the wrong emission."""
    await seed(triage_db, {"com.example.one": 1})
    await decide(triage_db, "com.example.one", "approve")
    await _seed_emission(triage_db, ["com.example.one"], branch="uadclaw/pixel-000000000001")
    await _seed_emission(triage_db, ["com.example.one"], branch="uadclaw/pixel-000000000009")

    async with triage_db() as session:
        rows = await triagestore.load_rows(session)
        standing = await triagestore.load_standing(session, "com.example.one")

    assert [row.shipped_branch for row in rows] == ["uadclaw/pixel-000000000009"], (
        "the row must show the NEWEST emission's branch, not the first"
    )
    assert standing.shipped_branch == "uadclaw/pixel-000000000009", (
        "the standing must agree with the row about which emission is latest"
    )


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


async def test_the_queue_and_the_card_agree_when_an_older_bundle_comes_back(db_env, triage_db):
    """One definition of "the current decision", asked through both readers.

    They had two. The queue took the NEWEST decision and asked whether it was current; the
    card took the newest decision that IS current, walking past newer ones. The two disagree
    the moment a row's `bundle_sha256` returns to a value an older decision was filed
    against, which the content-addressed bundle makes reachable: a corpus that shrinks back
    reproduces a hash. The queue then said undecided while the card rendered `reject` as
    current, over the same row, in the same render.

    The newest decision is the one that counts, and a package whose newest decision was filed
    against other evidence goes back in the queue — which is the safe reading either way: a
    reviewer looks again at a proposal nobody has judged, rather than a stale verdict
    silently deciding it.
    """
    await seed(triage_db, {"com.example.one": 1})
    await decide(triage_db, "com.example.one", "reject", reason="too vague", at=NOW)

    async def reclassify(bundle: str) -> None:
        async with triage_db() as session, session.begin():
            await store_classification(
                session,
                proposal("com.example.one", bundle=bundle),
                model=MODEL,
                thinking=True,
                usage={},
                attempts=1,
                at=NOW,
            )

    await reclassify("c" * 64)
    await decide(
        triage_db,
        "com.example.one",
        "approve",
        at=NOW + timedelta(minutes=1),
        bundle="c" * 64,
    )
    # The evidence reverts, so the content-addressed hash comes back with it.
    await reclassify(BUNDLE)

    async with triage_db() as session:
        card = await triagestore.load_candidate(session, "com.example.one")
    assert await queue(triage_db) == ["com.example.one"]
    assert card.decision is None, (
        f"the queue says undecided and the card shows {card.decision} as current"
    )
    assert [item.action for item in card.history] == ["approve", "reject"], (
        "the log is untouched: what changed is which entry counts, never what is kept"
    )


async def test_the_counts_and_the_rows_come_off_one_read(db_env, triage_db, monkeypatch):
    """The screen showed a "cleared" state computed from counts taken by a second query, so a
    decision landing between the two described rows that were no longer there. Counted rather
    than raced: one read is the property, and a race is not what a test can pin."""
    await seed(triage_db, {"com.example.one": 1, "com.example.two": 1})
    reads = []
    real = triagestore.load_rows

    async def counting(session):
        reads.append(1)
        return await real(session)

    monkeypatch.setattr(triagestore, "load_rows", counting)

    async with triage_db() as session:
        rows, counts = await triagestore.load_board(session, view="queue")

    assert len(reads) == 1
    assert [row.package for row in rows] == ["com.example.one", "com.example.two"]
    assert counts["queue"] == 2


async def test_load_rows_reads_its_three_dicts_on_one_snapshot(db_env, triage_db, monkeypatch):
    """`load_rows` runs the joined rows, the newest-decision dict and the shipped-branch dict
    as three statements, and Postgres reads committed PER STATEMENT — so without a REPEATABLE
    READ transaction a decision landing between them is visible to one and not the others,
    and the board renders a stale decision beside fresh rows (or the reverse).

    The competing write is a real second connection committing a real decision, fired from
    between the two reads by wrapping the decision dict. Under REPEATABLE READ the whole read
    sees the instant before the write and the row carries no decision; under READ COMMITTED
    the dict read takes a fresh snapshot and the row carries the new verdict — so removing
    `load_rows`'s `_begin_snapshot` call turns this red.
    """
    await seed(triage_db, {"com.example.one": 1})
    real = triagestore._latest_decisions
    fired: list[str] = []

    async def latest_then_let_somebody_else_decide(session):
        async with triage_db() as other, other.begin():
            await triagestore.decide(
                other, package="com.example.one", action="reject", reason="raced in", at=NOW
            )
        fired.append("decided")
        return await real(session)

    monkeypatch.setattr(triagestore, "_latest_decisions", latest_then_let_somebody_else_decide)

    async with triage_db() as session:
        rows = await triagestore.load_rows(session)

    # The interleave really happened, and it really landed: without both of these the snapshot
    # assertion below passes for the wrong reason.
    assert fired == ["decided"]
    async with triage_db() as session:
        result = await session.execute(select(PackageTriageDecision))
        assert [row.action for row in result.scalars()] == ["reject"]
    assert [row.decision for row in rows] == [None]


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


async def test_an_edit_survives_the_next_model_run(db_env, triage_db):
    """The whole reason the edit writes provenance rather than only a value, driven through
    both writers. Asserting the provenance string alone proves the edit is TAGGED; this is
    the leg that proves the tag does something, and it is what the model re-run would erase.
    """
    await seed(triage_db, {"com.example.one": 1})
    edited = "Notes application. Removing it loses locally stored notes."
    async with triage_db() as session, session.begin():
        await triagestore.apply_edit(
            session, package="com.example.one", edits={"description": edited}, at=NOW
        )

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

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == edited
    assert row.provenance["description"] == "human:triage"
    # …and the rest of the row is the new run's, or nothing was re-classified at all.
    assert row.bundle_sha256 == "c" * 64


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


def unknown_proposal(package: str):
    """What the model writes when it will not guess: the `unknown` sentinel and a declaration
    beside it. A normal, expected outcome rather than a failure — `classify.py` says so at
    length — so it is the shape the edit form meets most often on a weak candidate."""
    return Classification(
        package=package,
        bundle_sha256=BUNDLE,
        description=UNKNOWN,
        list=UadList.MISC,
        removal=Removal.ADVANCED,
        confidence=Confidence.LOW,
        unknown_fields=("description",),
        reasoning_brief="Nothing in the evidence says what this does.",
        provenance={"description": f"llm:{MODEL}", "removal": f"llm:{MODEL}"},
    )


async def test_an_unchanged_unknown_description_does_not_block_the_rest_of_the_edit(
    db_env, triage_db
):
    """`triage_card.html` prefills the description box with the stored value, so a reviewer
    who only touches the removal select still posts the `unknown` sentinel back. Seven
    characters against a twenty-character floor: validating before asking whether the field
    MOVED refused the whole submission, dropping the removal change they actually made, and
    told them to declare it in `unknown_fields` — a field the form has no control for.

    That is the ordinary submission for such a row rather than an edge case: the card renders
    `model answer (description)` as missing for exactly these.
    """
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        await store_classification(
            session,
            unknown_proposal("com.example.one"),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    async with triage_db() as session, session.begin():
        changed = await triagestore.apply_edit(
            session,
            package="com.example.one",
            edits={"description": UNKNOWN, "removal": "Expert"},
            at=NOW,
        )

    assert set(changed) == {"removal"}
    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.removal == "Expert"
    assert row.description == UNKNOWN, "an unchanged field is not an edit and is not rewritten"
    assert row.unknown_fields == ["description"], "so the model's own declaration still stands"


async def test_a_description_the_reviewer_actually_rewrites_is_still_validated(db_env, triage_db):
    """The other leg, and without it the test above is satisfied by a store that validates
    nothing: a value that MOVED meets every rule a model answer does, and a written
    description clears the `unknown` declaration the model made."""
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        await store_classification(
            session,
            unknown_proposal("com.example.one"),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    with pytest.raises(ValueError):
        async with triage_db() as session, session.begin():
            await triagestore.apply_edit(
                session, package="com.example.one", edits={"description": "too short"}, at=NOW
            )

    written = "Notes application. Removing it loses locally stored notes."
    async with triage_db() as session, session.begin():
        await triagestore.apply_edit(
            session, package="com.example.one", edits={"description": written}, at=NOW
        )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == written
    assert row.unknown_fields == []


async def test_a_fresh_edit_clears_the_mark_saying_an_earlier_one_was_superseded(db_env, triage_db):
    """The mark records the value it replaced. Left beside a value the reviewer has just
    written, it tells them their own answer was already overruled, which is a lie the card
    renders in their face."""
    await seed(triage_db, {"com.example.one": 1}, floor="Advanced")
    async with triage_db() as session, session.begin():
        row = await session.get(PackageClassification, "com.example.one")
        row.provenance = {
            **row.provenance,
            "removal_superseded": "rule:floor superseded human:triage Recommended",
        }

    async with triage_db() as session, session.begin():
        await triagestore.apply_edit(
            session, package="com.example.one", edits={"removal": "Expert"}, at=NOW
        )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.removal == "Expert"
    assert "removal_superseded" not in row.provenance


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

    assert "sources" in card.missing


# --- icons -----------------------------------------------


async def test_a_stored_icon_reaches_the_row_and_the_card(db_env, triage_db):
    """The half that could not run while `icon_mime` lived in another lane's worktree.

    Driven through `store_device_facts`, the writer production uses, rather than by setting
    the column: `factstore` merges the icon first-wins across observations, so a fixture that
    wrote the merged row directly would pin the reader against a row shape the writer may not
    produce. Both readers are asserted because they read the column two different ways — the
    queue selects it, the card gets it off the ORM object — and only one of them was ever
    exercised against a real column.
    """
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        await store_device_facts(
            session,
            device_key="pixel:device0",
            build="bp1a.260505.001",
            facts=[
                make_facts(
                    "com.example.one",
                    icon_bytes=b"\x89PNG\r\n\x1a\n" + b"0" * 32,
                    icon_mime="image/png",
                )
            ],
            observed_at=NOW,
        )

    async with triage_db() as session:
        rows = await triagestore.load_rows(session)
        card = await triagestore.load_candidate(session, "com.example.one", upstream=UPSTREAM)

    assert [row.has_icon for row in rows] == [True]
    assert card.has_icon is True


async def test_a_package_with_no_stored_icon_says_so_on_the_row_and_on_the_card(db_env, triage_db):
    """`has_icon` is what the templates branch on, so the screen never probes the icon route
    to find out whether there is anything behind it. Measured here against the column being
    absent, which is this checkout's state: the answer is False and nothing raises."""
    await seed(triage_db, {"com.example.one": 1})

    async with triage_db() as session:
        rows = await triagestore.load_rows(session)
        card = await triagestore.load_candidate(session, "com.example.one", upstream=UPSTREAM)

    assert [row.has_icon for row in rows] == [False]
    assert card.has_icon is False


async def test_the_card_refuses_a_package_with_no_proposal(db_env, triage_db):
    with pytest.raises(triagestore.TriageError, match="com.example.absent"):
        async with triage_db() as session:
            await triagestore.load_candidate(session, "com.example.absent", upstream=UPSTREAM)


async def _rows(session_factory):
    async with session_factory() as session:
        return await triagestore.load_queue(session, view="queue")


# --- a declaration and the value it describes ------------------------------------------------


async def declare_unknown(session_factory, package: str) -> None:
    async with session_factory() as session, session.begin():
        await store_human_edit(session, package, edits={"description": UNKNOWN}, at=NOW)


def assert_declaration_matches_description(row: PackageClassification) -> None:
    """`description == "unknown"` iff `"description" in unknown_fields`, on every row.

    One assertion rather than one per writer: the two halves say the same thing about the same
    field, and a row carrying one without the other asserts a contradiction in whichever
    direction it is missing. A description reading `unknown` with nothing declaring it is the
    expensive direction — that string ships into `uad_lists.json` as if a reviewer wrote it.
    """
    declared = "description" in (row.unknown_fields or [])
    assert declared == (row.description == UNKNOWN), (
        f"description={row.description!r} but unknown_fields={row.unknown_fields!r}"
    )


async def test_a_declared_unknown_description_keeps_its_declaration_across_a_model_re_run(
    db_env, triage_db
):
    """The declaration is half of a `human:` value and has to travel with the other half.

    `_preserved` carries `description` because provenance says `human:`, and `unknown_fields`
    is not in `HUMAN_OWNED_CANDIDATES` — so the model's fresh list overwrote it and the row
    was left reading `unknown` with nothing saying it was declared.
    """
    await seed(triage_db, {"com.example.one": 1})
    await declare_unknown(triage_db, "com.example.one")

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

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == UNKNOWN, "the human value itself is supposed to survive"
    assert row.unknown_fields == ["description"]
    assert_declaration_matches_description(row)


async def test_a_declared_unknown_description_keeps_its_declaration_across_a_park(
    db_env, triage_db
):
    """The second writer. A park against a NEW bundle clears the answer and then puts the
    preserved human half back, so it takes the same route and drops the same declaration."""
    await seed(triage_db, {"com.example.one": 1})
    await declare_unknown(triage_db, "com.example.one")

    async with triage_db() as session, session.begin():
        await park_package(
            session,
            "com.example.one",
            bundle_sha256="d" * 64,
            model=MODEL,
            thinking=True,
            reason="removal: answered below the floor three times",
            usage={},
            attempts=3,
            at=NOW,
        )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description == UNKNOWN
    assert row.unknown_fields == ["description"]
    assert_declaration_matches_description(row)


async def test_a_human_description_takes_the_model_s_declaration_off_the_row(db_env, triage_db):
    """The other direction, and the one nobody would think to look for: a reviewer writes a
    real description over a model answer that declared the field unknown. Preserving their
    words while keeping the model's declaration asserts the same contradiction backwards."""
    await seed(triage_db, {"com.example.one": 1})
    async with triage_db() as session, session.begin():
        await store_human_edit(
            session,
            "com.example.one",
            edits={"description": "Notes application. Removing it loses locally stored notes."},
            at=NOW,
        )

    async with triage_db() as session, session.begin():
        declared = proposal("com.example.one", bundle="e" * 64)
        await store_classification(
            session,
            Classification(
                package=declared.package,
                bundle_sha256=declared.bundle_sha256,
                description=UNKNOWN,
                list=declared.list,
                removal=declared.removal,
                confidence=declared.confidence,
                unknown_fields=("description",),
                reasoning_brief=declared.reasoning_brief,
                provenance=declared.provenance,
            ),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    async with triage_db() as session:
        row = await session.get(PackageClassification, "com.example.one")
    assert row is not None
    assert row.description is not None and row.description.startswith("Notes application")
    assert row.unknown_fields == []
    assert_declaration_matches_description(row)
