"""The branch-emission schema and the store that reads the approved set into it.

Two things are being defended here and they fail differently.

The first is the crash window. `branch_emission` is written BEFORE the git commit it describes
is attempted, so `commit_oid IS NULL` is a designed state meaning "the outcome of this
emission is not known", and the recovery path in `stages.branch_stage` branches on exactly
that read. A row that carried a `committed_at` with no oid, or claimed `reconciled` with
neither, would make that read answer wrongly — and answering wrongly there is how a second
branch lands on somebody's clone. So both pairs are CHECK constraints and both are exercised
against a real Postgres rather than asserted off the model.

The second is what may become an `ApprovedPackage` at all. `load_approved` is the lane that
builds them out of `package_analysis`, whose `floor` column is nullable while
`ApprovedPackage.floor` is not, so it is the one place a package whose rule ladder never ran
has to be refused rather than defaulted. `floor=row.floor or "Recommended"` is the spelling
that reads exactly like the safe version and can never fire, since `danger_rank("Recommended")`
is 0, and there is a test here that goes red for it.
"""

import uuid
from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy import select, text

from uadclaw import emissionstore
from uadclaw import jobs as jobs_module
from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import store_classification
from uadclaw.emission import EmissionError
from uadclaw.emissionstore import (
    EmissionStoreError,
    forget_intent,
    load_approved,
    load_emission,
    record_commit,
    record_intent,
)
from uadclaw.facts import ApkFacts
from uadclaw.factstore import store_device_facts
from uadclaw.ladder import Removal
from uadclaw.models import (
    JOB_KIND_NEEDS_SCRATCH,
    JOB_KIND_STAGES,
    BranchEmission,
    BranchEmissionPackage,
    JobKind,
    PackageAnalysis,
    PackageClassification,
)
from uadclaw.triagestore import record_decision

pytestmark = pytest.mark.usefixtures("db_env")

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
MODEL = "deepseek-v4-flash"
BUNDLE = "b" * 64


def utcnow() -> datetime:
    return datetime.now(UTC)


def _row(**overrides: object) -> BranchEmission:
    values: dict[str, object] = {
        "job_id": None,
        "vendor": "pixel",
        "branch": "uadclaw/pixel-0123456789ab",
        "repo_path": "/upstream",
        "list_path": "resources/assets/uad_lists.json",
        "base_commit": "a" * 40,
        "list_sha256": "b" * 64,
        "pipeline_version": "0.1.0",
        "pipeline_commit_sha": "c" * 40,
        "package_count": 1,
        "pr_body": "## pixel",
        "created_at": utcnow(),
        "reconciled": False,
    }
    values.update(overrides)
    return BranchEmission(**values)


# --- the job kind ----------------------------------------------------------------------------


def test_branch_emission_is_its_own_kind_walking_only_branch():
    """Its own kind rather than a tail on the classification walk, for the reason `llm` is
    one: appending it there would cut a branch out of whatever happened to be approved the
    moment a corroboration run finished, with nobody having decided the batch was ready."""
    assert JOB_KIND_STAGES[JobKind.BRANCH_EMISSION] == ("branch",)
    assert jobs_module.stages_for("branch_emission") == ("branch",)
    assert jobs_module.next_stage(None, "branch_emission") == "branch"
    assert jobs_module.next_stage("branch", "branch_emission") is None


def test_branch_emission_takes_no_scratch_lease():
    """It writes into the operator's git clone, which is a bind mount of its own. Taking the
    single-occupant lease would queue a branch behind a multi-hour Samsung unpack for a
    directory that unpack never touches."""
    assert JOB_KIND_NEEDS_SCRATCH[JobKind.BRANCH_EMISSION] is False
    assert jobs_module.needs_scratch("branch_emission") is False


# --- the crash-window CHECK constraints --------------------------------------------------------


async def test_a_pending_emission_is_a_row_with_no_commit(db_session_factory):
    """The designed state, and the whole reason the row is written before the commit: the
    batch survives a crash and the outcome is what gets recovered."""
    async with db_session_factory() as session, session.begin():
        session.add(_row())
    async with db_session_factory() as session:
        stored = (await session.execute(select(BranchEmission))).scalar_one()
    assert stored.commit_oid is None
    assert stored.committed_at is None
    assert stored.reconciled is False


@pytest.mark.parametrize(
    "half",
    [
        {"commit_oid": "d" * 40},
        {"committed_at": datetime(2026, 8, 13, tzinfo=UTC)},
    ],
)
async def test_half_an_outcome_is_refused_by_the_database(db_session_factory, half):
    """`commit_oid IS NULL` is read as "the outcome is unknown". A row carrying one half of
    the pair makes that read answer wrongly for an emission that is actually finished, which
    is the one mistake here that ends in a second branch."""
    async with db_session_factory() as session, session.begin():
        session.add(_row(**half))
        with pytest.raises(sqlalchemy.exc.IntegrityError, match="ck_branch_emission_commit_pair"):
            await session.flush()


async def test_a_reconciled_row_must_carry_the_commit_it_recovered(db_session_factory):
    """`reconciled` claims a recovery found an outcome, so a pending row cannot wear it."""
    async with db_session_factory() as session, session.begin():
        session.add(_row(reconciled=True))
        with pytest.raises(
            sqlalchemy.exc.IntegrityError, match="ck_branch_emission_reconciled_commit"
        ):
            await session.flush()


async def test_one_job_can_only_have_one_emission_row(db_session_factory):
    """What makes the branch name derivable from the job and a retry idempotent: a second row
    for one job would give the recovery path two intents to choose between."""
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.BRANCH_EMISSION.value, params={"vendor": "pixel"}
        )
        job_id = job.id
    async with db_session_factory() as session, session.begin():
        session.add(_row(job_id=job_id))
    async with db_session_factory() as session, session.begin():
        session.add(_row(job_id=job_id, branch="uadclaw/pixel-second"))
        with pytest.raises(sqlalchemy.exc.IntegrityError, match="uq_branch_emission_job_id"):
            await session.flush()


async def test_two_jobs_that_never_ran_do_not_collide_on_a_null_job_id(db_session_factory):
    """`job_id` is SET NULL rather than CASCADE, so pruning the job rows must not start
    refusing every emission but one. Postgres treats NULLs as distinct in a unique
    constraint; asserted rather than assumed, because the opposite reading would silently
    delete this table's history the first time a job was pruned."""
    async with db_session_factory() as session, session.begin():
        session.add(_row(job_id=None))
        session.add(_row(job_id=None, branch="uadclaw/pixel-other"))
    async with db_session_factory() as session:
        assert len((await session.execute(select(BranchEmission))).scalars().all()) == 2


async def test_a_deleted_emission_takes_its_packages_with_it(db_session_factory):
    """The child rows describe one emission and nothing else, so they cascade. Pinned because
    the parent's own FK is SET NULL and the two directions read alike in a schema diff."""
    async with db_session_factory() as session, session.begin():
        row = _row()
        session.add(row)
        await session.flush()
        session.add(
            BranchEmissionPackage(
                emission_id=row.id,
                package="com.example.one",
                bundle_sha256="e" * 64,
                uad_list="Oem",
                removal="Advanced",
                floor="Advanced",
            )
        )
        emission_id = row.id
    async with db_session_factory() as session, session.begin():
        await session.execute(
            sqlalchemy.delete(BranchEmission).where(BranchEmission.id == emission_id)
        )
    async with db_session_factory() as session:
        remaining = (await session.execute(select(BranchEmissionPackage))).scalars().all()
    assert remaining == []


async def test_the_same_package_cannot_ship_twice_in_one_emission(db_session_factory):
    """Two rows for one package in one batch make the child table disagree with the JSON that
    shipped, where `insert_entries` already refuses a duplicate outright."""
    async with db_session_factory() as session, session.begin():
        row = _row()
        session.add(row)
        await session.flush()
        for _ in range(2):
            session.add(
                BranchEmissionPackage(
                    emission_id=row.id,
                    package="com.example.one",
                    bundle_sha256="e" * 64,
                    uad_list="Oem",
                    removal="Advanced",
                    floor="Advanced",
                )
            )
        with pytest.raises(sqlalchemy.exc.IntegrityError, match="uq_branch_emission_package"):
            await session.flush()


def test_the_emission_tables_survive_the_mapped_list_trap():
    """A `list:` mapped_column rebinds the name for the rest of its class body and the next
    `Mapped[list[Any]]` raises at IMPORT time, so this is a property of the module loading at
    all. Asserted on the rename rather than on the import, because "the tests ran" already
    proves the import and proves nothing about which spelling was used."""
    assert "uad_list" in BranchEmissionPackage.__table__.columns
    assert "list" not in BranchEmissionPackage.__table__.columns


def test_a_branch_emission_job_is_a_uuid_keyed_job_like_any_other():
    """Guards the FK's type against a future kind keyed differently."""
    assert BranchEmission.__table__.c.job_id.type.python_type is uuid.UUID


# --- the approved set ------------------------------------------------------------------------


@pytest.fixture
async def store_db(db_session_factory):
    """`conftest`'s truncation covers these tables, but the decision log is the one whose
    leftovers decide a package for the NEXT test on this xdist worker, so it is cleared
    explicitly the way `test_triage_store` does."""
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
        },
    )


async def seed(
    session_factory,
    package: str,
    *,
    device_keys: tuple[str, ...] = ("pixel:oriole",),
    floor: str | None = "Recommended",
    bundle: str = BUNDLE,
    parked: bool = False,
) -> None:
    """One package, through the real writers.

    `store_device_facts` is what produces `package_facts.devices`, so the device keys the
    vendor grouping reads are the ones production writes rather than a literal typed here —
    which is the whole point, since the grouping reading a hand-built field would not prove it
    reads the field production fills.
    """
    async with session_factory() as session, session.begin():
        for key in device_keys:
            await store_device_facts(
                session,
                device_key=key,
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
                floor_rule="default" if floor else None,
                dependencies=[],
                needed_by=[],
            )
        )
    async with session_factory() as session, session.begin():
        await store_classification(
            session,
            proposal(package, bundle=bundle),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )
        if parked:
            row = await session.get(PackageClassification, package)
            row.parked = True
            row.description = None
            row.uad_list = None
            row.removal = None


async def decide(session_factory, package: str, action: str, *, bundle: str = BUNDLE) -> None:
    async with session_factory() as session, session.begin():
        await record_decision(
            session,
            package=package,
            bundle_sha256=bundle,
            action=action,
            reason="not worth an entry" if action == "reject" else None,
            at=NOW,
        )


async def approved_for(session_factory, vendor: str):
    async with session_factory() as session:
        return await load_approved(session, vendor=vendor)


async def test_an_approved_package_becomes_a_shippable_entry(store_db):
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")

    packages = await approved_for(store_db, "pixel")

    assert [item.package for item in packages] == ["com.example.one"]
    assert packages[0].removal == "Advanced"
    assert packages[0].floor == "Recommended"
    assert packages[0].uad_list == "Misc"
    assert packages[0].bundle_sha256 == BUNDLE
    assert packages[0].model == MODEL
    assert packages[0].device_keys == ("pixel:oriole",)
    # `labels` is a dead field upstream and nothing here derives one, so the truthful claim is
    # an empty list rather than a guess.
    assert packages[0].labels == ()


@pytest.mark.parametrize("action", ["reject", "defer", "edit", "reopen"])
async def test_only_an_approval_reaches_a_branch(store_db, action):
    """`edit` and `reopen` are the dangerous two: both look like progress and neither is a
    verdict. An edited candidate is a revision somebody still has to approve, and a reopened
    one is back in the queue by definition."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", action)

    assert await approved_for(store_db, "pixel") == ()


async def test_an_undecided_package_reaches_no_branch(store_db):
    """The positive leg's opposite: with no decision at all there is nothing approved, so a
    later test asserting an approval must be asserting something the decision caused."""
    await seed(store_db, "com.example.one")

    assert await approved_for(store_db, "pixel") == ()


async def test_an_approval_of_evidence_that_has_since_changed_does_not_ship(store_db):
    """The self-expiring half of the decision log, and the reason emission calls
    `current_decision` rather than reading `action`: re-classifying moves the bundle hash, the
    old approval stops being current on its own, and the package is back in the queue for
    somebody to look at again. Shipping it anyway would put an entry upstream that no human
    ever read in the form it shipped."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    async with store_db() as session, session.begin():
        await store_classification(
            session,
            proposal("com.example.one", bundle="c" * 64),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )

    assert await approved_for(store_db, "pixel") == ()


async def test_the_newest_decision_wins_even_when_it_reopens(store_db):
    """The log is append-only, so "approved once" is not "approved"."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    await decide(store_db, "com.example.one", "reopen")

    assert await approved_for(store_db, "pixel") == ()


# --- the vendor is provenance, never the name --------------------------------------------------


async def test_the_vendor_comes_off_the_device_key_and_not_the_package_name(store_db):
    """A name-prefix rule swept 70 Xiaomi packages into an OPPO bucket in this repo once
    already. `com.xiaomi.*` scanned on a Pixel is a Pixel package here, which is the whole
    difference between reading a field and guessing from one that looks like it."""
    await seed(store_db, "com.xiaomi.misettings", device_keys=("pixel:oriole",))
    await decide(store_db, "com.xiaomi.misettings", "approve")

    assert [item.package for item in await approved_for(store_db, "pixel")] == [
        "com.xiaomi.misettings"
    ]
    assert await approved_for(store_db, "xiaomi") == ()


async def test_a_package_two_oems_ship_is_neither_oems_to_review_alone(store_db):
    await seed(store_db, "com.example.both", device_keys=("pixel:oriole", "samsung:SM-S911U"))
    await decide(store_db, "com.example.both", "approve")

    assert await approved_for(store_db, "pixel") == ()
    assert await approved_for(store_db, "samsung") == ()
    assert [item.package for item in await approved_for(store_db, "shared")] == ["com.example.both"]


async def test_a_package_with_no_device_provenance_fails_every_vendors_load(store_db):
    """Deliberately NOT scoped to the requested vendor. A package with no provenance cannot be
    attributed to anybody, so "it is somebody else's batch's problem" is unavailable: skipping
    it would be a human-approved package that silently never ships."""
    async with store_db() as session, session.begin():
        session.add(PackageAnalysis(package="com.example.orphan", updated_at=NOW, floor="Advanced"))
        await store_classification(
            session,
            proposal("com.example.orphan"),
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )
    await decide(store_db, "com.example.orphan", "approve")

    for vendor in ("pixel", "samsung", "shared"):
        with pytest.raises(EmissionError, match="no device keys"):
            await approved_for(store_db, vendor)


async def test_packages_come_back_in_the_order_they_will_be_written(store_db):
    for package in ("com.example.zulu", "com.example.alpha", "com.example.mike"):
        await seed(store_db, package)
        await decide(store_db, package, "approve")

    packages = await approved_for(store_db, "pixel")

    assert [item.package for item in packages] == [
        "com.example.alpha",
        "com.example.mike",
        "com.example.zulu",
    ]


# --- the floor is the refusal point ------------------------------------------------------------


async def test_a_package_whose_ladder_never_ran_refuses_the_whole_batch(store_db):
    """`package_analysis.floor` is nullable and `ApprovedPackage.floor` is not, because a
    package with no floor has no bound for its rating to sit above. The tempting spelling is
    `floor=row.floor or "Recommended"`, which produces a floor of `danger_rank` 0 — present,
    checking nothing, and reading exactly like the safe version. Refused instead, naming the
    package and the stage to run."""
    await seed(store_db, "com.example.unfloored", floor=None)
    await decide(store_db, "com.example.unfloored", "approve")

    with pytest.raises(EmissionStoreError) as excinfo:
        await approved_for(store_db, "pixel")

    assert "com.example.unfloored" in str(excinfo.value)
    assert "rule_ladder" in str(excinfo.value)


async def test_a_missing_floor_refuses_rather_than_dropping_the_package(store_db):
    """The batch is refused whole, never trimmed to what is shippable. A package a human
    approved that quietly does not appear in the PR is the failure nobody notices."""
    await seed(store_db, "com.example.fine")
    await decide(store_db, "com.example.fine", "approve")
    await seed(store_db, "com.example.unfloored", floor=None)
    await decide(store_db, "com.example.unfloored", "approve")

    with pytest.raises(EmissionStoreError, match="com.example.unfloored"):
        await approved_for(store_db, "pixel")


async def test_another_vendors_missing_floor_does_not_fail_this_batch(store_db):
    """The counterpart bound: the refusal is real, and it belongs to the batch whose packages
    it describes. Emitting Pixel must not be blocked by a Samsung package nobody was asking
    about."""
    await seed(store_db, "com.example.fine")
    await decide(store_db, "com.example.fine", "approve")
    await seed(store_db, "com.example.unfloored", device_keys=("samsung:SM-S911U",), floor=None)
    await decide(store_db, "com.example.unfloored", "approve")

    assert [item.package for item in await approved_for(store_db, "pixel")] == ["com.example.fine"]
    with pytest.raises(EmissionStoreError, match="com.example.unfloored"):
        await approved_for(store_db, "samsung")


async def test_an_approved_parked_package_refuses_rather_than_shipping_a_blank(store_db):
    """A park is a recorded terminal state carrying no proposal, so there is nothing to ship.
    Reachable because `decide` will record a verdict against any package with a
    classification row, and the parked view is one a reviewer can act from."""
    await seed(store_db, "com.example.parked", parked=True)
    await decide(store_db, "com.example.parked", "approve")

    with pytest.raises(EmissionStoreError, match="PARKED"):
        await approved_for(store_db, "pixel")


# --- two reads are not one snapshot ------------------------------------------------------------


async def test_the_approved_set_is_read_on_one_snapshot(store_db, monkeypatch):
    """`load_rows` issues two statements and the detail read is a third, and Postgres reads
    committed PER STATEMENT — so without a REPEATABLE READ transaction a write landing between
    them is visible to one and not the others. On the triage screen that is a stale count.
    Here it decides which bytes are committed into somebody else's repository.

    The competing write is a real second connection committing a real change, and it is fired
    from between the two reads by wrapping the first. Under READ COMMITTED the detail read
    returns the NEW description; the assertion below is the OLD one, so flipping the isolation
    level in `emissionstore._begin_snapshot` turns this red.
    """
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    original_rows = emissionstore.load_rows
    fired: list[str] = []

    async def load_rows_then_let_somebody_else_commit(session):
        rows = await original_rows(session)
        async with store_db() as other, other.begin():
            await other.execute(
                sqlalchemy.update(PackageClassification)
                .where(PackageClassification.package == "com.example.one")
                .values(description="Rewritten while the emission was reading.")
            )
        fired.append("committed")
        return rows

    monkeypatch.setattr(emissionstore, "load_rows", load_rows_then_let_somebody_else_commit)
    packages = await approved_for(store_db, "pixel")

    # The interleave really happened, and it really landed: without both of these the snapshot
    # assertion below passes for the wrong reason.
    assert fired == ["committed"]
    async with store_db() as session:
        stored = await session.get(PackageClassification, "com.example.one")
        assert stored.description == "Rewritten while the emission was reading."
    assert packages[0].description.startswith("Vendor application com.example.one.")


# --- recording the run -------------------------------------------------------------------------


async def emission_job(session_factory) -> uuid.UUID:
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.BRANCH_EMISSION.value, params={"vendor": "pixel"}
        )
        return job.id


async def test_an_intent_records_the_batch_before_anything_is_committed(store_db):
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)

    async with store_db() as session, session.begin():
        emission_id = await record_intent(
            session,
            job_id=job_id,
            vendor="pixel",
            branch="uadclaw/pixel-0123456789ab",
            repo_path="/upstream",
            list_path="resources/assets/uad_lists.json",
            base_commit="a" * 40,
            list_sha256="b" * 64,
            pipeline_version="0.1.0",
            pipeline_commit_sha="c" * 40,
            pr_body="## pixel",
            packages=packages,
            at=NOW,
        )

    async with store_db() as session:
        record = await load_emission(session, job_id=job_id)
        children = (
            (
                await session.execute(
                    select(BranchEmissionPackage).where(
                        BranchEmissionPackage.emission_id == emission_id
                    )
                )
            )
            .scalars()
            .all()
        )
    assert record is not None
    assert record.commit_oid is None
    assert record.branch == "uadclaw/pixel-0123456789ab"
    assert record.list_sha256 == "b" * 64
    assert [(child.package, child.removal, child.floor) for child in children] == [
        ("com.example.one", "Advanced", "Recommended")
    ]


async def test_an_empty_batch_is_never_recorded_as_a_run(store_db):
    job_id = await emission_job(store_db)
    async with store_db() as session, session.begin():
        with pytest.raises(EmissionStoreError, match="no packages"):
            await record_intent(
                session,
                job_id=job_id,
                vendor="pixel",
                branch="uadclaw/pixel-0123456789ab",
                repo_path="/upstream",
                list_path="resources/assets/uad_lists.json",
                base_commit="a" * 40,
                list_sha256="b" * 64,
                pipeline_version="0.1.0",
                pipeline_commit_sha="c" * 40,
                pr_body="## pixel",
                packages=(),
                at=NOW,
            )


async def _intent(store_db, job_id, packages, **overrides) -> int:
    values: dict[str, object] = {
        "vendor": "pixel",
        "branch": "uadclaw/pixel-0123456789ab",
        "repo_path": "/upstream",
        "list_path": "resources/assets/uad_lists.json",
        "base_commit": "a" * 40,
        "list_sha256": "b" * 64,
        "pipeline_version": "0.1.0",
        "pipeline_commit_sha": "c" * 40,
        "pr_body": "## pixel",
    }
    values.update(overrides)
    async with store_db() as session, session.begin():
        return await record_intent(session, job_id=job_id, packages=packages, at=NOW, **values)


async def test_an_unfinished_intent_is_replaced_children_and_all(store_db):
    """It described a batch whose branch was rolled back, so leaving it would make the
    recovery path compare a live branch against a digest from a run that produced nothing."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)

    first = await _intent(store_db, job_id, packages)
    second = await _intent(store_db, job_id, packages, list_sha256="d" * 64)

    assert second != first
    async with store_db() as session:
        rows = (await session.execute(select(BranchEmission))).scalars().all()
        children = (await session.execute(select(BranchEmissionPackage))).scalars().all()
    assert [row.list_sha256 for row in rows] == ["d" * 64]
    assert [child.emission_id for child in children] == [second]


async def test_a_finished_emission_is_never_overwritten_by_a_new_intent(store_db):
    """The row is the only record of what went into a branch somebody may already have
    pushed."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)
    async with store_db() as session, session.begin():
        await record_commit(
            session, emission_id=emission_id, commit_oid="e" * 40, at=NOW, reconciled=False
        )

    with pytest.raises(EmissionStoreError, match="already emitted"):
        await _intent(store_db, job_id, packages, list_sha256="d" * 64)


@pytest.mark.parametrize("reconciled", [False, True])
async def test_closing_an_intent_records_how_the_commit_was_learned(store_db, reconciled):
    """`reconciled` is not decoration: False is the oid `emit_branch` returned and True is one
    read back off a branch a crashed run had already cut. They are different claims about how
    much this pipeline watched happen."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)

    async with store_db() as session, session.begin():
        await record_commit(
            session, emission_id=emission_id, commit_oid="e" * 40, at=NOW, reconciled=reconciled
        )

    async with store_db() as session:
        row = await session.get(BranchEmission, emission_id)
        assert row.commit_oid == "e" * 40
        assert row.committed_at == NOW
        assert row.reconciled is reconciled


async def test_recording_the_same_commit_twice_is_a_no_op(store_db):
    """The case the whole write-first ordering exists for: the commit landed and the write
    that records it raced. A retry must not be an error."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)

    for _ in range(2):
        async with store_db() as session, session.begin():
            await record_commit(
                session, emission_id=emission_id, commit_oid="e" * 40, at=NOW, reconciled=False
            )

    async with store_db() as session:
        assert (await session.get(BranchEmission, emission_id)).commit_oid == "e" * 40


async def test_a_second_different_commit_for_one_intent_is_refused(store_db):
    """Two commits answering one intent means something cut a second branch, which is the
    outcome the whole recovery path exists to prevent. Neither is overwritten."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)
    async with store_db() as session, session.begin():
        await record_commit(
            session, emission_id=emission_id, commit_oid="e" * 40, at=NOW, reconciled=False
        )

    async with store_db() as session, session.begin():
        with pytest.raises(EmissionStoreError, match="already carries"):
            await record_commit(
                session, emission_id=emission_id, commit_oid="f" * 40, at=NOW, reconciled=False
            )

    async with store_db() as session:
        assert (await session.get(BranchEmission, emission_id)).commit_oid == "e" * 40


async def test_a_job_that_never_emitted_has_no_record(store_db):
    job_id = await emission_job(store_db)
    async with store_db() as session:
        assert await load_emission(session, job_id=job_id) is None


async def test_forgetting_an_unfinished_intent_takes_its_packages_with_it(store_db):
    """The write that keeps an ordinary failed attempt retryable. Called only once the caller
    has read the clone back and found no branch, which is what separates "this emission left
    nothing behind" from "its outcome is unknown"."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)

    async with store_db() as session, session.begin():
        await forget_intent(session, emission_id=emission_id)

    async with store_db() as session:
        assert await load_emission(session, job_id=job_id) is None
        assert (await session.execute(select(BranchEmissionPackage))).scalars().all() == []


async def test_a_finished_emission_is_never_forgotten(store_db):
    """Its row is the only record of what went into a branch somebody may already have pushed,
    so the delete is refused rather than allowed to erase it."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    emission_id = await _intent(store_db, job_id, packages)
    async with store_db() as session, session.begin():
        await record_commit(
            session, emission_id=emission_id, commit_oid="e" * 40, at=NOW, reconciled=False
        )

    async with store_db() as session, session.begin():
        with pytest.raises(EmissionStoreError, match="not an unfinished intent"):
            await forget_intent(session, emission_id=emission_id)

    async with store_db() as session:
        assert (await load_emission(session, job_id=job_id)).commit_oid == "e" * 40


async def test_forgetting_an_emission_that_is_already_gone_is_a_no_op(store_db):
    """It runs on a failure path unwinding another exception, so it must not raise for a row a
    previous attempt already cleared."""
    async with store_db() as session, session.begin():
        await forget_intent(session, emission_id=987654)


async def test_the_record_carries_the_clone_the_emission_happened_in(store_db):
    """`UPSTREAM_REPO_PATH` is an operator setting and can be repointed between a crash and the
    retry, so the recovery has to be able to tell whether it is looking at the right
    checkout."""
    await seed(store_db, "com.example.one")
    await decide(store_db, "com.example.one", "approve")
    packages = await approved_for(store_db, "pixel")
    job_id = await emission_job(store_db)
    await _intent(store_db, job_id, packages, repo_path="/somewhere/else")

    async with store_db() as session:
        record = await load_emission(session, job_id=job_id)
    assert record.repo_path == "/somewhere/else"
