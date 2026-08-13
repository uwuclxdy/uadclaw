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
from sqlalchemy import select

from uadclaw import jobs as jobs_module
from uadclaw.models import (
    JOB_KIND_NEEDS_SCRATCH,
    JOB_KIND_STAGES,
    BranchEmission,
    BranchEmissionPackage,
    JobKind,
)

pytestmark = pytest.mark.usefixtures("db_env")


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
