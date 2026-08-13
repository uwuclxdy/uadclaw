"""Branch-emission persistence: the approved set out, the run that shipped it in.

The same split `classify.py`/`classifystore.py` draws — `emission.py` is pure functions over
value objects and never sees a session, and this module is the only place an emission becomes
a row or an approved triage decision becomes an `ApprovedPackage`.

Four properties this module owes the rest of the pipeline, each of which is a rule the repo
already carries rather than a preference:

- **"Current" has one definition and it lives in `triagestore`.** The approved set is the
  packages whose newest decision is `approve` AND was filed against the bundle on the row
  today, which is exactly `triagestore.current_decision` over `triagestore.load_rows`. Both
  are CALLED here rather than restated in a query of this module's own: there were two
  spellings of "is this decision current" once, they disagreed whenever a content-addressed
  bundle hash returned to an earlier value, and collapsing them to one is what fixed it. A
  third spelling in the module that decides what ships upstream is the worst place to put one.
  `edit` and `reopen` are not verdicts and never reach a branch.

- **The read is one snapshot, and here that is not a cosmetic concern.** `load_rows` issues
  two statements and Postgres reads committed per statement, so a decision landing between
  them is visible to one and not the other. On the triage screen that is a stale count; here
  it decides what gets committed into somebody else's repository, so the whole read runs in a
  REPEATABLE READ transaction and every statement in it sees one instant.

- **The vendor comes from `device_key`'s driver prefix and never from the package name.** A
  name-prefix rule swept 70 Xiaomi packages into an OPPO bucket in this repo once already, so
  the grouping goes through `emission.vendor_for`, which reads a field that exists rather than
  a heuristic over one that only looks like it does.

- **This is where a package with no computed floor is refused.** `package_analysis.floor` is
  nullable — NULL means the rule ladder has not run for that package — while
  `ApprovedPackage.floor` is not, because a package with no floor has no bound for its rating
  to sit above. `triagestore` and `views/corpus.py` both spell `floor=... if analysis else
  None`, which is right for a screen that renders a dash and wrong here.

  **What holds it is the `floor is None` leg of `_refusal`, and nothing else** — three tests
  go red when that leg is disabled. The dangerous spelling to know about is
  `floor=row.floor or "Recommended"` at the construction site, because `danger_rank`
  ("Recommended") is 0, so it yields a floor that is structurally present, semantically absent
  and reads exactly like the safe version. It is worth naming and it is NOT what any test
  catches: while `_refusal` stands, no None floor ever reaches that line, so planting it there
  changes no behaviour and the suite stays green (measured — 75 passed). That is the argument
  for the refusal living in `_refusal` rather than at the construction site: a guard placed
  where the bad value cannot arrive is a guard nothing can prove.
"""

import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.emission import ApprovedPackage, group_by_vendor, vendor_for
from uadclaw.models import (
    BranchEmission,
    BranchEmissionPackage,
    PackageAnalysis,
    PackageClassification,
    PackageFact,
)
from uadclaw.triagestore import QueueRow, current_decision, load_rows

logger = logging.getLogger(__name__)

# The one triage action that puts a package on a branch. Named rather than inlined so the
# other four are visibly excluded: `defer` and `reject` are the obvious ones, and `edit` and
# `reopen` are the ones that look like progress — an edited candidate is a revision somebody
# still has to approve, and a reopened one is back in the queue.
APPROVE = "approve"


class EmissionStoreError(RuntimeError):
    """An approved package cannot be built into something shippable, or an emission run cannot
    be recorded.

    A pipeline-order or corpus problem rather than a bug here: a package whose rule ladder
    never ran, a proposal that was parked, an emission row a second write would overwrite.
    Always fatal to the whole batch, for the reason `EmissionError` is — these bytes and the
    disclosure that explains them ship as one unit, so a package silently dropped from the
    batch is a package a human approved that nobody will ever notice did not ship.
    """


@dataclass(frozen=True, slots=True)
class EmissionRecord:
    """One `branch_emission` row, as the stage's recovery path reads it.

    A value object rather than the ORM row for this module's usual reason, plus one specific
    to it: the recovery path reads this after its session is gone and branches on
    `commit_oid is None`, so a lazily-loaded attribute there would raise inside the one code
    path whose whole job is to not get confused about what already happened.
    """

    id: int
    vendor: str
    branch: str
    # The work-tree root the emission actually happened in, so a recovery can tell whether the
    # currently-configured clone is the one that holds its branch. `UPSTREAM_REPO_PATH` is an
    # operator setting and can be repointed between a crash and the retry.
    repo_path: str
    list_path: str
    base_commit: str
    list_sha256: str
    # None means the outcome of this emission is not known: the row was written before the git
    # commit was attempted, and nothing has recorded one since. See `stages.branch_stage`.
    commit_oid: str | None


# --- reading -----------------------------------------------------------------------------


async def _begin_snapshot(session: AsyncSession) -> None:
    """Put this session's transaction on one snapshot for every statement it will run.

    Must be the first thing done on a fresh session: Postgres takes the snapshot at the
    transaction's first statement, and SQLAlchemy cannot change the isolation level of a
    connection that is already inside one.
    """
    await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})


def approved_names(rows: Iterable[QueueRow]) -> list[str]:
    """The packages a human approved against the proposal on the row today, name-ordered.

    `current_decision` and nothing else decides "current". A decision filed against an older
    bundle is history — the package went back in the queue when its evidence changed — so a
    stale approval can never put a package on a branch.
    """
    return sorted(
        row.package
        for row in rows
        if (decision := current_decision(row.decision)) is not None and decision.action == APPROVE
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One approved package as the database has it: whose batch it is in, and whether it can
    ship. `built` and `refusal` are exclusive — exactly one of them is set."""

    package: str
    device_keys: tuple[str, ...]
    built: ApprovedPackage | None
    refusal: str


async def _details(session: AsyncSession, packages: Sequence[str]) -> list[_Candidate]:
    """Every approved package's shippable form, or the reason it has none.

    A refusal is carried rather than raised per row, so the caller can decide which refusals
    are this vendor's problem: a package filed under another vendor must not fail this batch
    for a floor this batch was never going to check. Provenance is read for every row,
    refused or not, because it is what decides whose batch a package is in — and that has to
    be answerable before a refusal can be attributed to anyone.
    """
    statement = (
        select(
            PackageClassification.package,
            PackageClassification.description,
            PackageClassification.uad_list,
            PackageClassification.removal,
            PackageClassification.bundle_sha256,
            PackageClassification.model,
            PackageClassification.parked,
            PackageAnalysis.floor,
            PackageAnalysis.dependencies,
            PackageAnalysis.needed_by,
            PackageFact.devices,
        )
        .outerjoin(PackageAnalysis, PackageAnalysis.package == PackageClassification.package)
        .outerjoin(PackageFact, PackageFact.package == PackageClassification.package)
        .where(PackageClassification.package.in_(list(packages)))
        .order_by(PackageClassification.package)
    )
    candidates: list[_Candidate] = []
    for row in await session.execute(statement):
        keys = tuple(str(key) for key in (row.devices or ()))
        refusal = _refusal(
            parked=row.parked,
            description=row.description,
            uad_list=row.uad_list,
            removal=row.removal,
            floor=row.floor,
        )
        if refusal:
            candidates.append(_Candidate(row.package, keys, None, refusal))
            continue
        candidates.append(
            _Candidate(
                row.package,
                keys,
                ApprovedPackage(
                    package=row.package,
                    uad_list=row.uad_list,
                    description=row.description,
                    dependencies=tuple(str(name) for name in (row.dependencies or ())),
                    needed_by=tuple(str(name) for name in (row.needed_by or ())),
                    # Always empty. `labels` is a dead field upstream — one live entry carries
                    # one — and nothing in this pipeline derives a value for it, so an empty
                    # list is the truthful claim rather than a gap.
                    labels=(),
                    removal=row.removal,
                    floor=row.floor,
                    bundle_sha256=row.bundle_sha256,
                    model=row.model,
                    device_keys=keys,
                ),
                "",
            )
        )
    return candidates


def _refusal(
    *,
    parked: bool,
    description: str | None,
    uad_list: str | None,
    removal: str | None,
    floor: str | None,
) -> str:
    """Why this row cannot ship, or "".

    Ordered so the message names the first thing wrong rather than a consequence of it: a
    parked row has no proposal, and a row with no proposal has nothing to check a floor
    against.
    """
    if parked:
        return (
            "its classification is PARKED, so the model never produced a proposal for it and "
            "there is nothing to ship. Re-run the llm stage for it, or reject it in triage."
        )
    for field, value in (
        ("description", description),
        ("list", uad_list),
        ("removal", removal),
    ):
        if value is None:
            return (
                f"its classification carries no {field}. An approved package with a missing "
                "field was approved against a row that has since been rewritten; re-classify "
                "it and review it again."
            )
    if floor is None:
        # The refusal this module exists to make. A package whose rule ladder never ran has no
        # bound for its rating to sit above, so there is nothing to check `removal` against and
        # no `ApprovedPackage` for it may exist. Defaulting to `Recommended` here would be
        # `danger_rank` 0 — a floor that is present and checks nothing.
        return (
            "no rule-ladder floor has been computed for it, so its removal rating has no lower "
            "bound to be checked against and it cannot be emitted. Run a firmware_analysis job "
            "through rule_ladder over this corpus first."
        )
    return ""


async def load_approved(session: AsyncSession, *, vendor: str) -> tuple[ApprovedPackage, ...]:
    """Every package approved for one vendor, in the order it will be written.

    **Hand this a fresh session with no transaction open**: it puts the whole read on one
    REPEATABLE READ snapshot, and the isolation level cannot be set on a connection that is
    already inside a transaction.

    A package whose device keys name no driver at all fails the whole load rather than being
    skipped, and that refusal is deliberately NOT scoped to the requested vendor: a package
    with no device provenance cannot be attributed to any vendor, so it is not "somebody
    else's batch's problem" — it is a package a human approved that would otherwise never
    ship, with nothing raised. Everything a vendor can be decided for is then filtered first,
    so an unrelated vendor's missing floor cannot fail this batch.
    """
    await _begin_snapshot(session)
    rows = await load_rows(session)
    names = approved_names(rows)
    if not names:
        return ()

    mine: list[ApprovedPackage] = []
    for candidate in await _details(session, names):
        # Raises for a package with no usable provenance, whatever vendor was asked for.
        if vendor_for(candidate.device_keys) != vendor:
            continue
        if candidate.built is None:
            raise EmissionStoreError(
                f"load_approved: {candidate.package} is approved but {candidate.refusal}"
            )
        mine.append(candidate.built)
    if not mine:
        return ()
    logger.info(
        "emission: %d of %d approved package(s) are filed under %s", len(mine), len(names), vendor
    )
    # Through `group_by_vendor` rather than a `sorted()` here, so the order and the
    # duplicate check are the ones `insert_entries` and the PR body were written against.
    return group_by_vendor(mine)[vendor]


async def load_emission(session: AsyncSession, *, job_id: uuid.UUID) -> EmissionRecord | None:
    """This job's emission row, or None when it has never written one."""
    row = (
        await session.execute(select(BranchEmission).where(BranchEmission.job_id == job_id))
    ).scalar_one_or_none()
    if row is None:
        return None
    return EmissionRecord(
        id=row.id,
        vendor=row.vendor,
        branch=row.branch,
        repo_path=row.repo_path,
        list_path=row.list_path,
        base_commit=row.base_commit,
        list_sha256=row.list_sha256,
        commit_oid=row.commit_oid,
    )


# --- writing -----------------------------------------------------------------------------


async def record_intent(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    vendor: str,
    branch: str,
    repo_path: str,
    list_path: str,
    base_commit: str,
    list_sha256: str,
    pipeline_version: str,
    pipeline_commit_sha: str,
    pr_body: str,
    packages: Sequence[ApprovedPackage],
    at: datetime,
) -> int:
    """Record the batch this job is ABOUT to commit, and return the row's id.

    Written and committed before a single git command runs, which is what makes a crash
    between the two recoverable — see `stages.branch_stage` for why that ordering and not the
    other one. `commit_oid` stays NULL until `record_commit`, and that NULL is the whole
    signal.

    A previous intent for this job is REPLACED, children and all: it described a batch whose
    branch was rolled back, and leaving it would make the recovery path compare a live branch
    against a digest from a run that produced nothing. An intent that already carries a commit
    oid is never replaced — that emission happened, and overwriting the row would erase the
    only record of what went into it.
    """
    if not packages:
        raise EmissionStoreError(
            "record_intent: no packages. An emission with an empty batch would record a run "
            "that shipped nothing and cut a branch with no diff on it."
        )
    existing = (
        await session.execute(select(BranchEmission).where(BranchEmission.job_id == job_id))
    ).scalar_one_or_none()
    if existing is not None:
        if existing.commit_oid is not None:
            raise EmissionStoreError(
                f"record_intent: job {job_id} already emitted {existing.branch} at "
                f"{existing.commit_oid}. Overwriting that row would erase the only record of "
                "what went into a branch somebody may already have pushed."
            )
        logger.info(
            "emission: replacing the unfinished intent for job %s (%s)", job_id, existing.branch
        )
        await session.execute(delete(BranchEmission).where(BranchEmission.id == existing.id))
        await session.flush()

    row = BranchEmission(
        job_id=job_id,
        vendor=vendor,
        branch=branch,
        repo_path=repo_path,
        list_path=list_path,
        base_commit=base_commit,
        list_sha256=list_sha256,
        pipeline_version=pipeline_version,
        pipeline_commit_sha=pipeline_commit_sha,
        package_count=len(packages),
        pr_body=pr_body,
        created_at=at,
        reconciled=False,
    )
    session.add(row)
    await session.flush()
    for package in packages:
        session.add(
            BranchEmissionPackage(
                emission_id=row.id,
                package=package.package,
                bundle_sha256=package.bundle_sha256,
                uad_list=package.uad_list,
                removal=package.removal,
                # The number the rating was checked against at emission time, copied rather
                # than joined later: `package_analysis.floor` is current state and the next
                # corpus run moves it, which would silently rewrite the audit trail of a check
                # that already happened.
                floor=package.floor,
            )
        )
    return row.id


async def forget_intent(session: AsyncSession, *, emission_id: int) -> None:
    """Delete an intent whose emission is KNOWN to have rolled back, children and all.

    Called only after the caller has read the clone back and found no branch, which is what
    separates "this emission left nothing behind" from "this emission's outcome is unknown".
    The two are otherwise the same row, and telling them apart is what lets an ordinary failed
    attempt be retried while a crash in the recording window is refused instead of double
    shipping — see `stages.branch_stage`.

    A row carrying a commit oid is never deleted: that emission happened, and its row is the
    only record of what went into a branch somebody may already have pushed.
    """
    row = await session.get(BranchEmission, emission_id)
    if row is None:
        return
    if row.commit_oid is not None:
        raise EmissionStoreError(
            f"forget_intent: emission {emission_id} carries commit {row.commit_oid} on "
            f"{row.branch}, so it is not an unfinished intent and is not deleted. Its row is "
            "the only record of what went into that branch."
        )
    await session.execute(delete(BranchEmission).where(BranchEmission.id == emission_id))
    logger.info("emission %d discarded: its branch was rolled back", emission_id)


async def record_commit(
    session: AsyncSession,
    *,
    emission_id: int,
    commit_oid: str,
    at: datetime,
    reconciled: bool,
) -> None:
    """Close an intent with the commit it produced.

    `reconciled` distinguishes the two ways one arrives: False is the oid `emit_branch`
    returned, True is one a later run read back off a branch a crashed run had already cut.
    They are different claims about how much this pipeline actually watched happen, so the row
    records which.

    Re-recording the same oid is a no-op rather than an error, because a retry after the
    commit landed and the write raced is the case this whole ordering exists for. A DIFFERENT
    oid is refused: two commits answering one intent means something cut a second branch.
    """
    row = await session.get(BranchEmission, emission_id)
    if row is None:
        raise EmissionStoreError(
            f"record_commit: emission {emission_id} no longer exists, so the branch that was "
            "just committed is recorded nowhere. Do not push it until it has been checked by "
            "hand: `git log` in the clone still holds what went onto it."
        )
    if row.commit_oid is not None:
        if row.commit_oid == commit_oid:
            return
        raise EmissionStoreError(
            f"record_commit: emission {emission_id} already carries {row.commit_oid} and was "
            f"handed {commit_oid}. Two commits answering one intent means a second branch was "
            "cut for this job; neither is overwritten. Check the clone by hand."
        )
    row.commit_oid = commit_oid
    row.committed_at = at
    row.reconciled = reconciled
    logger.info(
        "emission %d closed at %s on %s (%s)",
        emission_id,
        commit_oid[:12],
        row.branch,
        "reconciled after a crash" if reconciled else "emitted",
    )
