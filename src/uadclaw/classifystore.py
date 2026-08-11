"""Classification persistence: candidates out, proposals and parks in.

Mirrors the `corpus.py`/`corpusstore.py` and `facts.py`/`factstore.py` split this repo
already draws — `bundle.py` and `classify.py` are pure functions over value objects and never
see a session, and this module is the only place a proposal becomes a row.

Two properties this module owes the rest of the pipeline:

- **It writes `package_classification` and nothing else.** `package_analysis` carries the
  graph edges and the removal floor and its contract is that nothing on it is ever model
  output. Keeping that true is a matter of never giving the model stage a column there, so
  the disjointness is structural rather than a rule somebody has to remember.
- **A re-run replaces only `llm:`-provenance fields.** The stored `provenance` map says who
  owns each field; a field tagged `human:` (a triage edit, task 9) survives the upsert
  untouched, which is what "a bad model run is re-runnable" actually requires.
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.bundle import PackageIdentity
from uadclaw.classify import Classification
from uadclaw.models import PackageAnalysis, PackageClassification, PackageFact

logger = logging.getLogger(__name__)

# Fields whose stored value is kept when a human owns it. Everything else on the row
# describes the CALL (model, usage, attempts) rather than the answer, so a re-run always
# rewrites it — those are facts about the run that just happened.
# Keyed by the ROW's attribute name, with the provenance map's key beside it: the provenance
# map speaks upstream's vocabulary (`list`) because that is what ships, while the column
# cannot be called `list` (see `models.PackageClassification.uad_list`).
HUMAN_OWNED_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("description", "description"),
    ("uad_list", "list"),
    ("removal", "removal"),
    ("confidence", "confidence"),
    ("reasoning_brief", "reasoning_brief"),
)

HUMAN_PREFIX = "human:"


class ClassificationStoreError(RuntimeError):
    """The corpus is not in a state the classification stage can run against. A
    pipeline-order problem rather than a model failure."""


async def load_identities(session: AsyncSession) -> dict[str, PackageIdentity]:
    """The bundle's identity half for every merged package.

    Read separately from the corpus because `CorpusPackage` is the deterministic core's view
    and carries no label or certificate: the graph, the ladder and the filter never look at
    either, and widening a type three pure modules consume to serve a fourth is how a shared
    type stops meaning anything.
    """
    result = await session.execute(
        select(
            PackageFact.package,
            PackageFact.label,
            PackageFact.label_unresolved,
            PackageFact.cert_issuer,
            PackageFact.version_code,
            PackageFact.has_conflict,
        )
    )
    return {
        row.package: PackageIdentity(
            label=row.label,
            label_unresolved=row.label_unresolved,
            cert_issuer=row.cert_issuer,
            version_code=row.version_code,
            has_conflict=row.has_conflict,
        )
        for row in result
    }


async def queued_packages(session: AsyncSession) -> list[str]:
    """Every package the filter put in the additions queue, name-ordered.

    `queued is true` rather than `is not false`: NULL means the filter stage has not run for
    that package, which is a different thing from "not queued" and must not silently become
    a classification candidate.
    """
    result = await session.execute(
        select(PackageAnalysis.package)
        .where(PackageAnalysis.queued.is_(True))
        .order_by(PackageAnalysis.package)
    )
    return [row[0] for row in result]


async def existing_bundle_hashes(session: AsyncSession) -> dict[str, str]:
    """Package -> the bundle sha256 its current row answers.

    The idempotence key for the whole stage: a package whose stored hash equals the hash of
    the bundle we would build now has already been asked this exact question, and asking it
    again costs money and returns a different answer, because the model is not reproducible.
    """
    result = await session.execute(
        select(PackageClassification.package, PackageClassification.bundle_sha256)
    )
    return {row.package: row.bundle_sha256 for row in result}


def select_candidates(
    *,
    queued: Sequence[str],
    bundles: Mapping[str, Any],
    existing: Mapping[str, str],
    packages: Sequence[str] = (),
    reclassify: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Which packages this job actually calls the API for.

    Ordered by package name and then truncated, so a capped run is a deterministic prefix
    rather than whatever the database happened to return first — two runs of `limit=10`
    against an unchanged corpus must spend money on the same ten packages.
    """
    if packages:
        wanted = [name for name in sorted(set(packages)) if name in bundles]
    else:
        wanted = [name for name in sorted(queued) if name in bundles]
    if not reclassify:
        wanted = [
            name for name in wanted if existing.get(name) != getattr(bundles[name], "sha256", None)
        ]
    if limit is not None:
        wanted = wanted[:limit]
    return wanted


def _preserved(existing: PackageClassification | None) -> dict[str, Any]:
    """Field values on an existing row that a re-run must not overwrite.

    A field whose recorded provenance starts with `human:` was set in triage, and the whole
    point of tagging provenance is that a bad model run can be re-run without walking over
    it. Everything else is the model's and is replaced.
    """
    if existing is None:
        return {}
    provenance = existing.provenance or {}
    keep: dict[str, Any] = {}
    for column, field in HUMAN_OWNED_CANDIDATES:
        owner = provenance.get(field)
        if isinstance(owner, str) and owner.startswith(HUMAN_PREFIX):
            keep[column] = getattr(existing, column)
            keep.setdefault("provenance", dict(provenance))
    if "provenance" in keep:
        # Only the human-owned entries survive in the map; the rest is rewritten by the
        # caller's fresh provenance, or a stale model id would outlive the model.
        keep["provenance"] = {
            field: owner
            for field, owner in provenance.items()
            if isinstance(owner, str) and owner.startswith(HUMAN_PREFIX)
        }
    return keep


async def store_classification(
    session: AsyncSession,
    classification: Classification,
    *,
    model: str,
    thinking: bool,
    usage: Mapping[str, Any],
    attempts: int,
    at: datetime,
) -> None:
    """Write one accepted proposal, keeping any field a human owns."""
    existing = await session.get(PackageClassification, classification.package)
    preserved = _preserved(existing)
    values: dict[str, Any] = {
        "package": classification.package,
        "created_at": at,
        "updated_at": at,
        "bundle_sha256": classification.bundle_sha256,
        "model": model,
        "thinking": thinking,
        "description": classification.description,
        "uad_list": str(classification.list),
        "removal": str(classification.removal),
        "confidence": str(classification.confidence),
        "unknown_fields": list(classification.unknown_fields),
        "reasoning_brief": classification.reasoning_brief,
        "provenance": {**dict(classification.provenance), **preserved.pop("provenance", {})},
        "usage": dict(usage),
        "attempts": attempts,
        "parked": False,
        "parked_reason": None,
    }
    values.update(preserved)
    await _upsert(session, values)


async def park_package(
    session: AsyncSession,
    package: str,
    *,
    bundle_sha256: str,
    model: str,
    thinking: bool,
    reason: str,
    usage: Mapping[str, Any],
    attempts: int,
    at: datetime,
) -> None:
    """Record a package the model could not answer acceptably.

    A row rather than an exception: one package that keeps coming back malformed must not
    cost the other 47, and a park with no row is a package that silently gets retried by
    every future run forever.

    **A previous proposal survives a park only when the park is for the SAME bundle.** The
    row's `bundle_sha256` is a claim about which evidence the proposal on it answers, so
    writing a new hash over an old answer makes the row assert something nobody validated —
    and it is not a cosmetic lie. Measured: a package accepted at `Recommended`, then a
    second device observes it `coreApp="true"`, so its floor becomes `Unsafe`; the model
    keeps answering `Recommended`, gets rejected three times and parks. Re-pointing the old
    answer at the new bundle leaves a stored `removal` of `Recommended` on a row naming a
    bundle whose floor is `Unsafe` — the exact observable this whole repo is shaped to
    prevent, reached without `raise_to_floor` ever being called, and permanent, because the
    matching hash then makes `select_candidates` skip the package forever.

    So a park against new evidence clears the proposal. A park against the same evidence
    keeps it, because there the row's claim is still true: that answer did answer that
    bundle, and this run simply failed to improve on it. A field a HUMAN owns survives
    either way — the clearing is about an unvalidated model answer, and a triage edit was
    never the model's to discard.
    """
    existing = await session.get(PackageClassification, package)
    preserved = _preserved(existing)
    values: dict[str, Any] = {
        "package": package,
        "created_at": at,
        "updated_at": at,
        "bundle_sha256": bundle_sha256,
        "model": model,
        "thinking": thinking,
        "usage": dict(usage),
        "attempts": attempts,
        "parked": True,
        "parked_reason": reason,
    }
    if existing is None or existing.bundle_sha256 != bundle_sha256:
        values["description"] = None
        values["uad_list"] = None
        values["removal"] = None
        values["confidence"] = None
        values["reasoning_brief"] = None
        values["provenance"] = preserved.pop("provenance", {})
        values["unknown_fields"] = []
        values.update(preserved)
    await _upsert(session, values)
    logger.warning(
        "classification parked: package=%s attempts=%d reason=%s", package, attempts, reason
    )


async def _upsert(session: AsyncSession, values: dict[str, Any]) -> None:
    statement = pg_insert(PackageClassification).values(values)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[PackageClassification.package],
            # `created_at` is excluded so it keeps meaning "first classified", and `package`
            # is the conflict key. Everything else named here is this run's.
            set_={
                column: statement.excluded[column]
                for column in values
                if column not in ("package", "created_at")
            },
        )
    )


async def require_queue(session: AsyncSession) -> list[str]:
    """`queued_packages`, refusing an empty queue.

    An empty queue is not "a corpus with nothing worth proposing": the classification stage
    runs after the filter, so zero queued packages means the filter never ran, and a
    classification job over nothing would succeed silently having done nothing at all.
    """
    queue = await queued_packages(session)
    if not queue:
        raise ClassificationStoreError(
            "require_queue: package_analysis has no package with queued = true. The "
            "classification stage runs over what the filter stage put in the additions "
            "queue, so this database either never ran a firmware_analysis job to completion "
            "or every package in it is already upstream. Run a firmware_analysis job first."
        )
    return queue
