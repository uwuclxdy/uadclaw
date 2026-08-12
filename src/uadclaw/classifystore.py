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
- **No `removal` reaches a row below that package's floor, whoever wrote it.** The check
  lives in `_upsert`, the one seam every writer here passes, because a check per caller is
  one the next caller forgets — and triage is that next caller: it writes `human:` ratings
  into these columns with no `classify.validate_response` anywhere in its path. It refuses
  rather than clamps, for the reason `classify.py`'s docstring gives at length.
- **Where those two collide, the floor wins and the loss is recorded.** The floor only rises,
  so a `human:` rating can go stale with nobody editing anything. Preserving it then makes
  every later write of that row refuse — the model's next answer, and the fallback park for
  the same package — so the package can never be classified again. `_preserved` drops a human
  `removal` the gate would refuse and `_apply_superseded` writes down that it did.
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.bundle import PackageIdentity

# `_check_description` is reached across the module boundary on purpose. A human edit ships
# into the same `uad_lists.json` a model answer does — the space-next-to-newline rule is a
# literal upstream maintainer request on PR #1180 — so both writers meet the same description
# rules, and a second spelling of a rule somebody else's reviewer enforces is worse than the
# underscore. Making it public belongs to `classify.py`, which this task does not own.
from uadclaw.classify import Classification, Confidence, UadList, _check_description
from uadclaw.ladder import Removal, danger_rank
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
# The dashboard is single-user by construction (`auth.py`), so the surface is the only
# identity there is to record. A username here would be one this app never asked for.
HUMAN_TRIAGE = f"{HUMAN_PREFIX}triage"

# How a human value the floor has overtaken is recorded on the row it left. A provenance key
# rather than a column: the map is already what says who owns each field, the card already
# renders it, and no migration buys a fact that is true of one row for one re-run. The value
# is `rule:`-owned on purpose — `_preserved` keeps only `human:` entries, so the mark expires
# with the supersession instead of outliving it and describing a row it no longer fits.
SUPERSEDED_SUFFIX = "_superseded"
SUPERSEDED_BY = "rule:floor superseded"

# Every spelling of a removal tier a stored floor may legally read. Compared as strings, never
# ranked as strings: `danger_rank` is what orders them (see `_floor_refusal`).
FLOOR_TIERS: frozenset[str] = frozenset(str(removal) for removal in Removal)

# What a triage edit may change, keyed by the UPSTREAM field name the form and the provenance
# map both speak, valued by the column it lands in. Deliberately not every column: `attempts`,
# `usage` and `model` describe the CALL, `bundle_sha256` is the question that was asked, and
# `reasoning_brief` is the model's own note — overwriting that one destroys the evidence a
# reviewer reads the proposal against. The reviewer's own words go on the decision instead.
EDITABLE_FIELDS: dict[str, str] = {
    "description": "description",
    "list": "uad_list",
    "removal": "removal",
    "confidence": "confidence",
}


class ClassificationStoreError(RuntimeError):
    """The corpus is not in a state the classification stage can run against. A
    pipeline-order problem rather than a model failure."""


class BelowFloorError(RuntimeError):
    """A write would have left a `removal` ranking below that package's computed floor.

    Deliberately NOT a `ClassificationStoreError`: that class means "the pipeline ran out of
    order", which a caller can reasonably decide to report and move on from, and a
    safety-critical refusal must never ride inside a class somebody is already catching
    broadly. Carries the package, the two tiers and where the floor came from, because the
    triage form that hit it has to say what to do next.
    """


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


async def _preserved(
    session: AsyncSession, existing: PackageClassification | None
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    """Field values on an existing row that a re-run must not overwrite, and the human-owned
    ones it could not carry.

    A field whose recorded provenance starts with `human:` was set in triage, and the whole
    point of tagging provenance is that a bad model run can be re-run without walking over
    it. Everything else is the model's and is replaced.

    **A preserved value the floor gate would refuse is not preserved.** The floor only rises,
    so a `human:` removal stored under a lower floor goes stale with nobody touching it — and
    carrying it into the next write is what made `_refuse_below_floor` read that write as the
    human's and rank the preserved value rather than the answer offered. A correct model
    answer AT the new floor was then refused, naming the old human value; the fallback park
    for the same package restored the same rating and refused too, and that raise escaped its
    `TaskGroup` and cancelled every sibling package in the job.

    The two rules this module owes collide on exactly that row, and the floor is the one that
    holds: it is the safety invariant, and "a human field survives a re-run" cannot be honoured
    by storing a value this same module refuses. What is given up is the human's `removal` on
    that one row — never their description, list or confidence, none of which the floor
    constrains — and it is recorded rather than dropped quietly, by `_apply_superseded`.

    Returns `(keep, superseded)`, the second keyed by the provenance field with
    `(what to write on the row, why it could not be carried)` beside it.
    """
    if existing is None:
        return {}, {}
    provenance = existing.provenance or {}
    keep: dict[str, Any] = {}
    superseded: dict[str, tuple[str, str]] = {}
    for column, field in HUMAN_OWNED_CANDIDATES:
        owner = provenance.get(field)
        if not (isinstance(owner, str) and owner.startswith(HUMAN_PREFIX)):
            continue
        value = getattr(existing, column)
        if column == "removal" and value is not None:
            refusal = await _floor_refusal(session, existing.package, str(value), owner)
            if refusal is not None:
                superseded[field] = (f"{SUPERSEDED_BY} {owner} {value}", refusal)
                continue
        keep[column] = value
    if keep:
        # Only the human-owned entries survive in the map; the rest is rewritten by the
        # caller's fresh provenance, or a stale model id would outlive the model. A superseded
        # field loses its entry along with its value: a `human:` provenance sitting over a
        # value the human never wrote is worse than either half of it alone, and the next
        # re-run would read it and preserve the model's answer as if a reviewer had set it.
        keep["provenance"] = {
            field: owner
            for field, owner in provenance.items()
            if isinstance(owner, str) and owner.startswith(HUMAN_PREFIX) and field not in superseded
        }
    return keep, superseded


def _apply_superseded(package: str, superseded: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    """Record every human value a write could not carry: a warning for the log and a
    `<field>_superseded` provenance entry for the card.

    Called only by a writer that is actually replacing the field. A park against the SAME
    bundle writes no proposal at all, so nothing there was superseded and nothing claims to
    have been — the human's rating is still on the row it was always on.
    """
    marks: dict[str, str] = {}
    for field, (mark, refusal) in superseded.items():
        marks[f"{field}{SUPERSEDED_SUFFIX}"] = mark
        logger.warning(
            "%s: a human %s was superseded by the removal floor and is not carried into this "
            "write. %s",
            package,
            field,
            refusal,
        )
    return marks


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
    preserved, superseded = await _preserved(session, existing)
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
        "provenance": {
            **dict(classification.provenance),
            **preserved.pop("provenance", {}),
            **_apply_superseded(classification.package, superseded),
        },
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
    preserved, superseded = await _preserved(session, existing)
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
        values["provenance"] = {
            **preserved.pop("provenance", {}),
            **_apply_superseded(package, superseded),
        }
        values["unknown_fields"] = []
        values.update(preserved)
    await _upsert(session, values)
    logger.warning(
        "classification parked: package=%s attempts=%d reason=%s", package, attempts, reason
    )


def _validated_edit(field: str, value: str) -> str:
    """One edited field's value, or a `ValueError` naming what is wrong with it.

    Every field here is an enum upstream already defines except `description`, which goes
    through the same checks a model answer does — see the import comment at the top.
    """
    if field == "description":
        _check_description(value, unknown_fields=())
        return value
    if field == "list":
        return str(UadList(value))
    if field == "removal":
        return str(Removal(value))
    if field == "confidence":
        return str(Confidence(value))
    raise ClassificationStoreError(f"store_human_edit: {field!r} is not an editable field")


async def store_human_edit(
    session: AsyncSession,
    package: str,
    *,
    edits: Mapping[str, str],
    at: datetime,
) -> dict[str, dict[str, Any]]:
    """Apply a reviewer's edits to an existing proposal, and say what actually changed.

    The third writer of `package_classification`, and the one with no `validate_response`
    anywhere in its path — which is why the floor gate lives in `_upsert` rather than in the
    classification stage. Everything here passes that seam.

    Returns `{field: {"from": old, "to": new}}` for the fields whose value actually moved, so
    the caller can log what was replaced. An edit that changes nothing writes nothing: a
    submitted-unchanged form is not a revision, and a log full of empty edits is a log nobody
    reads.

    Provenance is rewritten to `human:` for each changed field and left alone for the rest,
    because `_preserved` reads exactly that map on the next model run. A value written here
    without its provenance is an edit the next re-classification silently erases.
    """
    existing = await session.get(PackageClassification, package)
    if existing is None:
        raise ClassificationStoreError(
            f"store_human_edit: {package} has no classification row to edit. Only a package "
            "the model has already proposed something for can be edited; queue a "
            "classification job for it first."
        )

    changed: dict[str, dict[str, Any]] = {}
    values: dict[str, Any] = {}
    for field, raw in edits.items():
        if field not in EDITABLE_FIELDS:
            raise ClassificationStoreError(
                f"store_human_edit: {field!r} is not an editable field. Editable: "
                f"{', '.join(sorted(EDITABLE_FIELDS))}. `dependencies` and `neededBy` come "
                "from the corpus graph and are not any writer's to set here."
            )
        column = EDITABLE_FIELDS[field]
        current = getattr(existing, column)
        # The unchanged check runs BEFORE validation, because a field the form handed back
        # untouched is not an edit and validating it rejects the row rather than the input.
        # The card prefills the description box with the stored value, and on a row the model
        # declared unknown that value is the `unknown` sentinel — seven characters against a
        # twenty-character floor — so a reviewer who only moved the removal select had their
        # whole submission refused, and the message named `unknown_fields`, which the form has
        # no control for. The second check below stands for a `_validated_edit` that ever
        # normalises: a value that lands back on what is already stored is still not an edit.
        if raw == current:
            continue
        value = _validated_edit(field, raw)
        if value == current:
            continue
        changed[field] = {"from": current, "to": value}
        values[column] = value

    if not changed:
        return {}

    provenance = {**(existing.provenance or {})}
    for field in changed:
        provenance[field] = HUMAN_TRIAGE
        # A mark saying an earlier human value was superseded describes the value it replaced,
        # not the one being written now. Leaving it beside a fresh edit tells the next reviewer
        # their own answer was already overruled.
        provenance.pop(f"{field}{SUPERSEDED_SUFFIX}", None)
    values["provenance"] = provenance
    if "description" in changed:
        # A row carrying a written description while still declaring it unknown asserts two
        # contradictory things, and the `unknown` one is the model's, not the reviewer's.
        values["unknown_fields"] = [
            field for field in (existing.unknown_fields or []) if field != "description"
        ]
    # The NOT NULL columns the INSERT half of the upsert needs. The row exists — this function
    # refuses above when it does not — so the UPDATE half is what runs and these write back
    # what is already there.
    values.update(
        package=package,
        created_at=existing.created_at,
        updated_at=at,
        bundle_sha256=existing.bundle_sha256,
        model=existing.model,
        thinking=existing.thinking,
    )
    await _upsert(session, values)
    logger.info("triage edit: package=%s fields=%s", package, ", ".join(sorted(changed)))
    return changed


async def _floor_refusal(
    session: AsyncSession, package: str, proposed: str, owner: object
) -> str | None:
    """Why storing a `removal` of `proposed` under `owner` for `package` is refused, or None.

    One spelling for the two callers that must agree. `_refuse_below_floor` raises whatever
    this returns, and `_preserved` drops a value it refuses rather than carrying that value
    into a write it would then make impossible — a preserved rating the gate refuses is not a
    safety win, it is a row nothing can ever be written to again. Two spellings of the same
    decision is how the two drifted apart in the first place.

    Ranked with `ladder.danger_rank`, never with `<` on the values: `Removal` is a `StrEnum`
    and `"Expert" < "Recommended"` is True lexicographically, which inverts the whole ladder
    while every test that only checks the extremes stays green.

    **A missing floor is unknown, and unknown is not `Recommended`.** The two cases split by
    who is writing, because they have different amounts of guarantee behind them:

    - a model rating already cleared a floor `ladder.compute_floors` built from the live
      corpus, and `llm_stage` recomputes rather than reading this column precisely because a
      stored floor can lag the corpus. Refusing here would reject a validated answer over a
      missing denormalised copy — a pipeline-order check wearing a safety check's clothes. It
      is logged instead, naming the package: a gate that could not run did not pass. Only
      `_refuse_below_floor` reaches this leg, because `_preserved` asks about `human:` values
      alone, so the warning fires once per write rather than twice.
    - a `human:` rating has no upstream validator anywhere in its path, so a floor it cannot
      be checked against is the entire guarantee absent rather than a duplicate of one. That
      is refused.
    """
    human = isinstance(owner, str) and owner.startswith(HUMAN_PREFIX)
    analysis = await session.get(PackageAnalysis, package)
    stored = analysis.floor if analysis is not None else None
    if stored is None:
        if human:
            return (
                f"{package}: refusing a {owner} removal of {proposed} because the package has "
                "no computed floor. `package_analysis.floor` is NULL, which means the "
                "rule_ladder stage has not run for this package — not that its floor is "
                "Recommended. A human rating is the one this repo has no other validator "
                "for, so it is refused rather than written unchecked. Run a "
                "firmware_analysis job through rule_ladder first."
            )
        logger.warning(
            "floor gate could not run: package=%s removal=%s has no package_analysis.floor "
            "(rule_ladder has not written this package)",
            package,
            proposed,
        )
        return None

    if stored not in FLOOR_TIERS:
        return (
            f"{package}: package_analysis.floor reads {stored!r}, which is not a removal "
            f"tier. A floor nothing can rank against clears nothing, so the write of "
            f"{proposed} is refused rather than allowed past an unreadable bound."
        )

    if danger_rank(Removal(proposed)) < danger_rank(Removal(stored)):
        return (
            f"{package}: refusing to store removal {proposed} under a computed floor of "
            f"{stored} (set by {analysis.floor_rule}). The floor is a lower bound that may be "
            "raised and never lowered, and this is REFUSED rather than raised to the floor: "
            "an answer below it came from misreading the same evidence the description was "
            "written from, so correcting the number would keep the misreading and hide it."
        )
    return None


async def _refuse_below_floor(session: AsyncSession, values: dict[str, Any]) -> None:
    """Refuse a write whose `removal` ranks below the package's stored floor. The one seam
    every writer of that column passes; the decision itself is `_floor_refusal`."""
    proposed = values.get("removal")
    if proposed is None:
        return
    reason = await _floor_refusal(
        session, values["package"], proposed, (values.get("provenance") or {}).get("removal")
    )
    if reason is not None:
        raise BelowFloorError(reason)


async def _upsert(session: AsyncSession, values: dict[str, Any]) -> None:
    await _refuse_below_floor(session, values)
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
