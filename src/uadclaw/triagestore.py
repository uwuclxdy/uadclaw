"""Triage persistence: the ranked queue, one candidate's card, and the decision log.

Mirrors the `corpus.py`/`corpusstore.py` and `facts.py`/`factstore.py` split — the view
renders what this module returns and never issues a query of its own, and everything here is
a value object rather than an ORM row, so a template cannot touch a lazily-loaded attribute
after its session is gone.

Three properties this module owes the rest of the pipeline:

- **Decisions are an append-only log, never a status column.** Rejections are the only
  measurement of whether the funnel is improving and the raw material for prompt work, so
  what is kept is the series and not the current value. "Current" is derived by
  `current_decision` and nowhere else: the NEWEST decision, and only while its
  `bundle_sha256` matches the proposal on the row today.
- **The queue is ranked by `package_facts.device_count`, descending.** That column is stored
  rather than joined at read time precisely to be this ranking signal.
- **Nothing is filtered out for looking weak.** A conflicted package (134 of 147 shared
  packages carry one, all measured as ordinary signing-key rotation) and an uncorroborated one
  (13.6% corroborate overall, 0 of 12 for `com.android.*`) both reach the queue, flagged. A
  screen that hid either would hide most of the corpus.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.bundle import nearest_entries
from uadclaw.classifystore import ClassificationStoreError, store_human_edit
from uadclaw.models import (
    PackageAnalysis,
    PackageClassification,
    PackageCorroboration,
    PackageFact,
    PackageTriageDecision,
)
from uadclaw.upstream import UpstreamEntry, UpstreamList

logger = logging.getLogger(__name__)

# What a reviewer can do to a proposal. `edit` and `reopen` are deliberately not verdicts:
# an edit is a revision the reviewer still has to approve, and `reopen` is the way back out
# of a verdict — without it a mistaken reject would be permanent, because the log is
# append-only and nothing else returns a package to the queue.
ACTIONS: tuple[str, ...] = ("approve", "reject", "defer", "edit", "reopen")
VERDICTS: frozenset[str] = frozenset({"approve", "reject"})
OPEN_ACTIONS: frozenset[str] = frozenset({"edit", "reopen"})
REASON_REQUIRED: frozenset[str] = frozenset({"reject"})

# The views the screen offers, in the order it shows them. `queue` is the work; the other
# three are how a decided, deferred or unanswerable package stays reachable.
VIEWS: tuple[str, ...] = ("queue", "deferred", "decided", "parked")

# How many existing upstream entries the card shows for comparison. Same default as the
# bundle's style anchors, and the same rule, so the reviewer compares a proposal against what
# the model was shown when it wrote it.
ANCHOR_COUNT = 4


class TriageError(ValueError):
    """A triage action was refused. Bad input from the form (a rejection with no reason, a
    package with no proposal) rather than a bug, so the view renders it beside the control
    that raised it instead of failing the request."""


class ReasonRequired(TriageError):
    """A rejection arrived with no reason. Its own class rather than a string a caller has to
    match on: the screen answers this one in its own words, and every other `TriageError`
    falls back to the message here, which is written for a log."""


@dataclass(frozen=True, slots=True)
class Decision:
    action: str
    reason: str | None
    decided_at: datetime
    # Whether this decision was made about the proposal currently on the row. A decision
    # against an older bundle is history: the package is back in the queue and the card shows
    # what was said last time.
    current: bool


@dataclass(frozen=True, slots=True)
class QueueRow:
    package: str
    device_count: int
    removal: str | None
    floor: str | None
    confidence: str | None
    corroboration: str | None
    has_conflict: bool
    unknown: bool
    parked: bool
    decision: Decision | None


@dataclass(frozen=True, slots=True)
class Candidate:
    """One card. Flat and pre-rendered on purpose: every value here is a string or a tuple of
    strings, so the template does no work and no attribute on it can reach the database."""

    package: str
    device_count: int
    devices: tuple[str, ...]
    evidence: tuple[tuple[str, str], ...]
    has_conflict: bool
    conflicts: tuple[str, ...]

    description: str | None
    uad_list: str | None
    removal: str | None
    confidence: str | None
    reasoning_brief: str | None
    unknown_fields: tuple[str, ...]
    provenance: Mapping[str, str]
    bundle_sha256: str
    model: str
    parked: bool
    parked_reason: str | None

    floor: str | None
    floor_rule: str | None
    floor_reasons: tuple[tuple[str, str, str], ...]
    dependencies: tuple[str, ...]
    needed_by: tuple[str, ...]

    corroboration: str | None
    corroboration_reasoning: str | None
    corroboration_failure: str | None
    sources: tuple[tuple[str, str, bool], ...]

    anchors: tuple[UpstreamEntry, ...]
    decision: Decision | None
    history: tuple[Decision, ...]
    missing: tuple[str, ...]


# --- reading -------------------------------------------------------------------------------


DecisionRow = tuple[str, str, str | None, datetime]


async def _latest_decisions(session: AsyncSession) -> dict[str, DecisionRow]:
    """Package -> its newest decision (bundle, action, reason, when).

    `DISTINCT ON` with the id as the last sort key rather than a `max(decided_at)` group-by:
    two decisions can share a timestamp (a form submitted twice, a clock with second
    resolution), and a grouping keyed on time alone then returns both and picks by luck.
    """
    statement = (
        select(
            PackageTriageDecision.package,
            PackageTriageDecision.bundle_sha256,
            PackageTriageDecision.action,
            PackageTriageDecision.reason,
            PackageTriageDecision.decided_at,
        )
        .distinct(PackageTriageDecision.package)
        .order_by(
            PackageTriageDecision.package,
            PackageTriageDecision.decided_at.desc(),
            PackageTriageDecision.id.desc(),
        )
    )
    result = await session.execute(statement)
    return {
        row.package: (row.bundle_sha256, row.action, row.reason, row.decided_at) for row in result
    }


async def load_rows(session: AsyncSession) -> tuple[QueueRow, ...]:
    """Every package the model has answered, ranked, with its current decision attached.

    Ordered in SQL rather than in Python so the ranking has one spelling: `device_count`
    descending with NULLs LAST, then package name. Nulls last is not decoration — a package
    with no `package_facts` row sorts FIRST under Postgres's default for a descending order,
    which puts the least evidence at the top of a queue ranked by evidence.
    """
    statement = (
        select(
            PackageClassification,
            PackageFact.device_count,
            PackageFact.has_conflict,
            PackageAnalysis.floor,
            PackageCorroboration.status,
        )
        .outerjoin(PackageFact, PackageFact.package == PackageClassification.package)
        .outerjoin(PackageAnalysis, PackageAnalysis.package == PackageClassification.package)
        .outerjoin(
            PackageCorroboration, PackageCorroboration.package == PackageClassification.package
        )
        .order_by(PackageFact.device_count.desc().nulls_last(), PackageClassification.package)
    )
    result = await session.execute(statement)
    latest = await _latest_decisions(session)

    rows: list[QueueRow] = []
    for classification, device_count, has_conflict, floor, corroboration in result:
        rows.append(
            QueueRow(
                package=classification.package,
                device_count=device_count or 0,
                removal=classification.removal,
                floor=floor,
                confidence=classification.confidence,
                corroboration=corroboration,
                has_conflict=bool(has_conflict),
                unknown=bool(classification.unknown_fields),
                parked=classification.parked,
                decision=_decision(
                    latest.get(classification.package), classification.bundle_sha256
                ),
            )
        )
    return tuple(rows)


def _decision(row: DecisionRow | None, bundle_sha256: str) -> Decision | None:
    if row is None:
        return None
    bundle, action, reason, decided_at = row
    return Decision(
        action=action, reason=reason, decided_at=decided_at, current=bundle == bundle_sha256
    )


def current_decision(newest: Decision | None) -> Decision | None:
    """The decision that counts against the proposal on the row today, or None.

    One definition, because there were two and they disagreed. The queue took the NEWEST
    decision and asked whether it was current; the card took the newest decision that IS
    current, walking past newer ones filed against other evidence. A row's `bundle_sha256`
    can return to a value an older decision was filed against — the bundle is
    content-addressed and pure, so a corpus that shrinks back reproduces a hash — and the two
    then answered differently about the same row in the same render.

    The newest is the one that counts. A package whose newest decision was made about
    different evidence is undecided and goes back in the queue, which is the reading that
    loses nothing: a reviewer looks again rather than a stale verdict deciding a proposal
    nobody has read.
    """
    return newest if newest is not None and newest.current else None


def in_view(row: QueueRow, view: str) -> bool:
    """Which list a package belongs to.

    A decision only counts against the proposal it was made about, which is what lets a
    re-classification return a decided package to the queue with nobody deleting anything: the
    bundle hash moves, the old verdict stops being current, and the history is untouched.
    """
    counting = current_decision(row.decision)
    verdict = counting.action if counting else None
    if view == "parked":
        return row.parked
    if row.parked:
        return False
    if view == "deferred":
        return verdict == "defer"
    if view == "decided":
        return verdict in VERDICTS
    if view == "queue":
        return verdict is None or verdict in OPEN_ACTIONS
    raise TriageError(f"load_queue: {view!r} is not a view. Views: {', '.join(VIEWS)}")


async def load_queue(session: AsyncSession, *, view: str = "queue") -> tuple[QueueRow, ...]:
    return tuple(row for row in await load_rows(session) if in_view(row, view))


async def load_counts(session: AsyncSession) -> dict[str, int]:
    """How many packages sit in each view. The screen shows these on its filters, so a
    reviewer can see that a queue reading empty has 12 deferred behind it."""
    rows = await load_rows(session)
    return {view: sum(1 for row in rows if in_view(row, view)) for view in VIEWS}


def _evidence_rows(fact: PackageFact | None) -> tuple[tuple[str, str], ...]:
    """The `package_facts` signals, in the order a reviewer reads them.

    Presence signals first (what the package IS), then integration (what it is wired into),
    then the list-valued ones. A signal that is absent still gets its row with a dash, because
    "this package does not declare a shared uid" and "nobody looked" are different answers and
    a missing row reads as the second.
    """
    if fact is None:
        return ()

    def joined(values: Sequence[Any], limit: int = 8) -> str:
        items = [str(value) for value in values]
        if not items:
            return "-"
        if len(items) > limit:
            return f"{', '.join(items[:limit])} (+{len(items) - limit} more)"
        return ", ".join(items)

    flags = [
        name
        for name, value in (
            ("coreApp", fact.core_app),
            ("persistent", fact.persistent),
            ("priv-app", fact.priv_app),
            ("input method", fact.is_input_method),
            ("device admin", fact.is_device_admin),
            ("accessibility service", fact.is_accessibility_service),
            ("carrier service", fact.is_carrier_service),
            ("no code", not fact.has_code),
        )
        if value
    ]
    return (
        ("label", fact.label + (" (unresolved)" if fact.label_unresolved else "")),
        ("devices", joined(fact.devices)),
        ("partitions", joined(fact.partitions)),
        ("flags", ", ".join(flags) if flags else "-"),
        ("shared uid", fact.shared_user_id or "-"),
        ("cert issuer", fact.cert_issuer or "-"),
        ("version code", str(fact.version_code) if fact.version_code is not None else "-"),
        ("overlay target", fact.overlay_target or "-"),
        ("provides libraries", joined(fact.libraries)),
        ("requires libraries", joined(fact.uses_libraries_required)),
        ("provider authorities", joined(fact.provider_authorities)),
        ("declared queries", joined(fact.queries_packages)),
        ("protected broadcasts", joined(fact.protected_broadcasts, limit=4)),
        ("intent filters", joined(fact.intent_filters, limit=4)),
    )


def _conflict_lines(fact: PackageFact | None) -> tuple[str, ...]:
    """Each recorded disagreement as one line. Rendered, never filtered on."""
    if fact is None:
        return ()
    lines: list[str] = []
    for item in fact.conflicts or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field", "?"))
        values = item.get("values")
        rendered = "; ".join(str(value) for value in values) if isinstance(values, list) else "?"
        lines.append(f"{field}: {rendered}")
    return tuple(lines)


def _sources(corroboration: PackageCorroboration | None) -> tuple[tuple[str, str, bool], ...]:
    """The cited urls, each with whether the card may turn it into a link.

    A stored url reached this row through a search result a third party chose, and an `href`
    is a scheme away from being executable — `javascript:` and `data:` both run from one, and
    escaping does nothing about it because the value is not markup. So the scheme is checked
    here rather than in the template: http and https become anchors, anything else is shown as
    text and labelled. Dropping it instead would hide a citation the judge was handed, which is
    the one thing the corroboration gate exists to make visible.
    """
    items = tuple(
        (str(item.get("url", "")), str(item.get("title", "")))
        for item in (corroboration.sources if corroboration else [])
        if isinstance(item, dict) and item.get("url")
    )
    return tuple((url, title, url.startswith(("http://", "https://"))) for url, title in items)


def _missing(
    fact: PackageFact | None,
    analysis: PackageAnalysis | None,
    classification: PackageClassification,
    corroboration: PackageCorroboration | None,
    anchors: Sequence[UpstreamEntry],
) -> tuple[str, ...]:
    """The evidence classes this card does NOT have.

    The partial state, named rather than hidden. Every one of these is an ordinary outcome
    over the real corpus — corroboration lands at 13.6% — so a card that quietly rendered the
    same layout with three of its six sections empty would read as complete and tell the
    reviewer nothing about what they are deciding without.
    """
    missing: list[str] = []
    if fact is None:
        missing.append("device facts")
    if analysis is None or analysis.floor is None:
        missing.append("removal floor")
    if corroboration is None or corroboration.status != "corroborated":
        missing.append("corroboration")
    if not anchors:
        missing.append("upstream neighbours")
    if classification.unknown_fields:
        unknown = ", ".join(str(field) for field in classification.unknown_fields)
        missing.append(f"model answer ({unknown})")
    return tuple(missing)


async def load_candidate(
    session: AsyncSession, package: str, *, upstream: UpstreamList | None = None
) -> Candidate:
    """One package's whole card, or a `TriageError` naming the package.

    The anchors come from `bundle.nearest_entries`, the same function and the same rule
    (longest shared dotted prefix) the classification bundle used, so the entries the reviewer
    compares against are the entries the model was shown.
    """
    classification = await session.get(PackageClassification, package)
    if classification is None:
        raise TriageError(
            f"load_candidate: {package} has no classification row, so there is nothing to "
            "triage for it. It is either not in the additions queue or no classification job "
            "has answered it yet."
        )
    fact = await session.get(PackageFact, package)
    analysis = await session.get(PackageAnalysis, package)
    corroboration = await session.get(PackageCorroboration, package)

    anchors = nearest_entries(package, upstream, limit=ANCHOR_COUNT) if upstream is not None else ()
    result = await session.execute(
        select(PackageTriageDecision)
        .where(PackageTriageDecision.package == package)
        .order_by(PackageTriageDecision.decided_at.desc(), PackageTriageDecision.id.desc())
    )
    history = tuple(
        Decision(
            action=row.action,
            reason=row.reason,
            decided_at=row.decided_at,
            current=row.bundle_sha256 == classification.bundle_sha256,
        )
        for row in result.scalars()
    )
    reasons = tuple(
        (str(item.get("rule", "?")), str(item.get("floor", "?")), str(item.get("detail", "")))
        for item in (analysis.floor_reasons if analysis else [])
        if isinstance(item, dict)
    )

    return Candidate(
        package=package,
        device_count=fact.device_count if fact else 0,
        devices=tuple(str(device) for device in (fact.devices if fact else [])),
        evidence=_evidence_rows(fact),
        has_conflict=bool(fact.has_conflict) if fact else False,
        conflicts=_conflict_lines(fact),
        description=classification.description,
        uad_list=classification.uad_list,
        removal=classification.removal,
        confidence=classification.confidence,
        reasoning_brief=classification.reasoning_brief,
        unknown_fields=tuple(str(field) for field in classification.unknown_fields or []),
        provenance={
            str(key): str(value) for key, value in (classification.provenance or {}).items()
        },
        bundle_sha256=classification.bundle_sha256,
        model=classification.model,
        parked=classification.parked,
        parked_reason=classification.parked_reason,
        floor=analysis.floor if analysis else None,
        floor_rule=analysis.floor_rule if analysis else None,
        floor_reasons=reasons,
        dependencies=tuple(str(item) for item in (analysis.dependencies if analysis else [])),
        needed_by=tuple(str(item) for item in (analysis.needed_by if analysis else [])),
        corroboration=corroboration.status if corroboration else None,
        corroboration_reasoning=corroboration.reasoning if corroboration else None,
        corroboration_failure=corroboration.failure_reason if corroboration else None,
        sources=_sources(corroboration),
        anchors=anchors,
        decision=current_decision(history[0] if history else None),
        history=history,
        missing=_missing(fact, analysis, classification, corroboration, anchors),
    )


# --- writing -------------------------------------------------------------------------------


async def record_decision(
    session: AsyncSession,
    *,
    package: str,
    bundle_sha256: str,
    action: str,
    reason: str | None = None,
    edited_fields: Mapping[str, Any] | None = None,
    at: datetime,
) -> None:
    """Append one decision. Nothing in this module ever updates or deletes one."""
    if action not in ACTIONS:
        raise TriageError(
            f"record_decision: {action!r} is not an action. Actions: {', '.join(ACTIONS)}."
        )
    cleaned = (reason or "").strip()
    if action in REASON_REQUIRED and not cleaned:
        raise ReasonRequired(
            f"record_decision: a {action} needs a reason. It is the only measurement of "
            "whether the funnel is improving and the raw material for prompt work, so a "
            "rejection nobody can read is worse than one that was never recorded."
        )
    session.add(
        PackageTriageDecision(
            package=package,
            bundle_sha256=bundle_sha256,
            action=action,
            reason=cleaned or None,
            edited_fields=dict(edited_fields or {}),
            decided_at=at,
        )
    )
    logger.info("triage decision: package=%s action=%s", package, action)


async def decide(
    session: AsyncSession,
    *,
    package: str,
    action: str,
    reason: str | None = None,
    at: datetime,
) -> None:
    """Record a verdict against the proposal currently on the row.

    The bundle hash is read here rather than taken from the form, so a decision cannot be
    filed against a proposal that has been replaced since the card was rendered — that is the
    one field a stale tab would otherwise carry back and silently make current.
    """
    classification = await session.get(PackageClassification, package)
    if classification is None:
        raise TriageError(f"decide: {package} has no classification row to decide on.")
    await record_decision(
        session,
        package=package,
        bundle_sha256=classification.bundle_sha256,
        action=action,
        reason=reason,
        at=at,
    )


async def apply_edit(
    session: AsyncSession,
    *,
    package: str,
    edits: Mapping[str, str],
    at: datetime,
) -> dict[str, dict[str, Any]]:
    """Write a reviewer's edits and log what they replaced.

    The write goes through `classifystore`, which is the only module that touches
    `package_classification` and the one that carries the floor gate — an edited `removal`
    below its floor is refused there, on the same seam the model's writes pass.
    """
    try:
        changed = await store_human_edit(session, package, edits=edits, at=at)
    except ClassificationStoreError as exc:
        # Bad form input rather than a broken pipeline: re-raised as the class the view
        # renders beside the field, keeping the message that names what to fix.
        raise TriageError(str(exc)) from exc
    if not changed:
        return {}
    classification = await session.get(PackageClassification, package)
    assert classification is not None  # noqa: S101 - store_human_edit raises when it is not
    await record_decision(
        session,
        package=package,
        bundle_sha256=classification.bundle_sha256,
        action="edit",
        edited_fields=changed,
        at=at,
    )
    return changed
