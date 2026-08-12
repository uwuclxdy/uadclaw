"""Corroboration persistence: candidates out, search rows and verdicts in.

The same split `classify.py`/`classifystore.py` draws — `corroborate.py` is pure functions
over value objects and never sees a session, and this module is the only place a verdict or a
search result becomes a row.

Three properties this module owes the rest of the pipeline:

- **It writes `package_corroboration` and `package_search_results` and nothing else.**
  `package_classification` carries what the model proposed, and a corroboration run must not
  be able to touch a proposal even by mistake. Keeping that true is a matter of never giving
  this stage a column there.
- **Search rows are replaced wholesale per package, never folded into what was there.** A
  re-search is a fresh answer to the same query; upserting per URL would leave a result the
  new search no longer returns sitting on the package as if it had, which is the same
  stale-row shape `factstore` recomputes its merged rows to avoid.
- **A failure write always clears the sources.** A row whose `description_sha256` names a
  claim while its `sources` answered a different one is a lie in exactly the shape
  `classifystore.park_package` documents: the hash asserts which question the row answers.
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.classify import UNKNOWN
from uadclaw.corroborate import (
    RETRYABLE,
    Corroboration,
    CorroborationStatus,
    SourceEvidence,
    description_digest,
    provenance_for,
)
from uadclaw.models import PackageClassification, PackageCorroboration, PackageSearchResult

logger = logging.getLogger(__name__)


class CorroborationStoreError(RuntimeError):
    """The corpus is not in a state the corroborate stage can run against. A pipeline-order
    problem rather than a search or model failure."""


async def load_proposals(session: AsyncSession) -> dict[str, str]:
    """Package -> the description a live classification proposed for it, name-ordered.

    A parked row is excluded because it carries no accepted proposal at all; a description of
    exactly `unknown` is excluded because there is no claim in it to corroborate. That second
    exclusion is a real scope decision rather than a filter: the model saying "I cannot tell
    what this does" is a valid, expected outcome, and asking a judge whether the internet
    supports the word "unknown" spends a search query and a model call to learn nothing. Those
    packages reach triage with no corroboration row, which is the truthful state — nothing was
    claimed, so nothing was checked.
    """
    result = await session.execute(
        select(PackageClassification.package, PackageClassification.description)
        .where(
            PackageClassification.parked.is_(False),
            PackageClassification.description.is_not(None),
        )
        .order_by(PackageClassification.package)
    )
    return {row.package: row.description for row in result if row.description != UNKNOWN}


async def existing_verdicts(session: AsyncSession) -> dict[str, tuple[str, str]]:
    """Package -> `(description_sha256, status)` for every row already recorded.

    The idempotence input for the whole stage: a package whose stored hash equals the hash of
    the description we would judge now, in a status that is not retryable, has already been
    asked this exact question.
    """
    result = await session.execute(
        select(
            PackageCorroboration.package,
            PackageCorroboration.description_sha256,
            PackageCorroboration.status,
        )
    )
    return {row.package: (row.description_sha256, row.status) for row in result}


def select_candidates(
    *,
    proposals: Mapping[str, str],
    existing: Mapping[str, tuple[str, str]],
    packages: Sequence[str] = (),
    limit: int | None = None,
) -> list[str]:
    """Which packages this job actually searches and judges.

    A package is a candidate when its corroboration row is missing, stale by description hash,
    or in a retryable failure status (`search_failed`, `judge_failed`). `uncorroborated` is
    deliberately not retryable: it is an answer, and re-asking it every run would spend a
    query and a judge call per package forever.

    Ordered by package name and then truncated, so a capped run is a deterministic prefix
    rather than whatever the database happened to return first — two runs of `limit=10` over
    unchanged descriptions must spend the quota on the same ten packages.
    """
    if packages:
        wanted = [name for name in sorted(set(packages)) if name in proposals]
    else:
        wanted = sorted(proposals)
    retryable = {str(status) for status in RETRYABLE}
    chosen = [
        name
        for name in wanted
        if _is_candidate(name, proposals[name], existing.get(name), retryable)
    ]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def _is_candidate(
    package: str, description: str, row: tuple[str, str] | None, retryable: set[str]
) -> bool:
    if row is None:
        return True
    digest, status = row
    if digest != description_digest(package, description):
        return True
    return status in retryable


async def cached_sources(
    session: AsyncSession, package: str, *, fresh_after: datetime
) -> list[SourceEvidence]:
    """This package's stored search results, but only while they are inside the TTL.

    The whole reason search rows are keyed on the package NAME: inside the window this returns
    the evidence with no HTTP call at all, so re-judging a `judge_failed` package — or
    re-judging after a prompt change rewrote the description — spends zero search quota.

    Freshness is decided on the NEWEST row rather than per row, because a search writes its
    whole result set in one pass: a mixed-age set would mean rows from two different answers to
    one query, and the judge would be shown a merge order that never existed.
    """
    result = await session.execute(
        select(PackageSearchResult)
        .where(PackageSearchResult.package == package)
        .order_by(PackageSearchResult.position)
    )
    rows = list(result.scalars())
    if not rows or max(row.fetched_at for row in rows) < fresh_after:
        return []
    return [
        SourceEvidence(
            url=row.url,
            title=row.title,
            snippet=row.snippet,
            block=row.block,
            position=row.position,
            text=row.page_text,
            fetch_error=row.fetch_error,
        )
        for row in rows
    ]


async def store_search_results(
    session: AsyncSession,
    package: str,
    sources: Sequence[SourceEvidence],
    *,
    at: datetime,
) -> None:
    """Replace this package's search rows with what the search just returned.

    Delete-then-insert rather than an upsert per URL: a result the new search no longer
    returns has to stop being on the package, and an upsert would leave it there dated from a
    previous run, indistinguishable from a result this search actually found.
    """
    await session.execute(delete(PackageSearchResult).where(PackageSearchResult.package == package))
    for source in sources:
        session.add(
            PackageSearchResult(
                package=package,
                position=source.position,
                url=source.url,
                title=source.title,
                snippet=source.snippet,
                block=source.block,
                page_text=source.text,
                fetch_error=source.fetch_error,
                fetched_at=at,
            )
        )


async def store_verdict(
    session: AsyncSession,
    corroboration: Corroboration,
    *,
    model: str | None,
    thinking: bool | None,
    sources: Sequence[Mapping[str, str]],
    usage: Mapping[str, Any],
    attempts: int,
    at: datetime,
) -> None:
    """Write one judged (or code-decided) verdict.

    `sources` is `[{"url", "title"}]` resolved from the search rows rather than from the model,
    so a citation can never be labelled with a title nobody fetched.
    """
    await _upsert(
        session,
        {
            "package": corroboration.package,
            "created_at": at,
            "updated_at": at,
            "description_sha256": corroboration.description_sha256,
            "status": str(corroboration.status),
            "model": model,
            "thinking": thinking,
            "sources": [dict(entry) for entry in sources],
            "reasoning": corroboration.reasoning or None,
            "failure_reason": None,
            "provenance": dict(corroboration.provenance),
            "usage": dict(usage),
            "attempts": attempts,
        },
    )


async def record_failure(
    session: AsyncSession,
    package: str,
    *,
    description: str,
    status: CorroborationStatus,
    reason: str,
    model: str | None = None,
    thinking: bool | None = None,
    usage: Mapping[str, Any] | None = None,
    attempts: int = 0,
    at: datetime,
) -> None:
    """Record that this package could not be searched, or could not be judged.

    A row rather than an exception: the stage is per-package resumable and one flaky request
    must not discard the run's completed work.

    **The sources are always cleared.** A failure write re-points `description_sha256` at the
    description this run was judging, and leaving a previous run's `corroborated` sources
    beside the new hash would make the row assert that those sources support a claim nobody
    checked them against — the same shape `classifystore.park_package` documents for a park
    against new evidence. The status is the failure either way, so there is nothing the old
    sources could still be true of.
    """
    if status not in RETRYABLE:
        raise CorroborationStoreError(
            f"record_failure: {status} is a verdict, not a failure; write it with "
            "`store_verdict` so it carries its sources and provenance"
        )
    await _upsert(
        session,
        {
            "package": package,
            "created_at": at,
            "updated_at": at,
            "description_sha256": description_digest(package, description),
            "status": str(status),
            "model": model,
            "thinking": thinking,
            "sources": [],
            "reasoning": None,
            "failure_reason": reason,
            "provenance": provenance_for(model=None),
            "usage": dict(usage or {}),
            "attempts": attempts,
        },
    )
    logger.warning(
        "corroboration %s: package=%s attempts=%d reason=%s", status, package, attempts, reason
    )


async def _upsert(session: AsyncSession, values: dict[str, Any]) -> None:
    statement = pg_insert(PackageCorroboration).values(values)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[PackageCorroboration.package],
            # `created_at` is excluded so it keeps meaning "first corroborated", and `package`
            # is the conflict key. Everything else named here is this run's.
            set_={
                column: statement.excluded[column]
                for column in values
                if column not in ("package", "created_at")
            },
        )
    )


async def require_proposals(session: AsyncSession) -> dict[str, str]:
    """`load_proposals`, refusing the one empty result that means a stage never ran.

    Two different things produce zero proposals and only one of them is a pipeline-order
    error. An EMPTY `package_classification` means the `llm` stage never ran, and a
    corroborate job over that would succeed silently having checked nothing — refused, the way
    `classifystore.require_queue` refuses an empty additions queue. A `package_classification`
    that HAS rows, none of them a live non-`unknown` proposal, is a legitimate corpus state
    (the model parked everything, or could not tell what anything does) and there is genuinely
    nothing to corroborate, so it returns empty and the caller logs it.
    """
    proposals = await load_proposals(session)
    if proposals:
        return proposals
    classified = await session.scalar(select(PackageClassification.package).limit(1))
    if classified is None:
        raise CorroborationStoreError(
            "require_proposals: package_classification is empty, so the llm stage has never "
            "run against this database and there is no description to corroborate. Run a "
            "classification job first."
        )
    return {}
