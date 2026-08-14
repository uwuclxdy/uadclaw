"""The deterministic core's persistence: `package_facts` in, `package_analysis` out.

Mirrors the split `facts.py`/`factstore.py` already draws. `corpus.py`, `ladder.py` and
`filters.py` are pure functions over value objects and never see a session; this module is the
only place that turns an ORM row into a `CorpusPackage` and writes a verdict back.

The three stages write disjoint column sets onto the same row, so each upsert names only its
own columns: re-running the filter must not blank a floor, and re-running the ladder must not
blank the edges. Everything is recomputed from the corpus rather than folded into what was
there, which makes a second run over an unchanged corpus produce an identical row — the same
property `factstore.recompute_package_facts` exists for.
"""

import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from uadclaw.corpus import (
    DIALER_ACTION,
    HOME_ACTION,
    HOME_CATEGORY,
    SMS_DELIVER_ACTION,
    CorpusGraph,
    CorpusPackage,
)
from uadclaw.etcconfig import ConfigInputs, merge_config_inputs
from uadclaw.filters import FilterVerdict
from uadclaw.ladder import RemovalFloor
from uadclaw.models import DeviceScan, PackageAnalysis, PackageFact
from uadclaw.upstream import UpstreamList

logger = logging.getLogger(__name__)

# Rows per INSERT. Postgres binds one parameter per column per row and caps a statement at
# 65535 of them, so a corpus of a few thousand packages must not go up as one statement.
UPSERT_CHUNK_ROWS = 500


class CorpusStoreError(RuntimeError):
    """The corpus is not in a state this stage can run against. A pipeline-order problem
    rather than bad firmware."""


def _library_names(values: Iterable[Any]) -> tuple[str, ...]:
    """`[{"name": …, "version": …}]` (as `factstore` writes it) down to names."""
    names: list[str] = []
    for value in values or ():
        name = value.get("name") if isinstance(value, Mapping) else value
        if name and str(name) not in names:
            names.append(str(name))
    return tuple(names)


def _handles(filters: Iterable[Any], action: str, category: str | None = None) -> bool:
    for item in filters or ():
        if not isinstance(item, Mapping):
            continue
        if action not in (item.get("actions") or ()):
            continue
        if category is None or category in (item.get("categories") or ()):
            return True
    return False


def corpus_package(row: PackageFact) -> CorpusPackage:
    """One merged fact row as the deterministic core sees it."""
    filters = row.intent_filters or []
    return CorpusPackage(
        package=row.package,
        devices=tuple(str(device) for device in (row.devices or ())),
        partitions=tuple(str(partition) for partition in (row.partitions or ())),
        device_count=row.device_count,
        priv_app=row.priv_app,
        core_app=row.core_app,
        persistent=row.persistent,
        shared_user_id=row.shared_user_id,
        overlay_target=row.overlay_target,
        libraries=_library_names(row.libraries),
        static_libraries=_library_names(row.static_libraries),
        uses_libraries_required=tuple(str(name) for name in (row.uses_libraries_required or ())),
        uses_libraries_optional=tuple(str(name) for name in (row.uses_libraries_optional or ())),
        queries_packages=tuple(str(name) for name in (row.queries_packages or ())),
        provider_authorities=tuple(str(name) for name in (row.provider_authorities or ())),
        content_uri_authorities=tuple(str(name) for name in (row.content_uri_authorities or ())),
        is_input_method=row.is_input_method,
        handles_home=_handles(filters, HOME_ACTION, HOME_CATEGORY),
        handles_dialer=_handles(filters, DIALER_ACTION),
        handles_sms_deliver=_handles(filters, SMS_DELIVER_ACTION),
    )


async def load_corpus(session: AsyncSession) -> list[CorpusPackage]:
    """Every merged package, ordered by name so every stage sees the same corpus.

    `icon_bytes` is deferred per query rather than on the column: `corpus_package` reads
    twenty signals and no artwork, and this loads the WHOLE corpus at once. Measured against a
    throwaway database holding the real 312-package Pixel corpus (64 of them carrying an
    icon), the column is 283,597 of the 822,206 bytes of row payload this query would
    otherwise transfer — **34.5% of it**, for something nothing downstream of here reads.

    Per query and not `deferred=True` on the mapping, because a column-level default would
    turn `row.icon_bytes` into a lazy load that raises inside an async session for whoever
    touches it next.
    """
    result = await session.execute(
        select(PackageFact).options(defer(PackageFact.icon_bytes)).order_by(PackageFact.package)
    )
    return [corpus_package(row) for row in result.scalars()]


async def _upsert(session: AsyncSession, rows: Sequence[dict[str, Any]]) -> None:
    for start in range(0, len(rows), UPSERT_CHUNK_ROWS):
        chunk = rows[start : start + UPSERT_CHUNK_ROWS]
        statement = pg_insert(PackageAnalysis).values(chunk)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[PackageAnalysis.package],
                # Only this stage's columns: a stage that listed every column would blank the
                # other two stages' output every time it ran.
                set_={
                    column: statement.excluded[column] for column in chunk[0] if column != "package"
                },
            )
        )


async def store_graph(
    session: AsyncSession, corpus: Sequence[CorpusPackage], graph: CorpusGraph, *, at: datetime
) -> int:
    """Write the emitted edges and the recorded-not-emitted evidence for every package."""
    rows = [
        {
            "package": item.package,
            "updated_at": at,
            "dependencies": list(graph.dependencies_of(item.package)),
            "needed_by": list(graph.needed_by(item.package)),
            "edges": [edge.as_json() for edge in graph.edges_for(item.package)],
            "evidence": graph.evidence[item.package].as_json()
            if item.package in graph.evidence
            else {},
        }
        for item in corpus
    ]
    await _upsert(session, rows)
    return len(rows)


async def store_filter_verdicts(
    session: AsyncSession,
    verdicts: Mapping[str, FilterVerdict],
    *,
    upstream: UpstreamList,
    at: datetime,
) -> int:
    """Write queue membership, with the bytes of `uad_lists.json` that decided it."""
    provenance = upstream.provenance()
    rows = [
        {
            "package": package,
            "updated_at": at,
            "upstream_present": verdict is FilterVerdict.ALREADY_UPSTREAM,
            "queued": verdict is FilterVerdict.QUEUED,
            "filter_verdict": str(verdict),
            "upstream_provenance": provenance,
        }
        for package, verdict in sorted(verdicts.items())
    ]
    await _upsert(session, rows)
    return len(rows)


async def store_floors(
    session: AsyncSession, floors: Mapping[str, RemovalFloor], *, at: datetime
) -> int:
    """Write each package's floor and every rule behind it."""
    rows = [
        {
            "package": package,
            "updated_at": at,
            "floor": str(floor.floor),
            "floor_rule": str(floor.rule),
            "floor_reasons": [item.as_json() for item in floor.fired],
            "privapp_allowlisted": floor.privapp_allowlisted,
            "privapp_permission_count": floor.privapp_permission_count,
        }
        for package, floor in sorted(floors.items())
    ]
    await _upsert(session, rows)
    return len(rows)


async def record_config_inputs(
    session: AsyncSession, *, job_id: uuid.UUID | None, config: ConfigInputs
) -> None:
    """Park this job's parsed `/etc` inputs on its device-scan row.

    Scratch dies with the job while the ladder is corpus-wide, so a device's allowlist has to
    outlive its own scratch directory or a later device's ladder run silently loses it.
    """
    if job_id is None:
        return
    result = await session.execute(
        update(DeviceScan).where(DeviceScan.job_id == job_id).values(config_inputs=config.as_json())
    )
    if result.rowcount == 0:
        raise CorpusStoreError(
            f"record_config_inputs: job {job_id} has no device_scans row, so extract_facts "
            "never ran for it. The deterministic core runs over the corpus that stage "
            "produces; requeue the job from the acquire stage rather than analysing nothing."
        )


async def load_config_inputs(session: AsyncSession) -> ConfigInputs:
    """Every device's recorded `/etc` inputs, unioned.

    Union rather than "this job's", because a floor is a property of the package across the
    corpus: a static role held on one device and a privileged allowlist entry on another both
    have to reach the same package's rating.
    """
    result = await session.execute(select(DeviceScan.config_inputs))
    return merge_config_inputs(
        ConfigInputs.from_json(payload) for payload in result.scalars() if payload
    )


async def require_corpus(session: AsyncSession) -> list[CorpusPackage]:
    """`load_corpus`, refusing an empty one.

    An empty corpus is not "a corpus with nothing interesting in it": every stage here runs
    after `extract_facts`, so zero packages means the facts never landed, and a graph over
    nothing, a filter over nothing and a ladder over nothing all succeed silently.
    """
    corpus = await load_corpus(session)
    if not corpus:
        raise CorpusStoreError(
            "require_corpus: package_facts is empty. The corpus graph, filter and rule ladder "
            "all run over the merged facts extract_facts writes, so this job either skipped "
            "that stage or its facts were never committed; requeue it from the acquire stage."
        )
    return corpus
