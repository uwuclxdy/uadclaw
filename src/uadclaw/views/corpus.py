"""The corpus screen: what the deterministic core concluded, `package_facts` joined to
`package_analysis`. Read-only — see `docs/pipeline-design.md` §3-6 for what each stage writes.

Three things this view has to get right that are easy to get backwards, all measured and all
recorded on `models.py`/`ladder.py`:

- `has_conflict` is rendered, never filtered out by default. 134 of 147 shared packages carry
  it on the real corpus and every measured one is ordinary signing-key rotation, not a
  suspicion score.
- Sorting or filtering by `floor` never compares the raw string column. `Removal` is a
  `StrEnum` and `"Expert" < "Recommended"` lexicographically, which would invert the ladder in
  the UI exactly the way `ladder.py` warns against doing in code. `_FLOOR_RANK` maps the
  column through `ladder.danger_rank` before any `ORDER BY` touches it.
- `queued`/`upstream_present` are nullable until the `filter` stage has run for a package, and
  NULL is a third state ("not filtered yet") rather than a synonym for `False`. The tri-state
  filters below reach it with `.is_(None)`, never with a falsy check, and the row renderer
  keeps it a visually distinct badge.

A package name is bytes out of downloaded firmware, never bytes this pipeline chose, so every
place one is written into an `href` goes through `urlencode` in the templates — the record's
identity is not the record's path (see the repo's own rule on this).
"""

import logging
from dataclasses import dataclass
from math import ceil
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from sqlalchemy import Select, case, func, select

from uadclaw import web
from uadclaw.db import get_session_factory
from uadclaw.ladder import DANGER_ORDER, Removal, danger_rank
from uadclaw.models import PackageAnalysis, PackageFact

logger = logging.getLogger(__name__)

router = APIRouter()

# 393 rows today and it grows per device scanned (module map: `device_count` is the triage
# ranking signal, and this corpus only ever grows). Never render the table unbounded.
PAGE_SIZE = 50

# NULL/no floor yet ranks below Recommended (`else_=-1`) so an unanalyzed package sorts as the
# least dangerous rather than colliding with a real tier or crashing the ORDER BY.
_FLOOR_RANK = case(
    *[(PackageAnalysis.floor == str(removal), danger_rank(removal)) for removal in DANGER_ORDER],
    else_=-1,
)

SORTS: dict[str, Any] = {
    "device_count": PackageFact.device_count,
    "package": PackageFact.package,
    "floor": _FLOOR_RANK,
}
DEFAULT_SORT = "device_count"
DEFAULT_DIR: dict[str, str] = {"device_count": "desc", "package": "asc", "floor": "desc"}

# Query-string spelling for the queued/upstream tri-state filters. "null" is not a stand-in for
# "unset" — it is the one value that reaches `.is_(None)`, so a filter can ask for "not
# filtered yet" as its own answer instead of it being unreachable.
TRISTATE: dict[str, bool | None] = {"true": True, "false": False, "null": None}

FLOOR_TAG_CLASS: dict[str, str] = {
    str(Removal.RECOMMENDED): "tag-success",
    str(Removal.ADVANCED): "tag-info",
    str(Removal.EXPERT): "tag-warning",
    str(Removal.UNSAFE): "tag-danger",
}


@dataclass(frozen=True, slots=True)
class CorpusRow:
    """One list-view row. A plain value rather than the joined ORM rows, so the template never
    has to know whether a package has no `PackageAnalysis` row yet or one with a null column —
    both collapse to `None` here, which is the fact the UI actually needs to render."""

    package: str
    device_count: int
    floor: str | None
    floor_rule: str | None
    queued: bool | None
    upstream_present: bool | None
    has_conflict: bool


def _toggle_dir(column: str, sort: str, direction: str) -> str:
    """The direction a header click should switch to: flip if already sorted on this column,
    otherwise that column's own sensible default (e.g. floor defaults to most-dangerous-first,
    not the ordering that happens to be alphabetically first)."""
    if sort != column:
        return DEFAULT_DIR[column]
    return "desc" if direction == "asc" else "asc"


def _corpus_url(
    *,
    q: str,
    floor_filter: str,
    queued_filter: str,
    upstream_filter: str,
    sort: str,
    direction: str,
    page: int,
) -> str:
    params = {
        "q": q,
        "floor": floor_filter,
        "queued": queued_filter,
        "upstream": upstream_filter,
        "sort": sort,
        "dir": direction,
        "page": str(page),
    }
    kept = {
        key: value
        for key, value in params.items()
        if value and not (key == "page" and value == "1")
    }
    query = urlencode(kept)
    return f"/corpus?{query}" if query else "/corpus"


def _apply_filters(
    stmt: Select[Any], *, q: str, floor_filter: str, queued_filter: str, upstream_filter: str
) -> Select[Any]:
    if q:
        stmt = stmt.where(PackageFact.package.ilike(f"%{q}%"))
    if floor_filter in {str(removal) for removal in Removal}:
        stmt = stmt.where(PackageAnalysis.floor == floor_filter)
    if queued_filter in TRISTATE:
        stmt = stmt.where(PackageAnalysis.queued.is_(TRISTATE[queued_filter]))
    if upstream_filter in TRISTATE:
        stmt = stmt.where(PackageAnalysis.upstream_present.is_(TRISTATE[upstream_filter]))
    return stmt


def _joined(stmt: Select[Any]) -> Select[Any]:
    return stmt.select_from(PackageFact).join(
        PackageAnalysis, PackageAnalysis.package == PackageFact.package, isouter=True
    )


@router.get("/corpus")
async def corpus_screen(request: Request):
    params = request.query_params
    q = (params.get("q") or "").strip()
    # Validated at the boundary rather than ignored inside `_apply_filters`. An unrecognised
    # value used to fall through with no filter applied while the context still carried it, so
    # the operator got the whole corpus, an active "clear filters" button and a select showing
    # nothing selected — three parts of one screen disagreeing about whether a filter is on.
    # Same helper the jobs screen uses, so the two answer this in one voice.
    floor_filter, floor_warning = web.valid_filter(
        params.get("floor") or "", (str(removal) for removal in Removal)
    )
    queued_filter, queued_warning = web.valid_filter(params.get("queued") or "", TRISTATE)
    upstream_filter, upstream_warning = web.valid_filter(params.get("upstream") or "", TRISTATE)

    sort = params.get("sort") or DEFAULT_SORT
    if sort not in SORTS:
        sort = DEFAULT_SORT
    direction = params.get("dir") or ""
    if direction not in ("asc", "desc"):
        direction = DEFAULT_DIR[sort]
    try:
        requested_page = max(1, int(params.get("page") or "1"))
    except ValueError:
        requested_page = 1

    context: dict[str, Any] = {
        "active_nav": "corpus",
        "q": q,
        "floor_filter": floor_filter,
        "queued_filter": queued_filter,
        "upstream_filter": upstream_filter,
        "sort": sort,
        "dir": direction,
        "floors": [str(removal) for removal in Removal],
        "FLOOR_TAG_CLASS": FLOOR_TAG_CLASS,
        "filter_warning": floor_warning or queued_warning or upstream_warning,
    }
    filters_active = bool(q or floor_filter or queued_filter or upstream_filter)
    context["filters_active"] = filters_active

    filter_kwargs = {
        "q": q,
        "floor_filter": floor_filter,
        "queued_filter": queued_filter,
        "upstream_filter": upstream_filter,
    }

    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            count_stmt = _apply_filters(
                _joined(select(func.count(PackageFact.package))), **filter_kwargs
            )
            total = (await session.execute(count_stmt)).scalar_one()

            total_pages = max(1, ceil(total / PAGE_SIZE))
            page = min(requested_page, total_pages)

            order_col = SORTS[sort]
            order = order_col.desc() if direction == "desc" else order_col.asc()
            rows_stmt = (
                _apply_filters(_joined(select(PackageFact, PackageAnalysis)), **filter_kwargs)
                .order_by(order, PackageFact.package.asc())
                .offset((page - 1) * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
            result = (await session.execute(rows_stmt)).all()
        except web.DB_UNREACHABLE:
            # The shared tuple rather than a bare `Exception`, which is what this used to be:
            # a bug inside `_apply_filters` would have reached the operator as "the database
            # is not answering", which is a lie a screen tells and a health probe does not.
            # Not re-raised: a reachability failure here must read as "could not load", never
            # as an empty corpus, so it renders its own distinct state rather than a 500 the
            # router decorates blank. Logged in full first.
            logger.exception("corpus list query failed")
            context["error"] = True
            return web.page(request, "corpus.html", context)

    rows = [
        CorpusRow(
            package=fact.package,
            device_count=fact.device_count,
            floor=analysis.floor if analysis is not None else None,
            floor_rule=analysis.floor_rule if analysis is not None else None,
            queued=analysis.queued if analysis is not None else None,
            upstream_present=analysis.upstream_present if analysis is not None else None,
            has_conflict=fact.has_conflict,
        )
        for fact, analysis in result
    ]

    context.update(
        {
            "rows": rows,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "range_start": (page - 1) * PAGE_SIZE + 1 if rows else 0,
            "range_end": (page - 1) * PAGE_SIZE + len(rows),
            "prev_href": (
                _corpus_url(sort=sort, direction=direction, page=page - 1, **filter_kwargs)
                if page > 1
                else None
            ),
            "next_href": (
                _corpus_url(sort=sort, direction=direction, page=page + 1, **filter_kwargs)
                if page < total_pages
                else None
            ),
            "sort_links": {
                col: _corpus_url(
                    sort=col,
                    direction=_toggle_dir(col, sort, direction),
                    page=1,
                    **filter_kwargs,
                )
                for col in SORTS
            },
        }
    )
    return web.page(request, "corpus.html", context)


@router.get("/corpus/{package}")
async def corpus_detail(request: Request, package: str):
    context: dict[str, Any] = {
        "active_nav": "corpus",
        "package": package,
        "FLOOR_TAG_CLASS": FLOOR_TAG_CLASS,
    }
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            fact = await session.get(PackageFact, package)
            analysis = await session.get(PackageAnalysis, package)
        except web.DB_UNREACHABLE:
            # Same catch as the list view above, for the same reason.
            logger.exception("corpus detail query failed for %s", package)
            context["error"] = True
            return web.page(request, "corpus_detail.html", context)

    if fact is None:
        # 200, not 404: htmx 2.0.10's default `responseHandling` does not swap a 4xx/5xx
        # response (`swap:false, error:true`), so a boosted link landing here would otherwise
        # leave the previous page on screen with no visible feedback — the exact silent
        # failure ux-patterns names as the worst version of an error.
        context["not_found"] = True
        return web.page(request, "corpus_detail.html", context)

    context["fact"] = fact
    context["analysis"] = analysis
    return web.page(request, "corpus_detail.html", context)
