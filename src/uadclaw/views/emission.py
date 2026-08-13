"""The emission screen: the pipeline's final stage, the branches it committed into the
operator's own clone.

Read-only and reads the database only — this screen opens no repo, makes no request and runs
no git. The branch itself lives in a clone this pipeline does not own, so what the screen
shows is the row `stages.branch_stage` wrote: which packages, at what ratings, against which
floors, from which base commit, disclosed with which body.

Three things this view has to get right that are easy to get backwards:

- **`commit_oid IS NULL` is "the outcome is unknown", not a corrupt row.** The row is written
  before the git commit is attempted, so a worker killed between the two leaves a pending
  intent. The screen renders it as its own status rather than as "not committed", which would
  read as a failure that is actually just unknown.
- **`reconciled` is a second status, not a detail of the first.** A commit oid the pipeline
  watched happen (`emit_branch` returned it) and one a later run read back off a branch a
  crashed run had already cut are different claims about how much of the emission was watched,
  so the screen renders three badges — pending, emitted, reconciled — and never collapses the
  two "has a commit" states.
- **A package name reaches an `href` through `web.url_segment`, never the `|urlencode`
  filter.** Jinja's `urlencode` keeps `/` safe, which is right in a query string and wrong in a
  path: a package name carrying a slash would link at a different path than the record (the
  corpus view's own rule, for the same reason here).
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request

from uadclaw import web
from uadclaw.db import get_session_factory
from uadclaw.emissionstore import (
    EmissionSummary,
    count_emissions,
    list_emissions,
    load_emission_detail,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Newest N emissions the list renders. A cap rather than pagination: this is an operator log,
# not a full history table — the same idea as `views/jobs.py`'s `LIST_LIMIT`, smaller because
# the pipeline ships a handful of branches a day rather than a handful of jobs.
LIST_LIMIT = 25

# The three outcomes a row can carry, spelled off the two stored columns. Keyed on the DISPLAY
# key, never on the columns: `reconciled` is not a third value of `commit_oid`, it is an axis
# on top of it, and reading it as one is how the two "has a commit" states collapse.
_STATUS_TAG: dict[str, str] = {
    "pending": "tag-default",
    "emitted": "tag-success",
    "reconciled": "tag-warning",
}
_STATUS_LABEL: dict[str, str] = {
    "pending": "outcome unknown",
    "emitted": "emitted",
    "reconciled": "reconciled",
}


def _status(commit_oid: str | None, reconciled: bool) -> str:
    if commit_oid is None:
        return "pending"
    return "reconciled" if reconciled else "emitted"


def _stamp(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else ""


def _summary_row(row: EmissionSummary) -> dict[str, Any]:
    status = _status(row.commit_oid, row.reconciled)
    return {
        "id": row.id,
        "vendor": row.vendor,
        "branch": row.branch,
        "package_count": row.package_count,
        "created_at": _stamp(row.created_at),
        "status_label": _STATUS_LABEL[status],
        "status_tag": _STATUS_TAG[status],
    }


@router.get("/emission")
async def emission_screen(request: Request):
    context: dict[str, Any] = {
        "active_nav": "emission",
        "list_limit": LIST_LIMIT,
    }
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            rows = await list_emissions(session, limit=LIST_LIMIT)
            total = await count_emissions(session)
    except web.DB_UNREACHABLE:
        # The shared tuple rather than a bare `Exception`: a bug in a read would otherwise
        # reach the operator as "the database is not answering". Not re-raised, so a
        # reachability failure renders its own distinct state rather than a 500 or an empty
        # log — "nothing has shipped yet" is the most expensive lie this screen can tell.
        # Logged in full first.
        logger.exception("emission list query failed")
        context["error"] = True
        return web.page(request, "emission.html", context)

    context["rows"] = [_summary_row(row) for row in rows]
    context["total"] = total
    context["capped"] = total > LIST_LIMIT
    context["empty"] = total == 0
    return web.page(request, "emission.html", context)


@router.get("/emission/{emission_id}")
async def emission_detail(request: Request, emission_id: int):
    context: dict[str, Any] = {
        "active_nav": "emission",
        "emission_id": emission_id,
    }
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            detail = await load_emission_detail(session, emission_id=emission_id)
    except web.DB_UNREACHABLE:
        logger.exception("emission detail query failed for %s", emission_id)
        context["error"] = True
        return web.page(request, "emission_detail.html", context)

    if detail is None:
        # 200, not 404: htmx 2.0.10's default response handling does not swap a 4xx/5xx, so a
        # boosted link into a missing emission would otherwise leave the previous page on
        # screen with no visible feedback.
        context["not_found"] = True
        return web.page(request, "emission_detail.html", context)

    status = _status(detail.commit_oid, detail.reconciled)
    context["detail"] = {
        "id": detail.id,
        "vendor": detail.vendor,
        "branch": detail.branch,
        "repo_path": detail.repo_path,
        "list_path": detail.list_path,
        "base_commit": detail.base_commit,
        "list_sha256": detail.list_sha256,
        "pipeline_version": detail.pipeline_version,
        "pipeline_commit_sha": detail.pipeline_commit_sha,
        "package_count": detail.package_count,
        "pr_body": detail.pr_body,
        "created_at": _stamp(detail.created_at),
        "commit_oid": detail.commit_oid or "",
        "committed_at": _stamp(detail.committed_at),
        "reconciled": detail.reconciled,
        "status_label": _STATUS_LABEL[status],
        "status_tag": _STATUS_TAG[status],
        "packages": [
            {
                "package": p.package,
                "href": web.corpus_href(p.package),
                "uad_list": p.uad_list,
                "removal": p.removal,
                "floor": p.floor,
            }
            for p in detail.packages
        ],
    }
    return web.page(request, "emission_detail.html", context)
