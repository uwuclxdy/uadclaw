"""The telemetry screen: `stats.compute_stats`, rendered to answer one question — what should
`worker_pool_size` be. Read-only; never reimplements an aggregate `stats.py` already computes.

Reachable at the same URL two ways, and this is the trap `web.py`'s own docstring names: a
boosted nav click ALSO carries `HX-Request: true`, so `web.is_htmx` alone cannot tell "wants a
fragment" from "the browser is navigating here via hx-boost". `HX-Boosted` is the header that
actually distinguishes them. Only the periodic auto-refresh below issues a bare (non-boosted)
`hx-get`, so it is the only caller `_wants_fragment` answers yes to; a boosted nav click still
gets the full page, navbar included.
"""

import logging
from typing import Any

from fastapi import APIRouter, Request

from uadclaw import web
from uadclaw.db import get_session_factory
from uadclaw.settings import get_settings
from uadclaw.stats import compute_stats

logger = logging.getLogger(__name__)

router = APIRouter()

# How often the screen polls itself for fresh numbers. A named constant rather than a literal
# sprinkled into the template, so the one place that decides the cadence is code.
TELEMETRY_REFRESH_INTERVAL_SECONDS = 15


def _wants_fragment(request: Request) -> bool:
    """Whether this request is the periodic auto-refresh rather than a page navigation. See
    the module docstring: `HX-Request` alone cannot tell the two apart, because hx-boost sets
    it too."""
    return web.is_htmx(request) and request.headers.get("HX-Boosted", "").lower() != "true"


@router.get("/telemetry")
async def telemetry_screen(request: Request):
    context: dict[str, Any] = {
        "active_nav": "telemetry",
        "refresh_interval_seconds": TELEMETRY_REFRESH_INTERVAL_SECONDS,
    }
    settings = get_settings()
    session_factory = get_session_factory()
    stats = None
    async with session_factory() as session:
        try:
            stats = await compute_stats(session, lookback_seconds=settings.stats_lookback_seconds)
        except Exception:
            # Broad on purpose, same reasoning as `app.py`'s health check: a DNS failure or a
            # refused connection surfaces as a raw `OSError`/`socket.gaierror` out of asyncpg's
            # own `connect()`, never wrapped into `sqlalchemy.exc.SQLAlchemyError` — that
            # wrapping only happens once a connection already exists. Not re-raised: on the
            # auto-refreshing fragment this has to render as its own state, not vanish into an
            # unhandled 500 the poll loop cannot recover from. Logged in full first.
            logger.exception("telemetry stats query failed")
            context["error"] = True

    if stats is not None:
        context["stats"] = stats
        # A wedge (§2.2/M9 in stats.py) still reports real numbers; a genuinely fresh
        # install reports all-zero across every one of these independent signals at once.
        # That conjunction is what tells the two apart — any single zero is a normal reading.
        context["empty"] = (
            stats.queue_depth_now == 0
            and not stats.stage_durations
            and not stats.worker_utilization
            and stats.lease.total_acquisitions == 0
            and stats.lease.current_holder_job_id is None
        )
        context["active_worker_count"] = len(stats.worker_utilization)
        context["worker_utilization_pct"] = {
            worker.worker_id: (
                (worker.busy_seconds / stats.window_seconds * 100)
                if stats.window_seconds > 0
                else 0.0
            )
            for worker in stats.worker_utilization
        }

    if _wants_fragment(request):
        return web.partial(request, "partials/telemetry_content.html", context)
    return web.page(request, "telemetry.html", context)
