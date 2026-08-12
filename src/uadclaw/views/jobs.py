"""The jobs screen: every pipeline run, the stage each one is on, and the controls that
start one.

Four things shape this module and none of them is obvious from the markup.

**Every path here is two segments or more under `/jobs`.** `app.py` registers
`GET /jobs/{job_id}` (the JSON read route) on the router it includes FIRST, and Starlette
matches by path pattern before it validates a parameter. So `/jobs/anything` matches that
route and answers 422 for a non-UUID, no matter what this router declares. `/jobs/list/rows`
and `/jobs/{job_id}/detail` are shaped that way deliberately, not for taste.

**`list_available()` is live network I/O against a vendor and never runs on a page render.**
The Pixel index alone is 1.2 MB and 2293 refs across 58 devices, and the Samsung and Oppo
endpoints are reverse-engineered. Only `POST /jobs/launch/devices` fetches; every other
handler reads `_device_index_cache` and renders nothing when it is cold. A re-render after a
failed job submission therefore keeps the device list the operator already loaded without
spending a second request.

**An index that yields nothing is an error, never an empty state.** A terms-walled index
answers HTTP 200 with zero links, which is indistinguishable from "this vendor has nothing"
in the rendered output and completely different in what the operator must do next. The
drivers raise `EmptyFirmwareIndexError` for it and `_load_device_index` raises the same for a
driver that returns an empty list without raising.

**A rendered vendor failure answers 200.** htmx does not swap a 4xx or 5xx response by
default, so a 502 here would leave the operator staring at an unchanged region: the silent
failure, wearing a correct status code. The fragment's own job is to render the picker or say
why it cannot, and it does that successfully. Form submissions are plain browser posts for
the same reason inverted: they are not swapped by htmx at all, so a rejected one re-renders
the whole page at 422 with everything the operator typed still in it.
"""

import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw import firmware, web
from uadclaw import jobs as jobs_module
from uadclaw.db import get_session_factory
from uadclaw.models import (
    TERMINAL_STATES,
    Job,
    JobKind,
    JobStageRun,
    JobState,
    ScratchLease,
)
from uadclaw.settings import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Newest N jobs the list renders. A cap rather than pagination: this is a single-operator
# dashboard whose queue is jobs-per-day, and the count beside the table says when the cap bit.
LIST_LIMIT = 100

# How often a fragment showing a live job re-asks for itself. Only ever emitted while
# something can still change — see `poll_trigger`.
POLL_INTERVAL_SECONDS = 3

# How long one driver's parsed index is re-used for. The fetch behind it is a vendor request
# measured in seconds and megabytes, so picking a device, mistyping a build and resubmitting
# must not cost three of them.
DEVICE_INDEX_TTL_SECONDS = 300.0

# driver name -> (monotonic expiry, device options). Process-local and unbounded only by the
# driver count, which is six.
_device_index_cache: dict[str, tuple[float, tuple[dict[str, Any], ...]]] = {}

_STATE_TAG: dict[JobState, str] = {
    JobState.QUEUED: "tag-info",
    JobState.CLAIMED: "tag-info",
    JobState.RUNNING: "tag-accent",
    JobState.SUCCEEDED: "tag-success",
    JobState.FAILED: "tag-danger",
}

# What a walk step carries when no `job_stage_runs` row exists for it yet. Spelled out rather
# than left undefined: the template environment runs StrictUndefined, so a missing key raises
# mid-render instead of showing a blank cell.
_BLANK_RUN: dict[str, str] = {"started_at": "", "finished_at": "", "outcome": "", "seconds": ""}

_STAGE_TAG: dict[str, str] = {
    "done": "tag-success",
    "running": "tag-accent",
    "failed": "tag-danger",
    "pending": "tag-default",
}

# How exposed enabling a driver leaves the operator, as the tag that carries it. The order is
# `TermsRisk`'s own: published source, official source behind terms, restricted mirror,
# private endpoint with no published terms at all.
_RISK_TAG: dict[firmware.TermsRisk, str] = {
    firmware.TermsRisk.PUBLIC: "tag-success",
    firmware.TermsRisk.ACKNOWLEDGEMENT: "tag-info",
    firmware.TermsRisk.RESTRICTED: "tag-warning",
    firmware.TermsRisk.REVERSE_ENGINEERED: "tag-danger",
}

# The same four, spelled for a reader. A blanket `.replace("_", " ")` stood here and reads as
# equivalent on today's four members: only `reverse_engineered` carries an underscore. It is
# not equivalent on the fifth. A member added later would render as prose nobody wrote and
# nobody chose, which is indistinguishable on screen from a value that was handled, so the
# omission surfaces as a wrong word rather than as a missing one. An unmapped member falls
# through to its own spelling here for exactly that reason: it should look unhandled.
_RISK_LABEL: dict[firmware.TermsRisk, str] = {
    firmware.TermsRisk.PUBLIC: "public",
    firmware.TermsRisk.ACKNOWLEDGEMENT: "terms accepted",
    firmware.TermsRisk.RESTRICTED: "restricted mirror",
    firmware.TermsRisk.REVERSE_ENGINEERED: "reverse engineered",
}

# What each named firmware failure means to the operator, in the screen's own voice. The
# detail beside it is the exception's own message, which already names the setting to change.
# Ordered most specific first; none of these five is a subclass of another.
_ERROR_TITLES: tuple[tuple[type[Exception], str], ...] = (
    (firmware.UnknownFirmwareDriverError, "no driver by that name"),
    (firmware.FirmwareDriverDisabledError, "that driver is switched off"),
    (firmware.FirmwareTermsNotAcknowledgedError, "the terms for this source are not acknowledged"),
    (firmware.EmptyFirmwareIndexError, "the index answered with no builds"),
    (firmware.FirmwareDownloadError, "the source refused the transfer"),
)


# --- pure helpers -------------------------------------------------------------------------


def poll_trigger(states: Iterable[JobState | str]) -> str | None:
    """The `hx-trigger` a fragment showing these jobs carries, or None when every one of them
    has reached a terminal state.

    None is the load-bearing half. A fragment that keeps its trigger after the last job
    finished re-queries Postgres every few seconds for the rest of the browser tab's life,
    over a row that cannot change again, and nothing on screen ever looks wrong.

    **An EMPTY list keeps its trigger.** It has no terminal state, so it cannot satisfy the
    rule above, and the list this reads is the FILTERED one: an operator sitting on
    `?state=running` with nothing running got a fragment with no trigger, and no job queued
    afterwards ever appeared until they reloaded by hand. A dead database is the other empty
    case and is deliberately not this one — `_unreadable_list` drops the trigger itself, so a
    database that is down is asked once rather than every three seconds.
    """
    listed = [JobState(state) for state in states]
    if listed and all(state in TERMINAL_STATES for state in listed):
        return None
    return f"every {POLL_INTERVAL_SECONDS}s"


def _stamp(value: Any) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else ""


def _duration_seconds(job: Job) -> float | None:
    if job.started_at is None or job.finished_at is None:
        return None
    return (job.finished_at - job.started_at).total_seconds()


def stage_walk(kind: str, stage: str | None, state: JobState) -> tuple[dict[str, str], ...]:
    """This KIND's walk with a status per stage.

    Kind-scoped through `jobs.stages_for`, never the module-global `PIPELINE_STAGES`: a
    firmware job's walk ends at `rule_ladder`, and rendering the four stages it deliberately
    does not run would show every finished firmware job as 6 of 10 forever.
    """
    stages = jobs_module.stages_for(kind)
    current = jobs_module.next_stage(stage, kind)
    reached = stages.index(stage) if stage in stages else -1
    walk: list[dict[str, str]] = []
    for index, name in enumerate(stages):
        if index <= reached:
            status_name = "done"
        elif name != current:
            status_name = "pending"
        elif state is JobState.FAILED:
            status_name = "failed"
        elif state in (JobState.CLAIMED, JobState.RUNNING):
            status_name = "running"
        else:
            status_name = "pending"
        walk.append({"stage": name, "status": status_name, "tag": _STAGE_TAG[status_name]})
    return tuple(walk)


def job_row(job: Job) -> dict[str, Any]:
    """One job as the table and the detail header read it."""
    stages = jobs_module.stages_for(job.kind)
    reached = stages.index(job.stage) + 1 if job.stage in stages else 0
    duration = _duration_seconds(job)
    return {
        "id": str(job.id),
        "short_id": str(job.id)[:8],
        "kind": job.kind,
        "params": job.params,
        "state": str(job.state),
        "state_tag": _STATE_TAG[job.state],
        "live": job.state not in TERMINAL_STATES,
        "stage": job.stage or "",
        "current_stage": jobs_module.next_stage(job.stage, job.kind) or "",
        "stages_done": reached,
        "stages_total": len(stages),
        "progress_pct": round(reached * 100 / len(stages)) if stages else 0,
        "attempt": job.attempt,
        "worker_id": job.worker_id or "",
        "failure_reason": job.failure_reason or "",
        "log_tail": job.log_tail,
        "created_at": _stamp(job.created_at),
        "started_at": _stamp(job.started_at),
        "finished_at": _stamp(job.finished_at),
        "heartbeat_at": _stamp(job.heartbeat_at),
        "duration": f"{duration:.1f}s" if duration is not None else "",
    }


def _error_view(exc: Exception) -> dict[str, str]:
    for cls, title in _ERROR_TITLES:
        if isinstance(exc, cls):
            return {"title": title, "detail": str(exc), "name": type(exc).__name__}
    return {"title": "the index fetch failed", "detail": str(exc), "name": type(exc).__name__}


def _device_options(refs: list[firmware.FirmwareRef]) -> tuple[dict[str, Any], ...]:
    """One entry per device in an index listing, carrying the build the acquire stage would
    pick if the operator names none.

    "newest" comes from `firmware.select_ref` rather than from row order or a local sort, so
    the label cannot disagree with what the job actually downloads.
    """
    by_device: dict[str, list[firmware.FirmwareRef]] = {}
    for ref in refs:
        by_device.setdefault(ref.device, []).append(ref)
    options: list[dict[str, Any]] = []
    for device in sorted(by_device):
        group = by_device[device]
        newest = firmware.select_ref(group, device=device)
        marketing = next((ref.marketing_name for ref in group if ref.marketing_name), "")
        options.append(
            {
                "device": device,
                "build_count": len(group),
                "newest_build": newest.build,
                "marketing_name": marketing or "",
            }
        )
    return tuple(options)


def _terms_rows(settings: Settings) -> tuple[dict[str, Any], ...]:
    """Every ENABLED driver with the posture it declares in code. No I/O: a driver's
    constructor only reads settings and `terms()` only reports them."""
    rows: list[dict[str, Any]] = []
    for name in firmware.enabled_driver_names(settings):
        posture = firmware.get_driver(name, settings).terms()
        rows.append(
            {
                "name": name,
                "risk": _RISK_LABEL.get(posture.risk, str(posture.risk)),
                "risk_tag": _RISK_TAG[posture.risk],
                "summary": posture.summary,
                "source_url": posture.source_url,
                "acknowledged": posture.acknowledged,
                # Whether this row is worth surfacing before the terms table is opened: an
                # unacknowledged driver or a reverse-engineered one is risk state, not
                # reference detail, and progressive disclosure is for reference detail only.
                "needs_attention": (
                    not posture.acknowledged
                    or posture.risk is firmware.TermsRisk.REVERSE_ENGINEERED
                ),
            }
        )
    return tuple(rows)


# --- the device index -------------------------------------------------------------------


def cached_device_index(driver_name: str) -> tuple[dict[str, Any], ...] | None:
    """This driver's last listing if it is still fresh, and never a fetch. Every render path
    reads through here, so no page load can reach a vendor."""
    entry = _device_index_cache.get(driver_name)
    if entry is None:
        return None
    expires_at, options = entry
    if time.monotonic() >= expires_at:
        del _device_index_cache[driver_name]
        return None
    return options


async def load_device_index(driver_name: str, settings: Settings) -> tuple[dict[str, Any], ...]:
    """Fetch one driver's index and cache it. **This is the only live network call on this
    screen** and only `POST /jobs/launch/devices` reaches it."""
    cached = cached_device_index(driver_name)
    if cached is not None:
        return cached
    driver = firmware.get_driver(driver_name, settings)
    refs = await driver.list_available()
    if not refs:
        # Every shipped driver raises this itself. Repeated here because the alternative is a
        # rendered "no devices", which reads as a fact about the vendor and is a fact about a
        # terms wall, a rotated endpoint or a driver bug.
        raise firmware.EmptyFirmwareIndexError(
            f"load_device_index: driver {driver_name!r} answered with zero builds. An index "
            "that parses to nothing is a failure, not an empty catalogue: check the driver's "
            "terms acknowledgement and its configured devices."
        )
    options = _device_options(refs)
    _device_index_cache[driver_name] = (time.monotonic() + DEVICE_INDEX_TTL_SECONDS, options)
    return options


# --- context ------------------------------------------------------------------------------


async def _lease_view(session: AsyncSession) -> dict[str, Any] | None:
    """The single-occupant scratch lease, or None while nothing holds it. `scratch_lease` is
    a singleton table (its own CHECK constraint pins the id), so the one row is the lease."""
    lease = (await session.execute(select(ScratchLease).limit(1))).scalar_one_or_none()
    if lease is None or lease.holder_job_id is None:
        return None
    return {
        "job_id": str(lease.holder_job_id),
        "short_id": str(lease.holder_job_id)[:8],
        "worker_id": lease.holder_worker_id or "",
        "acquired_at": _stamp(lease.acquired_at),
        "heartbeat_at": _stamp(lease.heartbeat_at),
    }


async def _list_context(
    session: AsyncSession, *, state_filter: str, kind_filter: str
) -> dict[str, Any]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(LIST_LIMIT)
    if state_filter:
        stmt = stmt.where(Job.state == JobState(state_filter))
    if kind_filter:
        stmt = stmt.where(Job.kind == kind_filter)
    listed = list((await session.execute(stmt)).scalars())
    total = (await session.execute(select(func.count()).select_from(Job))).scalar_one()
    rows = [job_row(job) for job in listed]
    return {
        "jobs": rows,
        "job_total": total,
        "state_filter": state_filter,
        "kind_filter": kind_filter,
        "list_capped": total > LIST_LIMIT,
        "list_limit": LIST_LIMIT,
        # The poll asks for the filter it is showing. Spelled once, here, rather than again in
        # the template: two spellings drift, and the drifted one silently widens the table
        # back to every job three seconds after the operator narrowed it.
        "rows_url": _rows_url(state_filter, kind_filter),
        "poll": poll_trigger(job.state for job in listed),
    }


def _blank_form_values() -> dict[str, str]:
    return {"driver": "", "device": "", "build": "", "packages": "", "limit": "", "reclassify": ""}


def _unreadable_list(state_filter: str, kind_filter: str) -> dict[str, Any]:
    """What the list slot carries when its query could not run: no rows, no counts, and no
    poll trigger, so a database that is down is asked once rather than every three seconds."""
    return {
        "jobs": [],
        "job_total": 0,
        "state_filter": state_filter,
        "kind_filter": kind_filter,
        "list_capped": False,
        "list_limit": LIST_LIMIT,
        "rows_url": _rows_url(state_filter, kind_filter),
        "poll": None,
        "lease": None,
        "db_error": True,
    }


def _retry(element_id: str, retry_url: str) -> dict[str, str]:
    return {"element_id": element_id, "retry_url": retry_url}


def _rows_url(state_filter: str, kind_filter: str) -> str:
    return f"/jobs/list/rows?state={state_filter}&kind={kind_filter}"


async def _screen_context(
    *,
    state_filter: str = "",
    kind_filter: str = "",
    filter_warning: str = "",
    selected_driver: str = "",
    devices: tuple[dict[str, Any], ...] | None = None,
    device_error: dict[str, str] | None = None,
    form_error: dict[str, str] | None = None,
    form_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Everything `jobs.html` needs, in one place so no render path can miss a key. The
    environment runs StrictUndefined, so a missing one raises rather than rendering blank."""
    settings = get_settings()
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            context = await _list_context(
                session, state_filter=state_filter, kind_filter=kind_filter
            )
            context["lease"] = await _lease_view(session)
            context["db_error"] = False
    except web.DB_UNREACHABLE:
        # `web.DB_UNREACHABLE` carries which classes and why they were measured. What is
        # local to this screen: the launch controls and the driver postures need no database,
        # so one dead query must not take the whole screen down with it. Logged with its
        # traceback, never swallowed, and `/health` answers "is it up" for the operator.
        logger.exception("reading the run list failed")
        context = _unreadable_list(state_filter, kind_filter)
    values = _blank_form_values() | (form_values or {})
    if not selected_driver:
        selected_driver = values["driver"]
    if devices is None and selected_driver:
        devices = cached_device_index(selected_driver)
    drivers = _terms_rows(settings)
    context |= {
        "active_nav": "jobs",
        "filter_warning": filter_warning,
        "states": [str(state) for state in JobState],
        "kinds": [str(kind) for kind in JobKind],
        "drivers": drivers,
        # Drives the terms table's `<details>`: open by default and named in its own
        # always-visible `<summary>` when a driver needs a decision, collapsed otherwise. A
        # request/response test cannot observe an interactive open/close, but it can and does
        # pin these two values landing in the rendered markup.
        "unacknowledged_driver_count": sum(1 for d in drivers if not d["acknowledged"]),
        "terms_need_attention": any(d["needs_attention"] for d in drivers),
        "selected_driver": selected_driver,
        "devices": devices,
        "device_error": device_error,
        "form_error": form_error,
        "form": values,
        "classification_ceiling": settings.classification_max_packages,
        "unavailable": _retry("job-list", _rows_url(state_filter, kind_filter)),
    }
    return context


def _device_context(
    *,
    selected_driver: str,
    devices: tuple[dict[str, Any], ...] | None,
    device_error: dict[str, str] | None,
    form_values: dict[str, str],
) -> dict[str, Any]:
    """The device-picker fragment's own context, a strict subset of the page's."""
    return {
        "selected_driver": selected_driver,
        "devices": devices,
        "device_error": device_error,
        "form": _blank_form_values() | form_values,
    }


# --- routes -------------------------------------------------------------------------------


@router.get("/jobs")
async def jobs_screen(request: Request, state: str = "", kind: str = "") -> Response:
    """The screen. Renders from the database only: no driver is contacted here, whatever is
    selected."""
    state_filter, state_warning = web.valid_filter(state, (str(s) for s in JobState))
    kind_filter, kind_warning = web.valid_filter(kind, (str(k) for k in JobKind))
    context = await _screen_context(
        state_filter=state_filter,
        kind_filter=kind_filter,
        filter_warning=state_warning or kind_warning,
    )
    return web.page(request, "jobs.html", context)


@router.get("/jobs/list/rows")
async def jobs_rows(request: Request, state: str = "", kind: str = "") -> Response:
    """The list fragment the page polls while anything in it is still live."""
    state_filter, _ = web.valid_filter(state, (str(s) for s in JobState))
    kind_filter, _ = web.valid_filter(kind, (str(k) for k in JobKind))
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            context = await _list_context(
                session, state_filter=state_filter, kind_filter=kind_filter
            )
    except web.DB_UNREACHABLE:
        logger.exception("reading the run list failed")
        return web.partial(
            request,
            "partials/jobs_unavailable.html",
            {"unavailable": _retry("job-list", _rows_url(state_filter, kind_filter))},
        )
    return web.partial(request, "partials/jobs_rows.html", context)


@router.post("/jobs/launch/devices")
async def load_devices(request: Request) -> Response:
    """Fetch a driver's index, on the operator's explicit say-so and nowhere else."""
    form = await request.form()
    driver_name = str(form.get("driver") or "").strip()
    settings = get_settings()
    devices: tuple[dict[str, Any], ...] | None = None
    error: dict[str, str] | None = None
    if not driver_name:
        error = {
            "title": "pick a driver first",
            "detail": "choose which vendor's index to list, then load its devices.",
            "name": "",
        }
    else:
        try:
            devices = await load_device_index(driver_name, settings)
        except (firmware.FirmwareError, firmware.FirmwareInputError) as exc:
            # Rendered, not raised: this is the operator's own screen and the message names
            # the fix. Logged with its traceback so the worker's logs still carry the whole
            # failure. Every driver wraps its transport errors into these two types.
            logger.exception("listing %s devices failed", driver_name)
            error = _error_view(exc)
    form_values = {"driver": driver_name}
    if web.is_htmx(request):
        return web.partial(
            request,
            "partials/jobs_devices.html",
            _device_context(
                selected_driver=driver_name,
                devices=devices,
                device_error=error,
                form_values=form_values,
            ),
        )
    context = await _screen_context(
        selected_driver=driver_name,
        devices=devices,
        device_error=error,
        form_values=form_values,
    )
    return web.page(request, "jobs.html", context)


def _packages(raw: str) -> list[str]:
    return [name for name in raw.replace(",", " ").split() if name]


async def _unqueued(
    request: Request, kind: str, values: dict[str, str], *, selected_driver: str = ""
) -> Response:
    """The screen again, saying the run was not queued, at 200.

    200 rather than 422 or 503, and rendered rather than raised: the operator's input was
    fine, the database was not, and a 5xx would leave them looking at an unstyled error page
    having lost everything they typed. `_screen_context` renders its own unreadable-list
    state underneath, so the page is honest about both halves at once.
    """
    logger.exception("queueing a %s job failed", kind)
    context = await _screen_context(
        selected_driver=selected_driver,
        form_error={
            "title": f"the {kind} run was not queued",
            "detail": web.DB_UNREACHABLE_MESSAGE,
        },
        form_values=values,
    )
    return web.page(request, "jobs.html", context)


async def _create(kind: JobKind, params: dict[str, Any]) -> uuid.UUID:
    session_factory = get_session_factory()
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind=kind.value, params=params)
        return job.id


@router.post("/jobs/launch/firmware")
async def launch_firmware(request: Request) -> Response:
    """Queue a firmware analysis job.

    Only `driver`, `device` and an optional `build` are submitted. The url and the digest the
    picker already knows are deliberately left off: `acquire` re-resolves them against the
    index when the job actually runs, and a firmware url pinned into a queued job can expire
    before a worker reaches it.

    Validation is `jobs.create_job`'s and nothing here duplicates it.
    """
    form = await request.form()
    values = {
        "driver": str(form.get("driver") or "").strip(),
        "device": str(form.get("device") or "").strip(),
        "build": str(form.get("build") or "").strip(),
    }
    params: dict[str, Any] = {"driver": values["driver"], "device": values["device"]}
    if values["build"]:
        params["build"] = values["build"]
    try:
        job_id = await _create(JobKind.FIRMWARE_ANALYSIS, params)
    except jobs_module.JobValidationError as exc:
        context = await _screen_context(
            selected_driver=values["driver"],
            form_error={"title": "that firmware target was refused", "detail": str(exc)},
            form_values=values,
        )
        return web.page(
            request, "jobs.html", context, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
        )
    except web.DB_UNREACHABLE:
        return await _unqueued(request, "firmware", values, selected_driver=values["driver"])
    return RedirectResponse(f"/jobs/{job_id}/detail", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/jobs/launch/classification")
async def launch_classification(request: Request) -> Response:
    """Queue a classification job. This one bills a paid API, which is why it is its own
    control with its own cost line rather than a second button beside the firmware one."""
    form = await request.form()
    values = {
        "packages": str(form.get("packages") or "").strip(),
        "limit": str(form.get("limit") or "").strip(),
        "reclassify": "on" if form.get("reclassify") else "",
    }
    params: dict[str, Any] = {"reclassify": bool(values["reclassify"])}
    packages = _packages(values["packages"])
    if packages:
        params["packages"] = packages
    if values["limit"]:
        # Handed over as typed. `ClassificationJobParams` decides whether it is a number.
        params["limit"] = values["limit"]
    try:
        job_id = await _create(JobKind.CLASSIFICATION, params)
    except jobs_module.JobValidationError as exc:
        context = await _screen_context(
            form_error={"title": "that classification run was refused", "detail": str(exc)},
            form_values=values,
        )
        return web.page(
            request, "jobs.html", context, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT
        )
    except web.DB_UNREACHABLE:
        return await _unqueued(request, "classification", values)
    return RedirectResponse(f"/jobs/{job_id}/detail", status_code=status.HTTP_303_SEE_OTHER)


async def _detail_context(job_id: uuid.UUID) -> dict[str, Any] | None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None:
            return None
        runs = list(
            (
                await session.execute(
                    select(JobStageRun)
                    .where(JobStageRun.job_id == job_id)
                    .order_by(JobStageRun.started_at, JobStageRun.id)
                )
            ).scalars()
        )
    # The LAST run per stage: a reclaimed job re-runs a stage under a new attempt, and the
    # walk shows where the job is now rather than where an abandoned attempt got to.
    by_stage: dict[str, dict[str, str]] = {
        run.stage: {
            "started_at": _stamp(run.started_at),
            "finished_at": _stamp(run.finished_at),
            "outcome": run.outcome or "",
            "seconds": (
                f"{(run.finished_at - run.started_at).total_seconds():.1f}"
                if run.finished_at is not None
                else ""
            ),
        }
        for run in runs
    }
    walk = [
        step | by_stage.get(step["stage"], _BLANK_RUN)
        for step in stage_walk(job.kind, job.stage, job.state)
    ]
    return {
        "job": job_row(job),
        "walk": walk,
        "run_count": len(by_stage),
        "poll": poll_trigger([job.state]),
    }


def _detail_shell(job_id: uuid.UUID) -> dict[str, Any]:
    return {
        "active_nav": "jobs",
        "job_id": str(job_id),
        "short_id": str(job_id)[:8],
        "unavailable": _retry("job-panel", f"/jobs/{job_id}/detail/panel"),
    }


@router.get("/jobs/{job_id}/detail")
async def job_detail(request: Request, job_id: uuid.UUID) -> Response:
    try:
        context = await _detail_context(job_id)
    except web.DB_UNREACHABLE:
        logger.exception("reading run %s failed", job_id)
        return web.page(request, "jobs_detail.html", _detail_shell(job_id) | {"panel_ok": False})
    if context is None:
        return web.page(
            request,
            "jobs_missing.html",
            {"active_nav": "jobs", "job_id": str(job_id)},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return web.page(
        request, "jobs_detail.html", _detail_shell(job_id) | context | {"panel_ok": True}
    )


@router.get("/jobs/{job_id}/detail/panel")
async def job_detail_panel(request: Request, job_id: uuid.UUID) -> Response:
    """The detail fragment the page polls, and stops polling the moment the job is terminal."""
    try:
        context = await _detail_context(job_id)
    except web.DB_UNREACHABLE:
        logger.exception("reading run %s failed", job_id)
        return web.partial(
            request,
            "partials/jobs_unavailable.html",
            {"unavailable": _retry("job-panel", f"/jobs/{job_id}/detail/panel")},
        )
    if context is None:
        # 200 rather than 404 on purpose, and for the same reason a rendered vendor failure
        # is: htmx swaps neither a 404 nor a 5xx, so the panel would keep its trigger and
        # poll a row that no longer exists forever. This fragment carries no trigger, so the
        # swap is what stops the polling.
        return web.partial(request, "partials/jobs_gone.html", {"job_id": str(job_id)})
    return web.partial(request, "partials/jobs_detail_panel.html", context)
