"""FastAPI app factory."""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from uadclaw import jobs as jobs_module
from uadclaw import web
from uadclaw.auth import (
    SESSION_KEY,
    AuthMiddleware,
    log_in,
    log_out,
    safe_next,
    verify_password,
)
from uadclaw.db import get_engine, get_session_factory
from uadclaw.models import Job
from uadclaw.settings import get_settings
from uadclaw.stats import StatsResponse, compute_stats
from uadclaw.views import corpus as corpus_view
from uadclaw.views import icons as icons_view
from uadclaw.views import jobs as jobs_view
from uadclaw.views import telemetry as telemetry_view
from uadclaw.views import triage as triage_view

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    web: str
    db: str


class LoginRequest(BaseModel):
    password: str


class CreateJobRequest(BaseModel):
    kind: str
    # Shaped by `kind` and validated against that kind's model in `jobs.create_job`, which
    # is why this is not typed tighter here: the route is generic over job kinds.
    params: dict[str, Any] | None = None


class JobResponse(BaseModel):
    """The dashboard-pollable shape of a job. Deliberately not the ORM model itself —
    an API response is its own contract, not whatever columns happen to exist today."""

    id: uuid.UUID
    kind: str
    params: dict[str, Any]
    state: str
    stage: str | None
    attempt: int
    worker_id: str | None
    failure_reason: str | None
    log_tail: str
    created_at: datetime
    claimed_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    heartbeat_at: datetime | None

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        return cls(
            id=job.id,
            kind=job.kind,
            params=job.params,
            state=str(job.state),
            stage=job.stage,
            attempt=job.attempt,
            worker_id=job.worker_id,
            failure_reason=job.failure_reason,
            log_tail=job.log_tail,
            created_at=job.created_at,
            claimed_at=job.claimed_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            heartbeat_at=job.heartbeat_at,
        )


router = APIRouter()


@router.get("/health")
async def health() -> HealthResponse:
    """Process liveness (this handler ran) and DB liveness, reported separately."""
    db_status = "ok"
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        # Broad on purpose: a health check must survive every failure mode of the DB
        # driver (DNS failure, refused connection, protocol error), not just the ones
        # SQLAlchemy wraps as SQLAlchemyError. Logged, never swallowed.
        logger.exception("db health check failed")
        db_status = "error"
    return HealthResponse(web="ok", db=db_status)


@router.get("/login")
async def login_form(request: Request, next: str = "/") -> Response:
    """The login page. Already authenticated, so it goes straight where it was headed —
    a login form shown to someone who is logged in is a dead end that looks like a bug."""
    if request.session.get(SESSION_KEY):
        return RedirectResponse(safe_next(next), status_code=status.HTTP_303_SEE_OTHER)
    return web.page(request, "login.html", {"next": safe_next(next)})


@router.post("/login")
async def login(request: Request) -> Response:
    """One path, two callers, and the JSON contract is unchanged.

    A form post comes from the login page and wants a redirect; a JSON post comes from a
    test or a script and wants the 204 this route has always returned. Dispatching on the
    request's own content type rather than adding a second path keeps `PUBLIC_PATHS` at one
    login entry: a second public path is a second thing to get wrong.
    """
    is_form = request.headers.get("content-type", "").startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    )
    if is_form:
        form = await request.form()
        password = str(form.get("password") or "")
        destination = safe_next(str(form.get("next") or "/"))
    else:
        payload = LoginRequest.model_validate(await request.json())
        password = payload.password
        destination = "/"

    settings = get_settings()
    if not verify_password(password, settings.auth_password.get_secret_value()):
        if is_form:
            # Deliberately vague and deliberately not a 401: a wrong password on a form is
            # a re-render of the form, and naming which half was wrong tells an attacker
            # something the single-user model never wants to confirm.
            return web.page(
                request,
                "login.html",
                {"next": destination, "error": "wrong password."},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad credentials")

    log_in(request)
    if is_form:
        return RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/logout")
async def logout(request: Request) -> Response:
    """204 for a JSON caller, a redirect to the login page for the dashboard's form."""
    log_out(request)
    if request.headers.get("content-type", "").startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/stats")
async def stats() -> StatsResponse:
    """Usage/utilization aggregates for tuning `worker_pool_size`. Behind auth like every
    other route — protected by default-deny `AuthMiddleware`, no allowlist entry needed."""
    settings = get_settings()
    session_factory = get_session_factory()
    async with session_factory() as session:
        return await compute_stats(session, lookback_seconds=settings.stats_lookback_seconds)


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job_route(payload: CreateJobRequest) -> JobResponse:
    """Start a job the worker will pick up, as JSON. The jobs SCREEN posts a form to its own
    route and calls the same `jobs.create_job` underneath, so this stayed rather than being
    replaced: it is the scriptable half. Behind auth automatically, like every route."""
    session_factory = get_session_factory()
    async with session_factory() as session, session.begin():
        try:
            job = await jobs_module.create_job(session, kind=payload.kind, params=payload.params)
        except jobs_module.JobValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
            ) from exc
        await session.flush()
        return JobResponse.from_job(job)


@router.get("/jobs/{job_id}")
async def get_job_route(job_id: uuid.UUID) -> JobResponse:
    """State, stage, timestamps and log tail — the poll target "the worker claims jobs and
    reports progress the dashboard can poll" (task 2 todo) actually needs."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"no job with id {job_id}"
            )
        return JobResponse.from_job(job)


@router.get("/")
async def index(request: Request) -> Response:
    """The dashboard root. Triage is the landing screen because it is the pipeline's
    throughput bottleneck by design: everything else runs unattended, that queue does not.

    A JSON caller still gets the identity payload this route has always returned, so the
    health-style probes that poll it keep working.
    """
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/triage", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse({"app": "uadclaw"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await get_engine().dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="uadclaw", lifespan=lifespan)
    # Added first so it ends up innermost: SessionMiddleware (added second, runs first)
    # populates request.session before AuthMiddleware reads it. See auth.py docstring.
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret.get_secret_value(),
        session_cookie=settings.session_cookie_name,
        same_site="lax",
        https_only=settings.cookie_secure,
    )
    app.include_router(router)
    # One router per screen, so two screens can never collide in one file. Every one of
    # these is behind the default-deny middleware by existing; none needs an exemption.
    app.include_router(triage_view.router)
    app.include_router(jobs_view.router)
    app.include_router(corpus_view.router)
    app.include_router(telemetry_view.router)
    app.include_router(icons_view.router)
    web.mount_static(app)
    return app
