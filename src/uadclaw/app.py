"""FastAPI app factory."""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from uadclaw import jobs as jobs_module
from uadclaw.auth import AuthMiddleware, log_in, log_out, verify_password
from uadclaw.db import get_engine, get_session_factory
from uadclaw.models import Job
from uadclaw.settings import get_settings
from uadclaw.stats import StatsResponse, compute_stats

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


@router.post("/login", status_code=status.HTTP_204_NO_CONTENT)
async def login(payload: LoginRequest, request: Request) -> None:
    settings = get_settings()
    if not verify_password(payload.password, settings.auth_password.get_secret_value()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad credentials")
    log_in(request)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request) -> None:
    log_out(request)


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
    """Start a job the worker will pick up. No UI here — task 9 owns that; this is the
    substrate the dashboard is built on. Behind auth automatically, like every route."""
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
async def index() -> dict[str, str]:
    """Dashboard root placeholder; real UI lands in task 9."""
    return {"app": "uadclaw"}


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
    return app
