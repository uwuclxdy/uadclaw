"""FastAPI app factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from uadclaw.auth import AuthMiddleware, log_in, log_out, verify_password
from uadclaw.db import get_engine
from uadclaw.settings import get_settings

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    web: str
    db: str


class LoginRequest(BaseModel):
    password: str


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
    if not verify_password(payload.password, settings.auth_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad credentials")
    log_in(request)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request) -> None:
    log_out(request)


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
        secret_key=settings.session_secret,
        session_cookie=settings.session_cookie_name,
        same_site="lax",
        https_only=settings.cookie_secure,
    )
    app.include_router(router)
    return app
