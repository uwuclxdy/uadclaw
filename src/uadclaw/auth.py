"""Single-user session auth.

One credential from settings, a signed session cookie (Starlette's `SessionMiddleware`,
itsdangerous under the hood). Default-deny: `AuthMiddleware` runs on every http AND
websocket request and rejects anything not in `PUBLIC_PATHS`, so a route protects itself
just by existing — nobody has to remember to hang a dependency on it. This is also the
only way to cover `/docs`, `/redoc` and `/openapi.json`, which FastAPI registers on the
app instance itself and a router-scoped `Depends` can never see.
"""

import hmac

from starlette import status
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

SESSION_KEY = "authenticated"
PUBLIC_PATHS = frozenset({"/health", "/login"})


def verify_password(candidate: str, expected: str) -> bool:
    """Constant-time credential comparison."""
    return hmac.compare_digest(candidate, expected)


def log_in(request: Request) -> None:
    request.session[SESSION_KEY] = True


def log_out(request: Request) -> None:
    # Clears this browser's cookie only. A signed session cookie has no server-side store
    # to revoke against, so a copy captured before logout stays valid until it expires
    # (`max_age` on SessionMiddleware). Accepted tradeoff for a LAN-only single-user app;
    # not a gap to close here.
    request.session.pop(SESSION_KEY, None)


class AuthMiddleware:
    """Reject any http or websocket request without an authenticated session, except
    `PUBLIC_PATHS`. Only `lifespan` scopes pass through unconditionally — there is no
    request to authenticate and no path to check for app startup/shutdown. A prior version
    let `!= "http"` stand in for "just `lifespan`", which also waved every websocket
    through unauthenticated; `SessionMiddleware` populates `scope["session"]` for
    websocket scopes exactly the same as http, so there is no reason to special-case them.

    Must be added to the app BEFORE `SessionMiddleware` (Starlette runs the
    most-recently-added middleware first), so `SessionMiddleware` populates
    `request.session` before this checks it.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope, receive)
        if not connection.session.get(SESSION_KEY):
            if scope["type"] == "websocket":
                close = WebSocketClose(code=status.WS_1008_POLICY_VIOLATION)
                await close(scope, receive, send)
            else:
                response = JSONResponse({"detail": "not authenticated"}, status_code=401)
                await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
