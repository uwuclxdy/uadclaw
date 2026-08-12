"""Single-user session auth.

One credential from settings, a signed session cookie (Starlette's `SessionMiddleware`,
itsdangerous under the hood). Default-deny: `AuthMiddleware` runs on every http AND
websocket request and rejects anything not in `PUBLIC_PATHS`, so a route protects itself
just by existing — nobody has to remember to hang a dependency on it. This is also the
only way to cover `/docs`, `/redoc` and `/openapi.json`, which FastAPI registers on the
app instance itself and a router-scoped `Depends` can never see.
"""

import hmac
from urllib.parse import quote

from starlette import status
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

SESSION_KEY = "authenticated"
PUBLIC_PATHS = frozenset({"/health", "/login"})

# Path PREFIXES served without a session. A deliberate widening of the exemption mechanism,
# made 2026-08-12 for the dashboard and kept to one entry.
#
# Why a prefix at all: the login page has to be styled before anyone can log in, and the
# alternative — inlining its CSS — forks the design tokens into a second copy that drifts.
# What is behind it is stylesheets, the vendored htmx build and five font subsets: no
# package data, no job state, no credential. Everything that reads the database is an exact
# path and stays default-deny.
#
# **Every entry keeps its trailing slash, and that is load-bearing.** `"/static"` as a
# prefix would also exempt `/statistics` and `/staticky`, which is the classic prefix bug
# and the reason this is a tuple of explicit strings rather than a `startswith` over
# `PUBLIC_PATHS`. `tests/test_auth.py` pins both directions.
PUBLIC_PREFIXES: tuple[str, ...] = ("/static/",)


def is_public(path: str) -> bool:
    """Whether `path` may be served without an authenticated session."""
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


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
        if scope["type"] == "lifespan" or is_public(scope["path"]):
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope, receive)
        if not connection.session.get(SESSION_KEY):
            if scope["type"] == "websocket":
                close = WebSocketClose(code=status.WS_1008_POLICY_VIOLATION)
                await close(scope, receive, send)
            else:
                response = self._reject(connection)
                await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _reject(connection: HTTPConnection) -> JSONResponse | RedirectResponse:
        """How to say "not authenticated" to whoever asked.

        Three callers, three answers, and the JSON one stays the default so nothing that
        already treats a 401 body as the contract changes:

        - **htmx** gets a 401 carrying `HX-Redirect`. htmx swaps fragments into a live page,
          so a redirect it followed itself would paste the whole login page into whatever
          element the request targeted. The header tells it to navigate the window instead.
        - **A browser navigating** gets a 303 to the login form. A JSON 401 rendered as raw
          text is a dead end for the one caller that has a person behind it.
        - **Everything else** gets the JSON 401 this has always returned.

        Sniffed off `Accept` rather than off the path, because the same dashboard route can
        legitimately be fetched either way.
        """
        if connection.headers.get("HX-Request", "").lower() == "true":
            return JSONResponse(
                {"detail": "not authenticated"},
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"HX-Redirect": login_url(_target_of(connection))},
            )
        if "text/html" in connection.headers.get("accept", ""):
            return RedirectResponse(
                login_url(_target_of(connection)), status_code=status.HTTP_303_SEE_OTHER
            )
        return JSONResponse(
            {"detail": "not authenticated"}, status_code=status.HTTP_401_UNAUTHORIZED
        )


def _target_of(connection: HTTPConnection) -> str:
    path = connection.scope.get("path", "/")
    query = connection.scope.get("query_string", b"").decode("latin-1")
    return f"{path}?{query}" if query else path


def safe_next(candidate: str | None) -> str:
    """The post-login destination, or `/` when the candidate is not one.

    An open redirect on a login form is the classic way a session cookie leaves the LAN, and
    "starts with a slash" does NOT rule one out: `//evil.example` is a protocol-relative URL
    every browser resolves as absolute, and `/\\evil.example` is treated the same way by
    browsers that normalise the backslash. So the test is a single leading slash followed by
    neither a slash nor a backslash, and anything else falls back to the dashboard root
    rather than being repaired — a destination this could not parse is not a destination the
    user asked for.
    """
    if not candidate or not candidate.startswith("/"):
        return "/"
    if candidate.startswith(("//", "/\\")):
        return "/"
    if "\\" in candidate or "\n" in candidate or "\r" in candidate:
        return "/"
    return candidate


def login_url(target: str | None = None) -> str:
    """The login page, carrying where to go afterwards when that is somewhere worth going."""
    destination = safe_next(target)
    if destination == "/":
        return "/login"
    # `safe="/"` and nothing else: a destination carrying its own query string has to arrive
    # as ONE value. Leaving `?`, `=` and `&` unescaped makes `/login?next=/jobs?state=running`
    # parse as `next=/jobs` plus a stray `state` param, so the redirect quietly drops half of
    # where the user was going.
    return f"/login?next={quote(destination, safe='/')}"
