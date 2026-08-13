"""Server-rendered dashboard plumbing: the Jinja environment, the static mount, and the
one distinction htmx forces on every view.

The dashboard is server-rendered with htmx rather than a JSON API plus a client app, which
means a route can be asked for two different things at the same URL: a whole page when a
browser navigates to it, and a fragment when htmx swaps part of an existing page. `is_htmx`
is that question and `partial` is the answer, so no view has to reach into headers itself.

**Templates are autoescaped and every value that reaches one is untrusted.** Package names,
labels and manifest strings all come out of downloaded firmware, and search titles and page
text come off the open web. `select_autoescape` is on for `.html` in this module and nowhere
else, so a future template directory cannot quietly opt out of it.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote

import asyncpg
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from sqlalchemy.exc import SQLAlchemyError

from uadclaw.monogram import monogram_for

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Every way "the database is not answering" reaches a view, in one tuple because four screens
# each spelling their own is what let them drift into three different answers to one question.
#
# All three were measured, not guessed. A wrong password raises
# `asyncpg.exceptions.InvalidPasswordError`, which is a `PostgresError` and NOT a
# `SQLAlchemyError` — SQLAlchemy only wraps what happens once a connection exists — so it
# escaped the two-class handlers and 500'd, which is exactly the shape a wrongly-mounted
# secret produces in production. An unresolvable host raises a bare `socket.gaierror` and a
# refused or timed-out connection an `OSError`; `TimeoutError` is an `OSError` too, which is
# what `db.py`'s connect timeout surfaces as.
#
# Deliberately not a bare `Exception`: a bug inside a store or a stats aggregate would then
# reach the operator as "the database is not answering", which is a lie a screen tells and a
# health probe does not.
DB_UNREACHABLE: tuple[type[BaseException], ...] = (SQLAlchemyError, OSError, asyncpg.PostgresError)

# What the screen says when one of those is caught. One sentence for every screen: they were
# all answering the same question and none of them was answering it differently on purpose.
DB_UNREACHABLE_MESSAGE = (
    "the database is not answering, so this could not be read. nothing was lost; /health "
    "reports whether it is up."
)

# The vendored htmx build, pinned by filename so a cached page can never be served a
# different library than the one it was tested against. Verified 2026-08-12 against the npm
# registry: the `htmx.org@2.0.10` tarball's sha512 matches its published `dist.integrity`,
# and this file is byte-identical to `package/dist/htmx.min.js` inside it.
# sha256 71ea67185bfa8c98c39d31717c6fce5d852370fcdfd129db4543774d3145c0de
HTMX_FILENAME = "htmx-2.0.10.min.js"

# The environment is constructed here rather than left to `Jinja2Templates(directory=...)`,
# which builds one with a bare `select_autoescape()` — that escapes `.html`, `.htm` and
# `.xml` and leaves every other extension and every rendered string UNESCAPED. Escaping is
# the whole defence for this app's template inputs, so it is set to escape by default and
# opt out per extension, rather than the other way round.
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(default_for_string=True, default=True),
    undefined=StrictUndefined,
)
templates = Jinja2Templates(env=_env)


class NavItem(NamedTuple):
    key: str
    label: str
    href: str


# The dashboard's whole information architecture, in the order it is shown. A screen that is
# not here is a screen nobody can reach, which is the point: adding a route and forgetting
# the nav is the failure this list exists to make obvious.
NAV_ITEMS: tuple[NavItem, ...] = (
    NavItem("triage", "triage", "/triage"),
    NavItem("jobs", "jobs", "/jobs"),
    NavItem("corpus", "corpus", "/corpus"),
    NavItem("telemetry", "telemetry", "/telemetry"),
    NavItem("emission", "emission", "/emission"),
)

templates.env.globals["htmx_filename"] = HTMX_FILENAME
templates.env.globals["nav_items"] = NAV_ITEMS
templates.env.globals["active_nav"] = None
# Registered as a global rather than passed per view, so `partials/pkg_icon.html` can be
# imported by any template without every call site threading the derivation through its own
# context. One implementation is the point: two screens deriving it separately is how one
# package ended up with two different chips.
templates.env.globals["monogram_for"] = monogram_for


def url_segment(value: Any) -> str:
    """One path segment of a URL, from a string this pipeline did not write.

    Two separate escapes, because percent-encoding alone does not cover the second and an
    earlier version of this function claimed it did.

    Jinja's own `urlencode` is built for query strings and keeps `/` safe, which is correct
    there and wrong in a path: a package name carrying a slash spells a segment boundary the
    record never had. `safe=""` closes that.

    It does not close `.`, which is an ordinary unreserved character that `quote` has no
    reason to touch. A package named `..` therefore survives quoting intact and is resolved
    by the browser before the request is sent, so the encoding that stops a name inventing a
    boundary does nothing about a name that IS one. Only the segments made entirely of dots
    are relative-path tokens, so only those are rewritten, and a package name with dots in it
    (which is all of them) is untouched.
    """
    quoted = quote(str(value), safe="")
    if quoted and set(quoted) == {"."}:
        return quoted.replace(".", "%2E")
    return quoted


def corpus_href(package: Any) -> str:
    """The `/corpus/{package}` detail link, from a package name this pipeline did not write.

    One spelling, shared by `views/corpus.py` and `views/emission.py` for the reason
    `url_segment` is one function: a package name carrying a `/` must reach an href through
    `url_segment`, never the `|urlencode` filter (jinja's `url_quote` keeps `/` safe, so the
    slash comes back unescaped and the link points at a different path than the record). A
    second local spelling of this is the drift that split the monogram derivation in two.
    """
    return f"/corpus/{url_segment(package)}"


templates.env.filters["url_segment"] = url_segment


def is_htmx(request: Request) -> bool:
    """Whether htmx issued this request, rather than the browser navigating to it.

    htmx sets `HX-Request: true` on every request it makes. A boosted link sets it too, so
    this alone does not mean "wants a fragment" — it means "the page around this already
    exists". Views that render a fragment check this; views that always render a whole page
    ignore it and stay correct either way.
    """
    return request.headers.get("HX-Request", "").lower() == "true"


def valid_filter(raw: str, allowed: Iterable[str]) -> tuple[str, str]:
    """A filter value off the query string and a warning for one the URL made up.

    An unknown value falls back to "everything" rather than to an empty table, which would
    read as "no rows like that exist", and rather than to a silently unapplied filter, which
    shows the whole set behind a control claiming to narrow it. Shared because both list
    screens have the shape and only one of them was answering it.
    """
    value = raw.strip()
    if not value or value in set(allowed):
        return value, ""
    return "", f"ignored the unknown filter value {value!r} and listed everything instead."


def page(
    request: Request, template: str, context: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """A whole page: the named template, which extends `base.html`."""
    return templates.TemplateResponse(request, template, context or {}, **kwargs)


def partial(
    request: Request, template: str, context: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """A fragment htmx swaps into an existing page. Same call, named apart so a view reads
    as what it returns and a fragment template accidentally extending `base.html` is a
    reviewable mistake rather than an invisible one."""
    return templates.TemplateResponse(request, template, context or {}, **kwargs)


def mount_static(app: Any) -> None:
    """Serve `static/` at `/static`, which `auth.PUBLIC_PREFIXES` exempts from auth.

    That exemption is a deliberate security decision and its reasoning lives on
    `auth.PUBLIC_PREFIXES`, not here. What belongs here is the alternative it was chosen
    over: inlining the login page's styles instead would fork the design tokens into a
    second copy that drifts every time the first one changes.
    """
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
