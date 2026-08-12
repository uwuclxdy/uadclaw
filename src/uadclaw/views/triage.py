"""The triage screen: the ranked queue, one candidate at a time, keyboard-driven.

The pipeline's throughput bottleneck by design — everything else runs unattended and this
queue does not — so the interaction work here is the point rather than decoration.

The view issues no query of its own: `triagestore` owns the queue, the card and the decision
log, and this module turns a request into one of those calls and a template. Three things it
does own:

- **The five screen states are decided here, not in the template.** `state` is one of
  `loading`/`success`/`error`/`empty`/`partial` per section, so a template branch can never
  invent a sixth or collapse "settled and empty" into "still loading" — the failure NN/g
  describes, where an empty state renders from an unsettled request and the reader acts on it.
- **A failure of one section does not take the screen down.** A missing `uad_lists.json`
  costs the card its upstream neighbours and says so; it does not fail the request, because
  the floor, the evidence and the proposal are all still reviewable without it.
- **Every action is a form.** htmx swaps the board, and the same forms submit and redirect
  with no JavaScript at all. The keybinds click those buttons rather than calling an
  endpoint, so keyboard, mouse and no-JS all take one path.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Form, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.exc import SQLAlchemyError

from uadclaw import triagestore, web
from uadclaw.classifystore import BelowFloorError
from uadclaw.db import get_session_factory
from uadclaw.settings import get_settings
from uadclaw.triagestore import VIEWS, Candidate, QueueRow, ReasonRequired, TriageError
from uadclaw.upstream import UpstreamList, UpstreamListError, load_upstream_list

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/triage")

# (action, key, label, button variant) for the plain verdict buttons. The key is spelled HERE
# and the template reads it off this table rather than deriving one from the action name.
# Measured in a browser: `action[0]` gave `reopen` the letter `r`, the script binds whichever
# control comes first in the document, and pressing `r` to write a rejection reopened the
# package instead. The buttons and the keys they answer to have to come from one list.
ACTION_BUTTONS: tuple[tuple[str, str, str, str], ...] = (
    ("approve", "a", "approve", "btn-primary"),
    ("defer", "d", "defer", "btn-secondary"),
    ("reopen", "u", "reopen", "btn-ghost"),
)
# Reject and edit are not plain buttons: `r` focuses the reason field, because a rejection
# with no reason is refused and a key that fires a refusal teaches nothing, and `e` opens the
# edit form rather than submitting anything.
REJECT_KEY = "r"
EDIT_KEY = "e"

KEYBINDS: tuple[tuple[str, str], ...] = (
    *((key, action) for action, key, _, _ in ACTION_BUTTONS),
    (REJECT_KEY, "reject"),
    (EDIT_KEY, "edit"),
    ("j", "next"),
    ("k", "previous"),
)

# `uad_lists.json` is 1.6 MB of JSON and the card needs it on every keystroke-driven
# navigation, so it is parsed once per (path, mtime, size) rather than once per request. The
# key is the file's own identity rather than a clock: an operator who swaps the file in gets
# the new one on the next request, and nothing here has to be told to expire.
_UPSTREAM_CACHE: dict[tuple[str, int, int], UpstreamList] = {}

# How each action reads once it has happened. A map rather than a suffix rule: `defer`
# doubles its r and `approve` drops its e, so the rule is a table however it is spelled.
PAST_TENSE: dict[str, str] = {
    "approve": "approved",
    "reject": "rejected",
    "defer": "deferred",
    "edit": "edited",
    "reopen": "reopened",
}


@dataclass(frozen=True, slots=True)
class Board:
    """Everything one render of the screen needs. Built here so the template branches on
    named states rather than on the truthiness of half a dozen collections."""

    view: str
    rows: tuple[QueueRow, ...]
    counts: dict[str, int]
    candidate: Candidate | None
    # `empty` (nothing at all), `cleared` (this view is done, others are not), `ok`.
    state: str
    error: str | None = None
    notice: str | None = None
    undo: str | None = None
    upstream_error: str | None = None
    edit_open: bool = False


def _upstream(path: Path) -> tuple[UpstreamList | None, str | None]:
    """The operator's `uad_lists.json`, or the reason the card has no neighbours.

    Returned rather than raised: the anchors are one section of one card, and a screen that
    404s because a comparison file is missing has thrown away the floor, the evidence and the
    proposal along with it.
    """
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = _UPSTREAM_CACHE.get(key)
        if cached is None:
            cached = load_upstream_list(path)
            _UPSTREAM_CACHE.clear()
            _UPSTREAM_CACHE[key] = cached
        return cached, None
    except (OSError, UpstreamListError) as exc:
        logger.warning("triage: upstream list unavailable at %s: %s", path, exc)
        return None, (
            f"no upstream list at {path}, so this card has nothing to compare against. "
            "point UPSTREAM_LIST_PATH at a copy of uad_lists.json."
        )


async def _board(
    *,
    view: str,
    package: str | None,
    error: str | None = None,
    notice: str | None = None,
    undo: str | None = None,
    edit_open: bool = False,
) -> Board:
    settings = get_settings()
    session_factory = get_session_factory()
    upstream, upstream_error = _upstream(settings.upstream_list_path)

    try:
        async with session_factory() as session:
            counts = await triagestore.load_counts(session)
            rows = await triagestore.load_queue(session, view=view)
            selected = package if any(row.package == package for row in rows) else None
            if selected is None and package is not None and error is None:
                error = f"{package} is not in the {view} list. showing the top of it instead."
            if selected is None and rows:
                selected = rows[0].package
            candidate = (
                await triagestore.load_candidate(session, selected, upstream=upstream)
                if selected
                else None
            )
    except (SQLAlchemyError, OSError):
        # The read failed rather than settling empty, and those are different screens: an
        # empty state rendered from a failed read tells the reviewer their queue is done. The
        # traceback goes to the log; what reaches the screen is what to do about it.
        logger.exception("triage: the queue could not be read")
        return Board(
            view=view,
            rows=(),
            counts=dict.fromkeys(VIEWS, 0),
            candidate=None,
            state="error",
            error="the queue could not be read. the database is not answering; check it is up.",
            upstream_error=upstream_error,
        )

    if rows:
        state = "ok"
    elif any(counts.values()):
        state = "cleared"
    else:
        state = "empty"
    return Board(
        view=view,
        rows=rows,
        counts=counts,
        candidate=candidate,
        state=state,
        error=error,
        notice=notice,
        undo=undo,
        upstream_error=upstream_error,
        edit_open=edit_open,
    )


def _context(board: Board) -> dict[str, Any]:
    return {
        "active_nav": "triage",
        "board": board,
        "keybinds": KEYBINDS,
        "action_buttons": ACTION_BUTTONS,
        "reject_key": REJECT_KEY,
        "edit_key": EDIT_KEY,
        "views": VIEWS,
    }


def _render(request: Request, board: Board) -> Response:
    """A fragment when htmx asked for one, the whole page otherwise.

    The same board either way: an action performed from a keyboard and the same action
    performed by a no-JS browser following a redirect land on identical markup, so nothing
    about the screen depends on which one happened.
    """
    if web.is_htmx(request):
        return web.partial(request, "partials/triage_board.html", _context(board))
    return web.page(request, "triage.html", _context(board))


def _redirect(view: str, package: str | None, *, done: str, undo: str | None) -> RedirectResponse:
    """Post-redirect-get, carrying just enough for the confirmation to survive it.

    `done` is an action NAME rather than a message: the sentence is composed from
    `PAST_TENSE` on the way back out, so a crafted link cannot put words on the screen. The
    no-JS path needs this — htmx keeps the banner by swapping the board in place.
    """
    query = [f"view={quote(view, safe='')}", f"done={quote(done, safe='')}"]
    if package:
        query.append(f"package={quote(package, safe='')}")
    if undo:
        query.append(f"undo={quote(undo, safe='')}")
    return RedirectResponse("/triage?" + "&".join(query), status_code=status.HTTP_303_SEE_OTHER)


@router.get("")
async def triage_screen(
    request: Request,
    view: str = "queue",
    package: str | None = None,
    done: str | None = None,
    undo: str | None = None,
) -> Response:
    error = None
    if view not in VIEWS:
        error = f"{view!r} is not a list. showing the queue."
        view = "queue"
    notice = f"{PAST_TENSE[done]}." if done in PAST_TENSE else None
    return _render(
        request,
        await _board(
            view=view,
            package=package,
            error=error,
            notice=notice,
            undo=undo if notice else None,
        ),
    )


@router.post("/decide")
async def decide(
    request: Request,
    package: str = Form(...),
    action: str = Form(...),
    reason: str | None = Form(default=None),
    view: str = Form(default="queue"),
) -> Response:
    """Record one verdict and answer with the board it produced.

    A verdict moves the reviewer to the top of what is left, because the queue is ranked and
    the top is by definition the next most valuable thing to look at. An `edit` or a `reopen`
    keeps the package in front of them: neither is a verdict, and being thrown to a different
    package after editing one is how an edit gets left unapproved.
    """
    at = datetime.now(UTC)
    session_factory = get_session_factory()
    try:
        async with session_factory() as session, session.begin():
            await triagestore.decide(session, package=package, action=action, reason=reason, at=at)
    except ReasonRequired:
        # In the screen's own words. The store's message names the function and the invariant,
        # which is what a log wants; what a reviewer needs is the next action.
        return _render(
            request,
            await _board(
                view=view,
                package=package,
                error="a rejection needs a reason. one line is enough, and it is kept.",
            ),
        )
    except TriageError as exc:
        return _render(request, await _board(view=view, package=package, error=str(exc)))

    stay = action in triagestore.OPEN_ACTIONS
    undo = None if stay else package
    if web.is_htmx(request):
        board = await _board(
            view=view,
            package=package if stay else None,
            notice=f"{PAST_TENSE.get(action, action)} {package}.",
            undo=undo,
        )
        return _render(request, board)
    return _redirect(view, package if stay else None, done=action, undo=undo)


@router.post("/edit")
async def edit(
    request: Request,
    package: str = Form(...),
    description: str = Form(default=""),
    list_: str = Form(default="", alias="list"),
    removal: str = Form(default=""),
    confidence: str = Form(default=""),
    view: str = Form(default="queue"),
) -> Response:
    """Apply a reviewer's edits, which land as `human:`-provenance fields.

    Only fields the form actually sent are offered, so an empty box is "unchanged" rather
    than "set this to empty" — the alternative silently blanks a description whenever a
    browser omits a disabled field.
    """
    at = datetime.now(UTC)
    edits = {
        field: value.strip()
        for field, value in (
            ("description", description),
            ("list", list_),
            ("removal", removal),
            ("confidence", confidence),
        )
        if value.strip()
    }
    session_factory = get_session_factory()
    try:
        async with session_factory() as session, session.begin():
            changed = await triagestore.apply_edit(session, package=package, edits=edits, at=at)
    except (TriageError, BelowFloorError, ValueError) as exc:
        return _render(
            request,
            await _board(view=view, package=package, error=str(exc), edit_open=True),
        )

    if web.is_htmx(request):
        notice = (
            "edited " + ", ".join(sorted(changed)) + f" on {package}."
            if changed
            else f"nothing changed on {package}."
        )
        return _render(request, await _board(view=view, package=package, notice=notice))
    return _redirect(view, package, done="edit" if changed else "", undo=None)
