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

from uadclaw import triagestore, web
from uadclaw.classify import UNKNOWN
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
    "defer": "skipped",
    "edit": "edited",
    "reopen": "reopened",
}

# What each list is CALLED, against what it is. Display strings only: the value in the url,
# in `VIEWS` and in every branch of `triagestore.in_view` stays what it is, because a rename
# there is a migration and this is a label.
VIEW_LABELS: dict[str, str] = {
    "queue": "queue",
    "deferred": "skipped",
    "decided": "decided",
    "parked": "no answer",
}

# The corroboration statuses, in the reviewer's words. A MAP rather than `.replace("_", " ")`
# on whatever the column holds: that turns a status nobody wrote a label for into a sentence
# that looks authored, and `corroborated` has to become a count anyway, which no substitution
# can do. An unmapped value falls through as itself, which reads as the raw value it is.
CORROBORATION_LABELS: dict[str, str] = {
    "uncorroborated": "no sources",
    "search_failed": "search failed",
    "judge_failed": "check failed",
}


def corroboration_label(status: str | None, sources: int) -> str:
    """How a corroboration verdict reads on the card.

    `corroborated` is spelled as its evidence rather than as its name, because the count is
    what a reviewer decides on and the word was the one the user named as unclear.
    """
    if status is None:
        return "not searched yet"
    if status == "corroborated":
        return f"{sources} source{'' if sources == 1 else 's'}"
    return CORROBORATION_LABELS.get(status, status)


# Every badge this screen can render, with the one line that says what it means. Rendered
# once, in a `<details>` a keyboard reaches with one Tab, rather than as a `title` on each
# badge: a `title` on a non-focusable `<span>` exists for a pointer and for nothing else.
BADGE_MEANINGS: tuple[tuple[str, str], ...] = (
    ("removal", "the rating this entry would ship with, never below the minimum rating."),
    ("no answer", "deepseek could not answer. read the reason, then edit or reject."),
    ("conflict", "devices disagree on a fact. shown, never filtered on."),
    ("n sources", "how many sources back the description. 13.6% of packages have any."),
)


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
    # What takes focus once htmx has swapped the board in. Decided here rather than in the
    # script, because the server is what knows why this render happened: a swap leaves focus
    # on `<body>`, so the next Tab restarts at the top of the document and a refusal the
    # reviewer has to answer is somewhere behind them. One of `decide` (the column holding
    # the callouts and the card), `reason`, `edit`.
    focus: str = "decide"


def _valid_view(view: str) -> tuple[str, str | None]:
    """The list to render, and the reason it is not the one that was asked for.

    Parsed at the boundary rather than trusted, on all three entry points. The GET validated
    it and the two POSTs did not, so a form carrying a made-up view raised out of
    `load_queue` into a handler that catches only its own input errors — a 500 on a write.
    It also reaches an `href` in the rendered board, and a value drawn from a fixed set never
    needs encoding, which is the cheaper half of the same rule.
    """
    if view in VIEWS:
        return view, None
    return "queue", f"{view!r} is not a list. showing the queue."


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
            f"no upstream list at {path}. point UPSTREAM_LIST_PATH at a copy of uad_lists.json."
        )


async def _board(
    *,
    view: str,
    package: str | None,
    error: str | None = None,
    notice: str | None = None,
    undo: str | None = None,
    edit_open: bool = False,
    focus: str = "decide",
) -> Board:
    settings = get_settings()
    session_factory = get_session_factory()
    upstream, upstream_error = _upstream(settings.upstream_list_path)

    try:
        async with session_factory() as session:
            # One read for both, so the counts on the filters and the rows beside them
            # describe the same instant. See `triagestore.load_board`.
            rows, counts = await triagestore.load_board(session, view=view)
            selected = package if any(row.package == package for row in rows) else None
            if selected is None and package is not None and error is None:
                label = VIEW_LABELS[view]
                error = f"{package} is not in the {label} list. showing the top of it instead."
            if selected is None and rows:
                selected = rows[0].package
            candidate = (
                await triagestore.load_candidate(session, selected, upstream=upstream)
                if selected
                else None
            )
    except web.DB_UNREACHABLE:
        # The read failed rather than settling empty, and those are different screens: an
        # empty state rendered from a failed read tells the reviewer their queue is done. The
        # traceback goes to the log; what reaches the screen is what to do about it. Which
        # exception classes count as "not answering", and why each was measured, is
        # `web.DB_UNREACHABLE`'s — this screen's answer was the one the other three drifted
        # away from, so it is the one that moved to the shared seam.
        logger.exception("triage: the queue could not be read")
        return Board(
            view=view,
            rows=(),
            counts=dict.fromkeys(VIEWS, 0),
            candidate=None,
            state="error",
            error=f"the queue could not be read. {web.DB_UNREACHABLE_MESSAGE}",
            upstream_error=upstream_error,
            focus=focus,
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
        focus=focus,
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
        "view_labels": VIEW_LABELS,
        "badge_meanings": BADGE_MEANINGS,
        "corroboration_label": corroboration_label,
        # Passed as a callable rather than pre-rendered onto every row: a chip is derived
        # from the package name and nothing else, so a field on the row would be the same
        # value written twice.
        "monogram": triagestore.monogram,
        "unknown_value": UNKNOWN,
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
    view, error = _valid_view(view)
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
    view, view_error = _valid_view(view)
    session_factory = get_session_factory()
    try:
        async with session_factory() as session, session.begin():
            await triagestore.decide(session, package=package, action=action, reason=reason, at=at)
    except ReasonRequired:
        # In the screen's own words. The store's message names the function and the invariant,
        # which is what a log wants; what a reviewer needs is the next action — and the field
        # it is about, which is why focus goes there rather than to the top of the card.
        return _render(
            request,
            await _board(
                view=view,
                package=package,
                error=view_error or "a rejection needs a reason. one line, and it is kept.",
                focus="reason",
            ),
        )
    except TriageError as exc:
        return _render(
            request, await _board(view=view, package=package, error=view_error or str(exc))
        )
    except web.DB_UNREACHABLE:
        # A verdict is the one thing this screen exists to record, so a failure to record it
        # has to be visible. htmx swaps neither a 4xx nor a 5xx, so a 500 here disabled the
        # button, swapped nothing, and the reviewer's verdict vanished with no message at all
        # — the silent failure, wearing a correct status code.
        logger.exception("triage: recording a %s for %s failed", action, package)
        return _render(
            request,
            await _board(
                view=view,
                package=package,
                error=view_error or f"nothing was recorded. {web.DB_UNREACHABLE_MESSAGE}",
            ),
        )

    stay = action in triagestore.OPEN_ACTIONS
    undo = None if stay else package
    if web.is_htmx(request):
        board = await _board(
            view=view,
            package=package if stay else None,
            error=view_error,
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
    unknown_description: str | None = Form(default=None),
    list_: str = Form(default="", alias="list"),
    removal: str = Form(default=""),
    confidence: str = Form(default=""),
    view: str = Form(default="queue"),
) -> Response:
    """Apply a reviewer's edits, which land as `human:`-provenance fields.

    Only fields the form actually sent are offered, so an empty box is "unchanged" rather
    than "set this to empty" — the alternative silently blanks a description whenever a
    browser omits a disabled field.

    `unknown_description` is the control the description rule's own message asks for. That
    message is shared with the model path deliberately, so it tells a reviewer to declare the
    field in `unknown_fields`, and until this box existed the screen offered no way to do it:
    the only answers were a description long enough to pass a floor the reviewer could not
    honestly reach, or nothing. The box wins over the text area, because a reviewer who ticks
    it has said the box above is not an answer.
    """
    at = datetime.now(UTC)
    if unknown_description is not None:
        description = UNKNOWN
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
    view, view_error = _valid_view(view)
    session_factory = get_session_factory()
    try:
        async with session_factory() as session, session.begin():
            changed = await triagestore.apply_edit(session, package=package, edits=edits, at=at)
    except (TriageError, BelowFloorError, ValueError) as exc:
        return _render(
            request,
            await _board(
                view=view,
                package=package,
                error=view_error or str(exc),
                edit_open=True,
                focus="edit",
            ),
        )
    except web.DB_UNREACHABLE:
        # Same reason as `decide` above, plus one of its own: the form is still open and the
        # reviewer's typing is still in it, so the edit is retryable the moment the database
        # answers. A 500 would have thrown the text away along with the message.
        logger.exception("triage: applying an edit to %s failed", package)
        return _render(
            request,
            await _board(
                view=view,
                package=package,
                error=view_error or f"the edit was not saved. {web.DB_UNREACHABLE_MESSAGE}",
                edit_open=True,
                focus="edit",
            ),
        )

    if web.is_htmx(request):
        notice = (
            "edited " + ", ".join(sorted(changed)) + f" on {package}."
            if changed
            else f"nothing changed on {package}."
        )
        return _render(
            request, await _board(view=view, package=package, error=view_error, notice=notice)
        )
    return _redirect(view, package, done="edit" if changed else "", undo=None)
