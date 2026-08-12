"""The triage screen. Placeholder: the router exists and is wired so the nav resolves;
the screen itself is task 16's triage lane.
"""

from fastapi import APIRouter, Request

from uadclaw import web

router = APIRouter()


@router.get("/triage")
async def triage_screen(request: Request):
    return web.page(request, "triage.html", {"active_nav": "triage"})
