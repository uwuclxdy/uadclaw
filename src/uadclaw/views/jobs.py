"""The jobs screen. Placeholder: the router exists and is wired so the nav resolves;
the screen itself is task 16's jobs lane.
"""

from fastapi import APIRouter, Request

from uadclaw import web

router = APIRouter()


@router.get("/jobs")
async def jobs_screen(request: Request):
    return web.page(request, "jobs.html", {"active_nav": "jobs"})
