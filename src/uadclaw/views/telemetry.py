"""The telemetry screen. Placeholder: the router exists and is wired so the nav resolves;
the screen itself is task 16's telemetry lane.
"""

from fastapi import APIRouter, Request

from uadclaw import web

router = APIRouter()


@router.get("/telemetry")
async def telemetry_screen(request: Request):
    return web.page(request, "telemetry.html", {"active_nav": "telemetry"})
