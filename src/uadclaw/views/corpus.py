"""The corpus screen. Placeholder: the router exists and is wired so the nav resolves;
the screen itself is task 16's corpus lane.
"""

from fastapi import APIRouter, Request

from uadclaw import web

router = APIRouter()


@router.get("/corpus")
async def corpus_screen(request: Request):
    return web.page(request, "corpus.html", {"active_nav": "corpus"})
