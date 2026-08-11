"""ASGI entrypoint: `uvicorn uadclaw.asgi:app`."""

from uadclaw.app import create_app

app = create_app()
