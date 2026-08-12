"""`GET /icons/{package}`: the stored launcher icon, or a 404.

Behind the default-deny `AuthMiddleware` like every other route — it protects itself by
existing, and neither `PUBLIC_PATHS` nor `PUBLIC_PREFIXES` grows for it.

Not a screen: an `<img src>` wants bytes or a status, never a rendered error page. So this is
the one view in the package that answers a status rather than a 200 with a visible message,
and the screens that decide whether to point an `<img>` here read `has_icon` off their own
query rather than probing this route.

Three response headers carry the whole safety story, because the payload is a document
generated from bytes a vendor firmware image supplied:

- `Content-Type` is re-derived from a fixed set rather than echoed out of the column, so a row
  written by anything other than `icons.extract_icon` still cannot name its own type;
- `X-Content-Type-Options: nosniff` stops a browser re-reading a PNG body as HTML when it
  disagrees with the declared type — the classic content-sniffing XSS;
- `Content-Security-Policy` is what makes an SVG safe to serve. An SVG opened at its own URL
  is a document, not just an `<img>` payload: `default-src 'none'` blocks every script,
  stylesheet, font and fetch it could name, `sandbox` puts it in an opaque origin with
  scripting off, `frame-ancestors 'none'` keeps it out of anyone else's frame, and
  `base-uri`/`form-action 'none'` close the two directives `default-src` does not cover.
"""

import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import select

from uadclaw import web
from uadclaw.db import get_session_factory
from uadclaw.icons import ICON_MIMES, MAX_ICON_BYTES
from uadclaw.models import PackageFact

logger = logging.getLogger(__name__)

router = APIRouter()

# `package_facts.package` is a String(255): anything longer cannot name a row, so it is refused
# before it reaches a query. Deliberately a LENGTH bound and not a character class — a charset
# derived from the package names this pipeline has met would silently 404 every spelling it
# has not, and the query is parameterized either way.
MAX_PACKAGE_NAME = 255

ICON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'none'; sandbox; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    # Private because the whole dashboard is: this is behind a session cookie and must not
    # land in a shared cache. Short because a re-scan can replace the artwork and the `<img>`
    # carries no cache-buster.
    "Cache-Control": "private, max-age=300",
}


@router.get("/icons/{package}")
async def package_icon(package: str) -> Response:
    if len(package) > MAX_PACKAGE_NAME:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(PackageFact.icon_bytes, PackageFact.icon_mime).where(
                        PackageFact.package == package
                    )
                )
            ).first()
    except web.DB_UNREACHABLE:
        # Logged in full and answered as a status: an image cannot render its own error, and a
        # 200 with an error body would be served AS the icon.
        logger.exception("icon lookup failed for %s", package)
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    if row is None or not row.icon_bytes or row.icon_mime not in ICON_MIMES:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    if len(row.icon_bytes) > MAX_ICON_BYTES:
        # The write path caps this, so a row over the cap means something else wrote it. Loud
        # rather than served: the bound belongs to the response as much as to the column.
        logger.warning(
            "icon for %s is %d bytes, over the %d cap, and was not served",
            package,
            len(row.icon_bytes),
            MAX_ICON_BYTES,
        )
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    return Response(content=row.icon_bytes, media_type=row.icon_mime, headers=ICON_HEADERS)
