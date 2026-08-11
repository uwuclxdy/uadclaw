"""Pixel factory images from `developers.google.com/android/images`.

The index is behind a client-side terms wall, verified 2026-08-10 and again 2026-08-11: a
bare GET answers HTTP 200 with ~75 KB of prose and ZERO download links, the same shape a
healthy-but-empty index would have. With `Cookie: devsite_wall_acks=nexus-image-tos` the
same URL answers ~1.2 MB carrying 2293 factory zips. That acceptance is the operator's, so
the cookie is configuration (`pixel_terms_ack_cookie_*`), never a constant hidden in a fetch
helper, and a zero-link parse is a hard error rather than "no builds available".
"""

import logging
import re
from pathlib import Path

import httpx

from uadclaw.firmware import (
    EmptyFirmwareIndexError,
    FirmwareDriver,
    FirmwareError,
    FirmwareInputError,
    FirmwareRef,
    FirmwareTermsNotAcknowledgedError,
    TermsPosture,
    TermsRisk,
    download_to_file,
)
from uadclaw.settings import Settings

logger = logging.getLogger(__name__)

# One table row per build. Parsed row by row, not with one page-wide regex, so a row that
# breaks shape can never pair its neighbour's checksum with this row's URL.
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
# Neither the codename nor the build id can contain "-", so splitting the basename on it is
# unambiguous. Build ids are usually lowercase in the URL and uppercase in the page label,
# but not always (`razorg-JLS36C-factory-834eab41.zip`, the one exception in 2293 links), so
# both cases are accepted here and the canonical uppercase form is produced below.
_FACTORY_URL_RE = re.compile(
    r'href="(?P<url>https://dl\.google\.com/dl/android/aosp/'
    r"(?P<device>[A-Za-z0-9_]+)-(?P<build>[A-Za-z0-9._]+)-factory-[0-9a-f]+\.zip)\"",
)
_SHA256_RE = re.compile(r"<td>\s*(?P<sha256>[0-9a-f]{64})\s*</td>", re.IGNORECASE)
# e.g. data-label="tegu for Pixel 9a [CP2A.260705.006]". The build id is always the LAST
# bracket group, and the marketing name may itself contain brackets ("Nexus 7 [2013]
# (Mobile)") or be empty ("stallion for  [...]"), so it stays optional downstream.
_LABEL_RE = re.compile(r'data-label="[^"]*? for (?P<marketing_name>[^"]*?)\s*\[[^\[\]"]*\]"')
# e.g. <td>17.0.0 (CP2A.260705.006, Jul 2026)</td>
_VERSION_RE = re.compile(r"<td>\s*(?P<version>\d[\w.]*)\s*\(")


def parse_factory_index(html: str, *, source_url: str) -> list[FirmwareRef]:
    """Every factory build in the index page, in page order (oldest first per device).

    Raises `EmptyFirmwareIndexError` when the page parses to nothing. That is the whole
    point of this function existing separately from the fetch: the terms-walled page is a
    HTTP 200 with real content and no links, so "zero rows" has to be loud here rather than
    flowing on as an empty list a caller reads as "this source has no builds".
    """
    refs: list[FirmwareRef] = []
    for row in _ROW_RE.finditer(html):
        body = row.group(1)
        link = _FACTORY_URL_RE.search(body)
        if link is None:
            continue
        sha = _SHA256_RE.search(body)
        label = _LABEL_RE.search(body)
        version = _VERSION_RE.search(body)
        marketing = label.group("marketing_name").strip() if label else ""
        refs.append(
            FirmwareRef(
                driver=PixelDriver.name,
                device=link.group("device"),
                build=link.group("build").upper(),
                url=link.group("url"),
                sha256=sha.group("sha256").lower() if sha else None,
                android_version=version.group("version") if version else None,
                marketing_name=marketing or None,
            )
        )
    if not refs:
        raise EmptyFirmwareIndexError(
            f"parse_factory_index: {source_url} returned {len(html)} characters containing no "
            "factory-image links at all. That is what the terms wall looks like, not an empty "
            "catalogue: set PIXEL_TERMS_ACK_COOKIE_VALUE=nexus-image-tos (accepting Google's "
            "factory-image terms) and retry."
        )
    return refs


class PixelDriver(FirmwareDriver):
    name = "pixel"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._index_url = settings.pixel_index_url
        self._cookie_name = settings.pixel_terms_ack_cookie_name
        self._cookie_value = settings.pixel_terms_ack_cookie_value.strip()
        self._timeout = settings.firmware_http_timeout_seconds
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.ACKNOWLEDGEMENT,
            source_url=self._index_url,
            summary=(
                "Google's official factory-image terms. The index serves no download links "
                "until they are acknowledged; the operator records that acceptance as "
                f"{self._cookie_name}=nexus-image-tos in configuration."
            ),
            acknowledged=bool(self._cookie_value),
        )

    def _ack_headers(self) -> dict[str, str]:
        if not self._cookie_value:
            raise FirmwareTermsNotAcknowledgedError(
                "PixelDriver: Google's factory-image terms have not been acknowledged, so "
                f"{self._index_url} would answer 200 with zero download links. Set "
                "PIXEL_TERMS_ACK_COOKIE_VALUE=nexus-image-tos to record acceptance, or "
                "disable the driver with DISABLED_FIRMWARE_DRIVERS=pixel."
            )
        return {"Cookie": f"{self._cookie_name}={self._cookie_value}"}

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), follow_redirects=True), True

    async def list_available(self) -> list[FirmwareRef]:
        headers = self._ack_headers()
        client, owned = self._open_client()
        try:
            try:
                response = await client.get(self._index_url, headers=headers)
            except httpx.HTTPError as exc:
                raise FirmwareError(
                    f"PixelDriver.list_available: {self._index_url} is unreachable: {exc}"
                ) from exc
            if response.status_code != httpx.codes.OK:
                raise FirmwareError(
                    f"PixelDriver.list_available: {self._index_url} answered HTTP "
                    f"{response.status_code}; expected 200"
                )
            refs = parse_factory_index(response.text, source_url=self._index_url)
        finally:
            if owned:
                await client.aclose()
        logger.info("pixel index: %d factory builds", len(refs))
        return refs

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> Path:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"PixelDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        headers = self._ack_headers()
        client, owned = self._open_client()
        try:
            return await download_to_file(
                client,
                ref.url,
                dest_dir / ref.archive_filename,
                headers=headers,
                expected_sha256=ref.sha256,
            )
        finally:
            if owned:
                await client.aclose()
