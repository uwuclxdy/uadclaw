"""Xiaomi / Redmi / POCO full ROMs off Xiaomi's own CDN.

Xiaomi publishes no index. `XiaomiFirmwareUpdater/miui-updates-tracker` has scraped the OTA
check API into `data/latest.yml` continuously for years, and that file is the index this
driver reads: 3,550 flat entries carrying `codename`, `version`, `link`, `md5`, `date`,
`android`, `name`, `branch` and `method`, measured 2026-08-11.

Three measured facts shape everything below.

**The host in the index is not the host to download from.** Every `link` points at
`bigota.d.miui.com`, which is fronted by a CloudFront distribution that fails intermittently:
one probe run answered HTTP 503 with a 999-byte error page to a plain GET, another answered
200 to three consecutive GETs, both on 2026-08-11 and both against the same URL. HEAD passes
in every case, with a correct Content-Length — so a reachability check built on HEAD reports
a healthy source while the download fails. `ultimateota.d.miui.com` answers 403. Only
`cdnorg.d.miui.com` served the same path reliably (verified by a full 1,523,069,732-byte
download whose md5 matched the index), so `cdn_url` rewrites the host and nothing here ever
trusts a HEAD.

**Xiaomi publishes md5, not sha256**, for 3,454 of the 3,550 entries. Verifying at all means
verifying in md5, which is why `download_to_file` takes the algorithm the source used.

**Only the `Recovery` half of the index is openable.** The 2,028 `Recovery` entries are zips
carrying a `payload.bin` — the chain `uadclaw.unpack` already reads. The 1,522 `Fastboot`
entries are `.tgz`, which matches no magic in the dispatch table, so listing them would hand
the acquire stage builds that always fail one stage later. 1,267 of the 1,275 devices publish
both, so the filter costs almost no coverage.

A further 164 of those 2,028 rows publish a plain `http://` link (all of them pre-2016
devices). `FirmwareRef` refuses a non-https URL, so they are counted, logged and dropped
rather than downloaded in the clear; the whole index parses to 1,864 offers as of
2026-08-11.
"""

import datetime
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from uadclaw.firmware import (
    DownloadedArchive,
    EmptyFirmwareIndexError,
    FirmwareDriver,
    FirmwareError,
    FirmwareInputError,
    FirmwareRef,
    TermsPosture,
    TermsRisk,
    download_to_file,
    driver_client,
)
from uadclaw.settings import Settings

logger = logging.getLogger(__name__)

# The index's own hosts, and the one that actually serves. Rewriting is a host substitution
# on the same path — Xiaomi's CDNs are aliases of one origin, verified by downloading the
# same path from `cdnorg` and matching the md5 `bigota`'s entry published.
BROKEN_CDN_HOSTS = frozenset({"bigota.d.miui.com", "ultimateota.d.miui.com"})
WORKING_CDN_HOST = "cdnorg.d.miui.com"

# The download method whose archive this pipeline can open. See the module docstring.
OPENABLE_METHOD = "Recovery"

# How many malformed rows are named individually before the log just carries the count.
_MAX_LOGGED_SKIPS = 5


def cdn_url(url: str) -> str:
    """Point a Xiaomi firmware URL at the CDN host that serves it.

    Applied when the index is parsed AND again in `fetch`, because a ref can arrive from a
    job's params carrying whatever URL an operator pasted out of the tracker.
    """
    parts = urlsplit(url)
    if parts.hostname not in BROKEN_CDN_HOSTS:
        return url
    return urlunsplit(parts._replace(netloc=WORKING_CDN_HOST))


def _entry_date(entry: dict[str, Any]) -> datetime.date:
    """The build's date, or the earliest representable one for the 4 entries of 3,550 that
    carry none — which parks them behind every dated build rather than ahead of it."""
    value = entry.get("date")
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.min


def _optional_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] or None


def parse_latest_index(text: str, *, source_url: str) -> list[FirmwareRef]:
    """Every openable build in the tracker's `latest.yml`, oldest first.

    Sorted by the index's own `date` field rather than left in file order, because the
    driver-agnostic `select_ref` resolves "newest" as the last row for a device when the
    build id carries no date — and no Xiaomi version string does (`V14.0.44.0.TGOMIXM`).
    Measured 2026-08-11: 297 of the 1,270 devices with a `Recovery` build list a build that
    is NOT the newest one last, so file order alone silently resolves to a stale build for
    nearly a quarter of the catalogue.

    Raises `EmptyFirmwareIndexError` rather than returning an empty list, for the same reason
    the Pixel parser does: a source that answers 200 with nothing usable is a failure, not a
    catalogue with no builds in it.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FirmwareError(
            f"parse_latest_index: {source_url} is not parseable YAML ({exc}). The tracker "
            "publishes one flat mapping per build; check whether the file moved or the repo "
            "changed layout."
        ) from exc
    if not isinstance(document, list):
        raise FirmwareError(
            f"parse_latest_index: {source_url} parsed to {type(document).__name__}, not the "
            "list of build mappings this index has always been"
        )

    dated: list[tuple[datetime.date, int, FirmwareRef]] = []
    skipped: list[str] = []
    for position, entry in enumerate(document):
        if not isinstance(entry, dict):
            skipped.append(f"entry {position} is {type(entry).__name__}, not a mapping")
            continue
        if entry.get("method") != OPENABLE_METHOD:
            continue
        md5 = _optional_text(entry.get("md5"), limit=32)
        try:
            ref = FirmwareRef(
                driver=XiaomiDriver.name,
                device=str(entry.get("codename", "")),
                build=str(entry.get("version", "")),
                url=cdn_url(str(entry.get("link", ""))),
                md5=md5.lower() if md5 else None,
                android_version=_optional_text(entry.get("android"), limit=64),
                marketing_name=_optional_text(entry.get("name"), limit=128),
            )
        except ValueError as exc:
            skipped.append(f"entry {position} ({entry.get('codename')!r}): {exc}")
            continue
        dated.append((_entry_date(entry), position, ref))

    if skipped:
        logger.warning(
            "xiaomi index: %d of %d entries did not validate and are not on offer: %s",
            len(skipped),
            len(document),
            "; ".join(skipped[:_MAX_LOGGED_SKIPS]),
        )
    if not dated:
        raise EmptyFirmwareIndexError(
            f"parse_latest_index: {source_url} parsed {len(document)} entries and not one of "
            f"them is a usable {OPENABLE_METHOD} build. The tracker has published thousands "
            "continuously since 2015, so zero means the file moved, its schema changed, or "
            "the fetch got a placeholder — not that Xiaomi shipped nothing."
        )
    dated.sort(key=lambda row: (row[0], row[1]))
    return [ref for _date, _position, ref in dated]


class XiaomiDriver(FirmwareDriver):
    name = "xiaomi"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._index_url = settings.xiaomi_index_url
        self._timeout = settings.firmware_http_timeout_seconds
        self._max_archive_bytes = settings.max_firmware_archive_bytes
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.PUBLIC,
            source_url=self._index_url,
            summary=(
                "Xiaomi's own CDN, unauthenticated and ungated, indexed by the community "
                "XiaomiFirmwareUpdater tracker that has scraped it publicly for years with "
                "no recorded objection. Xiaomi has published no API and no terms grant, so "
                "this is tolerated rather than authorised, as is every source here: no "
                "OEM's terms were found to authorise automated bulk downloading."
            ),
            # Nothing to acknowledge: there is no gate, and no acceptance to record.
            acknowledged=True,
        )

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return driver_client(self._timeout), True

    async def list_available(self) -> list[FirmwareRef]:
        client, owned = self._open_client()
        try:
            try:
                response = await client.get(self._index_url)
            except httpx.HTTPError as exc:
                raise FirmwareError(
                    f"XiaomiDriver.list_available: {self._index_url} is unreachable: {exc}"
                ) from exc
            if response.status_code != httpx.codes.OK:
                raise FirmwareError(
                    f"XiaomiDriver.list_available: {self._index_url} answered HTTP "
                    f"{response.status_code}; expected 200"
                )
            refs = parse_latest_index(response.text, source_url=self._index_url)
        finally:
            if owned:
                await client.aclose()
        logger.info(
            "xiaomi index: %d %s builds across %d devices",
            len(refs),
            OPENABLE_METHOD.lower(),
            len({ref.device for ref in refs}),
        )
        return refs

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"XiaomiDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        url = cdn_url(ref.url)
        if url != ref.url:
            logger.info("xiaomi: %s rewritten to the CDN host that serves it (%s)", ref.url, url)
        digest = ref.published_digest()
        client, owned = self._open_client()
        try:
            return await download_to_file(
                client,
                url,
                dest_dir / ref.archive_filename,
                expected_digest=digest[1] if digest else None,
                digest_algorithm=digest[0] if digest else "sha256",
                max_bytes=self._max_archive_bytes,
            )
        finally:
            if owned:
                await client.aclose()
