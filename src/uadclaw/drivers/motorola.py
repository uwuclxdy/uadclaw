"""Motorola retail firmware from the lolinet mirror.

Motorola publishes no downloader but its own Windows-only Rescue and Smart Assistant.
`mirrors.lolinet.com/firmware/lenomola` has mirrored the same packages as a browsable tree
for years. The tree renders client-side through h5ai, but h5ai ships a JSON API that answers
a directory listing directly, which is what this driver reads (verified 2026-08-11):

    POST /_h5ai/public/index.php   action=get&items[href]=<path>&items[what]=1

Three measured facts shape this driver.

**There is no index document, only a tree**: 8 year directories, ~26 devices in each, 10-15
channel directories per device. Enumerating "everything on offer" would be thousands of
requests per job against a mirror that asks for non-commercial use, so `MOTOROLA_DEVICES`
names the codenames to crawl and the driver refuses rather than guessing.

**The API answers with the ancestor chain as well as the children.** A listing of one channel
came back with 68 items of which 5 were that channel's files; the rest were `/`, `/firmware/`
and every directory in between. Filtering to direct children is not tidiness, it is the
difference between reading a device's builds and reading the whole mirror's top level.

**A build id is not unique within a device.** Measured across 164 files on three devices, 16
of the 32 distinct (device, build) pairs appear in more than one channel — `V1TRS35H.60-33-5`
is published by 9 of `rtwo`'s 10 channels. So the channel is part of the ref's build
(`V1TRS35H.60-33-7_RETAIL`) rather than part of its device: putting it in `device` would make
one phone count as ten toward `package_facts.device_count`, which is the triage ranking
signal, and leaving it out entirely would let `select_ref` silently pick one channel's build
when the operator asked for another's.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

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

H5AI_API_PATH = "/_h5ai/public/index.php"
FIRMWARE_ROOT = "/firmware/lenomola/"

# Two filename shapes in the same tree, measured over 164 files:
#   XT2301-1_RTWO_BOOST_13_T1TRS33.43-48-41-3_subsidy-DISH_WILD_regulatory-...zip
#   RTWO_RETAIL_15_V1TRS35H.60-33-7_subsidy-DEFAULT_regulatory-...zip
#   XT2341-2_CANCUN_RETLA_14_UTLB34.102-62_a05daa66_release-keys_global_CFC.zip
# The leading fields are not a fixed count (the model SKU is present on some and absent on
# others), so the anchor is the trailing `_subsidy-` / `_<8 hex>_release-keys` marker and the
# two fields immediately before it. `^.*_` is greedy on purpose: it pins android/build to the
# LAST such pair, so a codename carrying a numeric field cannot be read as the version.
_FILENAME_RE = re.compile(
    r"^.*_(?P<android>\d{1,3})_(?P<build>[A-Za-z0-9][A-Za-z0-9.-]*)"
    r"_(?:subsidy-|[0-9a-f]{8}_release-keys)"
)
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_DEVICE_RE = re.compile(r"^[a-z0-9_]{1,64}$")

# How deep below `<year>/<device>/` the crawl goes looking for files. Measured layout is
# `official/<channel>/<file>`, so 2 is what exists and 3 is the margin.
_MAX_WALK_DEPTH = 3
# Per device. A tree that grew a symlink loop or a thousand channels becomes a named failure
# rather than an unbounded crawl of somebody else's mirror.
_MAX_LISTINGS_PER_DEVICE = 96


def _direct_children(items: list[dict[str, Any]], base: str) -> list[dict[str, Any]]:
    """The entries of `base` itself. h5ai answers with every ancestor directory as well, so
    an unfiltered read of `items` returns the mirror's root alongside the wanted listing."""
    children = []
    for item in items:
        href = str(item.get("href", ""))
        if not href.startswith(base) or href == base:
            continue
        if "/" in href[len(base) :].rstrip("/"):
            continue
        children.append(item)
    return children


def parse_firmware_filename(filename: str) -> tuple[str, str] | None:
    """`(android_version, build_id)` out of a mirror filename, or None when it is not one."""
    match = _FILENAME_RE.match(filename)
    if match is None:
        return None
    return match.group("android"), match.group("build")


class MotorolaDriver(FirmwareDriver):
    name = "motorola"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._mirror_url = settings.motorola_mirror_url.rstrip("/")
        self._devices = settings.motorola_device_names
        self._timeout = settings.firmware_http_timeout_seconds
        self._max_archive_bytes = settings.max_firmware_archive_bytes
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.RESTRICTED,
            source_url=f"{self._mirror_url}{FIRMWARE_ROOT}",
            summary=(
                "mirrors.lolinet.com, a third-party mirror of Motorola packages rather than "
                "Motorola's own infrastructure, which offers no downloader but a Windows-only "
                "desktop tool. The site states its files are free but NOT for commercial use, "
                "which is a real restriction on what may be done with what this driver "
                "downloads. No Motorola terms authorise automated bulk downloading either."
            ),
            # Nothing to acknowledge mechanically: the restriction is a stated condition of
            # use, not a gate, so there is no acceptance for the operator to record.
            acknowledged=True,
        )

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return driver_client(self._timeout), True

    async def _listdir(self, client: httpx.AsyncClient, path: str) -> list[dict[str, Any]]:
        url = f"{self._mirror_url}{H5AI_API_PATH}"
        try:
            response = await client.post(
                url, data={"action": "get", "items[href]": path, "items[what]": "1"}
            )
        except httpx.HTTPError as exc:
            raise FirmwareError(
                f"MotorolaDriver: {url} is unreachable while listing {path}: {exc}"
            ) from exc
        if response.status_code != httpx.codes.OK:
            raise FirmwareError(
                f"MotorolaDriver: {url} answered HTTP {response.status_code} while listing "
                f"{path}; expected 200"
            )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise FirmwareError(
                f"MotorolaDriver: the h5ai API answered non-JSON for {path}; the mirror may "
                "have dropped its JSON API or be serving an error page"
            ) from exc
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise FirmwareError(
                f"MotorolaDriver: the h5ai API answered no `items` list for {path} "
                f"({str(payload)[:200]}); the API contract changed"
            )
        return _direct_children(items, path)

    async def _year_directories(self, client: httpx.AsyncClient) -> list[str]:
        children = await self._listdir(client, FIRMWARE_ROOT)
        years = sorted(
            str(item["href"]) for item in children if str(item.get("href", "")).endswith("/")
        )
        if not years:
            raise EmptyFirmwareIndexError(
                f"MotorolaDriver: {self._mirror_url}{FIRMWARE_ROOT} lists no directories at "
                "all. It has carried 8 year directories since at least 2020, so zero means "
                "the mirror moved or the API stopped answering listings, not that Motorola "
                "published nothing."
            )
        return years

    async def _walk_zip_files(
        self, client: httpx.AsyncClient, base: str, *, budget: list[int], depth: int = 0
    ) -> list[dict[str, Any]]:
        """Every `.zip` under `base`, with the listing budget shared across the whole walk."""
        if depth > _MAX_WALK_DEPTH:
            return []
        if budget[0] <= 0:
            raise FirmwareError(
                f"MotorolaDriver: listing {base} would pass the {_MAX_LISTINGS_PER_DEVICE} "
                "directory-listing budget for one device. The mirror's layout grew deeper or "
                "wider than the measured `official/<channel>/` shape; look before raising it."
            )
        budget[0] -= 1
        found: list[dict[str, Any]] = []
        for item in await self._listdir(client, base):
            href = str(item.get("href", ""))
            if href.endswith("/"):
                found.extend(
                    await self._walk_zip_files(client, href, budget=budget, depth=depth + 1)
                )
            elif href.endswith(".zip"):
                found.append(item)
        return found

    def _ref_for(self, item: dict[str, Any], *, device: str) -> FirmwareRef | None:
        href = str(item["href"])
        segments = href.strip("/").split("/")
        filename = segments[-1]
        channel = segments[-2] if len(segments) >= 2 else ""
        parsed = parse_firmware_filename(filename)
        if parsed is None or not _CHANNEL_RE.match(channel):
            return None
        android, build_id = parsed
        try:
            return FirmwareRef(
                driver=self.name,
                device=device,
                build=f"{build_id}_{channel}",
                url=f"{self._mirror_url}{href}",
                android_version=android,
            )
        except ValueError as exc:
            logger.warning("motorola: %s did not validate as a ref: %s", href, exc)
            return None

    async def list_available(self) -> list[FirmwareRef]:
        if not self._devices:
            raise FirmwareInputError(
                "MotorolaDriver.list_available: no devices configured. lolinet publishes a "
                "directory tree and no index document, so enumerating every device would be "
                "thousands of requests against a non-commercial community mirror. Name the "
                "codenames to track, e.g. MOTOROLA_DEVICES=rtwo,bronco, or disable the driver "
                "with DISABLED_FIRMWARE_DRIVERS=motorola."
            )
        bad = [device for device in self._devices if not _DEVICE_RE.match(device)]
        if bad:
            raise FirmwareInputError(
                f"MotorolaDriver.list_available: MOTOROLA_DEVICES carries {bad!r}, which is not "
                "a lolinet codename (lowercase letters, digits and underscores, e.g. `rtwo` or "
                "`cancunf_retcn`)"
            )
        client, owned = self._open_client()
        refs: list[FirmwareRef] = []
        try:
            years = await self._year_directories(client)
            for device in self._devices:
                refs.extend(await self._device_refs(client, device, years))
        finally:
            if owned:
                await client.aclose()
        if not refs:
            raise EmptyFirmwareIndexError(
                f"MotorolaDriver.list_available: none of {list(self._devices)} resolved to a "
                f"single firmware zip under {self._mirror_url}{FIRMWARE_ROOT}. Check the "
                "codenames against the mirror: it files a device under the year it launched, "
                "so `rtwo` lives at /firmware/lenomola/2023/rtwo/."
            )
        logger.info(
            "motorola mirror: %d build(s) across %d device(s)",
            len(refs),
            len({ref.device for ref in refs}),
        )
        return refs

    async def _device_refs(
        self, client: httpx.AsyncClient, device: str, years: list[str]
    ) -> list[FirmwareRef]:
        """Every build for one device, oldest first.

        Every year directory is searched rather than only the first hit: the mirror files a
        device under its launch year, but a device that shipped across a boundary appears
        twice, and stopping at the first would silently drop half its builds.

        Oldest first by the mirror's own mtime, because `select_ref` resolves "newest" as the
        last row for a device when the build id carries no parseable date field, and a
        Motorola build id (`V1TRS35H.60-33-7`) does not.
        """
        budget = [_MAX_LISTINGS_PER_DEVICE]
        rows: list[tuple[int, FirmwareRef]] = []
        for year in years:
            # A year that does not carry this device lists nothing (h5ai answers a missing
            # path with the ancestor chain alone), so the walk costs one request and stops.
            for item in await self._walk_zip_files(client, f"{year}{device}/", budget=budget):
                ref = self._ref_for(item, device=device)
                if ref is not None:
                    rows.append((int(item.get("time") or 0), ref))
        if not rows:
            logger.warning(
                "motorola: %r resolved to no firmware zip under any of %d year directories",
                device,
                len(years),
            )
        rows.sort(key=lambda row: row[0])
        return [ref for _time, ref in rows]

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"MotorolaDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        # The mirror publishes an `.md5` sidecar for some builds and nothing for others, and
        # the h5ai listing carries no checksum at all, so a Motorola archive arrives
        # integrity-unverified and the acquire stage records that alongside its digest.
        client, owned = self._open_client()
        try:
            return await download_to_file(
                client,
                ref.url,
                dest_dir / ref.archive_filename,
                max_bytes=self._max_archive_bytes,
            )
        finally:
            if owned:
                await client.aclose()
