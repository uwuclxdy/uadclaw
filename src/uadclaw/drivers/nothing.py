"""Nothing Phone firmware from the `spike0en/nothing_archive` GitHub releases.

Nothing runs no public firmware index. `nothing_archive` re-uploads pulls from Nothing's own
OTA servers as one GitHub release per build — 229 releases measured 2026-08-11, tagged
`<Codename>_<Build>` (`FroggerPro_B4.1-260723-1820`), with the APK-bearing partitions in a
split 7-Zip volume set `<TAG>-image-logical.7z.001`, `.002`, ... up to `.006`.

Two measured facts shape this driver.

**A split 7z volume set is a raw split of one archive.** Volume `.001` opens with the 7z
magic and `.002`/`.003` carry no magic at all (measured on `FroggerPro_B4.1-260723-1820`), so
concatenating them in order reproduces the original `.7z` byte for byte. `fetch` therefore
streams the volumes into a single file and deletes each one as it is consumed: the driver
hands back one archive with one digest, `uadclaw.unpack` sees one container, and nothing is
left for the unpack stage's retention to miss — it deletes the archive it was told about, and
a sibling volume it was never told about would have survived the whole job.

**The published hashes do not cover the download.** `<TAG>-hash.sha256` lists the INNER image
files (`abl.img`, `aop.img`, ...), not the archives, so it can only be checked after
extraction. Nothing here pretends otherwise: every Nothing archive comes back
`integrity_verified=False` and the acquire stage logs it.
"""

import asyncio
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import IO, Any

import httpx

from uadclaw.firmware import (
    DownloadedArchive,
    EmptyFirmwareIndexError,
    FirmwareDownloadError,
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

# `<PREFIX>-image-logical.7z.001` and its siblings: the logical partitions, which is where
# every preinstalled APK lives. `-image-boot`, `-image-firmware` and the hash files are not
# read. The prefix is NOT assumed to equal the release tag — 10 of 229 releases name their
# assets after something else (`Spacewar-T1.5-230619-0042_1.5.5-image-logical.7z.001` under
# tag `Spacewar_T1.5-230619-0042`) and requiring the two to match dropped all ten silently.
_LOGICAL_RE = re.compile(r"^(?P<prefix>.+)-image-logical\.7z(?:\.(?P<volume>\d{3}))?$")
# The whole-image fallback. 7 of 229 releases publish it ALONGSIDE the logical set rather
# than instead of it, so the logical set wins outright: taking both would pair two archives
# that each claim to be volume 1.
_SINGLE_RE = re.compile(r"^(?P<prefix>.+)-image\.7z$")
# `FroggerPro_B4.1-260723-1820`. 6 of 229 tags are release-tooling artifacts
# (`0.0.0-dev+spacewar.230719`) that carry no codename and are skipped.
_TAG_RE = re.compile(r"^(?P<device>[A-Za-z0-9]{1,64})_(?P<build>[A-Za-z0-9][A-Za-z0-9._-]{0,63})$")

_PAGE_SIZE = 100
# 229 releases today. The cap turns a paginating endpoint that never stops into a named
# failure instead of a worker that requests pages until GitHub rate-limits it.
_MAX_PAGES = 20

_COPY_CHUNK_BYTES = 4 * 1024 * 1024


def _asset_volumes(assets: list[dict[str, Any]], *, tag: str) -> list[str]:
    """The archive's download URLs in volume order, or an empty list when this release ships
    no readable image archive at all (1 of 229 releases)."""
    logical: dict[str, list[tuple[int, str]]] = {}
    single: dict[str, list[tuple[int, str]]] = {}
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if not url.startswith("https://"):
            continue
        if match := _LOGICAL_RE.match(name):
            logical.setdefault(match.group("prefix"), []).append(
                (int(match.group("volume") or 1), url)
            )
        elif match := _SINGLE_RE.match(name):
            single.setdefault(match.group("prefix"), []).append((1, url))
    sets = logical or single
    if not sets:
        return []
    if len(sets) > 1:
        raise FirmwareError(
            f"_asset_volumes: release {tag} carries {len(sets)} differently-named image "
            f"archives ({sorted(sets)}); refusing to guess which one is this build's"
        )
    numbered = next(iter(sets.values()))
    numbers = [number for number, _url in numbered]
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        raise FirmwareError(
            f"_asset_volumes: release {tag} publishes 7z volumes {sorted(numbers)}, which is "
            f"not the complete run 1..{len(numbers)}. A gap means the release is incomplete, "
            "and joining what is there would produce a truncated archive."
        )
    return [url for _number, url in sorted(numbered)]


def parse_releases(payload: str, *, source_url: str) -> list[FirmwareRef]:
    """Every release carrying a readable image archive, oldest first.

    Oldest first because `select_ref` resolves "newest" as the last row for a device when the
    build id carries no parseable Android date field, and a Nothing build id
    (`B4.1-260723-1820`) does not. GitHub answers newest first, so the order is reversed here
    rather than left to chance.
    """
    try:
        releases = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FirmwareError(
            f"parse_releases: {source_url} did not answer JSON ({exc}); GitHub serves an error "
            "document with a 200 in some rate-limited shapes, so check the body"
        ) from exc
    if not isinstance(releases, list):
        raise FirmwareError(
            f"parse_releases: {source_url} answered {type(releases).__name__}, not the list of "
            f"releases this endpoint returns. GitHub reports rate limiting this way: "
            f"{str(releases)[:200]}"
        )

    refs: list[FirmwareRef] = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        tag = str(release.get("tag_name", ""))
        named = _TAG_RE.match(tag)
        if named is None:
            continue
        assets = release.get("assets")
        volumes = _asset_volumes(assets, tag=tag) if isinstance(assets, list) else []
        if not volumes:
            continue
        try:
            refs.append(
                FirmwareRef(
                    driver=NothingDriver.name,
                    device=named.group("device"),
                    build=named.group("build"),
                    url=volumes[0],
                    archive_suffix=".7z",
                )
            )
        except ValueError as exc:
            logger.warning("nothing: release %s did not validate as a ref: %s", tag, exc)
    if not refs:
        raise EmptyFirmwareIndexError(
            f"parse_releases: {source_url} returned {len(releases)} release(s) and not one of "
            "them carries an image archive. The archive has published one per build since "
            "2022, so zero means the repository moved, the asset naming changed, or GitHub "
            "answered a rate-limit body — not that Nothing shipped nothing."
        )
    refs.reverse()
    return refs


def _append_and_consume(source: Path, out: IO[bytes], digest: "hashlib._Hash") -> int:
    """Append one downloaded volume onto the joined archive, hash it into the running digest,
    and delete it. Blocking: every caller runs it through `asyncio.to_thread`.

    Deleting here rather than after the whole set is what keeps the peak on disk at "the
    joined archive plus one volume" instead of twice the archive.
    """
    copied = 0
    with source.open("rb") as fh:
        while chunk := fh.read(_COPY_CHUNK_BYTES):
            out.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
    source.unlink(missing_ok=True)
    return copied


class NothingDriver(FirmwareDriver):
    name = "nothing"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._releases_url = settings.nothing_releases_url.rstrip("/")
        self._timeout = settings.firmware_http_timeout_seconds
        self._max_archive_bytes = settings.max_firmware_archive_bytes
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.RESTRICTED,
            source_url=self._releases_url,
            summary=(
                "spike0en/nothing_archive, a community re-upload of Nothing's OTA images "
                "rather than Nothing's own endpoint. The assets themselves are ungated, but "
                "no Nothing licence covers redistributing them and no OEM's terms were found "
                "to authorise automated bulk downloading, so this sits above a first-party "
                "public source rather than beside one. Fetched without any GitHub credential, "
                "which caps it at GitHub's unauthenticated 60 requests/hour."
            ),
            # Nothing to acknowledge: there is no gate, and no acceptance to record.
            acknowledged=True,
        )

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return driver_client(self._timeout), True

    async def _get_json_text(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            response = await client.get(url, headers={"Accept": "application/vnd.github+json"})
        except httpx.HTTPError as exc:
            raise FirmwareError(f"NothingDriver: {url} is unreachable: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise FirmwareError(
                f"NothingDriver: {url} answered HTTP {response.status_code}; expected 200. "
                "GitHub answers 403 with a rate-limit body once the unauthenticated 60 "
                "requests/hour for this IP are spent, which is the usual cause."
            )
        return response.text

    async def list_available(self) -> list[FirmwareRef]:
        client, owned = self._open_client()
        pages: list[list[FirmwareRef]] = []
        try:
            for page in range(1, _MAX_PAGES + 1):
                url = f"{self._releases_url}?per_page={_PAGE_SIZE}&page={page}"
                text = await self._get_json_text(client, url)
                # Parsed per page, so the "zero usable releases" error names the page that
                # came back empty rather than an aggregate that hides which one did.
                try:
                    pages.append(parse_releases(text, source_url=url))
                except EmptyFirmwareIndexError:
                    if page == 1:
                        raise
                    break
                if len(json.loads(text)) < _PAGE_SIZE:
                    break
            else:
                raise FirmwareError(
                    f"NothingDriver.list_available: {self._releases_url} was still returning "
                    f"full pages after {_MAX_PAGES}; refusing to keep paging"
                )
        finally:
            if owned:
                await client.aclose()
        # Each page is already oldest-first within itself and GitHub pages newest-first, so
        # the pages are reversed and then concatenated to get one oldest-first run.
        refs = [ref for page_refs in reversed(pages) for ref in page_refs]
        logger.info(
            "nothing archive: %d build(s) across %d device(s)",
            len(refs),
            len({ref.device for ref in refs}),
        )
        return refs

    async def _volumes_for(self, client: httpx.AsyncClient, ref: FirmwareRef) -> list[str]:
        """The release's volume URLs, read back from the release itself rather than derived
        from `ref.url` by incrementing its suffix. Probing `.002`, `.003`, ... until a 404 is
        the shape where one transient 404 silently yields a truncated archive."""
        tag = f"{ref.device}_{ref.build}"
        text = await self._get_json_text(client, f"{self._releases_url}/tags/{tag}")
        try:
            release = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FirmwareError(f"NothingDriver.fetch: release {tag} did not answer JSON") from exc
        assets = release.get("assets") if isinstance(release, dict) else None
        volumes = _asset_volumes(assets, tag=tag) if isinstance(assets, list) else []
        if not volumes:
            raise FirmwareError(
                f"NothingDriver.fetch: release {tag} carries no image archive any more. The "
                "index this ref came from is stale; re-list before fetching."
            )
        if volumes[0] != ref.url:
            raise FirmwareInputError(
                f"NothingDriver.fetch: release {tag} serves its first volume from "
                f"{volumes[0]}, but this ref names {ref.url}. Refusing to download a set the "
                "ref does not describe; re-list the index and retry."
            )
        return volumes

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"NothingDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        client, owned = self._open_client()
        try:
            volumes = await self._volumes_for(client, ref)
            return await self._join_volumes(client, volumes, dest_dir / ref.archive_filename)
        finally:
            if owned:
                await client.aclose()

    async def _join_volumes(
        self, client: httpx.AsyncClient, volumes: list[str], dest: Path
    ) -> DownloadedArchive:
        """Stream each volume onto the end of one archive, hashing the joined bytes.

        The result is byte-identical to the undivided `.7z` the uploader split, so its sha256
        is reproducible by hand (`cat *.7z.00* | sha256sum`) rather than being an identity
        this pipeline invented for a set of files.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(dest.name + ".volumes")
        partial = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        written = 0
        try:
            with partial.open("wb") as out:
                for index, url in enumerate(volumes, start=1):
                    remaining = self._max_archive_bytes - written
                    if remaining <= 0:
                        raise FirmwareDownloadError(
                            f"NothingDriver.fetch: {dest.name} passed the "
                            f"{self._max_archive_bytes}-byte ceiling at volume {index} of "
                            f"{len(volumes)}. Raise MAX_FIRMWARE_ARCHIVE_BYTES if this "
                            "firmware is genuinely that large."
                        )
                    volume = await download_to_file(
                        client,
                        url,
                        staging / f"{index:03d}.part7z",
                        max_bytes=remaining,
                    )
                    written += await asyncio.to_thread(
                        _append_and_consume, volume.path, out, digest
                    )
        except BaseException:
            # Cancellation included: a reclaim lands here holding gigabytes across two
            # directories, and neither is tracked by anything else.
            await asyncio.to_thread(shutil.rmtree, staging, True)
            partial.unlink(missing_ok=True)
            raise
        await asyncio.to_thread(shutil.rmtree, staging, True)
        partial.replace(dest)
        logger.warning(
            "%s joined from %d volume(s), %d bytes, sha256=%s: the archive itself is "
            "integrity-unverified because the release's published hashes cover the image "
            "files inside it, not the download",
            dest.name,
            len(volumes),
            written,
            digest.hexdigest(),
        )
        return DownloadedArchive(path=dest, sha256=digest.hexdigest(), integrity_verified=False)
