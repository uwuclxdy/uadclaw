"""Firmware acquisition: the single interface every OEM driver implements, plus the registry
that resolves a driver name and honours its enable switch.

A driver answers exactly three questions: what builds exist (`list_available`), how to get
one onto disk (`fetch`), and what the operator agrees to by enabling it (`terms`). Nothing
downstream is keyed on the OEM — unpacking dispatches on container format (`uadclaw.unpack`)
— so a vendor changing archive layout needs no new driver, and a new OEM is one class.

Every driver is disableable through `Settings.disabled_firmware_drivers` without touching
another stage: the Samsung and Oppo endpoints (task 11) are reverse-engineered and will
break without warning, and the rest of the pipeline has to stay green when one does.
"""

import abc
import asyncio
import datetime
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uadclaw.settings import Settings

logger = logging.getLogger(__name__)

# Streaming chunk for multi-GB downloads: big enough that the per-chunk overhead is noise,
# small enough that a chunk never dominates the worker's memory.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class FirmwareError(RuntimeError):
    """A firmware source misbehaved: unreachable, unparseable, or serving something other
    than what it advertised. Distinct from `FirmwareInputError` because the fix is on the
    source's side (or is "retry later"), never in the caller's arguments."""


class FirmwareInputError(ValueError):
    """A bad operator/caller input reached this module. Distinct type from `FirmwareError`
    (and from a genuine bug) so an API handler can map it to 4xx rather than 5xx."""


class UnknownFirmwareDriverError(FirmwareInputError):
    """No driver by that name. Almost always a typo in a job's params."""


class FirmwareDriverDisabledError(FirmwareInputError):
    """The driver exists but the operator disabled it in configuration."""


class FirmwareTermsNotAcknowledgedError(FirmwareInputError):
    """The source is behind a terms wall and the operator has not recorded acceptance."""


class EmptyFirmwareIndexError(FirmwareError):
    """An index fetch succeeded and yielded zero builds.

    Its own type because this is the failure that most looks like success: the Pixel index
    answers HTTP 200 with ~75 KB of prose and no download links at all when the terms
    acknowledgement is missing. Treating that as "no builds available" would park the whole
    pipeline on a fixable configuration mistake, silently.
    """


class FirmwareDownloadError(FirmwareError):
    """The archive did not come down intact: bad status, a `Content-Length` disagreeing with
    the published size, truncated body, or a checksum that does not match the one the index
    published."""


class FirmwareRedirectError(FirmwareError):
    """A firmware fetch's redirect would downgrade https to http. Refused rather than followed.
    Carries the refused URL."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"refused a redirect downgrading https to http: {url!r}")


class _NoHttpsDowngradeTransport(httpx.AsyncBaseTransport):
    """Wraps the inner transport and refuses a redirect that downgrades https to http.

    httpx follows a 3xx to an http location without complaint, so the redirect is the one place
    a https-by-construction fetch can leak into the clear. An http request is refused only once
    an https request has already passed through this transport — that is the downgrade case, and
    it is the only one that can be a downgrade: no driver initiates an http URL on its own, since
    every default index/mirror/catalogue URL is https and `FirmwareRef.url` is validated
    `^https://` (Xiaomi's plain-http rows are refused at ref construction). A direct http URL an
    operator configured — a self-hosted mirror, say — is the FIRST request through a fresh client
    and passes, because the operator chose cleartext for a host they control; the guard is about
    a source silently downgrading a transfer, never about refusing cleartext by fiat. This beats
    a shared redirect-walk (`follow_redirects=False` plus a per-hop helper, the shape
    `brave.PageFetcher` uses) because it covers every `client.get`/`post`/`stream` call site in
    all six drivers automatically, where a redirect-walk would touch each call site. The address
    half (loopback/private/reserved) deliberately does not live here: a firmware CDN is a public
    host by definition, and `brave.PageFetcher` owns that gate for the URLs a third party
    chooses.

    Passing an explicit transport to `httpx.AsyncClient` disables httpx's env-proxy mounting
    (`allow_env_proxies = trust_env and transport is None`), so `HTTP_PROXY`/`HTTPS_PROXY` no
    longer route firmware traffic. Deliberate rather than an accident, and stated here so it
    never reads as one: the worker image pulls from public CDNs and sets no proxy, while
    `llm`/`brave` build their own clients and still honour the env. Flag it rather than
    widening it — a proxied firmware deployment is a follow-up, not this task.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self._seen_https = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "http" and self._seen_https:
            raise FirmwareRedirectError(str(request.url))
        if request.url.scheme == "https":
            self._seen_https = True
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        # `AsyncBaseTransport.aclose` is a no-op, and the inner transport's `aclose` is where
        # keep-alive connections return to the pool. Without this, every driver's
        # `finally: client.aclose()` closes nothing and the per-call pool leaks sockets.
        await self._inner.aclose()


def driver_client(
    timeout: float, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """The one client constructor every driver's `_open_client` uses.

    `follow_redirects=True` because firmware sources legitimately 3xx (a Samsung download
    authorisation, Xiaomi's CDN rewrite, an Oppo gate), but the wrapper transport refuses any
    redirect that would downgrade https to http before it is sent — see
    `_NoHttpsDowngradeTransport`. `transport` is a test seam: production wraps
    `httpx.AsyncHTTPTransport()`, tests hand in a `MockTransport`.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        transport=_NoHttpsDowngradeTransport(
            transport if transport is not None else httpx.AsyncHTTPTransport()
        ),
        follow_redirects=True,
    )


class TermsRisk(StrEnum):
    """How exposed the operator is by enabling a driver. Ordered loosely by how likely the
    source is to object, which is what the dashboard wants to surface."""

    PUBLIC = "public"  # published source, no gate, no restriction
    ACKNOWLEDGEMENT = "acknowledgement"  # official source behind terms the operator accepts
    RESTRICTED = "restricted"  # public mirror carrying a use restriction
    REVERSE_ENGINEERED = "reverse_engineered"  # private endpoint, no published terms at all


class TermsPosture(BaseModel):
    """A driver's own statement of what using it means, declared in code next to the driver
    rather than in a doc, so the dashboard can show it and a reviewer can diff it."""

    model_config = ConfigDict(frozen=True)

    risk: TermsRisk
    source_url: str
    summary: str
    # False when the source needs an acknowledgement the operator has not configured yet;
    # True when there is nothing to acknowledge or the acknowledgement is in place.
    acknowledged: bool


class FirmwareRef(BaseModel):
    """One downloadable build. Validated rather than trusted: a ref can arrive from a job's
    params (operator input), and `device`/`build` end up in a filename, so both are pinned
    to a charset that cannot spell a path segment, and the URL to https.
    """

    model_config = ConfigDict(frozen=True)

    driver: str = Field(pattern=r"^[a-z0-9_]{1,32}$")
    # `-` is in the DEVICE charset for the same reason it is in the build charset below: every
    # Samsung model name carries one (`SM-S928B`) and transliterating it changes the product's
    # identity, which then reaches a human in triage and `package_facts.device_count` as a
    # name no Samsung document spells. It cannot spell a separator or a traversal, and
    # `unpack.safe_component` already treats it as a harmless path component.
    device: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # `-` is in the charset because two of the six sources spell their build ids with it and
    # neither can be transliterated without changing the identity: a Nothing release tag is
    # `B4.1-260723-1820` and a Motorola build is `V1TRS35H.60-33-7`. It is the same charset
    # `unpack.safe_component` already treats as a harmless path component, and it cannot
    # spell a separator or a traversal.
    build: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    url: str = Field(pattern=r"^https://[^\s]{1,2048}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # What Xiaomi's index publishes, and all it publishes. Kept as its own field rather than
    # folded into `sha256` with an algorithm tag so that `sha256` always means sha256.
    md5: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    # The byte size the source published for this build. A lower bound only: zero cannot be a
    # downloadable archive, and the ceiling is `download_to_file`'s `max_bytes`, which knows
    # this box's scratch budget rather than the catalogue's.
    size: int | None = Field(default=None, gt=0)
    android_version: str | None = Field(default=None, max_length=64)
    # Marketing name ("Pixel 9 Pro Fold"). Not `model_*`: pydantic protects that prefix.
    marketing_name: str | None = Field(default=None, max_length=128)
    # The container this build arrives as. Unpacking dispatches on the bytes and never on
    # this, but an operator staring at a scratch directory should not be told a 7z is a zip.
    archive_suffix: str = Field(default=".zip", pattern=r"^\.[a-z0-9]{1,8}$")

    @property
    def archive_filename(self) -> str:
        """Built from the validated fields, never from the URL's basename: the bytes that
        identify a build are not the bytes that may spell a path."""
        return f"{self.driver}-{self.device}-{self.build}{self.archive_suffix}"

    def published_digest(self) -> tuple[str, str] | None:
        """`(algorithm, hexdigest)` the source published for this build, or None when it
        published nothing to check against. sha256 wins when a source publishes both."""
        if self.sha256 is not None:
            return ("sha256", self.sha256)
        if self.md5 is not None:
            return ("md5", self.md5)
        return None


@dataclass(frozen=True, slots=True)
class DownloadedArchive:
    """What came down, and whether anyone could prove it. `integrity_verified` is False when
    the source published no checksum to compare against — carried forward rather than assumed
    either way, since the whole pipeline downstream trusts these bytes."""

    path: Path
    sha256: str
    integrity_verified: bool


class FirmwareJobParams(BaseModel):
    """A `firmware_analysis` job's target as it arrives from the dashboard.

    Either names a build outright (`build` + `url`, the shape `list_available` hands back)
    or names a device and lets the acquire stage resolve it against the driver's index.
    `extra="forbid"` so a misspelt key is a 422 at creation rather than a job that runs for
    ten minutes and downloads the wrong thing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str = Field(pattern=r"^[a-z0-9_]{1,32}$")
    # Same charset as `FirmwareRef.device`, and the two must not disagree: a narrower one here
    # is a 422 on a device the driver itself offers (`SM-S928B`).
    device: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Same charset as `FirmwareRef.build` and for the same reason: a job that names a real
    # Nothing or Motorola build must be creatable, and the two must not disagree about what a
    # build id may spell — a narrower one here is a 422 on a build the driver itself offers.
    build: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    url: str | None = Field(default=None, pattern=r"^https://[^\s]{1,2048}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Alongside `sha256` because a source that publishes only md5 would otherwise have its
    # checksum silently dropped on this path: an operator who pins a Xiaomi build outright
    # would get an integrity-unverified download of a build whose digest the index knows.
    md5: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    def as_ref(self) -> FirmwareRef | None:
        """The ref these params name outright, or None when the build still has to be
        resolved against the driver's index."""
        if self.url is None or self.build is None:
            return None
        return FirmwareRef(
            driver=self.driver,
            device=self.device,
            build=self.build,
            url=self.url,
            sha256=self.sha256,
            md5=self.md5,
        )


# Android build ids carry their date in the middle field: CP2A.260705.006 -> 2026-07-05.
# 1,806 of the Pixel index's 2,293 builds parse this way; the other 487 are pre-2016 ids
# (NDE63H, JLS36C) with no date field at all.
_BUILD_DATE_RE = re.compile(r"\.(\d{6})\.")


def build_date(build: str) -> datetime.date | None:
    """The build's date, or None for an id that does not carry one."""
    match = _BUILD_DATE_RE.search(build)
    if match is None:
        return None
    try:
        return datetime.datetime.strptime(match.group(1), "%y%m%d").date()
    except ValueError:
        return None


def select_ref(refs: list[FirmwareRef], *, device: str, build: str | None = None) -> FirmwareRef:
    """Pick one build out of an index listing: the named build, or the newest one on offer for
    that device. Driver-agnostic on purpose — build selection is not something each OEM should
    reinvent.

    Newest is decided by the date in the build id, not by position. Measured against the real
    Pixel index on 2026-08-11: taking the last row per device agrees with the date for 56 of
    58 devices and is WRONG for two — crosshatch and blueline both list SP1A.210812.016.C2
    (2021-08-12) after RQ3A.211001.001 (2021-10-01). A lexicographic sort is not an option
    either: the letter prefix resets, so UQ1A (2024) sorts above CP2A (2026). Builds with no
    parseable date fall back to index order behind every dated build, which keeps the 487
    pre-2016 ids from ever outranking a real one.
    """
    for_device = [ref for ref in refs if ref.device == device]
    if not for_device:
        raise FirmwareInputError(
            f"select_ref: device {device!r} is not in this index ({len(refs)} builds across "
            f"{len({ref.device for ref in refs})} devices); check the codename"
        )
    if build is None:
        newest = max(
            enumerate(for_device),
            key=lambda pair: (
                build_date(pair[1].build) is not None,
                build_date(pair[1].build) or datetime.date.min,
                pair[0],
            ),
        )[1]
        return newest
    wanted = build.upper()
    for ref in for_device:
        if ref.build.upper() == wanted:
            return ref
    raise FirmwareInputError(
        f"select_ref: device {device!r} has no build {build!r} in this index; it offers "
        f"{len(for_device)} builds, most recently {for_device[-1].build}"
    )


class FirmwareDriver(abc.ABC):
    """The whole OEM-facing interface. Later drivers implement this and nothing else."""

    name: ClassVar[str]

    @abc.abstractmethod
    def terms(self) -> TermsPosture:
        """What enabling this driver commits the operator to, evaluated against current
        configuration (so `acknowledged` reflects reality, not intent)."""

    @abc.abstractmethod
    async def list_available(self) -> list[FirmwareRef]:
        """Every build the source currently offers, in the source's own order (oldest
        first where the source is ordered). Raises `EmptyFirmwareIndexError` rather than
        returning an empty list, because "the index parsed to nothing" is a failure."""

    @abc.abstractmethod
    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        """Download `ref` into `dest_dir` (created if absent) and return the archive with the
        digest of what actually arrived. Writes nothing outside `dest_dir`: the caller owns
        retention of that directory."""


DriverFactory = Callable[[Settings], FirmwareDriver]


def _driver_factories() -> dict[str, DriverFactory]:
    """Imported lazily so a driver module can import this one. One line per OEM; tasks 11's
    six drivers are six entries here and no change anywhere else."""
    from uadclaw.drivers.motorola import MotorolaDriver
    from uadclaw.drivers.nothing import NothingDriver
    from uadclaw.drivers.oppo import OppoDriver
    from uadclaw.drivers.pixel import PixelDriver
    from uadclaw.drivers.samsung import SamsungDriver
    from uadclaw.drivers.xiaomi import XiaomiDriver

    return {
        MotorolaDriver.name: MotorolaDriver,
        NothingDriver.name: NothingDriver,
        OppoDriver.name: OppoDriver,
        PixelDriver.name: PixelDriver,
        SamsungDriver.name: SamsungDriver,
        XiaomiDriver.name: XiaomiDriver,
    }


def driver_names() -> tuple[str, ...]:
    return tuple(sorted(_driver_factories()))


def enabled_driver_names(settings: Settings) -> tuple[str, ...]:
    disabled = settings.disabled_firmware_driver_names
    return tuple(name for name in driver_names() if name not in disabled)


def get_driver(name: str, settings: Settings) -> FirmwareDriver:
    """Resolve a driver by name, refusing an unknown or disabled one with distinct types so
    a typo and a deliberate shutoff never read the same in a failed job's reason."""
    factories = _driver_factories()
    factory = factories.get(name)
    if factory is None:
        raise UnknownFirmwareDriverError(
            f"get_driver: no firmware driver named {name!r}; known drivers are "
            f"{', '.join(sorted(factories))}"
        )
    if name in settings.disabled_firmware_driver_names:
        raise FirmwareDriverDisabledError(
            f"get_driver: firmware driver {name!r} is disabled by configuration "
            f"(DISABLED_FIRMWARE_DRIVERS={settings.disabled_firmware_drivers!r}); remove it "
            "from that list to run this driver again"
        )
    return factory(settings)


def _write_and_hash(fh: IO[bytes], digests: "list[hashlib._Hash]", chunk: bytes) -> None:
    fh.write(chunk)
    for digest in digests:
        digest.update(chunk)


# What a firmware index is allowed to have published. An allowlist rather than handing the
# name straight to `hashlib.new`, which happily accepts `md4` and every other broken or
# unexpected digest the local OpenSSL build exposes.
DIGEST_ALGORITHMS = frozenset({"sha256", "md5"})


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    expected_digest: str | None = None,
    digest_algorithm: str = "sha256",
    max_bytes: int | None = None,
    expected_size: int | None = None,
) -> DownloadedArchive:
    """Stream `url` to `dest`, verifying the checksum the index published when there is one.
    Returns the archive's sha256 alongside its path, so a caller can record what it got.

    `expected_size`, when set, is compared against the response's `Content-Length` before the
    first body byte is written, so a source serving the wrong archive fails before a multi-GB
    transfer rather than at the digest. A missing `Content-Length` proceeds — not every source
    sends one, and the digest is still the final gate.

    `digest_algorithm` is the algorithm the SOURCE published in, not the one this pipeline
    keeps: Xiaomi's index publishes md5 for 3,454 of its 3,550 entries and no sha256 at all,
    so verifying at all means verifying in md5. The returned sha256 is computed regardless
    and in the same pass, because that digest is the pipeline's own identity for the archive
    and every later stage records it.

    Downloads land on a `.part` sibling and are renamed only once the body is complete and
    the digest matches, so a truncated or corrupted transfer can never be mistaken for a
    finished archive by a later stage (or by a resumed attempt). Every failure mode deletes
    that sibling, including cancellation: a reclaim or a shutdown lands here mid-write, and a
    3.5 GB orphan is exactly the shape retention exists to prevent.

    The write and the hash both run in a worker thread. They are 3,500 iterations over a
    multi-GB body, and inline they would block the event loop this worker shares with every
    other slot and with the heartbeat task that keeps the job from being reclaimed.
    """
    if digest_algorithm not in DIGEST_ALGORITHMS:
        raise FirmwareInputError(
            f"download_to_file: {digest_algorithm!r} is not a digest algorithm this pipeline "
            f"verifies against; use one of {', '.join(sorted(DIGEST_ALGORITHMS))}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    sha256 = hashlib.sha256()
    published = sha256 if digest_algorithm == "sha256" else hashlib.new(digest_algorithm)
    digests = [sha256] if published is sha256 else [sha256, published]
    written = 0
    try:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code != httpx.codes.OK:
                await response.aread()
                raise FirmwareDownloadError(
                    f"download_to_file: {url} answered HTTP {response.status_code}; expected "
                    "200. Re-list the index — a firmware URL can expire or be withdrawn."
                )
            if expected_size is not None:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length.isdigit():
                    declared = int(content_length)
                    if declared != expected_size:
                        raise FirmwareDownloadError(
                            f"download_to_file: {url} advertised Content-Length {declared}, "
                            f"but the index published {expected_size} bytes; refusing to "
                            "start a transfer whose size cannot match"
                        )
            with partial.open("wb") as fh:
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise FirmwareDownloadError(
                            f"download_to_file: {url} exceeded the {max_bytes}-byte ceiling after "
                            f"{written} bytes. Raise MAX_FIRMWARE_ARCHIVE_BYTES if this firmware "
                            "is genuinely that large; scratch is sized for one image at a time."
                        )
                    await asyncio.to_thread(_write_and_hash, fh, digests, chunk)
    except (httpx.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise FirmwareDownloadError(
            f"download_to_file: {url} failed after {written} bytes: {exc}"
        ) from exc
    except BaseException:
        # Cancellation (worker shutdown, job reclaim) and every other exit path: the partial
        # file is worthless to anyone and nothing else is tracking it.
        partial.unlink(missing_ok=True)
        raise

    actual_sha256 = sha256.hexdigest()
    if expected_digest is not None and published.hexdigest() != expected_digest:
        partial.unlink(missing_ok=True)
        raise FirmwareDownloadError(
            f"download_to_file: {url} downloaded {written} bytes with {digest_algorithm} "
            f"{published.hexdigest()}, but the index published {expected_digest}. The partial "
            "file was deleted; retry the job."
        )
    if expected_digest is None:
        # Not fatal — some sources publish no checksum — but it must not be silent: the
        # archive is integrity-unverified and the caller records that alongside the digest.
        logger.warning(
            "%s came with no published checksum; recording it as integrity-unverified (sha256=%s)",
            dest.name,
            actual_sha256,
        )
    partial.replace(dest)
    logger.info("downloaded %s (%d bytes, sha256=%s)", dest.name, written, actual_sha256)
    return DownloadedArchive(
        path=dest, sha256=actual_sha256, integrity_verified=expected_digest is not None
    )
