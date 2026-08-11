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
import hashlib
import logging
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

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
    """The archive did not come down intact: bad status, truncated body, or a checksum that
    does not match the one the index published."""


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
    device: str = Field(pattern=r"^[A-Za-z0-9_]{1,64}$")
    build: str = Field(pattern=r"^[A-Za-z0-9._]{1,64}$")
    url: str = Field(pattern=r"^https://[^\s]{1,2048}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    android_version: str | None = Field(default=None, max_length=64)
    # Marketing name ("Pixel 9 Pro Fold"). Not `model_*`: pydantic protects that prefix.
    marketing_name: str | None = Field(default=None, max_length=128)

    @property
    def archive_filename(self) -> str:
        """Built from the validated fields, never from the URL's basename: the bytes that
        identify a build are not the bytes that may spell a path."""
        return f"{self.driver}-{self.device}-{self.build}.zip"


class FirmwareJobParams(BaseModel):
    """A `firmware_analysis` job's target as it arrives from the dashboard.

    Either names a build outright (`build` + `url`, the shape `list_available` hands back)
    or names a device and lets the acquire stage resolve it against the driver's index.
    `extra="forbid"` so a misspelt key is a 422 at creation rather than a job that runs for
    ten minutes and downloads the wrong thing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str = Field(pattern=r"^[a-z0-9_]{1,32}$")
    device: str = Field(pattern=r"^[A-Za-z0-9_]{1,64}$")
    build: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._]{1,64}$")
    url: str | None = Field(default=None, pattern=r"^https://[^\s]{1,2048}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

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
        )


def select_ref(refs: list[FirmwareRef], *, device: str, build: str | None = None) -> FirmwareRef:
    """Pick one build out of an index listing: the named build, or the last one the source
    lists for that device (sources list oldest first, so that is the newest). Driver-agnostic
    on purpose — build selection is not something each OEM should reinvent."""
    for_device = [ref for ref in refs if ref.device == device]
    if not for_device:
        raise FirmwareInputError(
            f"select_ref: device {device!r} is not in this index ({len(refs)} builds across "
            f"{len({ref.device for ref in refs})} devices); check the codename"
        )
    if build is None:
        return for_device[-1]
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
    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> Path:
        """Download `ref` into `dest_dir` (created if absent) and return the archive path.
        Writes nothing outside `dest_dir`: the caller owns retention of that directory."""


DriverFactory = Callable[[Settings], FirmwareDriver]


def _driver_factories() -> dict[str, DriverFactory]:
    """Imported lazily so a driver module can import this one. One line per OEM; tasks 11's
    six drivers are six entries here and no change anywhere else."""
    from uadclaw.drivers.pixel import PixelDriver

    return {PixelDriver.name: PixelDriver}


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


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Stream `url` to `dest`, verifying the checksum the index published when there is one.

    Downloads land on a `.part` sibling and are renamed only once the body is complete and
    the digest matches, so a truncated or corrupted transfer can never be mistaken for a
    finished archive by a later stage (or by a resumed attempt).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code != httpx.codes.OK:
                await response.aread()
                raise FirmwareDownloadError(
                    f"download_to_file: {url} answered HTTP {response.status_code}; expected "
                    "200. Re-list the index — a firmware URL can expire or be withdrawn."
                )
            with partial.open("wb") as fh:
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise FirmwareDownloadError(
            f"download_to_file: transport error fetching {url} after {written} bytes: {exc}"
        ) from exc

    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise FirmwareDownloadError(
            f"download_to_file: {url} downloaded {written} bytes with sha256 {actual}, but the "
            f"index published {expected_sha256}. The partial file was deleted; retry the job."
        )
    partial.replace(dest)
    logger.info("downloaded %s (%d bytes, sha256=%s)", dest.name, written, actual)
    return dest
