"""Unpacking, dispatched on what the bytes are rather than on which OEM produced them, plus
selective extraction of the files later stages actually read.

Every step is chosen by magic number, never by file extension or vendor: a Pixel factory zip
carries raw ext4 partition images directly (measured 2026-08-11 on oriole/cp2a.260705.006.a1
— no `super.img`, no sparse header), while the emulator system image is a GPT disk whose
`super` partition needs `lpunpack`, a Xiaomi recovery ROM carries a `payload.bin`, and a
Nothing build arrives as a 7-Zip archive of partition images. Same table serves all four, so
a vendor switching layouts needs no new driver.

One thing the magic cannot decide is ORDER. Motorola splits its super image across
`super.img_sparsechunk.0 … .13` inside its firmware zip, and the only record of which chunk
comes first is the suffix, so `_sparsechunk_sets` groups and orders on the name — and then
refuses any member whose bytes are not a sparse image, so a file that merely wears the name
still cannot get itself concatenated into a partition.

Extraction is selective by construction: an ext4 partition is listed once, the paths worth
having are chosen in Python, and only those are handed back to `7z`. That sidesteps two
measured traps at once — `7z x -r 'sysconfig/*'` extracts ZERO files while the listing proves
dozens exist, and `debugfs -R "ls -l /app"` walks nothing on an image where 7z finds 64 APKs
because the partition carries a nested `system/` root.

EROFS takes a different tool, measured 2026-08-11 and contrary to what
`docs/research/local-probe.md` records: 7-Zip 26.02 has NO EROFS handler at all (`7z i` lists
`Ext` and `SquashFS` and nothing else of the kind). Pointed at an EROFS image it silently
falls through to its gzip reader and lists ONE entry where `fsck.erofs --extract` recovers
105 — an under-extraction that raises nothing and looks exactly like a sparse partition. So
EROFS goes through `fsck.erofs`, and the archive type is forced on every 7z call rather than
sniffed, because that same sniffing is what produced the wrong answer.

Samsung adds two more steps and the deepest chain here — zip -> tar -> LZ4 frame -> sparse ->
super -> partition — and its member names carry none of it: the tars are called `.tar.md5`
and every image inside them `.img.lz4`, so both steps are taken from magic (`ustar` at 257,
`04 22 4d 18` at 0) and the suffixes are never read. The tar is walked as a STREAM straight
out of the zip member, because writing it out first would put an 11.47 GB copy on disk that
nothing reads twice. Measured 2026-08-12 on the real `SM-S911U`/`XAA` build.
"""

import asyncio
import hashlib
import logging
import os
import re
import shutil
import struct
import tarfile
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import IO

import lz4.frame

from uadclaw.settings import Settings

logger = logging.getLogger(__name__)


class UnpackError(RuntimeError):
    """An archive or image could not be turned into partition images or files."""


class UnsupportedContainerError(UnpackError):
    """The bytes are not a container this pipeline knows how to open. Its own type because
    the fix is "add a format to the dispatch table", not "retry"."""


class MissingToolError(UnpackError):
    """An external unpacking tool is not installed or not on PATH."""


class ToolFailedError(UnpackError):
    """An external tool ran and failed."""


class NothingExtractedError(UnpackError):
    """Extraction completed and produced nothing where something was expected.

    Its own type because this is the failure that looks like success: a glob that matches
    nothing, a partition list that came out empty, or an archive whose root is one level
    deeper than assumed all end here, and every one of them would otherwise flow downstream
    as "this build has no preinstalled apps". Raised per partition as well as overall — a
    build with four populated partitions and one silently empty one is the same bug wearing
    a smaller number.
    """


class UnsafePathError(UnpackError):
    """A name taken from archive or image metadata would have written outside the directory
    it was given, or could not be reduced to a usable path component.

    Its own type because the input is attacker-influenced: partition names come out of a GPT
    as raw UTF-16, member names out of a zip's central directory, and file paths out of a
    filesystem image, none of them written by us.
    """


class DuplicatePartitionError(UnpackError):
    """Two partitions in one image claim the same identity. Refused rather than resolved:
    silently letting the second overwrite the first loses a whole partition's APKs, and
    guessing which one the device would have mounted is not something this can know."""


class ContainerFormat(StrEnum):
    ZIP = "zip"
    SEVEN_ZIP = "seven_zip"  # Nothing ships its partitions as a split 7z volume set
    TAR = "tar"  # Samsung wraps each of its six firmware halves in one
    PAYLOAD_BIN = "payload_bin"  # A/B OTA payload
    LZ4_FRAME = "lz4_frame"  # Samsung compresses every partition image with one
    SPARSE_IMAGE = "sparse_image"  # Android sparse image -> simg2img
    SUPER_IMAGE = "super_image"  # dynamic partitions -> lpunpack
    GPT_DISK = "gpt_disk"  # whole-disk image with a partition table
    EXT4 = "ext4"
    EROFS = "erofs"
    UNKNOWN = "unknown"


# Enough to cover every magic offset below (the super geometry block starts at 4096).
_HEADER_BYTES = 8192

_ZIP_MAGIC = b"PK\x03\x04"
_SEVEN_ZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"
_PAYLOAD_MAGIC = b"CrAU"
_LZ4_FRAME_MAGIC = b"\x04\x22\x4d\x18"  # 0x184D2204 LE, LZ4 frame format
_SPARSE_MAGIC = b"\x3a\xff\x26\xed"  # 0xED26FF3A little-endian
_TAR_MAGIC = b"ustar"  # in the first header block, POSIX and GNU spell the rest differently
_GPT_MAGIC = b"EFI PART"  # at LBA 1
_LP_GEOMETRY_MAGIC = b"gDla"  # 0x616C4467 LE, at LP_PARTITION_RESERVED_BYTES (4096)
_EROFS_MAGIC = b"\xe2\xe1\xf5\xe0"  # 0xE0F5E1E2 LE, at EROFS_SUPER_OFFSET (1024)
_EXT4_MAGIC = b"\x53\xef"  # 0xEF53 LE, at superblock offset 0x38 (i.e. 1024 + 56)

_TAR_MAGIC_OFFSET = 257
_GPT_HEADER_OFFSET = 512
_LP_GEOMETRY_OFFSET = 4096
_EROFS_MAGIC_OFFSET = 1024
_EXT4_MAGIC_OFFSET = 1024 + 0x38

# How much of a stream is read at a time, and how much of it has to arrive before a decoded
# payload can be classified (`_HEADER_BYTES` covers every magic offset above).
_STREAM_CHUNK_BYTES = 4 * 1024 * 1024

# Recursion cap for the container chain (zip -> zip -> super -> partition is depth 4 on real
# firmware). A cap turns a pathological or hostile nesting into a named error instead of a
# worker that unpacks until the disk fills.
_MAX_UNPACK_DEPTH = 6

# Images that are never a filesystem, skipped before they are extracted rather than after.
# `super_empty.img` matters most: it is a 5,080-byte metadata template (measured on a real
# Pixel factory zip) that carries the LP geometry magic, so it detects as a super image and
# would send lpunpack chasing partitions that do not exist there.
_NON_FILESYSTEM_IMAGES = frozenset(
    {
        "super_empty",
        "boot",
        "init_boot",
        "vendor_boot",
        "vendor_kernel_boot",
        "dtbo",
        "vbmeta",
        "vbmeta_system",
        "vbmeta_vendor",
        "recovery",
        "radio",
        "bootloader",
        "modem",
        "abl",
        "xbl",
        "tz",
        "aop",
        "hyp",
        "keymaster",
        "devcfg",
        "qupfw",
        "featenabler",
        "imagefv",
        "multiimgoem",
        "shrm",
        "uefisecapp",
        "cpucp",
        "pvmfw",
    }
)

# Real filesystems that carry no preinstalled app on any build measured, so unpacking one
# costs a full decompression and yields nothing. Distinct from `_NON_FILESYSTEM_IMAGES`
# (those are not filesystems at all) and from `PARTITIONS_ALLOWED_EMPTY` (which says a
# partition ALREADY unpacked is allowed to yield nothing). `userdata` is why this exists: the
# Samsung `AP_`/`USERDATA_` pair ships one as 1.85 GB of LZ4 that un-sparses to the whole
# declared partition, and it is a fresh `/data` with not one preinstalled APK in it.
# `vm-bootsys` (the pVM guest's own root, 8 files) and `dspso` (the DSP firmware filesystem
# inside Samsung's bootloader tar, 80 files) are here on the same measured footing: real ext4,
# no APK, and not a partition the device ever installs an app into. Missing one of these is a
# loud failure and not a quiet one — `extract_artifacts` names the partition and says what to
# do about it — so the list grows a name per device met rather than guessing ahead.
_NO_APP_PARTITIONS = frozenset(
    {"userdata", "cache", "metadata", "persist", "omr", "misc", "vm-bootsys", "dspso"}
)

_TOOL_PROVIDERS = {
    # p7zip 16.02 does not read ext4 at all; 7-Zip >= 24 does. Neither reads EROFS.
    "7z": "7-Zip >= 24 (Arch: `7zip`, Debian: `7zip`; p7zip 16.02 will NOT do)",
    # Both ship in Arch's `android-tools`; Debian splits them and packages lpunpack nowhere at
    # all, so naming one package would send half the operators looking for something that does
    # not exist. The worker image compiles lpunpack from nmeum/android-tools for that reason.
    "lpunpack": "Arch: `android-tools`; Debian: no package provides it, build it from "
    "github.com/nmeum/android-tools as the worker image does",
    "simg2img": "Arch: `android-tools`, Debian: `android-sdk-libsparse-utils`",
    "fsck.erofs": "erofs-utils",
}

# Partitions that legitimately carry no APK and no config input, so yielding nothing is not
# an extraction failure for them. Measured: `system_other` (the inactive slot's staging
# partition, 429 entries on oriole, zero matches) and every `*_dlkm` partition (kernel modules
# only — 333 entries on oriole's vendor_dlkm, 105 on the emulator's system_dlkm). Anything
# else that comes back empty is a bug in the patterns or the extraction, and says so.
# `prism` and `optics` are Samsung's CSC pair and are kept rather than skipped: they are
# where a carrier build could put an app, and on the measured `SM-S911U` they hold 397 and 72
# entries with not one artifact among them. Listed here rather than in `_NO_APP_PARTITIONS`
# so that the day one of them does carry an APK, it is extracted instead of never opened.
#
# `odm` is the one entry here that COSTS something. It carries 4 APKs on Nothing's FroggerPro
# and is a 10-file stub on Samsung's SM-S911U — `etc/build.prop`, `etc/passwd`, four selinux
# files — so it is a partition that genuinely holds no app on one vendor and real apps on
# another, and this list is per-partition rather than per-vendor. Listing it buys Samsung a
# job that finishes at the price of Nothing's `odm` no longer raising if its extraction
# breaks. The alternative is to separate "the image yielded no file" from "the image yielded
# files and none was wanted", which the EROFS path cannot currently tell apart at all
# (`listed is None` below); that is a change to what this guard MEANS and wants deciding
# rather than slipping in.
PARTITIONS_ALLOWED_EMPTY = frozenset(
    {"system_other", "cache", "metadata", "userdata", "prism", "optics", "odm"}
)


def partition_may_be_empty(name: str) -> bool:
    return name in PARTITIONS_ALLOWED_EMPTY or name.endswith("_dlkm")


# Matched against the partition-relative path with a leading "/" prepended, so a pattern
# starting with `*/` matches both a nested root (`system/etc/...`) and a partition rooted
# directly at `etc/`. Measured on real firmware: `roles.xml` is simply absent from some
# builds, `default-permissions*.xml` is present and is a rule-ladder input, and every
# partition carries a populated `etc/sysconfig/`. The last three patterns are basename
# catch-alls for OEMs that file the same inputs somewhere else.
ARTIFACT_PATTERNS: tuple[str, ...] = (
    "*.apk",
    "*/etc/permissions/*.xml",
    "*/etc/sysconfig/*",
    "*/etc/default-permissions/*.xml",
    "*/privapp-permissions*.xml",
    "*/default-permissions*.xml",
    "*/roles.xml",
)


@dataclass(frozen=True, slots=True)
class PartitionImage:
    """One filesystem image. `name` is the identity with any A/B slot suffix stripped, so
    `system_a` from one device and `system` from another compare; `slot` keeps the spelling
    the source used, which is what keeps two slots' files apart on disk."""

    name: str
    path: Path
    fmt: ContainerFormat
    slot: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedArtifact:
    partition: str
    # Exactly as recorded inside the image, nested root included. This is what the ground
    # truth manifests are keyed on (`{partition}/{image_path}`).
    image_path: str
    # The canonical absolute path the file has on a running device. Task 4 keys facts on
    # this, so the nested-root difference between partitions cannot leak into them.
    device_path: str
    local_path: Path


@dataclass(frozen=True, slots=True)
class GptPartition:
    name: str
    offset: int
    size: int


def _classify_header(header: bytes) -> ContainerFormat:
    """Order matters: a sparse-wrapped super still has to be un-sparsed first, and a GPT
    disk's own filesystem-looking bytes live inside its partitions, not at its head."""

    def at(offset: int, magic: bytes) -> bool:
        return header[offset : offset + len(magic)] == magic

    if at(0, _SPARSE_MAGIC):
        return ContainerFormat.SPARSE_IMAGE
    if at(0, _ZIP_MAGIC):
        return ContainerFormat.ZIP
    if at(0, _SEVEN_ZIP_MAGIC):
        return ContainerFormat.SEVEN_ZIP
    if at(0, _PAYLOAD_MAGIC):
        return ContainerFormat.PAYLOAD_BIN
    if at(0, _LZ4_FRAME_MAGIC):
        return ContainerFormat.LZ4_FRAME
    # Before the deeper offsets: `ustar` in a tar's first header block is a longer and more
    # structured match than a two-byte ext4 magic that a tar's first FILE could carry at the
    # same offset by coincidence.
    if at(_TAR_MAGIC_OFFSET, _TAR_MAGIC):
        return ContainerFormat.TAR
    if at(_GPT_HEADER_OFFSET, _GPT_MAGIC):
        return ContainerFormat.GPT_DISK
    if at(_LP_GEOMETRY_OFFSET, _LP_GEOMETRY_MAGIC):
        return ContainerFormat.SUPER_IMAGE
    if at(_EROFS_MAGIC_OFFSET, _EROFS_MAGIC):
        return ContainerFormat.EROFS
    if at(_EXT4_MAGIC_OFFSET, _EXT4_MAGIC):
        return ContainerFormat.EXT4
    return ContainerFormat.UNKNOWN


def detect_format(path: Path, *, offset: int = 0) -> ContainerFormat:
    """Classify the container at `offset` in `path` by magic number alone. `offset` lets a
    GPT partition be classified in place, before deciding whether it is worth copying out."""
    with path.open("rb") as fh:
        fh.seek(offset)
        return _classify_header(fh.read(_HEADER_BYTES))


def read_gpt_partitions(path: Path) -> list[GptPartition]:
    """Parse the primary GPT. Done here rather than shelling out to `sfdisk` so the worker
    image needs one fewer package for a path only the local emulator image takes."""
    with path.open("rb") as fh:
        fh.seek(_GPT_HEADER_OFFSET)
        header = fh.read(92)
        if len(header) < 92 or header[:8] != _GPT_MAGIC:
            raise UnpackError(f"read_gpt_partitions: {path} has no GPT header at byte 512")
        entry_lba, entry_count, entry_size = struct.unpack_from("<QII", header, 72)
        if not (0 < entry_count <= 512) or not (128 <= entry_size <= 4096):
            raise UnpackError(
                f"read_gpt_partitions: {path} declares {entry_count} partition entries of "
                f"{entry_size} bytes, which is out of range for a GPT; the image is corrupt "
                "or is not the disk it claims to be"
            )
        sector = _GPT_HEADER_OFFSET  # LBA size; 512 for every Android image seen so far
        fh.seek(entry_lba * sector)
        raw = fh.read(entry_count * entry_size)

    partitions: list[GptPartition] = []
    for i in range(entry_count):
        entry = raw[i * entry_size : (i + 1) * entry_size]
        if len(entry) < 128 or entry[:16] == b"\x00" * 16:
            continue
        first_lba, last_lba = struct.unpack_from("<QQ", entry, 32)
        name = entry[56:128].decode("utf-16-le", errors="replace").rstrip("\x00")
        partitions.append(
            GptPartition(
                name=name, offset=first_lba * sector, size=(last_lba - first_lba + 1) * sector
            )
        )
    return partitions


def normalize_partition_name(name: str) -> str:
    """Strip an A/B slot suffix so the same partition compares across devices and slots."""
    for suffix in ("_a", "_b"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


# --- Untrusted names -> paths. Every path built from bytes this pipeline did not write goes
# through these three, and every filesystem write built from one asserts containment right
# before it happens. GPT partition names are raw UTF-16 out of a disk image, zip member names
# come from a downloaded archive's central directory, and file paths come out of a filesystem
# image: none of them is ours, and all three end up in a path. -------------------------------

_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_COMPONENT_CHARS = 96

# `super.img_sparsechunk.0` … `.13`, how Motorola splits a super image inside its firmware
# zip. Grouped and ordered by this name because nothing else records the order; what each
# member IS still comes from its magic (see `_unpack_sparsechunks`).
_SPARSECHUNK_RE = re.compile(r"^(?P<stem>.+)_sparsechunk\.(?P<index>\d+)$")


def safe_component(name: str, *, context: str) -> str:
    """Reduce an untrusted name to a single, harmless path component.

    Everything outside `[A-Za-z0-9._-]` becomes `_`, so a separator can never survive, and a
    name that reduces to nothing usable (empty, `.`, `..`, all separators) is refused rather
    than replaced with a default: a partition called `..` is not a partition this pipeline
    should be quietly renaming and carrying on with.
    """
    component = _SAFE_COMPONENT_RE.sub("_", Path(name).name.strip())[:_MAX_COMPONENT_CHARS]
    if not component or set(component) <= {"."}:
        raise UnsafePathError(
            f"{context}: {name!r} does not reduce to a usable path component; refusing to "
            "build a path from it"
        )
    return component


def ensure_within(root: Path, candidate: Path, *, context: str) -> Path:
    """Assert `candidate` is inside `root` immediately before writing to it.

    Belt and braces on top of `safe_component`: this also catches an absolute path arriving
    from somewhere that never went through it (`Path("/a/b") / "/etc/passwd"` is
    `/etc/passwd`, and every `exists()` check downstream of that happily passes).
    """
    resolved_root = root.resolve()
    resolved = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise UnsafePathError(
            f"{context}: {candidate} resolves to {resolved}, outside {resolved_root}; refusing "
            "to touch it"
        )
    return resolved


def unique_path(directory: Path, filename: str) -> Path:
    """A path in `directory` that no earlier caller took, deterministically.

    Two zip members `a/system.img` and `b/system.img` reduce to one basename, and an A/B
    super image carves `system_a` and `system_b`; without this the second write silently
    replaces the first and a whole partition's APKs vanish with nothing raised.
    """
    stem, dot, suffix = filename.partition(".")
    candidate = directory / filename
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{dot}{suffix}"
        counter += 1
    return candidate


def safe_archive_path(path: str, *, context: str) -> str:
    """Validate a path read out of an archive or filesystem listing before it is joined to a
    destination. Absolute paths and `..` segments are refused, not sanitised: normal firmware
    has neither (5,000+ entries measured across five partitions), so one is a malformed or
    hostile image and the job should stop rather than quietly relocate the file."""
    if not path or path.startswith("/") or path.startswith("\\"):
        raise UnsafePathError(f"{context}: refusing absolute path {path!r} from an image listing")
    parts = PurePosixPath(path).parts
    if any(part in {"..", "."} for part in parts) or ":" in parts[0]:
        raise UnsafePathError(
            f"{context}: refusing path {path!r} from an image listing: it contains a traversal "
            "or drive component"
        )
    return path


def canonical_device_path(partition: str, image_path: str) -> str:
    """The absolute path this file has on a running device.

    Measured on real firmware: `system.img` entries carry a nested `system/` root
    (`system/priv-app/...`) while `product.img`, `system_ext.img` and `vendor.img` are rooted
    directly at `app/`, `etc/`, `priv-app/`. Normalising here is what stops that difference
    from reaching task 4 as two spellings of the same file.
    """
    root = image_path.split("/", 1)[0]
    if root == partition:
        return f"/{image_path}"
    return f"/{partition}/{image_path}"


def matches_artifact_patterns(image_path: str, patterns: Sequence[str] = ARTIFACT_PATTERNS) -> bool:
    candidate = f"/{image_path}"
    return any(fnmatch(candidate, pattern) for pattern in patterns)


def _tool_path(name: str, configured: str | None = None) -> str:
    """Resolve a tool to an absolute path, naming what provides it when it is missing."""
    candidate = configured or name
    if "/" in candidate:
        if Path(candidate).is_file():
            return candidate
        raise MissingToolError(
            f"_tool_path: {name} was configured as {candidate!r} but no such file exists; "
            "fix the setting or install the tool"
        )
    resolved = shutil.which(candidate)
    if resolved is None:
        raise MissingToolError(
            f"_tool_path: {candidate!r} is not on PATH. Install it "
            f"({_TOOL_PROVIDERS.get(name, 'see docs/research/local-probe.md')}) or point the "
            "matching setting at it."
        )
    return resolved


def _drop_consumed_intermediate(source: Path, depth: int) -> None:
    """Delete an intermediate this module produced, never the caller's own input.

    At depth 0 `source` is the archive the acquire stage downloaded and still records in the
    job's state file; deleting it there turns a resumable failure into a full re-download.
    Reachable the moment a driver hands back a bare `super.img` or `payload.bin` rather than a
    zip, which is a normal shape for several OEMs.
    """
    if depth > 0:
        source.unlink(missing_ok=True)


async def _run_tool(argv: Sequence[str], *, ok_codes: Sequence[int] = (0,)) -> str:
    """Run an unpacking tool, returning its stdout. `create_subprocess_exec` + `communicate`
    rather than `subprocess.run`: these all run inside the worker's event loop, and both
    pipes must be drained concurrently or a chatty tool deadlocks on a full pipe buffer."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode not in ok_codes:
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise ToolFailedError(
            f"_run_tool: {argv[0]} exited {proc.returncode} running {' '.join(argv)}: "
            + " | ".join(tail)
        )
    if proc.returncode != 0:
        # 7z exits 1 on a non-fatal warning ("There are data after the end of archive" is
        # normal for a partition image with slack past the filesystem). Logged, not hidden.
        logger.info("%s exited %d (non-fatal)", argv[0], proc.returncode)
    return stdout.decode("utf-8", "replace")


def _safe_member_name(member: str) -> str:
    """The basename of a zip member, refusing anything that could escape the destination.
    Members come from an archive downloaded off the internet; they are never trusted."""
    return safe_component(member, context="_safe_member_name")


def _extract_zip_member(
    archive: Path, member: str, dest_dir: Path, *, max_bytes: int | None = None
) -> Path:
    """One member out, onto a path that is inside `dest_dir` and that no earlier member took.

    `a/system.img` and `b/system.img` share a basename; before `unique_path` the second wrote
    over the first and a whole partition's APKs disappeared with nothing raised.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_path(dest_dir, _safe_member_name(member))
    ensure_within(dest_dir, dest, context="_extract_zip_member")
    with zipfile.ZipFile(archive) as zf:
        declared = zf.getinfo(member).file_size
        if max_bytes is not None and declared > max_bytes:
            raise UnpackError(
                f"_extract_zip_member: {member!r} in {archive.name} declares {declared} bytes, "
                f"over the {max_bytes}-byte ceiling; raise max_firmware_archive_bytes if this "
                "firmware is genuinely that large"
            )
        with zf.open(member) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
    return dest


def _copy_slice(source: Path, dest: Path, *, root: Path, offset: int, size: int) -> Path:
    # Checked against `root`, never against `dest.parent`: the parent of an escaping path is
    # itself outside, so containment against it is a tautology that passes for exactly the
    # input it is supposed to catch.
    ensure_within(root, dest, context="_copy_slice")
    dest.parent.mkdir(parents=True, exist_ok=True)
    remaining = size
    with source.open("rb") as src, dest.open("wb") as out:
        src.seek(offset)
        while remaining > 0:
            chunk = src.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)
    if remaining > 0:
        raise UnpackError(
            f"_copy_slice: {source} ended {remaining} bytes short of the {size}-byte slice at "
            f"offset {offset}; the image is truncated"
        )
    return dest


async def unpack_to_partitions(
    source: Path, workdir: Path, *, settings: Settings
) -> list[PartitionImage]:
    """Open `source` (a factory zip, OTA payload, super image, sparse image, whole-disk image
    or a bare filesystem image) into the filesystem images it ultimately contains.

    Everything is written under `workdir`, which the caller owns and is expected to delete
    once the files worth keeping have been extracted out of it — these are the multi-GB
    intermediates retention exists for. `source` itself is never deleted; intermediates this
    function creates under `workdir` are, as soon as the next step has consumed them.
    """
    return _resolve_partition_identities(
        await _unpack(source, source.stem, workdir, settings, depth=0)
    )


def _resolve_partition_identities(partitions: list[PartitionImage]) -> list[PartitionImage]:
    """Settle A/B slots and refuse a genuine identity collision.

    `lpunpack` on a retrofitted A/B super emits `system_a` and `system_b`, which normalise to
    one name; extracting both would double every package, and letting the second overwrite
    the first loses one. Slot A wins (it is the slot a factory image populates) and slot B is
    dropped with a warning. Anything else that collides — two zip members named `system.img`
    under different directories — is ambiguous in a way nothing here can resolve, so it is a
    named error.
    """
    by_slot = {partition.name: partition for partition in partitions}
    kept: list[PartitionImage] = []
    for partition in partitions:
        if partition.name.endswith("_b") and partition.name[:-2] + "_a" in by_slot:
            logger.warning(
                "partition %s dropped: slot A of the same partition is present in this image",
                partition.name,
            )
            continue
        kept.append(partition)

    seen: dict[str, PartitionImage] = {}
    resolved: list[PartitionImage] = []
    for partition in kept:
        identity = normalize_partition_name(partition.name)
        if identity in seen:
            raise DuplicatePartitionError(
                f"_resolve_partition_identities: {partition.path.name} and "
                f"{seen[identity].path.name} both claim partition {identity!r}. Refusing to "
                "guess which one the device would mount; extract them as separate jobs."
            )
        seen[identity] = partition
        resolved.append(
            PartitionImage(
                name=identity, path=partition.path, fmt=partition.fmt, slot=partition.name
            )
        )
    return resolved


async def _unpack(
    source: Path, name: str, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    if depth > _MAX_UNPACK_DEPTH:
        raise UnpackError(
            f"_unpack: container nesting deeper than {_MAX_UNPACK_DEPTH} at {source}; refusing "
            "to keep unpacking"
        )
    fmt = await asyncio.to_thread(detect_format, source)
    logger.info("unpacking %s (%s) at depth %d", source.name, fmt, depth)
    match fmt:
        case ContainerFormat.ZIP:
            return await _unpack_zip(source, workdir, settings, depth)
        case ContainerFormat.SEVEN_ZIP:
            return await _unpack_7z(source, workdir, settings, depth)
        case ContainerFormat.TAR:
            return await _unpack_tar(source, workdir, settings, depth)
        case ContainerFormat.LZ4_FRAME:
            return await _unpack_lz4(source, name, workdir, settings, depth)
        case ContainerFormat.PAYLOAD_BIN:
            return await _unpack_payload(source, workdir, settings, depth)
        case ContainerFormat.SPARSE_IMAGE:
            return await _unpack_sparse(source, name, workdir, settings, depth)
        case ContainerFormat.SUPER_IMAGE:
            return await _unpack_super(source, workdir, settings, depth)
        case ContainerFormat.GPT_DISK:
            return await _unpack_gpt(source, workdir, settings, depth)
        case ContainerFormat.EXT4 | ContainerFormat.EROFS:
            # The name stays as the source spelled it (slot suffix included); collapsing
            # `system_a` to `system` here is what let two slots overwrite each other.
            return [
                PartitionImage(name=safe_component(name, context="_unpack"), path=source, fmt=fmt)
            ]
        case _:
            raise UnsupportedContainerError(
                f"_unpack: {source} is not a container this pipeline reads (leading bytes match "
                "no known magic). Add its format to the dispatch table in uadclaw.unpack."
            )


async def _unpack_zip(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    members = await asyncio.to_thread(_zip_members, source)
    nested_zips = [m for m in members if m.lower().endswith(".zip")]
    payloads = [m for m in members if Path(m).name == "payload.bin"]
    images = [m for m in members if m.lower().endswith(".img")]

    # A Pixel factory zip holds the partition images inside a nested `image-<device>-<build>
    # .zip` and keeps only bootloader/radio blobs at the top level, so the nested archive
    # wins outright rather than being merged with the outer one's images.
    if nested_zips:
        if images:
            logger.info(
                "%s: taking %d nested archive(s) and ignoring %d top-level .img member(s): %s",
                source.name,
                len(nested_zips),
                len(images),
                ", ".join(sorted(Path(image).name for image in images)),
            )
        found: list[PartitionImage] = []
        for member in nested_zips:
            extracted = await asyncio.to_thread(
                _extract_zip_member,
                source,
                member,
                workdir / "nested",
                max_bytes=settings.max_firmware_archive_bytes,
            )
            found.extend(await _unpack(extracted, extracted.stem, workdir, settings, depth + 1))
            # The nested archive is an intermediate like any other: once its members are out,
            # it is a duplicate copy of gigabytes nothing reads again.
            await asyncio.to_thread(extracted.unlink, True)
        return found

    if payloads:
        extracted = await asyncio.to_thread(
            _extract_zip_member,
            source,
            payloads[0],
            workdir / "payload",
            max_bytes=settings.max_firmware_archive_bytes,
        )
        return await _unpack(extracted, "payload", workdir, settings, depth + 1)

    found: list[PartitionImage] = []
    chunk_sets = _sparsechunk_sets(members)
    for stem, chunks in sorted(chunk_sets.items()):
        found.extend(await _unpack_sparsechunks(source, stem, chunks, workdir, settings, depth))

    for member in images:
        stem = Path(member).stem
        if normalize_partition_name(stem) in _NON_FILESYSTEM_IMAGES:
            continue
        extracted = await asyncio.to_thread(
            _extract_zip_member,
            source,
            member,
            workdir / "images",
            max_bytes=settings.max_firmware_archive_bytes,
        )
        try:
            found.extend(await _unpack(extracted, stem, workdir, settings, depth + 1))
        except UnsupportedContainerError:
            # A firmware archive carries plenty of images that are not filesystems and are
            # not on the skip list either (per-OEM firmware blobs). Dropping one is normal;
            # the caller's "no APKs at all" guard is what catches dropping them all.
            logger.info("skipping %s: not a filesystem image", member)
            await asyncio.to_thread(extracted.unlink, True)

    # Whatever the scan above did not claim is sniffed. A Samsung archive claims nothing —
    # its six members are named `.tar.md5` and this pipeline reads none of that — so all six
    # are opened by their magic, and each is walked as a stream rather than extracted, which
    # is what keeps an 11.47 GB copy of the `AP_` member off the disk.
    claimed = set(images) | {member for chunks in chunk_sets.values() for _index, member in chunks}
    seen: dict[tuple[str, str], Path] = {}
    tars = await asyncio.to_thread(
        _zip_tar_members, source, [member for member in members if member not in claimed]
    )
    for member in tars:
        extracted = await asyncio.to_thread(
            _stream_zip_tar,
            source,
            member,
            workdir / "tar",
            root=workdir,
            max_bytes=settings.max_firmware_archive_bytes,
            seen=seen,
        )
        logger.info("%s: %s yielded %d partition image(s)", source.name, member, len(extracted))
        found.extend(await _unpack_images(extracted, workdir, settings, depth + 1))

    if not found:
        raise UnsupportedContainerError(
            f"_unpack_zip: {source.name} contains no nested archive, no payload.bin, no "
            f"sparsechunk set, no tar carrying a partition image and no readable .img member "
            f"(it has {len(members)} entries); this is not a firmware archive shape the "
            "pipeline knows"
        )
    return found


def _sparsechunk_sets(members: Sequence[str]) -> dict[str, list[tuple[int, str]]]:
    """Group `super.img_sparsechunk.0 … .13` members by the image they rebuild.

    Motorola splits its super image across up to 14 chunk files, none of which ends in
    `.img`, so before this the whole set was invisible to the `.img` member scan and a
    Motorola firmware zip read as "no image members at all". The NAME is used only to group
    and order the set — nothing else can, since the order lives nowhere but the suffix — and
    `_unpack_sparsechunks` still refuses any member whose BYTES are not a sparse image.
    """
    sets: dict[str, list[tuple[int, str]]] = {}
    for member in members:
        match = _SPARSECHUNK_RE.match(PurePosixPath(member).name)
        if match is not None:
            sets.setdefault(match.group("stem"), []).append((int(match.group("index")), member))
    return sets


async def _unpack_sparsechunks(
    source: Path,
    stem: str,
    chunks: list[tuple[int, str]],
    workdir: Path,
    settings: Settings,
    depth: int,
) -> list[PartitionImage]:
    """Rebuild one raw image out of an ordered sparse-chunk set.

    `simg2img` takes the whole set in one call and writes one raw image. The set has to be
    complete: a missing chunk would produce a shorter image that still mounts and still
    yields APKs, which is the quietest possible way to lose a partition's worth of packages.
    """
    ordered = sorted(chunks)
    indices = [index for index, _member in ordered]
    if indices != list(range(len(indices))):
        raise UnpackError(
            f"_unpack_sparsechunks: {source.name} carries chunks {indices} for {stem!r}, which "
            f"is not the complete run 0..{len(indices) - 1}. A gap means the archive is "
            "incomplete; re-download it rather than rebuilding a short image."
        )

    chunk_dir = workdir / "chunks"
    extracted: list[Path] = []
    for _index, member in ordered:
        path = await asyncio.to_thread(
            _extract_zip_member,
            source,
            member,
            chunk_dir,
            max_bytes=settings.max_firmware_archive_bytes,
        )
        fmt = await asyncio.to_thread(detect_format, path)
        if fmt is not ContainerFormat.SPARSE_IMAGE:
            raise UnpackError(
                f"_unpack_sparsechunks: {member} is named as a sparse chunk of {stem!r} but its "
                f"leading bytes are {fmt}, not an Android sparse image. The name orders the "
                "set; the magic decides what it is, and these disagree."
            )
        extracted.append(path)

    raw_dir = workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = safe_component(Path(stem).stem, context="_unpack_sparsechunks")
    raw = unique_path(raw_dir, f"{name}.raw.img")
    await _run_tool([_tool_path("simg2img"), *(str(path) for path in extracted), str(raw)])
    for path in extracted:
        await asyncio.to_thread(path.unlink, True)
    logger.info("rebuilt %s from %d sparse chunk(s)", raw.name, len(extracted))
    return await _unpack(raw, name, workdir, settings, depth + 1)


async def _unpack_7z(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    """A 7-Zip archive of partition images (Nothing's `-image-logical.7z` volume set, joined
    into one file by the driver before it ever reaches here).

    `-t7z` forces the handler for the same reason the ext4 path does: the format is already
    known from the magic, and 7z's own sniffing is what mistook an EROFS image for a gzip
    stream and reported one file where there were 105.
    """
    records = await _list_7z_records(source)
    wanted = [
        record
        for record in records
        if record["Path"].lower().endswith(".img")
        and normalize_partition_name(PurePosixPath(record["Path"]).stem)
        not in _NON_FILESYSTEM_IMAGES
    ]
    for record in wanted:
        safe_archive_path(record["Path"], context=f"_unpack_7z[{source.name}]")
        declared = int(record.get("Size") or 0)
        if declared > settings.max_firmware_archive_bytes:
            raise UnpackError(
                f"_unpack_7z: {record['Path']!r} in {source.name} declares {declared} bytes, "
                f"over the {settings.max_firmware_archive_bytes}-byte ceiling; raise "
                "max_firmware_archive_bytes if this firmware is genuinely that large"
            )
    if not wanted:
        raise UnsupportedContainerError(
            f"_unpack_7z: {source.name} holds {len(records)} entries and none of them is a "
            "filesystem image this pipeline reads; this is not a firmware archive shape the "
            "pipeline knows"
        )

    out_dir = workdir / "sevenzip"
    await _extract_7z(source, [record["Path"] for record in wanted], out_dir)
    _drop_consumed_intermediate(source, depth)
    found: list[PartitionImage] = []
    for record in wanted:
        image = ensure_within(out_dir, Path(record["Path"]), context="_unpack_7z")
        if not image.is_file():
            raise NothingExtractedError(
                f"_unpack_7z: {record['Path']} was listed inside {source.name} but is not on "
                "disk after extraction; 7z silently skipped it"
            )
        try:
            found.extend(await _unpack(image, image.stem, workdir, settings, depth + 1))
        except UnsupportedContainerError:
            logger.info("skipping %s: not a filesystem image", record["Path"])
            await asyncio.to_thread(image.unlink, True)
    if not found:
        raise UnpackError(
            f"_unpack_7z: none of the {len(wanted)} image(s) in {source.name} is a filesystem "
            "this pipeline reads"
        )
    return found


async def _list_7z_records(archive: Path) -> list[dict[str, str]]:
    listing = await _run_tool(
        [_tool_path("7z"), "l", "-slt", "-ba", "-t7z", str(archive)], ok_codes=(0, 1)
    )
    return [record for record in parse_7z_records(listing) if record.get("Path")]


async def _extract_7z(archive: Path, names: Sequence[str], dest: Path) -> None:
    """Hand 7z the exact members to pull, through a list file for the same reasons the ext4
    path uses one: no argument-length ceiling and no ambiguity with a leading dash."""
    dest.mkdir(parents=True, exist_ok=True)
    list_file = archive.with_suffix(archive.suffix + ".7zlist")
    await asyncio.to_thread(list_file.write_text, "\n".join(names) + "\n", "utf-8")
    await _run_tool(
        [
            _tool_path("7z"),
            "x",
            "-y",
            "-bd",
            "-bso0",
            "-scsUTF-8",
            "-t7z",
            f"-o{dest}",
            str(archive),
            f"@{list_file}",
        ],
        ok_codes=(0, 1),
    )
    await asyncio.to_thread(list_file.unlink, True)


def _zip_members(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return [info.filename for info in zf.infolist() if not info.is_dir()]


# --- tar and LZ4, the two steps Samsung adds. Neither is decided from a name: `AP_….tar.md5`
# is a plain tar with an md5 line appended, and every image inside it wears `.img.lz4`, so
# both would be trivial to dispatch on and both are dispatched on magic instead. ------------

# What a partition image can be once it has been decoded. A tar entry that lands outside this
# set is dropped exactly as a non-filesystem `.img` member of a zip is: an `AP_` tar carries
# a 1.2 GB `meta-data/fota.zip` next to the partitions, and recursing into that would unpack
# a whole second OTA package nothing downstream reads.
_PARTITION_FORMATS = frozenset(
    {
        ContainerFormat.SPARSE_IMAGE,
        ContainerFormat.SUPER_IMAGE,
        ContainerFormat.EXT4,
        ContainerFormat.EROFS,
    }
)

_SKIPPED_TAR_PARTITIONS = _NON_FILESYSTEM_IMAGES | _NO_APP_PARTITIONS


def _decoded_chunks(
    payload: IO[bytes], head: bytes, *, framed: bool, max_bytes: int, context: str
) -> Iterator[bytes]:
    """One entry's payload, LZ4-decoded when its own leading bytes said it was a frame.

    Streamed rather than decoded in one call: an LZ4 frame declares no output size anywhere
    in its header, so `lz4.frame.decompress` on a Samsung `super.img.lz4` builds an 11 GB
    `bytes` in the worker's heap before returning any of it. `max_bytes` is the same ceiling
    the zip path applies to a declared member size, applied here to what actually comes out,
    since nothing declares it up front — which is also what bounds a decompression bomb.
    """
    decompressor = lz4.frame.LZ4FrameDecompressor() if framed else None
    produced = 0
    chunk = head
    while chunk:
        decoded = decompressor.decompress(chunk) if decompressor is not None else chunk
        if decoded:
            produced += len(decoded)
            if produced > max_bytes:
                raise UnpackError(
                    f"{context}: decoded more than the {max_bytes}-byte ceiling without reaching "
                    "the end of the entry; raise max_firmware_archive_bytes if this firmware is "
                    "genuinely that large"
                )
            yield decoded
        chunk = payload.read(_STREAM_CHUNK_BYTES)


def _extract_tar_partitions(
    stream: IO[bytes],
    dest_dir: Path,
    *,
    root: Path,
    context: str,
    max_bytes: int,
    seen: dict[tuple[str, str], Path],
) -> list[tuple[str, Path]]:
    """Every partition image inside one tar, without the tar itself ever being written.

    `tarfile`'s `r|` stream mode walks a non-seekable file object, which is exactly what a zip
    member is, so the 11.47 GB `AP_` member is decompressed once on its way past rather than
    landing on disk to be read again.

    Each entry is classified twice from its own bytes: once raw, which is what says whether it
    is an LZ4 frame, and once on the decoded head, because what has to be known is what the
    payload IS. `seen` is keyed on the partition name AND the sha256 of the decoded image, so
    a build that ships one partition in two of its tars (`CSC_` and `HOME_CSC_` carry
    byte-identical `prism` and `optics`) resolves rather than colliding, while two DIFFERENT
    partitions that happen to hold the same bytes — two near-empty images of one size is not
    an exotic shape — stay two partitions. There is nothing to guess between two copies of one
    partition, which is what separates this from `DuplicatePartitionError`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    found: list[tuple[str, Path]] = []
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            safe_archive_path(member.name, context=context)
            payload = tar.extractfile(member)
            if payload is None:
                continue
            raw_head = payload.read(_HEADER_BYTES)
            raw_fmt = _classify_header(raw_head)
            framed = raw_fmt is ContainerFormat.LZ4_FRAME
            if not framed and raw_fmt not in _PARTITION_FORMATS:
                logger.info("%s: %s is %s, not a partition image", context, member.name, raw_fmt)
                continue
            # The suffix is dropped only because the magic already said the payload was a
            # frame; it is bookkeeping for the name on disk, never the reason to decode.
            entry = PurePosixPath(member.name)
            filename = safe_component(entry.stem if framed else entry.name, context=context)
            name = Path(filename).stem
            if normalize_partition_name(name) in _SKIPPED_TAR_PARTITIONS:
                logger.info("%s: skipping %s, it carries no artifact", context, member.name)
                continue

            chunks = _decoded_chunks(
                payload,
                raw_head,
                framed=framed,
                max_bytes=max_bytes,
                context=f"{context}[{member.name}]",
            )
            head = bytearray()
            for chunk in chunks:
                head += chunk
                if len(head) >= _HEADER_BYTES:
                    break
            fmt = _classify_header(bytes(head[:_HEADER_BYTES]))
            if fmt not in _PARTITION_FORMATS:
                logger.info(
                    "%s: %s decodes to %s, not a partition image", context, member.name, fmt
                )
                continue

            dest = unique_path(dest_dir, filename)
            ensure_within(root, dest, context=context)
            digest = hashlib.sha256()
            with dest.open("wb") as out:
                out.write(head)
                digest.update(head)
                for chunk in chunks:
                    out.write(chunk)
                    digest.update(chunk)
            key = (name, digest.hexdigest())
            if key in seen:
                logger.info(
                    "%s: %s is byte-identical to %s, already taken from an earlier member",
                    context,
                    member.name,
                    seen[key].name,
                )
                dest.unlink(missing_ok=True)
                continue
            seen[key] = dest
            found.append((name, dest))
    return found


def _stream_zip_tar(
    archive: Path,
    member: str,
    dest_dir: Path,
    *,
    root: Path,
    max_bytes: int,
    seen: dict[tuple[str, str], Path],
) -> list[tuple[str, Path]]:
    with zipfile.ZipFile(archive) as zf, zf.open(member) as stream:
        return _extract_tar_partitions(
            stream,
            dest_dir,
            root=root,
            context=f"_stream_zip_tar[{member}]",
            max_bytes=max_bytes,
            seen=seen,
        )


def _zip_tar_members(archive: Path, members: Sequence[str]) -> list[str]:
    """Which of `members` are tars, decided by reading each one's leading bytes.

    Costs one 8 KB decompression per member, which is why it is handed only the members the
    name-shaped scan did not already claim — and why a Samsung archive, whose six members
    claim nothing, gets all six sniffed.
    """
    tars: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for member in members:
            with zf.open(member) as stream:
                if _classify_header(stream.read(_HEADER_BYTES)) is ContainerFormat.TAR:
                    tars.append(member)
    return tars


async def _unpack_images(
    images: Sequence[tuple[str, Path]], workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    """Recurse into images already classified as partition formats, dropping the ones that
    turn out not to be a filesystem this pipeline reads."""
    found: list[PartitionImage] = []
    for name, path in images:
        try:
            found.extend(await _unpack(path, name, workdir, settings, depth + 1))
        except UnsupportedContainerError:
            logger.info("skipping %s: not a filesystem image", path.name)
            await asyncio.to_thread(path.unlink, True)
    return found


async def _unpack_tar(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    def read() -> list[tuple[str, Path]]:
        with source.open("rb") as stream:
            return _extract_tar_partitions(
                stream,
                workdir / "tar",
                root=workdir,
                context=f"_unpack_tar[{source.name}]",
                max_bytes=settings.max_firmware_archive_bytes,
                seen={},
            )

    images = await asyncio.to_thread(read)
    if not images:
        raise NothingExtractedError(
            f"_unpack_tar: {source.name} holds no entry whose bytes are a partition image; a "
            "tar handed here directly is expected to be the one carrying them"
        )
    _drop_consumed_intermediate(source, depth)
    found = await _unpack_images(images, workdir, settings, depth)
    if not found:
        raise UnpackError(
            f"_unpack_tar: none of the {len(images)} image(s) in {source.name} is a filesystem "
            "this pipeline reads"
        )
    return found


async def _unpack_lz4(
    source: Path, name: str, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    raw_dir = workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = unique_path(raw_dir, f"{safe_component(name, context='_unpack_lz4')}.img")

    def decode() -> None:
        with source.open("rb") as src, raw.open("wb") as out:
            head = src.read(_STREAM_CHUNK_BYTES)
            for chunk in _decoded_chunks(
                src,
                head,
                framed=True,
                max_bytes=settings.max_firmware_archive_bytes,
                context=f"_unpack_lz4[{source.name}]",
            ):
                out.write(chunk)

    await asyncio.to_thread(decode)
    _drop_consumed_intermediate(source, depth)
    return await _unpack(raw, name, workdir, settings, depth + 1)


async def _unpack_sparse(
    source: Path, name: str, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    raw_dir = workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = unique_path(raw_dir, f"{safe_component(name, context='_unpack_sparse')}.raw.img")
    await _run_tool([_tool_path("simg2img"), str(source), str(raw)])
    _drop_consumed_intermediate(source, depth)
    return await _unpack(raw, name, workdir, settings, depth + 1)


def unpopulated_slot_b(stem: str, siblings: Sequence[str]) -> bool:
    """Whether an unreadable partition image is a slot B that was never written.

    lpunpack writes every partition the super metadata declares, and on a retrofitted A/B
    device the B slot has extents but no content until the first OTA lands. Measured
    2026-08-11 on Motorola `rtwo`: 5 of the 6 `_b` images lpunpack produced are ZERO bytes,
    matching no magic, while `system_b` is a real 1.6 MB EROFS — and one zero-byte file was
    enough to fail the whole build with "not a container this pipeline reads".

    The A sibling has to be present for this to be droppable. Without that check this is
    "ignore any partition we cannot read", which is how a whole partition's packages go
    missing with nothing raised; `_resolve_partition_identities` drops the same image a
    moment later for the same reason, so nothing extra is lost here.
    """
    return stem.endswith("_b") and f"{stem[:-2]}_a" in set(siblings)


async def _unpack_super(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    parts_dir = workdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    await _run_tool([_tool_path("lpunpack"), str(source), str(parts_dir)])
    _drop_consumed_intermediate(source, depth)
    found: list[PartitionImage] = []
    images = sorted(parts_dir.glob("*.img"))
    stems = [image.stem for image in images]
    for image in images:
        try:
            found.extend(await _unpack(image, image.stem, workdir, settings, depth + 1))
        except UnsupportedContainerError:
            if not unpopulated_slot_b(image.stem, stems):
                raise
            logger.info(
                "partition %s is an unwritten slot B (%d bytes); slot A of the same partition "
                "is present",
                image.stem,
                image.stat().st_size,
            )
            await asyncio.to_thread(image.unlink, True)
    if not found:
        raise UnpackError(
            f"_unpack_super: lpunpack produced no filesystem image from {source.name}; the super "
            "image is empty or holds only partitions this pipeline cannot read"
        )
    return found


async def _unpack_gpt(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    partitions = await asyncio.to_thread(read_gpt_partitions, source)
    slices_dir = workdir / "slices"
    found: list[PartitionImage] = []
    for part in partitions:
        # Classified in place first: copying a multi-GB partition out only to discover it is
        # a vbmeta blob is the one avoidable cost on this path.
        fmt = await asyncio.to_thread(detect_format, source, offset=part.offset)
        if fmt is ContainerFormat.UNKNOWN:
            continue
        # The name is raw UTF-16 out of the disk image. A partition called `../../pwned`
        # otherwise carved attacker-controlled bytes outside the scratch directory, and the
        # same name then escaped a second time as the extraction destination.
        name = safe_component(part.name or f"part{part.offset}", context="_unpack_gpt")
        if normalize_partition_name(name) in _NON_FILESYSTEM_IMAGES:
            continue
        slices_dir.mkdir(parents=True, exist_ok=True)
        carved = await asyncio.to_thread(
            _copy_slice,
            source,
            unique_path(slices_dir, f"{name}.img"),
            root=workdir,
            offset=part.offset,
            size=part.size,
        )
        found.extend(await _unpack(carved, name, workdir, settings, depth + 1))
    if not found:
        raise UnpackError(
            f"_unpack_gpt: {source.name} has a GPT with {len(partitions)} partitions and none of "
            "them is a filesystem or super image"
        )
    return found


async def _unpack_payload(
    source: Path, workdir: Path, settings: Settings, depth: int
) -> list[PartitionImage]:
    """A/B OTA payload. Every partition is dumped and the non-filesystems are dropped right
    after, rather than selected up front from `-l` output: the listing is a human-facing
    table with no stable machine format, and mis-parsing it would silently skip a partition
    full of APKs. The cost is transient disk for a few boot images, which the deletion below
    reclaims immediately."""
    out_dir = workdir / "payload-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    await _run_tool(
        [
            _tool_path("payload-dumper-go", settings.payload_dumper_path),
            "-o",
            str(out_dir),
            str(source),
        ]
    )
    _drop_consumed_intermediate(source, depth)
    found: list[PartitionImage] = []
    for image in sorted(out_dir.glob("*.img")):
        if normalize_partition_name(image.stem) in _NON_FILESYSTEM_IMAGES:
            await asyncio.to_thread(image.unlink, True)
            continue
        try:
            found.extend(await _unpack(image, image.stem, workdir, settings, depth + 1))
        except UnsupportedContainerError:
            logger.info("skipping payload partition %s: not a filesystem image", image.name)
            await asyncio.to_thread(image.unlink, True)
    if not found:
        raise UnpackError(
            f"_unpack_payload: payload-dumper-go produced no filesystem image from {source.name}"
        )
    return found


def parse_7z_records(text: str) -> list[dict[str, str]]:
    """Regular-file entries out of `7z l -slt -ba` output, each as its whole key/value block.

    `-slt` emits one `Key = Value` block per entry separated by blank lines, which parses
    unambiguously; the default column layout does not (paths contain spaces). Directories and
    links are dropped here so no caller has to remember to.
    """
    records: list[dict[str, str]] = []
    record: dict[str, str] = {}

    def flush() -> None:
        if not record.get("Path"):
            return
        if record.get("Folder") == "+":
            return
        if record.get("Symbolic Link"):
            return
        if record.get("Mode", "").startswith(("l", "d")):
            return
        records.append(dict(record))

    for line in text.splitlines():
        if not line.strip():
            flush()
            record = {}
            continue
        key, sep, value = line.partition(" = ")
        if sep:
            record[key.strip()] = value.strip()
    flush()
    return records


def parse_7z_listing(text: str) -> list[str]:
    """Regular-file paths out of `7z l -slt -ba` output."""
    return [record["Path"] for record in parse_7z_records(text)]


async def _list_ext4(image: Path) -> list[str]:
    """`-text` forces the ext4 handler. Without it 7z sniffs, and its sniffing is what
    mistakes an EROFS image for a gzip stream and reports one file instead of a hundred —
    the format is already known here from the magic, so there is nothing to guess."""
    listing = await _run_tool(
        [_tool_path("7z"), "l", "-slt", "-ba", "-text", str(image)], ok_codes=(0, 1)
    )
    return parse_7z_listing(listing)


async def _extract_ext4(image: Path, names: Sequence[str], dest: Path) -> None:
    """Hand 7z the exact paths to pull. A list file rather than argv: the set can run to
    hundreds of entries, and 7z's own `@list` form has no argument-length ceiling and no
    ambiguity with a filename that starts with a dash."""
    dest.mkdir(parents=True, exist_ok=True)
    list_file = image.with_suffix(image.suffix + ".7zlist")
    await asyncio.to_thread(list_file.write_text, "\n".join(names) + "\n", "utf-8")
    await _run_tool(
        [
            _tool_path("7z"),
            "x",
            "-y",
            "-bd",
            "-bso0",
            "-scsUTF-8",
            "-text",
            f"-o{dest}",
            str(image),
            f"@{list_file}",
        ],
        ok_codes=(0, 1),
    )
    await asyncio.to_thread(list_file.unlink, True)


def _harvest_erofs_staging(staging: Path, dest: Path, patterns: Sequence[str]) -> list[str]:
    """`os.walk(followlinks=False)` rather than `rglob`: not descending into a symlinked
    directory is what stops a partition image's own symlink from walking the host filesystem,
    and on `rglob` that is an interpreter default (3.13 added `recurse_symlinks`) rather than
    anything this code states. Stated here instead."""
    moved: list[str] = []
    for root, _dirs, files in os.walk(staging, followlinks=False):
        for filename in sorted(files):
            path = Path(root) / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            if not matches_artifact_patterns(relative, patterns):
                continue
            target = ensure_within(dest, Path(relative), context="_harvest_erofs_staging")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved.append(relative)
    return sorted(moved)


async def _extract_erofs(image: Path, dest: Path, patterns: Sequence[str]) -> list[str]:
    """The EROFS path, because 7z has no EROFS handler at all (measured: it reports one file
    on an image holding 105). `fsck.erofs --extract` has no selective mode and no listing
    mode, so the whole partition is unpacked into a temporary tree, the wanted paths are
    moved out, and the tree is deleted — the caller still ends up with only the selected
    files, at the cost of transient disk for one partition."""
    staging = unique_path(image.parent, image.name + ".erofs-out")
    await _run_tool([_tool_path("fsck.erofs"), f"--extract={staging}", str(image)])
    dest.mkdir(parents=True, exist_ok=True)
    try:
        return await asyncio.to_thread(_harvest_erofs_staging, staging, dest, patterns)
    finally:
        await asyncio.to_thread(shutil.rmtree, staging, True)


async def extract_artifacts(
    partitions: Sequence[PartitionImage],
    dest_dir: Path,
    *,
    patterns: Sequence[str] = ARTIFACT_PATTERNS,
) -> list[ExtractedArtifact]:
    """Pull every APK plus the config inputs later stages read out of `partitions`, into
    `dest_dir/<partition>/<path-inside-the-image>`.

    Emptiness is an error at BOTH scales. A partition that yields nothing raises unless it is
    one that legitimately holds nothing (see `partition_may_be_empty`), because a whole
    partition matching nothing is the same bug as the whole build matching nothing, only
    quieter: four populated partitions out of five still returns hundreds of APKs and a job
    that records SUCCEEDED. And a run that yields no APK anywhere raises regardless, since
    that is indistinguishable from a build with no preinstalled apps.
    """
    artifacts: list[ExtractedArtifact] = []
    for partition in partitions:
        dest = ensure_within(dest_dir, Path(partition.name), context="extract_artifacts")
        if partition.fmt is ContainerFormat.EROFS:
            listed = None
            wanted = await _extract_erofs(partition.path, dest, patterns)
        else:
            entries = await _list_ext4(partition.path)
            listed = len(entries)
            wanted = [path for path in entries if matches_artifact_patterns(path, patterns)]
            # Validated BEFORE the extractor sees them, not after: an absolute or traversing
            # path must never be handed to a tool that will act on it.
            for path in wanted:
                safe_archive_path(path, context=f"extract_artifacts[{partition.name}]")
            if wanted:
                await _extract_ext4(partition.path, wanted, dest)

        if not wanted:
            if not partition_may_be_empty(partition.name):
                raise NothingExtractedError(
                    f"extract_artifacts: partition {partition.name!r} "
                    f"({partition.path.name}, {partition.fmt}) yielded no artifact at all out of "
                    f"{'an unlisted image' if listed is None else f'{listed} entries'}. A "
                    "partition matching nothing is an extraction failure, not an empty "
                    "partition: check the artifact patterns against a listing of this image, or "
                    "add it to PARTITIONS_ALLOWED_EMPTY if it genuinely carries no app."
                )
            logger.info(
                "partition %s yielded nothing, which is expected for this partition",
                partition.name,
            )
            continue

        for path in wanted:
            local = ensure_within(dest, Path(path), context="extract_artifacts")
            if not local.is_file():
                raise UnpackError(
                    f"extract_artifacts: {partition.name}/{path} was listed inside "
                    f"{partition.path.name} but is not on disk after extraction; the extraction "
                    "tool silently skipped it"
                )
            artifacts.append(
                ExtractedArtifact(
                    partition=partition.name,
                    image_path=path,
                    device_path=canonical_device_path(partition.name, path),
                    local_path=local,
                )
            )
        logger.info(
            "partition %s: extracted %d artifact(s) out of %s",
            partition.name,
            len(wanted),
            "an unlisted image" if listed is None else f"{listed} entries",
        )

    apks = sum(1 for artifact in artifacts if artifact.image_path.endswith(".apk"))
    if apks == 0:
        raise NothingExtractedError(
            "extract_artifacts: not one APK came out of "
            f"{len(partitions)} partition(s) ({', '.join(p.name for p in partitions) or 'none'}). "
            "That is an extraction failure, not a build without apps: check the artifact "
            "patterns against a `7z l` of the partition image."
        )
    logger.info(
        "extracted %d artifacts (%d APKs) from %d partitions",
        len(artifacts),
        apks,
        len(partitions),
    )
    return artifacts
