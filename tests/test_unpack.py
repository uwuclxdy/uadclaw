"""Container-format detection, GPT parsing, path normalisation and artifact selection.

Fast: every image here is a handful of synthetic bytes carrying the real magic numbers at
their real offsets. The chain itself (7z, lpunpack, simg2img) is exercised against a real
multi-GB image in `test_unpack_chain_heavy.py`.
"""

import asyncio
import struct
import zipfile
from pathlib import Path

import pytest

from uadclaw import unpack as unpack_module
from uadclaw.settings import Settings
from uadclaw.unpack import (
    ContainerFormat,
    DuplicatePartitionError,
    PartitionImage,
    UnpackError,
    UnsafePathError,
    _extract_zip_member,
    _harvest_erofs_staging,
    canonical_device_path,
    detect_format,
    ensure_within,
    extract_artifacts,
    matches_artifact_patterns,
    normalize_partition_name,
    parse_7z_listing,
    read_gpt_partitions,
    safe_archive_path,
    unpack_to_partitions,
)


def make_settings() -> Settings:
    return Settings(
        postgres_password="test-only-password",
        auth_password="test-only-admin-password",
        session_secret="test-only-session-secret",
    )


EXT4_MAGIC_AT = (1024 + 0x38, b"\x53\xef")
EROFS_MAGIC_AT = (1024, b"\xe2\xe1\xf5\xe0")
SUPER_MAGIC_AT = (4096, b"gDla")
SPARSE_MAGIC_AT = (0, b"\x3a\xff\x26\xed")


def magic_blob(*placements: tuple[int, bytes], size: int = 8192) -> bytes:
    buffer = bytearray(size)
    for offset, magic in placements:
        buffer[offset : offset + len(magic)] = magic
    return bytes(buffer)


def write_at(path, *placements: tuple[int, bytes], size: int = 8192):
    path.write_bytes(magic_blob(*placements, size=size))
    return path


def test_detects_every_container_by_magic_not_by_extension(tmp_path):
    zip_path = tmp_path / "factory.bin"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("image-comet-ad1a.zip", b"nested")

    cases = {
        # A raw ext4 partition image straight out of a Pixel factory zip: leading bytes are
        # zeroes, the only signal is the superblock magic at 0x438.
        "ext4.img": (write_at(tmp_path / "ext4.img", EXT4_MAGIC_AT), ContainerFormat.EXT4),
        "erofs.img": (write_at(tmp_path / "erofs.img", EROFS_MAGIC_AT), ContainerFormat.EROFS),
        "super.img": (
            write_at(tmp_path / "super.img", SUPER_MAGIC_AT),
            ContainerFormat.SUPER_IMAGE,
        ),
        "sparse.img": (
            write_at(tmp_path / "sparse.img", SPARSE_MAGIC_AT),
            ContainerFormat.SPARSE_IMAGE,
        ),
        "disk.img": (write_at(tmp_path / "disk.img", (512, b"EFI PART")), ContainerFormat.GPT_DISK),
        "payload.bin": (
            write_at(tmp_path / "payload.bin", (0, b"CrAU")),
            ContainerFormat.PAYLOAD_BIN,
        ),
        "factory.bin": (zip_path, ContainerFormat.ZIP),
        "blob.img": (write_at(tmp_path / "blob.img"), ContainerFormat.UNKNOWN),
    }
    for name, (path, expected) in cases.items():
        assert detect_format(path) is expected, name


def test_a_sparse_wrapped_super_reads_as_sparse_first(tmp_path):
    """Order matters: an Android-sparse super carries BOTH magics, and un-sparsing has to
    happen before lpunpack ever sees it."""
    path = write_at(tmp_path / "super.img", SPARSE_MAGIC_AT, SUPER_MAGIC_AT)

    assert detect_format(path) is ContainerFormat.SPARSE_IMAGE


def _synthetic_gpt(path, partitions: list[tuple[str, int, int]]):
    """A minimal but real GPT: header at LBA 1, 128-byte entries at LBA 2."""
    sector = 512
    entry_size = 128
    entry_count = len(partitions)
    entries = bytearray()
    for name, first_lba, last_lba in partitions:
        entry = bytearray(entry_size)
        entry[0:16] = b"\x01" * 16  # non-null type guid
        struct.pack_into("<QQ", entry, 32, first_lba, last_lba)
        encoded = name.encode("utf-16-le")
        entry[56 : 56 + len(encoded)] = encoded
        entries += entry

    header = bytearray(92)
    header[0:8] = b"EFI PART"
    struct.pack_into("<QII", header, 72, 2, entry_count, entry_size)

    total = (max(last for _, _, last in partitions) + 1) * sector
    buffer = bytearray(total)
    buffer[sector : sector + len(header)] = header
    buffer[2 * sector : 2 * sector + len(entries)] = entries
    path.write_bytes(bytes(buffer))
    return path


def test_gpt_partitions_are_parsed_without_shelling_out(tmp_path):
    path = _synthetic_gpt(tmp_path / "disk.img", [("vbmeta", 2048, 4095), ("super", 4096, 8191)])

    partitions = read_gpt_partitions(path)

    assert [(p.name, p.offset, p.size) for p in partitions] == [
        ("vbmeta", 2048 * 512, 2048 * 512),
        ("super", 4096 * 512, 4096 * 512),
    ]


def test_a_bogus_partition_table_is_a_named_error(tmp_path):
    path = tmp_path / "disk.img"
    buffer = bytearray(8192)
    buffer[512:520] = b"EFI PART"
    struct.pack_into("<QII", buffer, 512 + 72, 2, 99999, 128)  # absurd entry count
    path.write_bytes(bytes(buffer))

    with pytest.raises(UnpackError):
        read_gpt_partitions(path)


def test_canonical_device_path_undoes_the_nested_root_trap():
    """Measured on real firmware: `system.img` entries carry a nested `system/` root while
    product/system_ext/vendor are rooted at `app/`, `etc/`, `priv-app/`. Both spellings have
    to collapse to the path the file actually has on a device."""
    assert (
        canonical_device_path("system", "system/priv-app/Settings/Settings.apk")
        == "/system/priv-app/Settings/Settings.apk"
    )
    assert (
        canonical_device_path("product", "app/arcore-1.48/arcore-1.48.apk")
        == "/product/app/arcore-1.48/arcore-1.48.apk"
    )
    assert (
        canonical_device_path("system", "system/etc/permissions/privapp-permissions-platform.xml")
        == "/system/etc/permissions/privapp-permissions-platform.xml"
    )
    assert canonical_device_path("vendor", "overlay/x.apk") == "/vendor/overlay/x.apk"


def test_slot_suffixes_collapse():
    assert normalize_partition_name("system_a") == "system"
    assert normalize_partition_name("system_ext_b") == "system_ext"
    assert normalize_partition_name("product") == "product"
    assert normalize_partition_name("vendor_dlkm") == "vendor_dlkm"


def test_artifact_patterns_match_both_nested_and_rooted_partitions():
    wanted = [
        "system/priv-app/Settings/Settings.apk",
        "app/arcore/arcore.apk",
        "system/etc/permissions/privapp-permissions-platform.xml",
        "etc/permissions/privapp-permissions-google.xml",
        "etc/sysconfig/preinstalled-packages-platform.xml",
        "system/etc/sysconfig/framework-sysconfig.xml",
        "etc/default-permissions/default-permissions-google.xml",
        "system/etc/permissions/roles.xml",
    ]
    for path in wanted:
        assert matches_artifact_patterns(path), path

    for path in ["system/lib64/libc.so", "etc/fonts.xml", "media/audio/ringtones/x.ogg"]:
        assert not matches_artifact_patterns(path), path


async def test_pixel_factory_layout_unpacks_without_lpunpack(tmp_path):
    """The measured real layout (oriole, 2026-08-11): outer zip -> nested
    `image-<device>-<build>.zip` -> raw ext4 partition images, no super.img and no sparse
    header anywhere. The nested archive must win over the outer zip's bootloader blobs, and
    `super_empty.img` — a 5 KB metadata template that carries the super magic — must never
    reach lpunpack.
    """
    inner = tmp_path / "image-oriole-cp2a.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("system.img", magic_blob(EXT4_MAGIC_AT))
        zf.writestr("product.img", magic_blob(EXT4_MAGIC_AT))
        zf.writestr("super_empty.img", magic_blob(SUPER_MAGIC_AT))
    outer = tmp_path / "oriole-cp2a-factory-1234abcd.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("oriole-cp2a/image-oriole-cp2a.zip", inner.read_bytes())
        zf.writestr("oriole-cp2a/bootloader-oriole-slider.img", b"\x00" * 64)

    partitions = await unpack_to_partitions(outer, tmp_path / "work", settings=make_settings())

    assert sorted(p.name for p in partitions) == ["product", "system"]
    assert {p.fmt for p in partitions} == {ContainerFormat.EXT4}


async def test_a_gpt_disk_carves_only_its_filesystem_partitions(tmp_path):
    disk = _synthetic_gpt(tmp_path / "disk.img", [("vbmeta", 2, 3), ("system", 4, 40)])
    raw = bytearray(disk.read_bytes())
    raw[4 * 512 + 1024 + 0x38 : 4 * 512 + 1024 + 0x38 + 2] = b"\x53\xef"
    disk.write_bytes(bytes(raw))

    partitions = await unpack_to_partitions(disk, tmp_path / "work", settings=make_settings())

    assert [(p.name, p.fmt) for p in partitions] == [("system", ContainerFormat.EXT4)]


def test_erofs_harvest_moves_only_the_wanted_paths(tmp_path):
    """EROFS is extracted whole (fsck.erofs has no selective mode) and filtered afterwards,
    so the filter is what keeps the result selective."""
    staging = tmp_path / "staging"
    for relative in [
        "priv-app/Settings/Settings.apk",
        "etc/sysconfig/google.xml",
        "lib64/libc.so",
        "etc/build.prop",
    ]:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    dest = tmp_path / "out"
    dest.mkdir()

    moved = _harvest_erofs_staging(staging, dest, ["*.apk", "*/etc/sysconfig/*"])

    assert sorted(moved) == ["etc/sysconfig/google.xml", "priv-app/Settings/Settings.apk"]
    assert (dest / "priv-app/Settings/Settings.apk").is_file()
    assert not (dest / "lib64/libc.so").exists()


def test_7z_listing_parse_skips_folders_and_symlinks():
    listing = (
        "Path = system/priv-app/Settings/Settings.apk\n"
        "Folder = -\n"
        "Size = 8606\n"
        "Symbolic Link = \n"
        "Mode = -rw-r--r--\n"
        "\n"
        "Path = system/priv-app\n"
        "Folder = +\n"
        "Mode = drwxr-xr-x\n"
        "\n"
        "Path = system/bin/sh\n"
        "Folder = -\n"
        "Symbolic Link = mksh\n"
        "Mode = lrwxrwxrwx\n"
        "\n"
        "Path = etc/sysconfig/google.xml\n"
        "Folder = -\n"
        "Symbolic Link = \n"
        "Mode = -rw-r--r--\n"
    )

    assert parse_7z_listing(listing) == [
        "system/priv-app/Settings/Settings.apk",
        "etc/sysconfig/google.xml",
    ]


# --- untrusted names reaching a filesystem path -------------------------------------------
#
# One defect, three doors: a GPT partition name, a zip member name and a path out of an image
# listing are all attacker-influenced and all end up joined to a destination directory.


def test_a_gpt_partition_name_cannot_escape_the_work_directory(tmp_path):
    """`../../pwned` as a partition name carved attacker-controlled bytes outside scratch,
    then escaped a second time as the extraction destination. Reproduced end to end."""
    work = tmp_path / "work"
    escape_target = tmp_path / "pwned.img"
    disk = _synthetic_gpt(tmp_path / "disk.img", [("../../pwned", 4, 40)])
    raw = bytearray(disk.read_bytes())
    raw[4 * 512 + 1024 + 0x38 : 4 * 512 + 1024 + 0x38 + 2] = b"\x53\xef"
    disk.write_bytes(bytes(raw))

    partitions = asyncio.run(unpack_to_partitions(disk, work, settings=make_settings()))

    assert [p.name for p in partitions] == ["pwned"]
    assert not escape_target.exists()
    assert not (tmp_path / "slices").exists()
    for partition in partitions:
        assert partition.path.resolve().is_relative_to(work.resolve())


def test_a_partition_name_that_is_only_dots_is_refused(tmp_path):
    disk = _synthetic_gpt(tmp_path / "disk.img", [("..", 4, 40)])
    raw = bytearray(disk.read_bytes())
    raw[4 * 512 + 1024 + 0x38 : 4 * 512 + 1024 + 0x38 + 2] = b"\x53\xef"
    disk.write_bytes(bytes(raw))

    with pytest.raises(UnsafePathError):
        asyncio.run(unpack_to_partitions(disk, tmp_path / "work", settings=make_settings()))


def test_two_zip_members_with_one_basename_do_not_overwrite_each_other(tmp_path):
    """`a/system.img` and `b/system.img` both reduced to one path, so the second silently
    replaced the first and a whole partition's APKs disappeared."""
    archive = tmp_path / "firmware.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/system.img", magic_blob(EXT4_MAGIC_AT))
        zf.writestr("b/system.img", magic_blob(EXT4_MAGIC_AT, size=9000))

    first = _extract_zip_member(archive, "a/system.img", tmp_path / "out")
    second = _extract_zip_member(archive, "b/system.img", tmp_path / "out")

    assert first != second
    assert first.stat().st_size != second.stat().st_size  # neither clobbered the other


def test_an_ambiguous_partition_identity_is_refused_rather_than_resolved(tmp_path):
    archive = tmp_path / "firmware.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/system.img", magic_blob(EXT4_MAGIC_AT))
        zf.writestr("b/system.img", magic_blob(EXT4_MAGIC_AT))

    with pytest.raises(DuplicatePartitionError) as excinfo:
        asyncio.run(unpack_to_partitions(archive, tmp_path / "work", settings=make_settings()))

    assert "system" in str(excinfo.value)


def test_ab_slots_keep_both_files_and_settle_on_slot_a(tmp_path):
    """`lpunpack` on a retrofitted A/B super emits `system_a` and `system_b`; both used to
    normalise to `system` and the second extraction wrote over the first."""
    disk = _synthetic_gpt(tmp_path / "disk.img", [("system_a", 4, 20), ("system_b", 21, 40)])
    raw = bytearray(disk.read_bytes())
    for lba in (4, 21):
        raw[lba * 512 + 1024 + 0x38 : lba * 512 + 1024 + 0x38 + 2] = b"\x53\xef"
    disk.write_bytes(bytes(raw))

    partitions = asyncio.run(
        unpack_to_partitions(disk, tmp_path / "work", settings=make_settings())
    )

    assert [p.name for p in partitions] == ["system"]
    assert [p.slot for p in partitions] == ["system_a"]
    carved = sorted(path.name for path in (tmp_path / "work" / "slices").iterdir())
    assert carved == ["system_a.img", "system_b.img"]  # slot B kept its own file, unclobbered


def test_safe_archive_path_refuses_absolute_and_traversing_entries():
    for path in ("/etc/passwd", "../../etc/passwd", "a/../../b", "\\windows\\x"):
        with pytest.raises(UnsafePathError):
            safe_archive_path(path, context="test")
    assert safe_archive_path("system/priv-app/X/X.apk", context="test")


def test_ensure_within_refuses_an_absolute_join(tmp_path):
    """`Path("/a/b") / "/etc/passwd"` is `/etc/passwd`, and every `exists()` check downstream
    of that happily passes."""
    with pytest.raises(UnsafePathError):
        ensure_within(tmp_path, Path("/etc/passwd"), context="test")
    assert ensure_within(tmp_path, Path("a/b.apk"), context="test") == (tmp_path / "a/b.apk")


# --- emptiness is an error at both scales -------------------------------------------------


def _stub_extraction(monkeypatch, listings: dict[str, list[str]]):
    """Drive `extract_artifacts` without the toolchain: `listings` maps a partition image's
    stem to what `7z l` would have reported for it."""

    async def _list(image: Path) -> list[str]:
        return listings[image.stem]

    async def _extract(image: Path, names, dest: Path) -> None:
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

    monkeypatch.setattr(unpack_module, "_list_ext4", _list)
    monkeypatch.setattr(unpack_module, "_extract_ext4", _extract)


def _image(tmp_path: Path, name: str, fmt=ContainerFormat.EXT4) -> PartitionImage:
    path = tmp_path / f"{name}.img"
    path.write_bytes(magic_blob(EXT4_MAGIC_AT))
    return PartitionImage(name=name, path=path, fmt=fmt)


async def test_a_partition_that_yields_nothing_is_a_named_error(tmp_path, monkeypatch):
    """The guard used to be computed across ALL partitions, so on a real device four
    populated partitions and one silently empty one still returned hundreds of APKs and a job
    recorded SUCCEEDED."""
    _stub_extraction(
        monkeypatch,
        {
            "system": ["system/priv-app/A/A.apk"],
            "product": ["media/audio/x.ogg", "lib64/libc.so"],
        },
    )
    partitions = [_image(tmp_path, "system"), _image(tmp_path, "product")]

    with pytest.raises(unpack_module.NothingExtractedError) as excinfo:
        await extract_artifacts(partitions, tmp_path / "out")

    assert "'product'" in str(excinfo.value)
    assert "2 entries" in str(excinfo.value)


async def test_a_partition_that_never_carries_apps_may_be_empty(tmp_path, monkeypatch):
    """`system_other` and every `*_dlkm` genuinely hold no app — measured 429 and 333 entries
    with zero matches on oriole — so their emptiness is a rule, not silence."""
    _stub_extraction(
        monkeypatch,
        {"system": ["system/priv-app/A/A.apk"], "vendor_dlkm": ["lib/modules/x.ko"]},
    )
    partitions = [_image(tmp_path, "system"), _image(tmp_path, "vendor_dlkm")]

    artifacts = await extract_artifacts(partitions, tmp_path / "out")

    assert [artifact.image_path for artifact in artifacts] == ["system/priv-app/A/A.apk"]


async def test_an_erofs_partition_before_an_ext4_one_does_not_crash(tmp_path, monkeypatch):
    """`entries` was bound only in the ext4 branch and read for every partition. `_unpack_zip`
    iterates members in zip order, not sorted, so a build whose first .img is EROFS raised
    UnboundLocalError on the very first partition — and Android 13+ ships vendor/odm as
    EROFS."""
    _stub_extraction(monkeypatch, {"system": ["system/priv-app/A/A.apk"]})

    async def _no_erofs_artifacts(image, dest, patterns):
        return []

    monkeypatch.setattr(unpack_module, "_extract_erofs", _no_erofs_artifacts)
    partitions = [
        _image(tmp_path, "system_dlkm", fmt=ContainerFormat.EROFS),
        _image(tmp_path, "system"),
    ]

    artifacts = await extract_artifacts(partitions, tmp_path / "out")

    assert [artifact.partition for artifact in artifacts] == ["system"]


async def test_an_absolute_path_in_a_listing_never_reaches_the_extractor(tmp_path, monkeypatch):
    """`dest / "/etc/passwd"` is `/etc/passwd`, and the `is_file()` check after extraction
    then passes, so `local_path` pointed outside scratch where task 4 reads it and retention
    never deletes it."""
    handed_to_extractor: list[list[str]] = []
    _stub_extraction(monkeypatch, {"system": ["system/a/A.apk", "/etc/cron.d/pwned.apk"]})
    real_extract = unpack_module._extract_ext4

    async def _spy(image, names, dest):
        handed_to_extractor.append(list(names))
        await real_extract(image, names, dest)

    monkeypatch.setattr(unpack_module, "_extract_ext4", _spy)

    with pytest.raises(UnsafePathError):
        await extract_artifacts([_image(tmp_path, "system")], tmp_path / "out")

    assert handed_to_extractor == []  # refused before the extractor could act on it
