"""Container-format detection, GPT parsing, path normalisation and artifact selection.

Fast: every image here is a handful of synthetic bytes carrying the real magic numbers at
their real offsets. The chain itself (7z, lpunpack, simg2img) is exercised against a real
multi-GB image in `test_unpack_chain_heavy.py`.
"""

import struct
import zipfile

import pytest

from uadclaw.settings import Settings
from uadclaw.unpack import (
    ContainerFormat,
    UnpackError,
    _harvest_erofs_staging,
    canonical_device_path,
    detect_format,
    matches_artifact_patterns,
    normalize_partition_name,
    parse_7z_listing,
    read_gpt_partitions,
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
