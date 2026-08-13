"""Container-format detection, GPT parsing, path normalisation and artifact selection.

Fast: every image here is a handful of synthetic bytes carrying the real magic numbers at
their real offsets. The chain itself (7z, lpunpack, simg2img) is exercised against a real
multi-GB image in `test_unpack_chain_heavy.py`.
"""

import asyncio
import io
import shutil
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

import lz4.frame
import pytest

from uadclaw import unpack as unpack_module
from uadclaw.settings import Settings
from uadclaw.unpack import (
    ContainerFormat,
    DuplicatePartitionError,
    NothingExtractedError,
    PartitionImage,
    UnpackError,
    UnsafePathError,
    _extract_tar_partitions,
    _extract_zip_member,
    _harvest_erofs_staging,
    _sparsechunk_sets,
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
    unpopulated_slot_b,
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


def write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def write_tar(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    return write_bytes(path, tar_bytes(entries))


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
        # Samsung's two, both named to say the opposite of what they hold.
        "AP_S911USQS8FZG1.md5": (
            write_tar(tmp_path / "AP_S911USQS8FZG1.md5", [("super.img.lz4", b"\x00" * 32)]),
            ContainerFormat.TAR,
        ),
        "vendor.img": (
            write_bytes(tmp_path / "vendor.img", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
            ContainerFormat.LZ4_FRAME,
        ),
        "logical.bin": (
            write_at(tmp_path / "logical.bin", (0, b"7z\xbc\xaf\x27\x1c")),
            ContainerFormat.SEVEN_ZIP,
        ),
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


# --- Samsung's two extra steps: zip -> tar -> LZ4 frame -> sparse -> super -> partition.
# Every fixture below is named to LIE about what it holds, because the whole point is that
# nothing in the chain reads a name to decide a format. ---------------------------------------


# Spelled out rather than imported: a fixture built by iterating the set under test shrinks
# with it, so dropping a name would silently stop being tested. The equality assertion below
# is what makes a name added to the module fail here until it is pinned too.
NO_APP_PARTITIONS = (
    "userdata",
    "cache",
    "metadata",
    "persist",
    "omr",
    "misc",
    "vm-bootsys",
    "dspso",
)


def test_the_no_app_partition_set_is_exactly_what_the_fixtures_below_pin():
    assert set(NO_APP_PARTITIONS) == unpack_module._NO_APP_PARTITIONS


def tar_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def samsung_shaped_zip(path: Path, members: list[tuple[str, bytes]]) -> Path:
    """Deflated, like the real decrypted archive: not one member ends in `.img` or `.zip`, so
    the name-shaped scan claims nothing and every member has to be opened on its magic."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members:
            zf.writestr(name, payload)
    return path


def test_a_tar_and_an_lz4_frame_are_detected_from_their_magic(tmp_path):
    """`.tar.md5` is a plain tar with an md5 line appended, so its magic is `ustar` at 257 —
    257 bytes in, where no extension can be seen."""
    tar_path = tmp_path / "AP_S911USQS8FZG1_meta_OS16.tar.md5"
    tar_path.write_bytes(tar_bytes([("boot.img.lz4", b"\x00" * 32)]))
    frame = tmp_path / "super.img"
    frame.write_bytes(lz4.frame.compress(magic_blob(EXT4_MAGIC_AT)))

    assert detect_format(tar_path) is ContainerFormat.TAR
    assert detect_format(frame) is ContainerFormat.LZ4_FRAME


async def test_a_samsung_shaped_zip_of_tars_reaches_the_partition_images(tmp_path):
    """The measured layout (`SM-S911U`/`XAA`, 2026-08-12): six `.tar.md5` members, every image
    inside them `.img.lz4`, and a 1.2 GB `meta-data/fota.zip` sitting next to the partitions
    in the `AP_` tar. Recursing into that zip would unpack a whole second OTA package."""
    ext4 = magic_blob(EXT4_MAGIC_AT)
    erofs = magic_blob(EROFS_MAGIC_AT)
    archive = samsung_shaped_zip(
        tmp_path / "samsung-SM-S911U-S911USQS8FZG1_XAA.zip",
        [
            (
                "AP_S911USQS8FZG1_meta_OS16.tar.md5",
                tar_bytes(
                    [
                        ("boot.img.lz4", lz4.frame.compress(magic_blob())),
                        ("system.img.lz4", lz4.frame.compress(ext4)),
                        ("odm.img.lz4", lz4.frame.compress(erofs)),
                        ("meta-data/fota.zip", b"PK\x03\x04" + b"\x00" * 256),
                        # Every member of `_NO_APP_PARTITIONS`, each a real ext4 image, so the
                        # set is pinned as a set rather than at the one name a fixture needed.
                        *(
                            (f"{skipped}.img.lz4", lz4.frame.compress(ext4))
                            for skipped in NO_APP_PARTITIONS
                        ),
                    ]
                ),
            ),
            (
                "USERDATA_VZW_S911USQS8FZG1.tar.md5",
                tar_bytes([("userdata.img.lz4", lz4.frame.compress(ext4))]),
            ),
        ],
    )

    partitions = await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert sorted((p.name, str(p.fmt)) for p in partitions) == [
        ("odm", "erofs"),
        ("system", "ext4"),
    ]
    # `boot` is a skipped non-filesystem, every name in `NO_APP_PARTITIONS` is a real ext4
    # image with no app on it, and the nested zip is not a partition image at all.
    assert not (tmp_path / "work" / "tar" / "fota.zip").exists()


async def test_a_zip_member_that_is_an_lz4_frame_is_decoded_on_its_own(tmp_path):
    """The chain's other entry into LZ4: a member the name-shaped scan claims as an `.img`,
    whose bytes turn out to be a frame. It reaches `_unpack` directly rather than through the
    tar walk, so the dispatch table needs the format as well as the tar reader."""
    archive = tmp_path / "firmware.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("system.img", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT)))

    partitions = await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert [(p.name, str(p.fmt)) for p in partitions] == [("system", "ext4")]


def test_neither_the_lz4_suffix_nor_its_absence_decides_anything(tmp_path):
    """Both directions of the same rule. `vendor.img` wears no `.lz4` and is decoded because
    its bytes are a frame; `system.img.lz4` wears one and is taken verbatim because its bytes
    are not. The second keeps `.img` in its partition name, which is the honest outcome of
    never consulting the suffix: the name is a label, the magic is the decision.
    """
    framed = magic_blob(EXT4_MAGIC_AT)
    verbatim = magic_blob(EXT4_MAGIC_AT, size=9000)
    stream = io.BytesIO(
        tar_bytes([("vendor.img", lz4.frame.compress(framed)), ("system.img.lz4", verbatim)])
    )

    found = _extract_tar_partitions(
        stream,
        tmp_path / "out",
        root=tmp_path,
        context="test",
        max_bytes=1 << 20,
        seen={},
    )

    assert [name for name, _path in found] == ["vendor", "system.img"]
    assert [path.read_bytes() for _name, path in found] == [framed, verbatim]


async def test_one_image_shipped_in_two_tars_resolves_instead_of_colliding(tmp_path):
    """`CSC_` and `HOME_CSC_` both carry `prism` and `optics`, byte-identical on the measured
    build. Two partitions claiming one identity is normally refused, and rightly — but there
    is nothing to guess between two copies of the same bytes."""
    prism = magic_blob(EXT4_MAGIC_AT)
    csc = tar_bytes([("prism.img.lz4", lz4.frame.compress(prism))])
    archive = samsung_shaped_zip(
        tmp_path / "fw.zip",
        [("CSC_OYN_S911UOYN8FZG1.tar.md5", csc), ("HOME_CSC_OYN_S911UOYN8FZG1.tar.md5", csc)],
    )

    partitions = await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert [p.name for p in partitions] == ["prism"]


async def test_two_tars_carrying_DIFFERENT_images_under_one_name_are_still_refused(tmp_path):
    """The dedupe above must not become "the second copy is always droppable": two partitions
    that really disagree are the case `DuplicatePartitionError` exists for."""
    archive = samsung_shaped_zip(
        tmp_path / "fw.zip",
        [
            (
                "CSC_OYN.tar.md5",
                tar_bytes([("prism.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT)))]),
            ),
            (
                "HOME_CSC_OYN.tar.md5",
                tar_bytes(
                    [("prism.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT, size=9000)))]
                ),
            ),
        ],
    )

    with pytest.raises(DuplicatePartitionError):
        await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())


def test_two_entries_with_one_name_in_a_tar_do_not_overwrite_each_other(tmp_path):
    stream = io.BytesIO(
        tar_bytes(
            [
                ("a/system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
                ("b/system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT, size=9000))),
            ]
        )
    )

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert len({path for _name, path in found}) == 2
    assert all(path.is_file() for _name, path in found)


def test_two_different_partitions_holding_the_same_bytes_stay_two_partitions(tmp_path):
    """The dedupe is keyed on the partition AND its content, not content alone: two
    near-empty images of one size hashing the same is not an exotic shape, and collapsing
    them would lose a partition with nothing raised."""
    same = lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))
    stream = io.BytesIO(tar_bytes([("optics.img.lz4", same), ("prism.img.lz4", same)]))

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert [name for name, _path in found] == ["optics", "prism"]


@pytest.mark.parametrize("entry", ["../../pwned.img", "/etc/passwd.img"])
def test_a_tar_entry_name_that_would_escape_is_refused(tmp_path, entry):
    """Entry names come out of an archive downloaded off the internet, exactly like a zip
    member's or a GPT partition's, and get the same treatment."""
    stream = io.BytesIO(tar_bytes([(entry, lz4.frame.compress(magic_blob(EXT4_MAGIC_AT)))]))

    with pytest.raises(UnsafePathError):
        _extract_tar_partitions(
            stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
        )


def test_an_lz4_frame_declares_its_output_size_only_when_it_chooses_to(tmp_path):
    """The measurement `_decoded_chunks` rests on, and the correction to what it used to say.
    FLG bit 3 is an OPTIONAL content size: Samsung's own producer sets it (`0x6c` on all 31
    entries of the measured `BL_` tar) and python-lz4's default does not, so a frame may or
    may not declare its output and the ceiling is counted on the way out."""
    declared = lz4.frame.compress(b"A" * 100000, store_size=True)
    silent = lz4.frame.compress(b"A" * 100000, store_size=False)

    assert declared[4] & 0b1000  # FLG bit 3, measured 2026-08-12 as 0x4c
    assert lz4.frame.get_frame_info(declared)["content_size"] == 100000
    assert not silent[4] & 0b1000  # 0x40
    assert lz4.frame.get_frame_info(silent)["content_size"] == 0


def test_one_decompress_call_cannot_return_the_whole_frame(tmp_path):
    """4 MiB of compressed zeros returns 1016 MiB in ONE allocation without `max_length`, and
    every ceiling in this module is checked on what came back — so the ceiling never sees the
    spike. Bounding the call is what bounds the worker's heap."""
    payload = io.BytesIO(lz4.frame.compress(b"\x00" * (16 * 1024 * 1024)))

    sizes = [
        len(chunk)
        for chunk in unpack_module._decoded_chunks(
            payload,
            payload.read(unpack_module._STREAM_CHUNK_BYTES),
            framed=True,
            max_bytes=1 << 30,
            context="test",
        )
    ]

    assert sum(sizes) == 16 * 1024 * 1024
    assert max(sizes) <= unpack_module._DECODE_MAX_LENGTH


def test_a_truncated_lz4_entry_is_refused_rather_than_decoded_short(tmp_path):
    """Measured: a frame cut at 50% yields 2,490,368 of 5,000,000 bytes and raises nothing, so
    whatever chokes on the prefix downstream reports the wrong layer — and when the prefix
    still classifies, the partition is dropped in silence instead. `_copy_slice` already
    refuses a short slice for the same reason."""
    whole = lz4.frame.compress(magic_blob(EXT4_MAGIC_AT, size=5_000_000))
    stream = io.BytesIO(tar_bytes([("system.img.lz4", whole[: len(whole) // 2])]))
    out = tmp_path / "out"

    with pytest.raises(UnpackError, match="truncated"):
        _extract_tar_partitions(
            stream, out, root=tmp_path, context="test", max_bytes=1 << 30, seen={}
        )

    # And nothing half-written survives: a partial image classifies and mounts like a whole one.
    assert list(out.iterdir()) == []


def test_an_entry_holding_two_lz4_frames_is_decoded_whole(tmp_path):
    """A second frame used to be discarded as `unused_data`: measured, a two-frame member
    yielded 4000 of 8000 bytes with nothing raised."""
    tail = b"the second frame" * 8
    stream = io.BytesIO(
        tar_bytes(
            [
                (
                    "system.img.lz4",
                    lz4.frame.compress(magic_blob(EXT4_MAGIC_AT)) + lz4.frame.compress(tail),
                )
            ]
        )
    )

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert [name for name, _path in found] == ["system"]
    assert found[0][1].read_bytes() == magic_blob(EXT4_MAGIC_AT) + tail


def test_the_ceiling_is_for_the_whole_tar_and_not_for_each_entry(tmp_path):
    """Per entry, an archive of N entries writes N times the ceiling into a scratch lease
    sized for one."""
    image = magic_blob(EXT4_MAGIC_AT)
    stream = io.BytesIO(
        tar_bytes([(f"p{index}.img.lz4", lz4.frame.compress(image)) for index in range(3)])
    )

    with pytest.raises(UnpackError, match="ceiling"):
        _extract_tar_partitions(
            stream,
            tmp_path / "out",
            root=tmp_path,
            context="test",
            max_bytes=2 * len(image) + 1,
            seen={},
        )


def test_a_tar_member_bigger_than_the_ceiling_is_refused_before_it_is_walked(tmp_path):
    """Every other zip member has its declared size checked against the ceiling; a tar member
    is streamed rather than extracted and skipped that check entirely. Sized so the ceiling
    the entries themselves decode past cannot stand in for it: the member declares 10,240
    bytes and the one image inside it decodes to 1,100."""
    member = tar_bytes(
        [("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT, size=1100)))]
    )
    archive = samsung_shaped_zip(tmp_path / "fw.zip", [("AP_x.tar.md5", member)])

    with zipfile.ZipFile(archive) as zf:
        assert zf.getinfo("AP_x.tar.md5").file_size == len(member) == 10240

    with pytest.raises(UnpackError, match="declares 10240 bytes"):
        unpack_module._stream_zip_tar(
            archive,
            "AP_x.tar.md5",
            tmp_path / "out",
            root=tmp_path,
            max_bytes=1200,
            seen={},
        )


def test_an_entry_named_as_a_partition_that_is_not_one_stops_the_job(tmp_path):
    """The silent half of the measured loss: `product.img.lz4` compressed with anything but an
    LZ4 frame (zstd here) was dropped with a `logger.info` and the job recorded SUCCEEDED with
    two partitions' APKs out of four. The name says what the entry should hold and the magic
    says what it is; a disagreement is refused, exactly as it is for a sparsechunk member."""
    stream = io.BytesIO(
        tar_bytes(
            [
                ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
                ("product.img.lz4", b"\x28\xb5\x2f\xfd" + b"\x00" * 4096),
            ]
        )
    )

    with pytest.raises(UnpackError) as excinfo:
        _extract_tar_partitions(
            stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
        )

    assert "product" in str(excinfo.value)


def test_an_entry_named_as_a_partition_that_decodes_to_junk_stops_the_job(tmp_path):
    """The other half: a corrupt `product.img.lz4` that still decodes returns bytes matching no
    magic, and the second gate dropped it as quietly as the first."""
    stream = io.BytesIO(
        tar_bytes(
            [
                ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
                ("product.img.lz4", lz4.frame.compress(b"\x00" * 8192)),
            ]
        )
    )

    with pytest.raises(UnpackError) as excinfo:
        _extract_tar_partitions(
            stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
        )

    assert "product" in str(excinfo.value)


def test_an_entry_claiming_a_skipped_partition_is_still_only_skipped(tmp_path):
    """Measured across all 65 file entries of the six `SM-S911U` tars: 19 claim an image and 7
    of those (`vbmeta`, `vbmeta_system`, `boot`, `init_boot`, `vendor_boot`, `dtbo`,
    `recovery`) decode to no partition format at all. Every one is on the skip list, and the
    claim rule must lose to it or the real archive fails at its first entry."""
    stream = io.BytesIO(
        tar_bytes(
            [
                ("boot.img.lz4", lz4.frame.compress(magic_blob())),
                ("vbmeta_system.img.lz4", b"\x00" * 4096),
                ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
            ]
        )
    )

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert [name for name, _path in found] == ["system"]


def test_an_entry_whose_name_claims_nothing_is_dropped_and_never_named(tmp_path):
    """The early raw-format gate earns its keep here rather than only on `fota.zip`: `...` is a
    legal tar entry name that `safe_component` refuses outright, so without the gate deciding
    first, one junk entry fails the whole job instead of being ignored."""
    stream = io.BytesIO(
        tar_bytes(
            [
                ("...", b"\x00" * 4096),
                ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
            ]
        )
    )

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert [name for name, _path in found] == ["system"]


def test_an_lz4_entry_that_decodes_past_the_ceiling_is_a_named_error(tmp_path):
    """An LZ4 frame's content size is optional and nothing here reads it, so the only bound on
    what a hostile entry expands to is what is counted on the way out."""
    stream = io.BytesIO(
        tar_bytes([("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT, size=1 << 20)))])
    )

    with pytest.raises(UnpackError, match="ceiling"):
        _extract_tar_partitions(
            stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=4096, seen={}
        )


async def test_a_tar_holding_no_partition_image_is_a_named_error(tmp_path):
    """`NON-HLOS.bin` is on no skip list, so this reaches the gate that classifies what an
    entry DECODED to rather than the one that reads its name."""
    path = tmp_path / "CP_S911USQS8FZG1.tar.md5"
    path.write_bytes(tar_bytes([("NON-HLOS.bin.lz4", lz4.frame.compress(magic_blob()))]))

    with pytest.raises(NothingExtractedError):
        await unpack_to_partitions(path, tmp_path / "work", settings=make_settings())


def test_an_entry_that_decodes_to_an_archive_is_not_a_partition_image(tmp_path):
    """The measured trap: the `AP_` tar carries a 1.2 GB `meta-data/fota.zip` beside the
    partitions. Its raw bytes give it away here, but an OEM compressing the same thing would
    pass the first gate, so what an entry decodes TO is checked as well as what it arrives as.
    """
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as zf:
        zf.writestr("payload.bin", b"CrAU" + b"\x00" * 64)
    stream = io.BytesIO(
        tar_bytes(
            [
                ("meta-data/fota.zip.lz4", lz4.frame.compress(nested.getvalue())),
                ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
            ]
        )
    )

    found = _extract_tar_partitions(
        stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
    )

    assert [name for name, _path in found] == ["system"]


needs_7z = pytest.mark.skipif(shutil.which("7z") is None, reason="needs 7-Zip >= 24 on PATH")
needs_simg = pytest.mark.skipif(
    shutil.which("img2simg") is None or shutil.which("simg2img") is None,
    reason="needs img2simg/simg2img from android-tools",
)


@needs_7z
async def test_a_7z_archive_of_partition_images_unpacks(tmp_path):
    """Nothing's shape once the driver has joined its volume set: one 7z whose members are
    partition images, plus boot blobs that are on the skip list."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "system.img").write_bytes(magic_blob(EXT4_MAGIC_AT))
    (source / "product.img").write_bytes(magic_blob(EROFS_MAGIC_AT))
    (source / "vbmeta.img").write_bytes(magic_blob())
    archive = tmp_path / "image-logical.7z"
    subprocess.run(  # noqa: S603
        ["7z", "a", "-t7z", "-bso0", "-bsp0", str(archive), str(source / "*")],
        check=True,
        capture_output=True,
    )

    assert detect_format(archive) is ContainerFormat.SEVEN_ZIP
    partitions = await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert sorted((p.name, str(p.fmt)) for p in partitions) == [
        ("product", "erofs"),
        ("system", "ext4"),
    ]


def test_sparsechunk_members_group_and_order_by_their_suffix():
    """Motorola splits its super across up to 14 chunks and the ORDER lives nowhere but the
    name, so `.10` must sort after `.9` rather than beside `.1`."""
    members = [
        "super.img_sparsechunk.10",
        "super.img_sparsechunk.2",
        "super.img_sparsechunk.0",
        "boot.img",
        "other.img_sparsechunk.0",
    ]

    sets = _sparsechunk_sets(members)

    assert sorted(sets) == ["other.img", "super.img"]
    assert sorted(sets["super.img"]) == [
        (0, "super.img_sparsechunk.0"),
        (2, "super.img_sparsechunk.2"),
        (10, "super.img_sparsechunk.10"),
    ]


async def test_a_sparsechunk_set_with_a_gap_is_refused_rather_than_rebuilt_short(tmp_path):
    """A missing chunk produces a shorter image that still mounts and still yields APKs,
    which is the quietest way to lose a partition's packages."""
    archive = tmp_path / "moto.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("super.img_sparsechunk.0", magic_blob(SPARSE_MAGIC_AT))
        zf.writestr("super.img_sparsechunk.2", magic_blob(SPARSE_MAGIC_AT))

    with pytest.raises(UnpackError) as excinfo:
        await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert "[0, 2]" in str(excinfo.value)


async def test_a_sparsechunk_member_whose_bytes_are_not_sparse_is_refused(tmp_path):
    """The name orders the set; the magic decides what it is. A file that merely wears the
    name must not get itself concatenated into a partition."""
    archive = tmp_path / "moto.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("super.img_sparsechunk.0", magic_blob(SPARSE_MAGIC_AT))
        zf.writestr("super.img_sparsechunk.1", magic_blob(EXT4_MAGIC_AT))

    with pytest.raises(UnpackError) as excinfo:
        await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert "super.img_sparsechunk.1" in str(excinfo.value)
    assert "ext4" in str(excinfo.value)


def sparse_bomb(*, total_blocks: int, block_size: int = 4096) -> bytes:
    """44 bytes declaring `total_blocks * block_size` of output.

    One `CHUNK_TYPE_FILL` chunk covers every declared block in four bytes of payload, which is
    what makes a sparse header's declared size unbounded by the file's own. Measured
    2026-08-12: 44 bytes declaring 1,073,741,824 made `simg2img` write fully allocated blocks
    (`st_blocks * 512` at the apparent size — not one sparse hole) until a `ulimit -f` cap
    killed it with SIGXFSZ, which is the only thing that stopped it.
    """
    header = struct.pack("<IHHHHIIII", 0xED26FF3A, 1, 0, 28, 12, block_size, total_blocks, 1, 0)
    return header + struct.pack("<HHII", 0xCAC2, 0, total_blocks, 16) + b"\xff\xff\xff\xff"


def _refuse_to_run_tools(monkeypatch):
    async def _never(argv, **kwargs):
        raise AssertionError(f"a tool ran that should have been refused first: {argv}")

    monkeypatch.setattr(unpack_module, "_run_tool", _never)


async def test_a_sparse_image_declaring_past_the_ceiling_never_reaches_simg2img(
    tmp_path, monkeypatch
):
    """`simg2img` has no size flag — its whole usage is `simg2img <sparse> <raw>` — so the
    declared output is refused from the header or it is not refused at all."""
    _refuse_to_run_tools(monkeypatch)
    image = tmp_path / "system.img"
    image.write_bytes(sparse_bomb(total_blocks=9_000_000))  # 36.86 GB declared out of 44 bytes

    with pytest.raises(UnpackError, match="ceiling"):
        await unpack_to_partitions(image, tmp_path / "work", settings=make_settings())

    assert not (tmp_path / "work" / "raw").exists()


async def test_a_sparsechunk_set_declaring_past_the_ceiling_never_reaches_simg2img(
    tmp_path, monkeypatch
):
    """The same tool, the same absence of a ceiling, reached through Motorola's chunk set."""
    _refuse_to_run_tools(monkeypatch)
    archive = tmp_path / "moto.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for index in range(2):
            zf.writestr(f"super.img_sparsechunk.{index}", sparse_bomb(total_blocks=9_000_000))

    with pytest.raises(UnpackError, match="ceiling"):
        await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())


@needs_simg
async def test_a_partition_image_that_unpacks_to_nothing_readable_is_not_dropped(tmp_path):
    """A tar entry reaches `_unpack_images` only after its own decoded bytes classified as a
    partition format, so one that then reads as no container at all is a partition LOST, not a
    blob it was right to ignore. Dropping it left a job SUCCEEDED with `system` alone."""
    block = 4096
    archive = samsung_shaped_zip(
        tmp_path / "fw.zip",
        [
            (
                "AP_x.tar.md5",
                tar_bytes(
                    [
                        ("system.img.lz4", lz4.frame.compress(magic_blob(EXT4_MAGIC_AT))),
                        (
                            "product.img.lz4",
                            lz4.frame.compress(
                                sparse_chunk(
                                    total_blocks=1,
                                    block_size=block,
                                    skip_blocks=0,
                                    data=bytes(block),
                                )
                            ),
                        ),
                    ]
                ),
            )
        ],
    )

    with pytest.raises(unpack_module.UnsupportedContainerError):
        await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())


def sparse_chunk(*, total_blocks: int, block_size: int, skip_blocks: int, data: bytes) -> bytes:
    """One member of a real sparsechunk set, in the format measured off Motorola's own.

    Every chunk of a set declares the WHOLE image's `total_blks` (2,426,880 on `rtwo`) and
    opens with a DONT_CARE chunk skipping to its own offset, so `simg2img c0 … cN out` writes
    each one at its absolute position rather than appending. Building this by hand rather
    than with `img2simg`: img2simg emits an image whose blocks start at zero, so two of them
    both claim block 0 and the second silently overwrites the first — which is a property of
    that shortcut, not of a real chunk set.
    """
    blocks = len(data) // block_size
    trailing = total_blocks - skip_blocks - blocks
    chunks = [struct.pack("<HHII", 0xCAC1, 0, blocks, 12 + len(data)) + data]
    if skip_blocks:
        chunks.insert(0, struct.pack("<HHII", 0xCAC3, 0, skip_blocks, 12))
    if trailing:
        # simg2img refuses a file whose chunks do not account for every declared block, so a
        # chunk that carries the middle of an image is bracketed by DONT_CAREs on both sides.
        chunks.append(struct.pack("<HHII", 0xCAC3, 0, trailing, 12))
    header = struct.pack(
        "<IHHHHIIII", 0xED26FF3A, 1, 0, 28, 12, block_size, total_blocks, len(chunks), 0
    )
    return header + b"".join(chunks)


@needs_simg
async def test_a_real_sparsechunk_set_rebuilds_one_image(tmp_path):
    """Two real Android sparse chunks, joined by simg2img in one call, back to the ext4 image
    they were carved from. Blobs carrying only the sparse magic would prove the dispatch and
    nothing about the rebuild."""
    block = 4096
    raw = magic_blob(EXT4_MAGIC_AT, (block + 16, b"second-half"), size=2 * block)
    archive = tmp_path / "moto.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for index, half in enumerate((raw[:block], raw[block:])):
            zf.writestr(
                f"super.img_sparsechunk.{index}",
                sparse_chunk(total_blocks=2, block_size=block, skip_blocks=index, data=half),
            )

    partitions = await unpack_to_partitions(archive, tmp_path / "work", settings=make_settings())

    assert [(p.name, str(p.fmt)) for p in partitions] == [("super", "ext4")]
    assert partitions[0].path.read_bytes() == raw


def test_an_unwritten_slot_b_is_droppable_only_when_slot_a_is_there():
    """Measured on Motorola `rtwo`: lpunpack writes every partition the super declares, and 5
    of the 6 `_b` images are ZERO bytes. Dropping any unreadable partition instead would be
    how a whole partition's packages go missing with nothing raised."""
    both = ["product_a", "product_b", "system_a", "system_b"]

    assert unpopulated_slot_b("product_b", both) is True
    assert unpopulated_slot_b("product_a", both) is False
    assert unpopulated_slot_b("product_b", ["product_b", "system_a"]) is False
    assert unpopulated_slot_b("odm", both) is False


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

    moved, listed = _harvest_erofs_staging(staging, dest, ["*.apk", "*/etc/sysconfig/*"])

    assert sorted(moved) == ["etc/sysconfig/google.xml", "priv-app/Settings/Settings.apk"]
    assert (dest / "priv-app/Settings/Settings.apk").is_file()
    assert not (dest / "lib64/libc.so").exists()
    # The other half of the answer: four files came out of the image and two were wanted, which
    # is a partition with no app in it rather than an extraction that produced nothing.
    assert listed == 4


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


@pytest.mark.parametrize("path", [".", "./"])
def test_a_name_that_is_only_a_dot_is_bad_input_and_not_a_crash(path):
    """`PurePosixPath(".").parts` is empty, so reading `parts[0]` raised IndexError — a
    genuine-bug type for what is attacker-controlled input off a tar's header block."""
    with pytest.raises(UnsafePathError):
        safe_archive_path(path, context="test")


def test_a_tar_entry_named_only_a_dot_does_not_crash_the_worker(tmp_path):
    """Reached the moment a tar is walked: entry names go into `safe_archive_path` raw, and a
    regular file named `.` is a legal thing to put in a tar."""
    stream = io.BytesIO(
        tar_bytes([(".", b"\x00" * 4096), ("system.img.lz4", lz4.frame.compress(magic_blob()))])
    )

    with pytest.raises(UnsafePathError):
        _extract_tar_partitions(
            stream, tmp_path / "out", root=tmp_path, context="test", max_bytes=1 << 20, seen={}
        )


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


# Spelled out rather than imported, for the same reason `NO_APP_PARTITIONS` above is: a
# fixture derived from the set under test shrinks with it.
ALLOWED_EMPTY = (
    "system_other",
    "cache",
    "metadata",
    "userdata",
    "prism",
    "optics",
    "odm",
    "dsp",
    "my_bigball",
    "my_carrier",
    "my_engineering",
    "my_heytap",
    "my_manifest",
)


def test_the_allowed_empty_set_is_exactly_what_the_test_below_pins():
    assert set(ALLOWED_EMPTY) == unpack_module.PARTITIONS_ALLOWED_EMPTY


@pytest.mark.parametrize("partition", ALLOWED_EMPTY)
async def test_an_allowed_empty_partition_whose_image_yields_nothing_still_raises(
    tmp_path, monkeypatch, partition
):
    """The two meanings of empty, kept apart. Naming a partition here says its image may match
    no PATTERN; it never says the image may hold no FILE, and no vendor ships an empty
    filesystem — Samsung's `odm` stub still holds 10. Collapsed into one rule, adding `odm`
    for Samsung took Nothing's `odm` (4 APKs on FroggerPro) out of the guard entirely."""
    _stub_extraction(monkeypatch, {"system": ["system/priv-app/A/A.apk"], partition: []})
    partitions = [_image(tmp_path, "system"), _image(tmp_path, partition)]

    with pytest.raises(unpack_module.NothingExtractedError) as excinfo:
        await extract_artifacts(partitions, tmp_path / "out")

    assert f"{partition!r}" in str(excinfo.value)
    assert "no file at all" in str(excinfo.value)


async def test_an_erofs_partition_that_yields_no_file_is_told_apart_from_one_that_matches_none(
    tmp_path, monkeypatch
):
    """The EROFS path could not tell the two apart at all — it reported the wanted paths and
    nothing else — so `fsck.erofs` producing an empty tree read exactly like a partition with
    no app on it. Both halves pinned in one test, because it is the difference that matters."""
    _stub_extraction(monkeypatch, {"system": ["system/priv-app/A/A.apk"]})
    yielded: list[int] = []

    async def _erofs(image, dest, patterns):
        return [], yielded.pop()

    monkeypatch.setattr(unpack_module, "_extract_erofs", _erofs)
    partitions = [_image(tmp_path, "odm", fmt=ContainerFormat.EROFS), _image(tmp_path, "system")]

    yielded.append(10)  # Samsung's 10-file `odm` stub: files out, no artifact among them
    assert [a.partition for a in await extract_artifacts(partitions, tmp_path / "out")] == [
        "system"
    ]

    yielded.append(0)  # the same partition when the extractor produced nothing at all
    with pytest.raises(unpack_module.NothingExtractedError, match="no file at all"):
        await extract_artifacts(partitions, tmp_path / "out2")


@pytest.mark.parametrize("partition", ALLOWED_EMPTY)
async def test_every_allowed_empty_partition_really_is_allowed_to_be_empty(
    tmp_path, monkeypatch, partition
):
    """`odm` is the entry that costs something: it holds 4 APKs on Nothing's FroggerPro and is
    a 10-file stub on Samsung's SM-S911U, so listing it buys Samsung a job that finishes and
    gives up the guard on Nothing's odm matching nothing. Pinned here so the trade is visible
    rather than reachable only through a multi-GB heavy run."""
    _stub_extraction(
        monkeypatch,
        {"system": ["system/priv-app/A/A.apk"], partition: ["etc/build.prop", "etc/passwd"]},
    )
    partitions = [_image(tmp_path, "system"), _image(tmp_path, partition)]

    artifacts = await extract_artifacts(partitions, tmp_path / "out")

    assert [artifact.image_path for artifact in artifacts] == ["system/priv-app/A/A.apk"]


async def test_an_erofs_partition_before_an_ext4_one_does_not_crash(tmp_path, monkeypatch):
    """`entries` was bound only in the ext4 branch and read for every partition. `_unpack_zip`
    iterates members in zip order, not sorted, so a build whose first .img is EROFS raised
    UnboundLocalError on the very first partition — and Android 13+ ships vendor/odm as
    EROFS."""
    _stub_extraction(monkeypatch, {"system": ["system/priv-app/A/A.apk"]})

    async def _no_erofs_artifacts(image, dest, patterns):
        return [], 105  # 105 kernel modules out, none of them an artifact

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
