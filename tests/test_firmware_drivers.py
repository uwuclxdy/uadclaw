"""The Xiaomi, Nothing, Motorola and Samsung drivers: index parsing, ordering, terms and fetch.

No network. Each driver reads a trimmed slice of its real index out of `tests/fixtures/`:

- `xiaomi_latest_index.yml` — 9 entry blocks copied byte for byte out of the tracker's 3,550,
  chosen to carry every shape the parser has to survive (both `method` values, a plain-http
  link, a null md5, a device whose last row is the older build, a link already on the CDN
  host the driver rewrites to).
- `nothing_releases.json` — 5 whole release objects out of the API's 229, with only the
  `author`, `uploader` and `body` keys dropped: a 3-volume set, a release publishing both a
  volume set and a whole-image archive, one whose assets are named after something other than
  its tag, one with no image archive at all, and one release-tooling tag with no codename.
- `motorola_h5ai_listings.json` — the h5ai API's real responses for the 7 paths one `rtwo`
  crawl touches, ancestor chains included, because filtering those out is load-bearing.
- `samsung_version_index.xml`, `samsung_binary_inform.xml`, `samsung_access_denied.xml` — the
  three real documents `SM-S911U`/`XAA` returned on 2026-08-11, byte for byte. The inform
  response is what the decryption key is derived from, so the key this suite asserts is the
  one measured against the live server rather than one recomputed from the same code.

Every HTTP call goes through an `httpx.MockTransport` that records what was sent.
"""

import hashlib
import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uadclaw.drivers.motorola import MotorolaDriver, parse_firmware_filename
from uadclaw.drivers.nothing import NothingDriver, parse_releases
from uadclaw.drivers.samsung import (
    AES_BLOCK_BYTES,
    DOWNLOAD_URL,
    MAX_INDEX_PROBES,
    SamsungDriver,
    SamsungProtocolError,
    auth_signature,
    binary_init_body,
    decrypt_archive,
    logic_check,
    normalize_version,
    parse_binary_inform,
    parse_version_index,
)
from uadclaw.drivers.xiaomi import (
    WORKING_CDN_HOST,
    XiaomiDriver,
    cdn_url,
    parse_latest_index,
)
from uadclaw.firmware import (
    EmptyFirmwareIndexError,
    FirmwareDownloadError,
    FirmwareDriverDisabledError,
    FirmwareError,
    FirmwareInputError,
    FirmwareJobParams,
    FirmwareRef,
    TermsRisk,
    driver_names,
    enabled_driver_names,
    get_driver,
    select_ref,
)
from uadclaw.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def make_settings(**overrides) -> Settings:
    base = {
        "postgres_password": "test-only-password",
        "auth_password": "test-only-admin-password",
        "session_secret": "test-only-session-secret",
    }
    return Settings(**(base | overrides))


def xiaomi_index() -> str:
    return (FIXTURES / "xiaomi_latest_index.yml").read_text(encoding="utf-8")


def nothing_index() -> str:
    return (FIXTURES / "nothing_releases.json").read_text(encoding="utf-8")


def motorola_listings() -> dict[str, dict]:
    return json.loads((FIXTURES / "motorola_h5ai_listings.json").read_text(encoding="utf-8"))


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- the registry carries all four drivers, and each disables on its own -------------------


def test_every_driver_is_registered_and_resolvable():
    assert driver_names() == ("motorola", "nothing", "pixel", "samsung", "xiaomi")
    for name in driver_names():
        assert get_driver(name, make_settings()).name == name


@pytest.mark.parametrize("disabled", ["xiaomi", "nothing", "motorola", "samsung"])
def test_disabling_one_driver_leaves_the_others_resolvable(disabled):
    """Task 11's whole point: a source that breaks is switched off without touching a stage."""
    settings = make_settings(disabled_firmware_drivers=disabled)

    with pytest.raises(FirmwareDriverDisabledError) as excinfo:
        get_driver(disabled, settings)

    assert disabled in str(excinfo.value)
    assert enabled_driver_names(settings) == tuple(
        name for name in driver_names() if name != disabled
    )
    for name in enabled_driver_names(settings):
        assert get_driver(name, settings).name == name


def test_all_four_new_drivers_can_be_disabled_at_once():
    settings = make_settings(disabled_firmware_drivers="xiaomi, nothing ,motorola,samsung")

    assert enabled_driver_names(settings) == ("pixel",)


# --- Xiaomi ---------------------------------------------------------------------------------


def test_xiaomi_index_parses_recovery_builds_only():
    """The 1,522 Fastboot rows are `.tgz`, which matches no magic in the dispatch table, so
    offering one would hand the acquire stage a build that always fails at unpack."""
    refs = parse_latest_index(xiaomi_index(), source_url="fixture")

    assert len(refs) == 7
    assert [ref.build for ref in refs if ref.device == "water_global"] == [
        "V14.0.24.0.TGOMIXM",
        "V14.0.44.0.TGOMIXM",
    ]
    assert not [ref for ref in refs if ref.url.endswith(".tgz")]


def test_xiaomi_index_drops_the_plain_http_rows():
    """164 of the real index's 2,028 Recovery rows publish an `http://` link. `FirmwareRef`
    refuses one, and the alternative to dropping them is downloading firmware in the clear."""
    refs = parse_latest_index(xiaomi_index(), source_url="fixture")

    assert "HM2013023" not in {ref.device for ref in refs}
    assert all(ref.url.startswith("https://") for ref in refs)


def test_xiaomi_carries_the_md5_the_index_published_not_a_sha256():
    refs = parse_latest_index(xiaomi_index(), source_url="fixture")
    water = next(ref for ref in refs if ref.build == "V14.0.24.0.TGOMIXM")

    assert water.sha256 is None
    assert water.md5 == "e2f5f8046340876d29d99683d1600f26"
    assert water.published_digest() == ("md5", "e2f5f8046340876d29d99683d1600f26")
    assert water.marketing_name == "Redmi A2 / A2+ / POCO C51 Global"
    assert water.android_version == "13.0"


def test_xiaomi_rewrites_the_broken_cdn_host_and_leaves_others_alone():
    """`bigota` is fronted by a CloudFront distribution that answers HEAD 200 with a correct
    Content-Length while a plain GET intermittently fails; only `cdnorg` served reliably."""
    refs = parse_latest_index(xiaomi_index(), source_url="fixture")
    hosts = {ref.url.split("/")[2] for ref in refs}

    assert "bigota.d.miui.com" not in hosts
    assert "ultimateota.d.miui.com" not in hosts
    assert hosts == {
        WORKING_CDN_HOST,
        "bkt-sgp-miui-ota-update-alisgp.oss-ap-southeast-1.aliyuncs.com",
    }
    assert (
        cdn_url("https://ultimateota.d.miui.com/a/b.zip") == f"https://{WORKING_CDN_HOST}/a/b.zip"
    )
    assert cdn_url("https://example.invalid/a/b.zip") == "https://example.invalid/a/b.zip"


def test_xiaomi_newest_build_comes_from_the_index_date_not_the_row_order():
    """No Xiaomi version string carries a date, so `select_ref` falls back to row order — and
    175 of the real index's devices list their OLDER build last."""
    refs = parse_latest_index(xiaomi_index(), source_url="fixture")

    assert select_ref(refs, device="air_global").build == "OS2.0.208.0.VGQMIXM"
    assert select_ref(refs, device="alioth").build == "OS1.0.10.0.TKHCNXM"


def test_xiaomi_index_with_no_usable_row_is_an_error_not_an_empty_catalogue():
    fastboot_only = (
        "- codename: water_global\n"
        "  version: V14.0.44.0.TGOMIXM\n"
        "  link: https://bigota.d.miui.com/a/b.tgz\n"
        "  md5: null\n"
        "  method: Fastboot\n"
    )

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        parse_latest_index(fastboot_only, source_url="https://example.invalid/latest.yml")

    assert "https://example.invalid/latest.yml" in str(excinfo.value)


def test_xiaomi_index_that_is_not_a_list_of_builds_is_refused():
    with pytest.raises(FirmwareError):
        parse_latest_index("404: Not Found\n", source_url="fixture")
    with pytest.raises(FirmwareError):
        parse_latest_index("- a\n  b: [", source_url="fixture")


def test_xiaomi_terms_do_not_claim_a_grant_nobody_gave():
    posture = XiaomiDriver(make_settings()).terms()

    assert posture.risk is TermsRisk.PUBLIC
    assert posture.acknowledged is True
    assert "tolerated rather than authorised" in posture.summary


async def test_xiaomi_fetch_verifies_the_published_md5(tmp_path):
    body = b"xiaomi-recovery-rom" * 64
    driver = XiaomiDriver(
        make_settings(), client=mock_client(lambda _r: httpx.Response(200, content=body))
    )
    import hashlib

    ref = FirmwareRef(
        driver="xiaomi",
        device="water_global",
        build="V14.0.24.0.TGOMIXM",
        url="https://cdnorg.d.miui.com/a/b.zip",
        md5=hashlib.md5(body).hexdigest(),  # noqa: S324 - the digest Xiaomi's index publishes
    )

    archive = await driver.fetch(ref, tmp_path)

    assert archive.integrity_verified is True
    assert archive.sha256 == hashlib.sha256(body).hexdigest()
    assert archive.path.read_bytes() == body


async def test_xiaomi_fetch_refuses_a_body_whose_md5_is_wrong(tmp_path):
    driver = XiaomiDriver(
        make_settings(), client=mock_client(lambda _r: httpx.Response(200, content=b"truncated"))
    )
    ref = FirmwareRef(
        driver="xiaomi",
        device="water_global",
        build="V1",
        url="https://cdnorg.d.miui.com/a/b.zip",
        md5="0" * 32,
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await driver.fetch(ref, tmp_path)

    assert "md5" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_job_params_carry_the_md5_so_a_pinned_xiaomi_build_is_still_verified():
    """Without `md5` here, pinning a build outright downgrades it to an unverified download
    of an archive whose digest the index publishes."""
    params = FirmwareJobParams(
        driver="xiaomi",
        device="water_global",
        build="V14.0.24.0.TGOMIXM",
        url="https://cdnorg.d.miui.com/a/b.zip",
        md5="e2f5f8046340876d29d99683d1600f26",
    )

    ref = params.as_ref()

    assert ref is not None
    assert ref.published_digest() == ("md5", "e2f5f8046340876d29d99683d1600f26")


async def test_xiaomi_fetch_rewrites_a_bigota_url_that_arrived_from_job_params(tmp_path):
    """A ref can be created straight from a job's params carrying whatever URL an operator
    pasted out of the tracker, which never went through the index parser."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"rom")

    driver = XiaomiDriver(make_settings(), client=mock_client(handler))
    ref = FirmwareRef(
        driver="xiaomi",
        device="water_global",
        build="V1",
        url="https://bigota.d.miui.com/V1/miui_WATER.zip",
    )

    await driver.fetch(ref, tmp_path)

    assert [str(request.url) for request in seen] == [
        f"https://{WORKING_CDN_HOST}/V1/miui_WATER.zip"
    ]


# --- Nothing ---------------------------------------------------------------------------------


def test_nothing_releases_parse_into_device_and_build():
    refs = parse_releases(nothing_index(), source_url="fixture")

    assert len(refs) == 3
    assert {ref.device for ref in refs} == {"FroggerPro", "Spacewar"}
    frogger = next(ref for ref in refs if ref.device == "FroggerPro")
    assert frogger.build == "B4.1-260723-1820"
    assert frogger.url.endswith("FroggerPro_B4.1-260723-1820-image-logical.7z.001")
    assert frogger.archive_suffix == ".7z"
    assert frogger.archive_filename == "nothing-FroggerPro-B4.1-260723-1820.7z"


def test_nothing_skips_a_release_with_no_image_archive_and_a_tag_with_no_codename():
    """1 of 229 releases publishes only hash files; 6 carry release-tooling tags like
    `0.0.0-dev+spacewar.230719`, which name no device at all."""
    refs = parse_releases(nothing_index(), source_url="fixture")

    assert "S1.0-220705-2027-GLO" not in {ref.build for ref in refs}
    assert all("dev+" not in ref.build for ref in refs)


def test_nothing_release_serving_both_archive_shapes_takes_the_logical_set():
    """7 of 229 releases publish `-image.7z` ALONGSIDE the volume set rather than instead of
    it; taking both pairs two archives that each claim to be volume 1."""
    refs = parse_releases(nothing_index(), source_url="fixture")
    both = next(ref for ref in refs if ref.build == "U2.5-240119-1910")

    assert both.url.endswith("-image-logical.7z.001")


def test_nothing_asset_names_need_not_match_the_release_tag():
    """10 of 229 releases name their assets after something other than the tag; requiring the
    two to match dropped all ten and read as "this build has no archive"."""
    refs = parse_releases(nothing_index(), source_url="fixture")
    renamed = next(ref for ref in refs if ref.build == "T1.5-230619-0042")

    assert renamed.url.endswith("Spacewar-T1.5-230619-0042_1.5.5-image-logical.7z.001")


def test_nothing_releases_come_back_oldest_first():
    """GitHub answers newest first and `select_ref` resolves "newest" as the last row for a
    device, since a Nothing build id carries no Android date field."""
    refs = parse_releases(nothing_index(), source_url="fixture")

    assert [ref.build for ref in refs if ref.device == "Spacewar"] == [
        "T1.5-230619-0042",
        "U2.5-240119-1910",
    ]
    assert select_ref(refs, device="Spacewar").build == "U2.5-240119-1910"


def test_nothing_rate_limit_body_is_an_error_not_an_empty_catalogue():
    """GitHub reports an exhausted unauthenticated quota as a JSON object, not a list."""
    with pytest.raises(FirmwareError) as excinfo:
        parse_releases('{"message": "API rate limit exceeded"}', source_url="fixture")

    assert "rate limit" in str(excinfo.value)


def test_nothing_empty_release_list_is_an_error():
    with pytest.raises(EmptyFirmwareIndexError):
        parse_releases("[]", source_url="fixture")


def test_nothing_terms_say_it_is_a_re_upload_rather_than_the_oem():
    posture = NothingDriver(make_settings()).terms()

    assert posture.risk is TermsRisk.RESTRICTED
    assert "re-upload" in posture.summary


async def test_nothing_fetch_joins_the_volume_set_into_one_archive(tmp_path):
    """A split 7z is a raw split of one archive, so joining reproduces it byte for byte —
    and leaves the unpack stage one file to delete instead of three it was never told about.
    """
    import hashlib

    volumes = {"001": b"7z\xbc\xaf\x27\x1c" + b"a" * 512, "002": b"b" * 512, "003": b"c" * 300}
    release = json.loads(nothing_index())
    frogger = next(r for r in release if r["tag_name"] == "FroggerPro_B4.1-260723-1820")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/tags/FroggerPro_B4.1-260723-1820"):
            return httpx.Response(200, text=json.dumps(frogger))
        suffix = str(request.url)[-3:]
        return httpx.Response(200, content=volumes[suffix])

    driver = NothingDriver(make_settings(), client=mock_client(handler))
    ref = next(
        r for r in parse_releases(nothing_index(), source_url="f") if r.device == "FroggerPro"
    )

    archive = await driver.fetch(ref, tmp_path)

    joined = volumes["001"] + volumes["002"] + volumes["003"]
    assert archive.path.read_bytes() == joined
    assert archive.sha256 == hashlib.sha256(joined).hexdigest()
    # The published hashes cover the images INSIDE the archive, not the download.
    assert archive.integrity_verified is False
    assert [path.name for path in tmp_path.iterdir()] == ["nothing-FroggerPro-B4.1-260723-1820.7z"]
    assert len(seen) == 4


async def test_nothing_fetch_refuses_a_ref_the_release_does_not_serve(tmp_path):
    """Guards the job-params path: an operator-supplied URL that is not this tag's first
    volume would otherwise have its siblings taken from a release it does not belong to."""
    release = json.loads(nothing_index())
    frogger = next(r for r in release if r["tag_name"] == "FroggerPro_B4.1-260723-1820")
    driver = NothingDriver(
        make_settings(),
        client=mock_client(lambda _r: httpx.Response(200, text=json.dumps(frogger))),
    )
    ref = FirmwareRef(
        driver="nothing",
        device="FroggerPro",
        build="B4.1-260723-1820",
        url="https://example.invalid/elsewhere-image-logical.7z.001",
        archive_suffix=".7z",
    )

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.fetch(ref, tmp_path)

    assert "example.invalid" in str(excinfo.value)


async def test_nothing_a_failed_volume_leaves_no_partial_archive_behind(tmp_path):
    """The set is gigabytes across two directories, and a failure halfway through is the
    shape retention exists to prevent."""
    release = json.loads(nothing_index())
    frogger = next(r for r in release if r["tag_name"] == "FroggerPro_B4.1-260723-1820")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tags/FroggerPro_B4.1-260723-1820"):
            return httpx.Response(200, text=json.dumps(frogger))
        if str(request.url).endswith("002"):
            return httpx.Response(503, content=b"gone")
        return httpx.Response(200, content=b"7z\xbc\xaf\x27\x1c" + b"a" * 64)

    driver = NothingDriver(make_settings(), client=mock_client(handler))
    ref = next(
        r for r in parse_releases(nothing_index(), source_url="f") if r.device == "FroggerPro"
    )

    with pytest.raises(FirmwareDownloadError):
        await driver.fetch(ref, tmp_path)

    assert list(tmp_path.iterdir()) == []


# --- Motorola ---------------------------------------------------------------------------------


def motorola_client(listings: dict[str, dict] | None = None) -> httpx.AsyncClient:
    served = motorola_listings() if listings is None else listings

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        path = httpx.QueryParams(body).get("items[href]", "")
        # The real API answers an unknown path with the ancestor chain and no children.
        return httpx.Response(200, json=served.get(path, {"items": []}))

    return mock_client(handler)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "RTWO_RETAIL_15_V1TRS35H.60-33-7_subsidy-DEFAULT_regulatory-DEFAULT_cid50_CFC.xml.zip",
            ("15", "V1TRS35H.60-33-7"),
        ),
        (
            "XT2301-1_RTWO_BOOST_13_T1TRS33.43-48-41-3_subsidy-DISH_WILD_regulatory-D.xml.zip",
            ("13", "T1TRS33.43-48-41-3"),
        ),
        (
            "XT2341-2_CANCUN_RETLA_14_UTLB34.102-62_a05daa66_release-keys_global_CFC.zip",
            ("14", "UTLB34.102-62"),
        ),
        ("README.html", None),
        ("TinyFastbootScript.zip", None),
    ],
)
def test_motorola_filename_shapes_that_share_one_tree(filename, expected):
    """The leading fields are not a fixed count — the model SKU is present on some files and
    absent on others — so the anchor is the trailing marker, not a field index."""
    assert parse_firmware_filename(filename) == expected


async def test_motorola_lists_every_channel_of_a_configured_device():
    driver = MotorolaDriver(make_settings(motorola_devices="rtwo"), client=motorola_client())

    refs = await driver.list_available()

    # RETAIL 5, RETEU 5, Boost 4 in the fixture; the mirror serves 10 channels for `rtwo`.
    assert len(refs) == 14
    assert {ref.device for ref in refs} == {"rtwo"}
    channels = {ref.build.rsplit("_", 1)[1] for ref in refs}
    assert channels == {"RETAIL", "RETEU", "Boost"}
    assert all(ref.url.startswith("https://mirrors.lolinet.com/firmware/lenomola/") for ref in refs)


async def test_motorola_keeps_the_channel_in_the_build_not_in_the_device():
    """16 of 32 measured (device, build) pairs are published by more than one channel. In
    `device` the channel would make one phone count as ten toward the triage ranking signal;
    dropped entirely it would let `select_ref` answer with a channel nobody asked for."""
    driver = MotorolaDriver(make_settings(motorola_devices="rtwo"), client=motorola_client())

    refs = await driver.list_available()
    shared = [ref.build for ref in refs if ref.build.startswith("V1TRS35H.60-33-7_")]

    assert sorted(shared) == ["V1TRS35H.60-33-7_RETAIL", "V1TRS35H.60-33-7_RETEU"]
    assert len({ref.build for ref in refs}) == len(refs)


async def test_motorola_newest_build_comes_from_the_mirrors_mtime():
    """A Motorola build id carries no Android date field, so `select_ref` falls back to row
    order and the driver owns making that order chronological."""
    driver = MotorolaDriver(make_settings(motorola_devices="rtwo"), client=motorola_client())

    refs = await driver.list_available()

    assert select_ref(refs, device="rtwo").build == "V1TRS35H.60-33-7_RETEU"
    assert select_ref(refs, device="rtwo", build="T1TR33.43-20-56_RETAIL").android_version == "13"


async def test_motorola_reads_only_the_requested_directorys_children():
    """h5ai answers with every ancestor directory as well: one channel listing came back with
    68 items of which 5 were that channel's files, the rest `/`, `/firmware/` and so on."""
    listings = motorola_listings()
    root = listings["/firmware/lenomola/"]["items"]

    assert any(item["href"] == "/" for item in root), "fixture lost the ancestor chain"
    driver = MotorolaDriver(make_settings(motorola_devices="rtwo"), client=motorola_client())

    refs = await driver.list_available()

    assert all("/firmware/lenomola/2023/rtwo/official/" in ref.url for ref in refs)


async def test_motorola_without_configured_devices_refuses_rather_than_crawling():
    driver = MotorolaDriver(make_settings(), client=motorola_client())

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.list_available()

    assert "MOTOROLA_DEVICES" in str(excinfo.value)


async def test_motorola_refuses_a_codename_that_is_not_one():
    driver = MotorolaDriver(make_settings(motorola_devices="../../etc"), client=motorola_client())

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.list_available()

    assert "../../etc" in str(excinfo.value)


async def test_motorola_device_that_resolves_to_nothing_is_an_error():
    driver = MotorolaDriver(
        make_settings(motorola_devices="nosuchdevice"), client=motorola_client()
    )

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        await driver.list_available()

    assert "nosuchdevice" in str(excinfo.value)


async def test_motorola_terms_carry_the_non_commercial_restriction():
    posture = MotorolaDriver(make_settings()).terms()

    assert posture.risk is TermsRisk.RESTRICTED
    assert "NOT for commercial use" in posture.summary


async def test_motorola_fetch_records_the_archive_as_integrity_unverified(tmp_path):
    """The h5ai listing carries no checksum at all, so nothing here can claim otherwise."""
    driver = MotorolaDriver(
        make_settings(motorola_devices="rtwo"),
        client=mock_client(lambda _r: httpx.Response(200, content=b"PK\x03\x04moto")),
    )
    ref = FirmwareRef(
        driver="motorola",
        device="rtwo",
        build="V1TRS35H.60-33-7_RETAIL",
        url="https://mirrors.lolinet.com/firmware/lenomola/2023/rtwo/official/RETAIL/x.zip",
    )

    archive = await driver.fetch(ref, tmp_path)

    assert archive.integrity_verified is False
    assert archive.path.name == "motorola-rtwo-V1TRS35H.60-33-7_RETAIL.zip"


# --- Samsung ---------------------------------------------------------------------------------

SAMSUNG_MODEL = "SM-S911U"
SAMSUNG_REGION = "XAA"
SAMSUNG_BUILD = "S911USQS8FZG1_XAA"
SAMSUNG_VERSION = "S911USQS8FZG1/S911UOYN8FZG1/S911USQS8FZG1/S911USQS8FZG1"
SAMSUNG_BINARY_NAME = "SM-S911U_2_20260708221800_qd55e39o59_fac.zip.enc4"
# Measured against the live server on 2026-08-11, not recomputed here: this is the key that
# actually decrypted the first block of the real 11,565,187,312-byte archive to `PK\x03\x04`.
SAMSUNG_KEY = bytes.fromhex("b7f921b15f9e3004f4241aaa2b7f4a82")


def samsung_settings(**overrides) -> Settings:
    return make_settings(
        **{"samsung_models": SAMSUNG_MODEL, "samsung_regions": SAMSUNG_REGION, **overrides}
    )


def samsung_fixture(name: str) -> str:
    return (FIXTURES / f"samsung_{name}.xml").read_text(encoding="utf-8")


def samsung_index_url(model: str = SAMSUNG_MODEL, region: str = SAMSUNG_REGION) -> str:
    return f"https://fota-cloud-dn.ospserver.net/firmware/{region}/{model}/version.xml"


def pkcs7_encrypt(plaintext: bytes, key: bytes) -> bytes:
    padding = AES_BLOCK_BYTES - len(plaintext) % AES_BLOCK_BYTES
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return encryptor.update(plaintext + bytes([padding]) * padding) + encryptor.finalize()


def samsung_plain_archive() -> bytes:
    """A stand-in for the factory zip: a real zip, because the decrypt asserts the magic."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("AP_S911USQS8FZG1_meta_OS16.tar.md5", b"tar-bytes" * 512)
        zf.writestr("BL_S911USQS8FZG1.tar.md5", b"bl-bytes" * 128)
    return buffer.getvalue()


def samsung_client(
    *,
    seen: list[httpx.Request] | None = None,
    index: dict[str, str] | None = None,
    inform: str | None = None,
    body: bytes | None = None,
    declared_size: int | None = None,
) -> httpx.AsyncClient:
    """The whole chain: the public index, the three FUS posts, and the binary.

    Each FUS response carries its OWN `NONCE` header, so a client that keeps signing with the
    first one is visibly wrong rather than accidentally fine.

    `BINARY_BYTE_SIZE` is rewritten to the length actually served, because the real fixture
    declares the real 11.5 GB build and `fetch` compares the two. `declared_size` forces them
    apart for the test that is about exactly that mismatch.
    """
    served_index = (
        {samsung_index_url(): samsung_fixture("version_index")} if index is None else index
    )
    served_inform = inform if inform is not None else samsung_fixture("binary_inform")
    if inform is None:
        size = declared_size if declared_size is not None else len(body or b"")
        served_inform = served_inform.replace(
            "<BINARY_BYTE_SIZE><Data>11565187312</Data>",
            f"<BINARY_BYTE_SIZE><Data>{size}</Data>",
        )
    nonces = iter(["nonce-generate-01", "nonce-inform-002", "nonce-init-0003"])

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        url = str(request.url)
        if "version.xml" in url:
            document = served_index.get(url)
            if document is None:
                return httpx.Response(403, text=samsung_fixture("access_denied"))
            return httpx.Response(200, text=document)
        if url.startswith(DOWNLOAD_URL):
            return httpx.Response(200, content=body if body is not None else b"")
        headers = {"NONCE": next(nonces)}
        if url.endswith("GenerateNonce.do"):
            return httpx.Response(200, text="", headers=headers)
        if url.endswith("BinaryInform.do"):
            return httpx.Response(
                200,
                text=served_inform,
                headers=headers,
            )
        return httpx.Response(
            200,
            text="<FUSMsg><FUSBody><Results><Status>S00</Status></Results></FUSBody></FUSMsg>",
            headers=headers,
        )

    return mock_client(handler)


def samsung_ref(build: str = SAMSUNG_BUILD, device: str = SAMSUNG_MODEL) -> FirmwareRef:
    return FirmwareRef(driver="samsung", device=device, build=build, url=samsung_index_url(device))


def test_samsung_index_puts_the_csc_in_the_build_and_the_model_in_the_device():
    """Two CSCs of one model are two regional firmware lines of ONE phone: the CSC in `device`
    would make that phone count twice toward `package_facts.device_count`."""
    ref = parse_version_index(
        samsung_fixture("version_index"),
        model=SAMSUNG_MODEL,
        region=SAMSUNG_REGION,
        source_url=samsung_index_url(),
    )

    assert ref.device == "SM-S911U"
    assert ref.build == SAMSUNG_BUILD
    assert ref.android_version == "16"
    # A Samsung build has no download URL until an authenticated inform call produces one, so
    # the ref names the document the offer came from.
    assert ref.url == samsung_index_url()
    assert ref.archive_filename == "samsung-SM-S911U-S911USQS8FZG1_XAA.zip"


def test_samsung_normalizes_the_three_part_index_version_to_the_four_fus_wants():
    assert normalize_version("A/B/C") == "A/B/C/A"
    assert normalize_version("A/B//D") == "A/B/A/D"
    assert normalize_version(SAMSUNG_VERSION) == SAMSUNG_VERSION


async def test_samsung_reads_a_403_as_a_pair_that_does_not_exist_rather_than_a_failure():
    """`SM-A546B` answers 403 under DBT, XEO, BTU and ATT and 200 under EUX. Coding that as an
    error would make one wrong pairing fail the whole grid."""
    driver = SamsungDriver(
        samsung_settings(samsung_models="SM-S911U,SM-A546B", samsung_regions="XAA,DBT"),
        client=samsung_client(),
    )

    refs = await driver.list_available()

    assert [(ref.device, ref.build) for ref in refs] == [("SM-S911U", SAMSUNG_BUILD)]


async def test_samsung_warns_about_a_model_that_resolved_nothing_at_all(caplog):
    """A pair that does not exist is normal — most of a model x CSC grid legitimately does
    not. A MODEL with no live CSC anywhere is the shape a typo takes, and it would otherwise
    read as a phone being tracked while the grid carries none of it."""
    driver = SamsungDriver(
        samsung_settings(samsung_models="SM-S911U,SM-TYPO9Z"), client=samsung_client()
    )

    with caplog.at_level(logging.WARNING):
        refs = await driver.list_available()

    assert [ref.device for ref in refs] == [SAMSUNG_MODEL]
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert any("SM-TYPO9Z" in message for message in warnings), warnings
    assert not any(SAMSUNG_MODEL in message for message in warnings), warnings


async def test_samsung_grid_where_no_pair_exists_is_an_error_not_an_empty_catalogue():
    driver = SamsungDriver(
        samsung_settings(samsung_models="SM-X999Z"), client=samsung_client(index={})
    )

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        await driver.list_available()

    assert "SM-X999Z/XAA" in str(excinfo.value)


async def test_samsung_index_answering_200_with_no_latest_build_is_an_error():
    """A pair that does not exist answers 403, so a 200 carrying nothing is a shape change."""
    driver = SamsungDriver(
        samsung_settings(),
        client=samsung_client(
            index={samsung_index_url(): "<versioninfo><firmware/></versioninfo>"}
        ),
    )

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        await driver.list_available()

    assert "no <latest> build" in str(excinfo.value)


@pytest.mark.parametrize(
    ("models", "regions"),
    [("", "XAA"), ("SM-S911U", ""), ("", "")],
)
async def test_samsung_without_a_configured_grid_refuses_rather_than_guessing(models, regions):
    driver = SamsungDriver(
        samsung_settings(samsung_models=models, samsung_regions=regions), client=samsung_client()
    )

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.list_available()

    assert "SAMSUNG_MODELS" in str(excinfo.value)


@pytest.mark.parametrize(
    ("models", "regions", "offender"),
    [
        ("../../etc", "XAA", "../../etc"),
        ("SM-S911U", "xaa", "xaa"),
        ("SM-S911U", "XA", "XA"),
        ("sm-s911u", "XAA", "sm-s911u"),
    ],
)
async def test_samsung_refuses_a_model_or_csc_that_is_not_one(models, regions, offender):
    """Both reach a URL, so both are refused rather than transliterated."""
    driver = SamsungDriver(
        samsung_settings(samsung_models=models, samsung_regions=regions), client=samsung_client()
    )

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.list_available()

    assert offender in str(excinfo.value)


async def test_samsung_refuses_a_grid_past_the_probe_ceiling():
    """The grid MULTIPLIES: models times CSCs is the request count against someone else's
    index, so it is capped on the product rather than on either axis."""
    models = ",".join(f"SM-X{index:03d}" for index in range(MAX_INDEX_PROBES // 2 + 1))
    driver = SamsungDriver(
        samsung_settings(samsung_models=models, samsung_regions="XAA,EUX"), client=samsung_client()
    )

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.list_available()

    assert str(MAX_INDEX_PROBES) in str(excinfo.value)


async def test_samsung_newest_resolves_to_the_last_configured_csc():
    """No Samsung version string carries a parseable date, so `select_ref` falls back to row
    order — and across two CSCs of one model there is no "newer", only the operator's order."""
    eux_index = samsung_fixture("version_index").replace("S911USQS8FZG1", "S911BXXU9GZH2")
    driver = SamsungDriver(
        samsung_settings(samsung_regions="XAA,EUX"),
        client=samsung_client(
            index={
                samsung_index_url(): samsung_fixture("version_index"),
                samsung_index_url(region="EUX"): eux_index,
            }
        ),
    )

    refs = await driver.list_available()

    assert [ref.build for ref in refs] == [SAMSUNG_BUILD, "S911BXXU9GZH2_EUX"]
    assert select_ref(refs, device=SAMSUNG_MODEL).build == "S911BXXU9GZH2_EUX"
    assert select_ref(refs, device=SAMSUNG_MODEL, build=SAMSUNG_BUILD).build == SAMSUNG_BUILD


def test_samsung_inform_derives_the_decryption_key_measured_against_the_live_server():
    """`md5(logic_check(BINARY_SW_VERSION, LOGIC_VALUE_FACTORY))`. The expected value is the
    key that really decrypted the real archive, so this fails if either input is read from the
    wrong field — which is exactly how every retired client broke."""
    binary = parse_binary_inform(
        samsung_fixture("binary_inform"), model=SAMSUNG_MODEL, region=SAMSUNG_REGION
    )

    assert binary.key == SAMSUNG_KEY
    assert binary.filename == SAMSUNG_BINARY_NAME
    assert binary.model_path == "/neofus/911/"
    assert binary.size == 11565187312
    assert binary.version == SAMSUNG_VERSION
    assert binary.model_type == "9"
    assert binary.region == SAMSUNG_REGION
    assert binary.download_url == (f"{DOWNLOAD_URL}?file=/neofus/911/{SAMSUNG_BINARY_NAME}")
    # https, not the plain http samloader-rs uses: this pipeline never pulls firmware in the
    # clear, and the host serves the same path over TLS (measured 2026-08-11, 206 + a correct
    # Content-Range).
    assert binary.download_url.startswith("https://")


def test_samsung_inform_reads_binary_sw_version_and_not_the_field_retired_clients_read():
    """`LATEST_FW_VERSION` is ABSENT from this response and `BINARY_SW_VERSION` is present, so
    a client reading only the former derives its key from nothing and decrypts to garbage of
    exactly the right size."""
    document = samsung_fixture("binary_inform")
    assert "LATEST_FW_VERSION" not in document

    fallback = document.replace("BINARY_SW_VERSION", "LATEST_FW_VERSION")
    binary = parse_binary_inform(fallback, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)

    assert binary.version == SAMSUNG_VERSION
    assert binary.key == SAMSUNG_KEY


def test_samsung_key_comes_from_the_factory_logic_value_and_not_the_home_one():
    """The real response carries the SAME value in both fields, so no assertion against it can
    tell the two apart — swapping them there is an equivalent mutant. This forces them apart so
    the choice is pinned, and `LOGIC_VALUE_FACTORY` is the one FUS derives the factory
    binary's key from."""
    document = samsung_fixture("binary_inform")
    assert document.count("xs6z6jreet76o8j3") == 2, "fixture no longer carries both values"
    split = document.replace(
        "<LOGIC_VALUE_HOME><Data>xs6z6jreet76o8j3</Data>",
        "<LOGIC_VALUE_HOME><Data>zzzzzzzzzzzzzzzz</Data>",
    )

    binary = parse_binary_inform(split, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)

    assert binary.key == SAMSUNG_KEY
    # Positive control: the home value really would produce a different key, so the assertion
    # above is discriminating rather than insensitive to the field it names.
    home_only = split.replace("LOGIC_VALUE_FACTORY", "LOGIC_VALUE_UNUSED")
    assert parse_binary_inform(home_only, model=SAMSUNG_MODEL, region=SAMSUNG_REGION).key != (
        SAMSUNG_KEY
    )


def test_samsung_inform_that_is_not_s00_is_a_protocol_error():
    denied = samsung_fixture("binary_inform").replace(
        "<Status>S00</Status>", "<Status>408</Status>"
    )

    with pytest.raises(SamsungProtocolError) as excinfo:
        parse_binary_inform(denied, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)

    assert "408" in str(excinfo.value)


@pytest.mark.parametrize("dropped", ["BINARY_NAME", "MODEL_PATH", "LOGIC_VALUE"])
def test_samsung_inform_missing_a_field_the_download_needs_is_a_protocol_error(dropped):
    document = samsung_fixture("binary_inform").replace(dropped, "REMOVED_FIELD")

    with pytest.raises(SamsungProtocolError) as excinfo:
        parse_binary_inform(document, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)

    assert dropped in str(excinfo.value)


@pytest.mark.parametrize(
    "binary_name",
    [
        "../../etc/passwd",
        "SM-S911U.zip.enc4&file=/other/thing",
        "SM S911U.zip.enc4",
        "",
    ],
)
def test_samsung_refuses_a_binary_name_that_could_reshape_the_download_url(binary_name):
    """`BINARY_NAME` and `MODEL_PATH` arrive inside a document downloaded from a third party
    and are the only untrusted values in the request URL."""
    document = samsung_fixture("binary_inform").replace(SAMSUNG_BINARY_NAME, binary_name)

    with pytest.raises(SamsungProtocolError):
        parse_binary_inform(document, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)


@pytest.mark.parametrize("model_path", ["/neofus/911", "neofus/911/", "/neofus/../911/"])
def test_samsung_refuses_a_model_path_that_is_not_one(model_path):
    document = samsung_fixture("binary_inform").replace(
        "<MODEL_PATH><Data>/neofus/911/</Data>", f"<MODEL_PATH><Data>{model_path}</Data>"
    )

    with pytest.raises(SamsungProtocolError):
        parse_binary_inform(document, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)


@pytest.mark.parametrize("declared", ["11565187313", "0", "-16", "eleven"])
def test_samsung_refuses_a_declared_size_that_is_not_a_whole_number_of_aes_blocks(declared):
    """A `.enc4` is padded to the 16-byte block, so a size that is not a multiple of one is a
    protocol change rather than a small firmware."""
    document = samsung_fixture("binary_inform").replace(
        "<BINARY_BYTE_SIZE><Data>11565187312</Data>", f"<BINARY_BYTE_SIZE><Data>{declared}</Data>"
    )

    with pytest.raises(SamsungProtocolError):
        parse_binary_inform(document, model=SAMSUNG_MODEL, region=SAMSUNG_REGION)


def test_samsung_document_carrying_a_dtd_is_refused_unparsed():
    """Same threat and the same mitigation `etcconfig` applies to a vendor image's config
    files: `xml.etree` expands internal entities and no Samsung document has a DTD."""
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE versioninfo [<!ENTITY a "aaaaaaaaaa">]>'
        "<versioninfo><firmware><version><latest>&a;</latest></version></firmware></versioninfo>"
    )

    with pytest.raises(SamsungProtocolError) as excinfo:
        parse_version_index(bomb, model=SAMSUNG_MODEL, region=SAMSUNG_REGION, source_url="fixture")

    assert "DTD" in str(excinfo.value)


def test_samsung_document_past_the_byte_ceiling_is_refused_unparsed():
    with pytest.raises(SamsungProtocolError) as excinfo:
        parse_binary_inform("<FUSMsg>" + " " * (1024 * 1024), model="M", region="XAA")

    assert "ceiling" in str(excinfo.value)


def test_samsung_auth_signature_encrypts_the_raw_nonce_and_never_decrypts_it():
    """One AES-128-ECB block over `nonce[:16]` right-padded with ASCII `0`. Every published
    Python client tries to DECRYPT this value first, gets an empty string, and reads as a
    network problem rather than as a protocol change."""
    assert auth_signature("abc") == "cc10cf6cabc638ad81d70765ffe4b83c"
    # Padding is applied to the same 16 bytes a longer nonce would be truncated to, so these
    # two must agree.
    assert auth_signature("abc0000000000000") == auth_signature("abc")
    assert auth_signature("abc00000000000009999") == auth_signature("abc")
    assert len(auth_signature("")) == 32


def test_samsung_init_logic_check_is_computed_over_the_filename_slice():
    """The call whose OMISSION answers HTTP 401 after the whole transfer. Its `LOGIC_CHECK`
    input is `filename[-25:-9]` — sliced from the END, so it is independent of how long the
    model name is."""
    binary = parse_binary_inform(
        samsung_fixture("binary_inform"), model=SAMSUNG_MODEL, region=SAMSUNG_REGION
    )
    nonce = "nonce-init-0003"

    body = binary_init_body(binary, nonce=nonce)

    assert SAMSUNG_BINARY_NAME[-25:-9] == "0_qd55e39o59_fac"
    assert f"<LOGIC_CHECK><Data>{logic_check('0_qd55e39o59_fac', nonce)}</Data>" in body
    assert f"<BINARY_NAME><Data>{SAMSUNG_BINARY_NAME}</Data>" in body
    assert "<DEVICE_MODEL_TYPE><Data>9</Data>" in body
    assert "<DEVICE_LOCAL_CODE><Data>XAA</Data>" in body


def test_samsung_terms_state_the_reverse_engineered_posture_rather_than_a_public_one():
    posture = SamsungDriver(samsung_settings()).terms()

    assert posture.risk is TermsRisk.REVERSE_ENGINEERED
    assert "no credential is presented" in posture.summary


async def test_samsung_fetch_walks_the_whole_handshake_and_decrypts_what_it_downloads(tmp_path):
    """The end-to-end shape, hermetically: index, nonce, inform, init, binary — in that order,
    each request signed with the nonce the PREVIOUS response issued."""
    plaintext = samsung_plain_archive()
    seen: list[httpx.Request] = []
    driver = SamsungDriver(
        samsung_settings(),
        client=samsung_client(seen=seen, body=pkcs7_encrypt(plaintext, SAMSUNG_KEY)),
    )

    archive = await driver.fetch(samsung_ref(), tmp_path)

    assert [request.url.path.rsplit("/", 1)[-1] for request in seen] == [
        "version.xml",
        "NF_SmartDownloadGenerateNonce.do",
        "NF_SmartDownloadBinaryInform.do",
        "NF_SmartDownloadBinaryInitForMass.do",
        "NF_SmartDownloadBinaryForMass.do",
    ]
    # The nonce ROTATES on every response carrying one, and the next request must be signed
    # with the newest. A client that signs everything with the first works until it does not.
    assert 'nonce="nonce-inform-002"' in seen[3].headers["authorization"]
    assert 'nonce="nonce-init-0003"' in seen[4].headers["authorization"]
    assert auth_signature("nonce-init-0003") in seen[4].headers["authorization"]
    assert seen[4].headers["user-agent"] == "SMART 2.0"

    assert archive.path.read_bytes() == plaintext
    assert archive.path.name == "samsung-SM-S911U-S911USQS8FZG1_XAA.zip"
    assert archive.sha256 == hashlib.sha256(plaintext).hexdigest()
    # FUS publishes no digest of the plaintext this pipeline keeps.
    assert archive.integrity_verified is False
    # The encrypted form is decrypted IN PLACE and renamed, so nothing is left beside it.
    assert [path.name for path in tmp_path.iterdir()] == [archive.path.name]


async def test_samsung_fetch_refuses_a_wrong_key_rather_than_keeping_the_garbage(tmp_path):
    """A wrong key yields a file of exactly the right size and complete garbage, which no
    exception reports on its own."""
    plaintext = samsung_plain_archive()
    wrong = bytes(16)
    driver = SamsungDriver(
        samsung_settings(), client=samsung_client(body=pkcs7_encrypt(plaintext, wrong))
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await driver.fetch(samsung_ref(), tmp_path)

    assert "padding" in str(excinfo.value) or "zip magic" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


async def test_samsung_fetch_refuses_a_body_that_decrypts_to_something_other_than_a_zip(tmp_path):
    """Valid padding is not proof: it is one byte value in 256 by luck, so the magic is
    checked too."""
    driver = SamsungDriver(
        samsung_settings(),
        client=samsung_client(body=pkcs7_encrypt(b"7z\xbc\xaf'\x1c" * 8, SAMSUNG_KEY)),
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await driver.fetch(samsung_ref(), tmp_path)

    assert "zip magic" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


async def test_samsung_fetch_refuses_a_body_shorter_than_the_size_fus_declared(tmp_path):
    """FUS publishes no digest, so the declared byte size is the only thing the source says
    about the bytes it serves. Truncated on a 16-byte boundary, the decrypt would blame the
    key instead of the transfer — which sends a reader to the wrong half of the protocol."""
    plaintext = samsung_plain_archive()
    whole = pkcs7_encrypt(plaintext, SAMSUNG_KEY)
    driver = SamsungDriver(
        samsung_settings(),
        client=samsung_client(body=whole[:-AES_BLOCK_BYTES], declared_size=len(whole)),
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await driver.fetch(samsung_ref(), tmp_path)

    assert "truncated" in str(excinfo.value)
    assert str(len(whole)) in str(excinfo.value)
    assert str(len(whole) - AES_BLOCK_BYTES) in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


async def test_samsung_fetch_refuses_a_build_the_index_no_longer_publishes(tmp_path):
    """Samsung's index carries only the CURRENT build per pair, so a stale ref cannot be
    resolved — and resolving it to whatever is current would download a build nobody named."""
    driver = SamsungDriver(samsung_settings(), client=samsung_client())

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.fetch(samsung_ref(build="S911USQS7EYH1_XAA"), tmp_path)

    assert "S911USQS7EYH1_XAA" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


async def test_samsung_fetch_refuses_a_pair_the_index_stopped_publishing_at_all(tmp_path):
    driver = SamsungDriver(samsung_settings(), client=samsung_client(index={}))

    with pytest.raises(FirmwareError) as excinfo:
        await driver.fetch(samsung_ref(), tmp_path)

    assert "403" in str(excinfo.value)


@pytest.mark.parametrize("build", ["S911USQS8FZG1", "S911USQS8FZG1_xaa", "_XAA"])
async def test_samsung_fetch_refuses_a_build_that_is_not_a_samsung_build_id(build, tmp_path):
    driver = SamsungDriver(samsung_settings(), client=samsung_client())

    with pytest.raises(FirmwareInputError) as excinfo:
        await driver.fetch(samsung_ref(build=build), tmp_path)

    assert "<PDA>_<CSC>" in str(excinfo.value)


async def test_samsung_fetch_refuses_a_binary_resolved_for_another_region(tmp_path):
    """Asked for XAA and given a binary for another CSC: that is firmware for a phone nobody
    asked about, and it would be recorded under this device's name."""
    driver = SamsungDriver(
        samsung_settings(),
        client=samsung_client(
            inform=samsung_fixture("binary_inform").replace(
                "<BINARY_LOCAL_CODE><Data>XAA</Data>", "<BINARY_LOCAL_CODE><Data>EUX</Data>"
            )
        ),
    )

    with pytest.raises(SamsungProtocolError) as excinfo:
        await driver.fetch(samsung_ref(), tmp_path)

    assert "EUX" in str(excinfo.value)


def test_samsung_decrypt_strips_the_padding_and_digests_the_plaintext(tmp_path):
    """The digest covers what this pipeline KEEPS, not the padded form that was on the wire —
    otherwise the archive's recorded identity is of a file nobody has."""
    plaintext = samsung_plain_archive()
    path = tmp_path / "archive.zip.enc4"
    path.write_bytes(pkcs7_encrypt(plaintext, SAMSUNG_KEY))

    digest = decrypt_archive(path, SAMSUNG_KEY, context="test")

    assert path.read_bytes() == plaintext
    assert digest == hashlib.sha256(plaintext).hexdigest()
    assert path.stat().st_size == len(plaintext)


def test_samsung_decrypt_spans_the_chunk_boundary_it_reads_in(tmp_path):
    """The in-place decrypt reads in 4 MB chunks and holds back the final block for the
    padding, so an archive several chunks long is the case that separates a working seek from
    one that rewrites the same offset."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("AP.tar.md5", bytes(range(256)) * 40_000)
    plaintext = buffer.getvalue()
    assert len(plaintext) > 4 * 1024 * 1024 * 2, "fixture must span more than two read chunks"
    path = tmp_path / "archive.zip.enc4"
    path.write_bytes(pkcs7_encrypt(plaintext, SAMSUNG_KEY))

    digest = decrypt_archive(path, SAMSUNG_KEY, context="test")

    assert path.read_bytes() == plaintext
    assert digest == hashlib.sha256(plaintext).hexdigest()


@pytest.mark.parametrize("size", [0, 17])
def test_samsung_decrypt_refuses_a_file_that_is_not_whole_aes_blocks(size, tmp_path):
    path = tmp_path / "archive.zip.enc4"
    path.write_bytes(b"x" * size)

    with pytest.raises(FirmwareDownloadError) as excinfo:
        decrypt_archive(path, SAMSUNG_KEY, context="test")

    assert "16-byte" in str(excinfo.value)


@pytest.mark.parametrize("declared_padding", [0, AES_BLOCK_BYTES + 1, 200])
def test_samsung_decrypt_refuses_padding_no_pkcs7_encoder_would_write(declared_padding, tmp_path):
    plaintext = samsung_plain_archive()
    forged = plaintext + bytes([declared_padding]) * (
        AES_BLOCK_BYTES - len(plaintext) % AES_BLOCK_BYTES
    )
    encryptor = Cipher(algorithms.AES(SAMSUNG_KEY), modes.ECB()).encryptor()
    path = tmp_path / "archive.zip.enc4"
    path.write_bytes(encryptor.update(forged) + encryptor.finalize())

    with pytest.raises(FirmwareDownloadError) as excinfo:
        decrypt_archive(path, SAMSUNG_KEY, context="test")

    assert "padding" in str(excinfo.value)


@pytest.mark.parametrize(
    "driver_class",
    [XiaomiDriver, NothingDriver, MotorolaDriver, SamsungDriver],
    ids=lambda c: c.name,
)
async def test_a_driver_refuses_a_ref_belonging_to_another_driver(driver_class, tmp_path):
    driver = driver_class(
        samsung_settings(motorola_devices="rtwo"),
        client=mock_client(lambda _r: httpx.Response(200, content=b"")),
    )
    ref = FirmwareRef(driver="pixel", device="comet", build="A.1", url="https://x/y.zip")

    with pytest.raises(FirmwareInputError):
        await driver.fetch(ref, tmp_path)
