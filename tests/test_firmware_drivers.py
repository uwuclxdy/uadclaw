"""The Xiaomi, Nothing and Motorola drivers: index parsing, ordering, terms and fetch.

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

Every HTTP call goes through an `httpx.MockTransport` that records what was sent.
"""

import json
from pathlib import Path

import httpx
import pytest

from uadclaw.drivers.motorola import MotorolaDriver, parse_firmware_filename
from uadclaw.drivers.nothing import NothingDriver, parse_releases
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
    assert driver_names() == ("motorola", "nothing", "pixel", "xiaomi")
    for name in driver_names():
        assert get_driver(name, make_settings()).name == name


@pytest.mark.parametrize("disabled", ["xiaomi", "nothing", "motorola"])
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


def test_all_three_new_drivers_can_be_disabled_at_once():
    settings = make_settings(disabled_firmware_drivers="xiaomi, nothing ,motorola")

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


@pytest.mark.parametrize(
    "driver_class", [XiaomiDriver, NothingDriver, MotorolaDriver], ids=lambda c: c.name
)
async def test_a_driver_refuses_a_ref_belonging_to_another_driver(driver_class, tmp_path):
    driver = driver_class(
        make_settings(motorola_devices="rtwo"),
        client=mock_client(lambda _r: httpx.Response(200, content=b"")),
    )
    ref = FirmwareRef(driver="pixel", device="comet", build="A.1", url="https://x/y.zip")

    with pytest.raises(FirmwareInputError):
        await driver.fetch(ref, tmp_path)
