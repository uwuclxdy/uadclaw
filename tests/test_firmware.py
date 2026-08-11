"""Driver interface, registry, and the Pixel index parse.

No network: the index tests run against `tests/fixtures/pixel_factory_index.html`, a trimmed
verbatim slice of the real page (see the comment at the top of that file), and every HTTP
call goes through an `httpx.MockTransport` that records what was actually sent — which is the
only way to prove the terms acknowledgement cookie leaves the process.
"""

import hashlib
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from uadclaw.drivers.pixel import PixelDriver, parse_factory_index
from uadclaw.firmware import (
    EmptyFirmwareIndexError,
    FirmwareDownloadError,
    FirmwareDriverDisabledError,
    FirmwareInputError,
    FirmwareRef,
    FirmwareTermsNotAcknowledgedError,
    TermsRisk,
    UnknownFirmwareDriverError,
    download_to_file,
    driver_names,
    enabled_driver_names,
    get_driver,
    select_ref,
)
from uadclaw.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "pixel_factory_index.html"


def make_settings(**overrides) -> Settings:
    """Explicit kwargs beat every other settings source, so this is independent of whatever
    .env or secrets directory happens to exist on the box running the tests."""
    base = {
        "postgres_password": "test-only-password",
        "auth_password": "test-only-admin-password",
        "session_secret": "test-only-session-secret",
        "pixel_terms_ack_cookie_value": "nexus-image-tos",
    }
    return Settings(**(base | overrides))


def index_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --- index parsing ----------------------------------------------------------------------


def test_index_parses_into_device_build_and_url():
    refs = parse_factory_index(index_html(), source_url="fixture")

    assert len(refs) == 6
    assert {ref.device for ref in refs} == {"comet", "stallion", "razorg"}
    comet = next(ref for ref in refs if ref.build == "AD1A.240530.030")
    assert comet.driver == "pixel"
    assert comet.url == (
        "https://dl.google.com/dl/android/aosp/comet-ad1a.240530.030-factory-77dca584.zip"
    )
    assert comet.sha256 == "77dca5840fa82483f836934ca634ee332728009444e9d2dc125a251bb9ac869e"
    assert comet.android_version == "14.0.0"
    assert comet.marketing_name == "Pixel 9 Pro Fold"


def test_index_build_id_is_canonical_uppercase_even_when_the_url_is_not():
    """`razorg-JLS36C-factory-834eab41.zip` is the one link in 2293 whose URL carries an
    uppercase build id; every other one is lowercase. Both must land in the same shape or
    the same build resolves differently depending on which row it came from."""
    refs = parse_factory_index(index_html(), source_url="fixture")
    razorg = [ref for ref in refs if ref.device == "razorg"]

    assert sorted(ref.build for ref in razorg) == ["JLS36C", "JLS36I"]
    assert razorg[0].marketing_name == "Nexus 7 [2013] (Mobile)"


def test_index_row_without_a_marketing_name_still_parses():
    refs = parse_factory_index(index_html(), source_url="fixture")
    stallion = next(ref for ref in refs if ref.device == "stallion")

    assert stallion.marketing_name is None
    assert stallion.build.startswith("CP2A.")


def test_zero_link_page_is_a_hard_error_not_an_empty_catalogue():
    """The terms-walled page is an HTTP 200 carrying real prose and no links at all. If that
    parsed to `[]`, a missing cookie would read as "this source has no builds"."""
    walled = "<html><body><h1>Terms and conditions</h1><p>Accept to continue.</p></body></html>"

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        parse_factory_index(walled, source_url="https://developers.google.com/android/images")

    message = str(excinfo.value)
    assert "PIXEL_TERMS_ACK_COOKIE_VALUE=nexus-image-tos" in message
    assert "https://developers.google.com/android/images" in message


# --- the acknowledgement cookie ----------------------------------------------------------


async def test_list_available_sends_the_acknowledgement_cookie():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=index_html())

    driver = PixelDriver(
        make_settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    refs = await driver.list_available()

    assert len(refs) == 6
    assert len(seen) == 1
    assert seen[0].headers["cookie"] == "devsite_wall_acks=nexus-image-tos"


async def test_unacknowledged_terms_refuse_before_any_request_is_made():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="<html>terms</html>")

    driver = PixelDriver(
        make_settings(pixel_terms_ack_cookie_value=""),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(FirmwareTermsNotAcknowledgedError) as excinfo:
        await driver.list_available()

    assert seen == []
    assert "PIXEL_TERMS_ACK_COOKIE_VALUE=nexus-image-tos" in str(excinfo.value)


def test_terms_posture_reports_whether_the_operator_acknowledged():
    accepted = PixelDriver(make_settings()).terms()
    unaccepted = PixelDriver(make_settings(pixel_terms_ack_cookie_value="")).terms()

    assert accepted.risk is TermsRisk.ACKNOWLEDGEMENT
    assert accepted.acknowledged is True
    assert accepted.source_url == "https://developers.google.com/android/images"
    assert unaccepted.acknowledged is False


# --- registry and per-driver enable/disable ----------------------------------------------


def test_registry_resolves_a_known_driver():
    driver = get_driver("pixel", make_settings())

    assert driver.name == "pixel"
    assert driver_names() == ("pixel",)


def test_unknown_driver_name_is_refused_by_name():
    with pytest.raises(UnknownFirmwareDriverError) as excinfo:
        get_driver("pixle", make_settings())

    assert "pixle" in str(excinfo.value)


def test_a_disabled_driver_is_refused_with_its_own_error():
    settings = make_settings(disabled_firmware_drivers="pixel")

    with pytest.raises(FirmwareDriverDisabledError):
        get_driver("pixel", settings)
    assert enabled_driver_names(settings) == ()
    assert driver_names() == ("pixel",)


def test_disable_list_tolerates_spacing_and_unrelated_names():
    settings = make_settings(disabled_firmware_drivers=" samsung , oppo ")

    assert settings.disabled_firmware_driver_names == frozenset({"samsung", "oppo"})
    assert enabled_driver_names(settings) == ("pixel",)


# --- ref selection and validation ---------------------------------------------------------


def test_select_ref_takes_the_named_build_or_the_newest():
    refs = parse_factory_index(index_html(), source_url="fixture")

    assert select_ref(refs, device="razorg", build="jls36c").build == "JLS36C"
    assert select_ref(refs, device="razorg").build == "JLS36I"


def test_select_ref_refuses_an_unknown_device_or_build():
    refs = parse_factory_index(index_html(), source_url="fixture")

    with pytest.raises(FirmwareInputError):
        select_ref(refs, device="nothere")
    with pytest.raises(FirmwareInputError):
        select_ref(refs, device="comet", build="ZZ1A.999999.001")


def test_ref_fields_that_reach_a_filename_cannot_spell_a_path():
    with pytest.raises(ValidationError):
        FirmwareRef(driver="pixel", device="../../etc", build="A", url="https://x/y.zip")
    with pytest.raises(ValidationError):
        FirmwareRef(driver="pixel", device="comet", build="a/b", url="https://x/y.zip")
    with pytest.raises(ValidationError):
        FirmwareRef(driver="pixel", device="comet", build="A", url="http://x/y.zip")

    ref = FirmwareRef(driver="pixel", device="comet", build="AD1A.1", url="https://x/y.zip")
    assert ref.archive_filename == "pixel-comet-AD1A.1.zip"


# --- download ------------------------------------------------------------------------------


async def test_download_verifies_the_published_checksum(tmp_path):
    body = b"factory-zip-bytes" * 100
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    )
    dest = tmp_path / "firmware.zip"

    await download_to_file(
        client, "https://x/y.zip", dest, expected_sha256=hashlib.sha256(body).hexdigest()
    )

    assert dest.read_bytes() == body


async def test_download_with_a_bad_checksum_leaves_nothing_behind(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"truncated"))
    )
    dest = tmp_path / "firmware.zip"

    with pytest.raises(FirmwareDownloadError):
        await download_to_file(client, "https://x/y.zip", dest, expected_sha256="0" * 64)

    assert list(tmp_path.iterdir()) == []


async def test_fetch_refuses_a_ref_from_another_driver(tmp_path):
    driver = PixelDriver(
        make_settings(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b""))
        ),
    )
    ref = FirmwareRef(driver="xiaomi", device="comet", build="A", url="https://x/y.zip")

    with pytest.raises(FirmwareInputError):
        await driver.fetch(ref, tmp_path)
