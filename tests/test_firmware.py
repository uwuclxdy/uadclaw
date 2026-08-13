"""Driver interface, registry, and the Pixel index parse.

No network: the index tests run against `tests/fixtures/pixel_factory_index.html`, a trimmed
verbatim slice of the real page (see the comment at the top of that file), and every HTTP
call goes through an `httpx.MockTransport` that records what was actually sent — which is the
only way to prove the terms acknowledgement cookie leaves the process.
"""

import asyncio
import datetime
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
    FirmwareError,
    FirmwareInputError,
    FirmwareRedirectError,
    FirmwareRef,
    FirmwareTermsNotAcknowledgedError,
    TermsRisk,
    UnknownFirmwareDriverError,
    build_date,
    download_to_file,
    driver_client,
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
    assert driver_names() == ("motorola", "nothing", "oppo", "pixel", "samsung", "xiaomi")


def test_unknown_driver_name_is_refused_by_name():
    with pytest.raises(UnknownFirmwareDriverError) as excinfo:
        get_driver("pixle", make_settings())

    assert "pixle" in str(excinfo.value)


def test_a_disabled_driver_is_refused_with_its_own_error():
    settings = make_settings(disabled_firmware_drivers="pixel")

    with pytest.raises(FirmwareDriverDisabledError):
        get_driver("pixel", settings)
    assert enabled_driver_names(settings) == ("motorola", "nothing", "oppo", "samsung", "xiaomi")
    assert driver_names() == ("motorola", "nothing", "oppo", "pixel", "samsung", "xiaomi")


def test_disable_list_tolerates_spacing_and_unrelated_names():
    # `vivo` and `huawei` are the OEMs the design names and neither has a driver, so this keeps
    # testing what it is named for. It used to say `samsung`, then `oppo`, and each stopped
    # being an unrelated name the moment that driver landed — while going on passing.
    settings = make_settings(disabled_firmware_drivers=" vivo , huawei ")

    assert settings.disabled_firmware_driver_names == frozenset({"vivo", "huawei"})
    assert enabled_driver_names(settings) == (
        "motorola",
        "nothing",
        "oppo",
        "pixel",
        "samsung",
        "xiaomi",
    )


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
        client, "https://x/y.zip", dest, expected_digest=hashlib.sha256(body).hexdigest()
    )

    assert dest.read_bytes() == body


async def test_download_with_a_bad_checksum_leaves_nothing_behind(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"truncated"))
    )
    dest = tmp_path / "firmware.zip"

    with pytest.raises(FirmwareDownloadError):
        await download_to_file(client, "https://x/y.zip", dest, expected_digest="0" * 64)

    assert list(tmp_path.iterdir()) == []


async def test_download_refuses_a_digest_algorithm_that_is_not_on_the_allowlist(tmp_path):
    """`hashlib.new` accepts `md4` and whatever else the local OpenSSL exposes; an index
    naming one must fail here rather than be verified against a broken digest."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(FirmwareInputError) as excinfo:
        await download_to_file(
            client,
            "https://x/y.zip",
            tmp_path / "f.zip",
            expected_digest="0" * 32,
            digest_algorithm="md4",
        )

    assert "md4" in str(excinfo.value)
    assert seen == []
    assert list(tmp_path.iterdir()) == []


async def test_download_returns_a_sha256_even_when_it_verified_an_md5(tmp_path):
    """The source's algorithm decides what is CHECKED; the pipeline's identity for the
    archive is always sha256, and `PipelineState.archive_sha256` records it."""
    body = b"xiaomi-rom" * 32
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
    )

    archive = await download_to_file(
        client,
        "https://x/y.zip",
        tmp_path / "f.zip",
        expected_digest=hashlib.md5(body).hexdigest(),  # noqa: S324 - what Xiaomi publishes
        digest_algorithm="md5",
    )

    assert archive.integrity_verified is True
    assert archive.sha256 == hashlib.sha256(body).hexdigest()


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


# --- build selection -----------------------------------------------------------------------


def _ref(device: str, build: str) -> FirmwareRef:
    return FirmwareRef(driver="pixel", device=device, build=build, url="https://x/y.zip")


def test_newest_build_comes_from_the_date_not_the_row_order():
    """Measured against the real index on 2026-08-11: last-row-per-device agrees with the
    date for 56 of 58 devices and is WRONG for crosshatch and blueline, which both list
    SP1A.210812.016.C2 (2021-08-12) AFTER RQ3A.211001.001 (2021-10-01)."""
    refs = [_ref("crosshatch", "RQ3A.211001.001"), _ref("crosshatch", "SP1A.210812.016.C2")]

    assert select_ref(refs, device="crosshatch").build == "RQ3A.211001.001"


def test_lexicographic_order_is_not_chronological_order():
    """The letter prefix resets, so UQ1A (2024) sorts above CP2A (2026)."""
    refs = [_ref("tegu", "UQ1A.240105.004"), _ref("tegu", "CP2A.260705.006")]

    assert select_ref(refs, device="tegu").build == "CP2A.260705.006"


def test_undated_build_ids_fall_back_to_index_order_behind_dated_ones():
    """487 of the index's 2,293 builds are pre-2016 ids carrying no date field at all."""
    assert build_date("NDE63H") is None
    assert build_date("CP2A.260705.006") == datetime.date(2026, 7, 5)

    undated_only = [_ref("razorg", "JLS36C"), _ref("razorg", "JLS36I")]
    assert select_ref(undated_only, device="razorg").build == "JLS36I"

    mixed = [_ref("razorg", "RQ3A.211001.001"), _ref("razorg", "JLS36I")]
    assert select_ref(mixed, device="razorg").build == "RQ3A.211001.001"


# --- download: the loop, the cleanup, the ceiling, the integrity verdict --------------------


async def test_the_download_never_writes_or_hashes_on_the_event_loop(tmp_path, monkeypatch):
    """~3,500 iterations over a 3.5 GB archive shared the loop with every other worker slot
    and with the heartbeat task that keeps the job from being reclaimed."""
    threads: list[str] = []
    real_to_thread = asyncio.to_thread

    async def _record(func, *args, **kwargs):
        threads.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _record)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"a" * 4096))
    )

    await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip")

    assert threads and set(threads) == {"_write_and_hash"}


async def test_a_download_with_no_published_checksum_is_recorded_as_unverified(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"bytes"))
    )

    archive = await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip")

    assert archive.integrity_verified is False
    assert archive.sha256 == hashlib.sha256(b"bytes").hexdigest()
    assert archive.path.is_file()


async def test_a_cancelled_download_leaves_no_partial_behind(tmp_path):
    """A reclaim or a worker shutdown lands here mid-write; only `httpx.HTTPError` was
    caught, so a multi-GB `.part` survived every other exit path."""

    def _handler(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    with pytest.raises(asyncio.CancelledError):
        await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip")

    assert list(tmp_path.iterdir()) == []


async def test_a_disk_error_mid_write_is_a_named_error_with_no_partial(tmp_path, monkeypatch):
    def _boom(fh, digest, chunk):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("uadclaw.firmware._write_and_hash", _boom)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"a" * 4096))
    )

    with pytest.raises(FirmwareDownloadError):
        await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip")

    assert list(tmp_path.iterdir()) == []


async def test_a_download_over_the_ceiling_stops_and_cleans_up(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"a" * 8192))
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip", max_bytes=4096)

    assert "ceiling" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


async def test_a_content_length_disagreeing_with_the_published_size_is_refused_before_any_write(
    tmp_path,
):
    """The verify line for todo 15: a source serving the wrong archive is caught from its
    `Content-Length` header, before a multi-GB transfer, on the raised type and both numbers —
    never on log text."""
    body = b"factory-zip-bytes" * 100
    read = {"bytes": 0}

    class BodyCanary(httpx.AsyncByteStream):
        # The body is a canary: the size check must refuse before a single body byte is
        # streamed, so any read here is the regression the test exists to catch.
        async def __aiter__(self):
            read["bytes"] += len(body)
            yield body

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, stream=BodyCanary(), headers={"Content-Length": "999999"}
            )
        )
    )

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await download_to_file(client, "https://x/y.zip", tmp_path / "f.zip", expected_size=12345)

    message = str(excinfo.value)
    assert "999999" in message
    assert "12345" in message
    assert read["bytes"] == 0
    assert list(tmp_path.iterdir()) == []


async def test_a_content_length_agreeing_with_the_published_size_proceeds(tmp_path):
    body = b"factory-zip-bytes" * 100
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})
        )
    )

    archive = await download_to_file(
        client, "https://x/y.zip", tmp_path / "f.zip", expected_size=len(body)
    )

    assert archive.path.read_bytes() == body


async def test_a_missing_content_length_proceeds_when_a_size_is_expected(tmp_path):
    """Some sources send no `Content-Length`; the digest stays the final gate for those."""
    body = b"factory-zip-bytes" * 100
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
    )

    archive = await download_to_file(
        client, "https://x/y.zip", tmp_path / "f.zip", expected_size=len(body)
    )

    assert archive.path.read_bytes() == body


# --- the ack cookie goes to exactly one host ------------------------------------------------


async def test_the_terms_cookie_is_not_sent_to_the_download_host(tmp_path):
    """The terms wall is on the index only — a bare request for a factory zip answers 200,
    measured 2026-08-11 — and httpx does not strip a caller-supplied Cookie across a
    cross-origin redirect, so sending it here hands the acknowledgement to the CDN and to
    wherever the CDN points next."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"zip")

    driver = PixelDriver(
        make_settings(), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    ref = FirmwareRef(driver="pixel", device="comet", build="A.1", url="https://dl.example/y.zip")

    await driver.fetch(ref, tmp_path)

    assert len(seen) == 1
    assert "cookie" not in seen[0].headers


async def test_a_redirected_index_is_refused_rather_than_followed():
    """Following it would replay the acknowledgement cookie at whatever host answered."""
    driver = PixelDriver(
        make_settings(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(302, headers={"location": "https://evil.example/i"})
            )
        ),
    )

    with pytest.raises(FirmwareError) as excinfo:
        await driver.list_available()

    assert "evil.example" in str(excinfo.value)


# --- the shared driver client refuses a downgrading redirect --------------------------------


def _downgrading_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(302, headers={"location": "http://example.com/next"})


async def test_driver_client_refuses_a_downgrade_on_a_plain_get():
    client = driver_client(5.0, transport=httpx.MockTransport(_downgrading_handler))
    try:
        with pytest.raises(FirmwareRedirectError) as excinfo:
            await client.get("https://example.com/start")
        assert str(excinfo.value.url) == "http://example.com/next"
    finally:
        await client.aclose()


async def test_driver_client_refuses_a_downgrade_on_a_stream(tmp_path):
    """The download half: `download_to_file` streams over the shared client, and the refusal
    must surface as the named error — not be folded into `FirmwareDownloadError` — and leave no
    partial behind."""
    client = driver_client(5.0, transport=httpx.MockTransport(_downgrading_handler))
    try:
        with pytest.raises(FirmwareRedirectError) as excinfo:
            await download_to_file(client, "https://example.com/start", tmp_path / "f.zip")
        assert str(excinfo.value.url) == "http://example.com/next"
        assert list(tmp_path.iterdir()) == []
    finally:
        await client.aclose()


async def test_driver_client_still_follows_an_https_to_https_redirect(tmp_path):
    """Refusing the downgrade must not refuse the legitimate hop: firmware sources 3xx within
    https (Samsung's download authorisation, Xiaomi's CDN rewrite, an Oppo gate)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.com/archive.zip"})
        return httpx.Response(200, content=b"zip-bytes")

    client = driver_client(5.0, transport=httpx.MockTransport(handler))
    try:
        archive = await download_to_file(client, "https://example.com/start", tmp_path / "f.zip")
        assert archive.path.read_bytes() == b"zip-bytes"
    finally:
        await client.aclose()


class _RecordingTransport(httpx.AsyncBaseTransport):
    """A fake inner transport that records `aclose`, so the wrapper's delegation is observable."""

    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    async def aclose(self) -> None:
        self.closed = True


async def test_driver_client_does_not_refuse_a_direct_http_fetch():
    """The guard refuses a DOWNGRADE, not cleartext by fiat: an operator-configured http
    index/mirror is the first request through a fresh client and must pass — the operator chose
    that host. Only an http request after an https one (a downgrade) is refused."""
    client = driver_client(
        5.0, transport=httpx.MockTransport(lambda _r: httpx.Response(200, text="index"))
    )
    try:
        response = await client.get("http://example.com/index")
        assert response.status_code == 200
    finally:
        await client.aclose()


async def test_driver_client_close_reaches_the_inner_transport():
    """The wrapper's `aclose` must delegate, or the inner keep-alive pool leaks. The default
    `AsyncBaseTransport.aclose` is a no-op, so this pins the override rather than the base."""
    inner = _RecordingTransport()
    client = driver_client(5.0, transport=inner)
    await client.aclose()
    assert inner.closed is True


async def test_all_six_drivers_build_their_client_through_the_shared_factory(monkeypatch):
    """Every driver's `_open_client` routes through the shared factory. Replacing `driver_client`
    in every driver module with a recorder proves each `_open_client` calls it rather than
    constructing `httpx.AsyncClient` directly. This pins `_open_client` only — it does not watch
    a future driver building a client somewhere else, which stays a grep-level invariant."""
    real = driver_client
    seen: list[float] = []

    def recording(timeout: float):
        seen.append(timeout)
        return real(timeout, transport=httpx.MockTransport(lambda _r: httpx.Response(200)))

    for name in driver_names():
        monkeypatch.setattr(f"uadclaw.drivers.{name}.driver_client", recording)
    opened: list[httpx.AsyncClient] = []
    try:
        for name in driver_names():
            driver = get_driver(name, make_settings())
            client, owned = driver._open_client()
            assert owned is True
            opened.append(client)
    finally:
        for client in opened:
            await client.aclose()

    assert len(seen) == len(driver_names())
    assert set(seen) == {make_settings().firmware_http_timeout_seconds}
