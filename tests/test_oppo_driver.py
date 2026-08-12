"""The OPPO/OnePlus/realme driver: the catalogue index, the update protocol, and the fallback.

No network. Five real documents out of `tests/fixtures/`:

- `oppo_catalogue.json` — 7 rows copied byte for byte out of the catalogue's 179, carrying every
  shape the parser has to survive: all four OPlus gate hosts, a Xiaomi row and an OnePlus-NA row
  that belong to other drivers, and a model whose two rows are in the WRONG chronological order,
  because that is what `select_ref` gets wrong when the driver does not sort.
- `oppo_update_request.json` — the plaintext body and headers of the request `RMX3706` answered
  `200` to on 2026-08-12, `protectedKey` dropped because it is random per call. The request this
  module builds is compared against it field by field rather than against itself.
- `oppo_update_rmx3706.json`, `oppo_update_plk110.json` — two decrypted `200` documents from the
  same probe. One resolves to a signed CDN link, the other to another `/downloadCheck` gate,
  which is the whole reason the URL allowlist covers both.
- `oppo_update_rmx3706_chain.json` — the answer to the same phone asked at the build it already
  has. It is the only captured response whose `otaVersion` and `realOtaVersion` name different
  builds, so it is the only evidence for which of them identifies a package.

**The endpoint is mocked against a test-generated RSA keypair**, substituted for a region's real
public key, so the handler unwraps `protectedKey` with the private half exactly as the server
does and answers under the key it recovered. Nothing weaker can pin the wrapping: a mock that
was simply handed the session key could not tell the key's base64 TEXT from its 32 raw bytes,
and only one of those gets a `200` from the live endpoint.
"""

import base64
import hashlib
import json
import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uadclaw.drivers import oppo
from uadclaw.drivers.oppo import (
    DEVICE_ID,
    GATE_USER_ID,
    IMEI,
    REGIONS,
    OppoDriver,
    OppoProtocolError,
    build_update_request,
    decrypt_update_response,
    download_host_allowed,
    marketing_major,
    ota_version_branch,
    parse_catalogue,
    protected_key_header,
    resolved_package,
    synthesize_ota_version,
)
from uadclaw.firmware import (
    EmptyFirmwareIndexError,
    FirmwareDownloadError,
    FirmwareError,
    FirmwareInputError,
    FirmwareRef,
    TermsRisk,
    select_ref,
)
from uadclaw.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"

CN_GATE_HOST = "component-ota-cn.allawntech.com"
# The catalogue row this suite's fetch tests are written against.
PLK110_BUILD = "PLK110_11.A.72_0720_202607301131"
PLK110_GATE = f"https://{CN_GATE_HOST}/downloadCheck?c=deadbeef&id=6a212ab99e207b01827c4fee"
# Deliberately not the request's IV: a driver that decrypted the response under the IV it sent
# would pass every round-trip test that reused one.
RESPONSE_IV = bytes(range(16))


def make_settings(**overrides) -> Settings:
    base = {
        "postgres_password": "test-only-password",
        "auth_password": "test-only-admin-password",
        "session_secret": "test-only-session-secret",
    }
    return Settings(**(base | overrides))


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def catalogue() -> dict:
    return json.loads((FIXTURES / "oppo_catalogue.json").read_text(encoding="utf-8"))


def captured_request() -> dict:
    return json.loads((FIXTURES / "oppo_update_request.json").read_text(encoding="utf-8"))


def captured_response(name: str) -> dict:
    return json.loads((FIXTURES / f"oppo_update_{name}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _test_keypair() -> tuple[rsa.RSAPrivateKey, str]:
    """A 2048-bit keypair standing in for a region's, generated once for the whole module."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, base64.b64encode(public_der).decode("ascii")


def unwrap_session_key(header: str, private_key: rsa.RSAPrivateKey) -> bytes:
    """What the server does with `protectedKey`: unwrap, then base64-decode the text inside."""
    wrapped = base64.b64decode(json.loads(header)["SCENE_1"]["protectedKey"])
    text = private_key.decrypt(
        wrapped,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None
        ),
    )
    return base64.b64decode(text, validate=True)


def decrypt_request(content: bytes, key: bytes) -> dict:
    params = json.loads(json.loads(content)["params"])
    decryptor = Cipher(algorithms.AES(key), modes.CTR(base64.b64decode(params["iv"]))).decryptor()
    plain = decryptor.update(base64.b64decode(params["cipher"])) + decryptor.finalize()
    return json.loads(plain)


def seal_response(document: dict, key: bytes, *, response_code: int = 200) -> str:
    if response_code != 200:
        return json.dumps({"responseCode": response_code, "errMsg": None})
    encryptor = Cipher(algorithms.AES(key), modes.CTR(RESPONSE_IV)).encryptor()
    cipher = encryptor.update(json.dumps(document).encode("utf-8")) + encryptor.finalize()
    return json.dumps(
        {
            "responseCode": 200,
            "errMsg": None,
            "body": json.dumps(
                {
                    "cipher": base64.b64encode(cipher).decode("ascii"),
                    "iv": base64.b64encode(RESPONSE_IV).decode("ascii"),
                }
            ),
        }
    )


@pytest.fixture
def cn_region(monkeypatch) -> oppo.OppoRegion:
    """The CN region with this module's public key in place of OPlus's."""
    _private_key, public_key_b64 = _test_keypair()
    region = replace(REGIONS["CN"], public_key_b64=public_key_b64)
    monkeypatch.setitem(oppo.REGION_BY_GATE_HOST, CN_GATE_HOST, region)
    return region


def plk110_ref(md5: str) -> FirmwareRef:
    return FirmwareRef(
        driver="oppo",
        device="PLK110",
        build=PLK110_BUILD,
        url=PLK110_GATE,
        md5=md5,
        android_version="16",
    )


def endpoint_handler(
    region: oppo.OppoRegion,
    *,
    document: dict | None = None,
    response_code: int = 200,
    model: str = "PLK110",
    payload: bytes = b"",
    seen: list[httpx.Request] | None = None,
):
    private_key, _public = _test_keypair()

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if str(request.url) == region.endpoint:
            key = unwrap_session_key(request.headers["protectedKey"], private_key)
            assert len(key) == 32
            # The server reads the model out of the encrypted body, not out of the header.
            assert decrypt_request(request.content, key)["model"] == model
            return httpx.Response(
                200, text=seal_response(document or {}, key, response_code=response_code)
            )
        return httpx.Response(200, content=payload)

    return handler


# --- the protocol -------------------------------------------------------------------------


def test_each_region_pairs_the_update_endpoint_with_its_own_gate_host():
    # Four near-identical table entries, and the only thing that would surface a copy-paste
    # between two of them is that `component-otapc-X` and `component-ota-X` have to agree: the
    # gate host is what a `FirmwareRef` carries, and it is what picks the endpoint to ask.
    for region in REGIONS.values():
        suffix = region.endpoint.removeprefix("https://component-otapc-").split("/")[0]
        assert region.gate_host == f"component-ota-{suffix}"
        assert region.language in {"en-EN", "zh-CN"}
    assert set(oppo.REGION_BY_GATE_HOST) == {region.gate_host for region in REGIONS.values()}
    assert len(REGIONS) == 4


def test_the_device_id_is_the_sha256_of_a_null_imei_and_nothing_else():
    # Pinned to the value the endpoint answered 200 to, not recomputed from the same expression
    # this module uses: no device state reaches this protocol and this is what proves it.
    assert IMEI == "000000000000000"
    assert DEVICE_ID == "14BDCD6FD64180AF5E7791DF91B6AF8E9A3E7BC844997EB8C29252706DF97CA5"


def test_the_synthesized_ota_version_carries_the_tail_the_endpoint_resolves_on():
    # `_0000_000000000000` answered 2100 on a device that answered 200 to this one.
    assert synthesize_ota_version("RMX3706", "A") == "RMX3706_11.A.00_0001_100000000000"
    assert synthesize_ota_version("PKC110", "C") == "PKC110_11.C.00_0001_100000000000"


def test_the_branch_is_read_off_a_real_build_rather_than_guessed_over_four_letters():
    assert ota_version_branch("PLK110_11.A.72_0720_202607301131", model="PLK110") == "A"
    assert ota_version_branch("NE2210_11.J.94_4940_202606031902", model="NE2210") == "J"
    assert ota_version_branch("CPH2449_11.H.15_3150_202607172114", model="CPH2449") == "H"
    # Another model's build never yields a branch: it would synthesize a query about a phone
    # this ref does not name.
    assert ota_version_branch("PLK110_11.A.72_0720_202607301131", model="PKC110") is None
    # A Xiaomi row out of the same catalogue.
    assert (
        ota_version_branch("zorn_eea_global-ota_full-OS3.0.302.0-user-16.0", model="zorn") is None
    )


def test_the_android_major_comes_from_the_catalogues_marketing_version():
    assert marketing_major("CPH2655_15.0.0.832(EX01)") == "15"
    assert marketing_major("PLK110_16.0.1.310(CN01)") == "16"
    assert marketing_major("RMX3706_13.1.0.118(SP01CN01)") == "13"
    assert marketing_major("nonsense") is None


def test_the_request_this_module_builds_is_the_one_the_endpoint_answered_200_to():
    captured = captured_request()
    body = captured["plain_body"]
    _wire, headers = build_update_request(
        model="RMX3706",
        ota_version=body["otaVersion"],
        region=REGIONS["CN"],
        android_major="13",
        key=b"k" * 32,
        iv=b"v" * 16,
        now_ms=int(body["time"]),
    )
    sent_iv = base64.b64decode(json.loads(json.loads(_wire)["params"])["iv"])
    assert sent_iv == b"v" * 16
    rebuilt = decrypt_request(_wire.encode(), b"k" * 32)

    assert rebuilt == body
    assert {k: v for k, v in headers.items() if k != "protectedKey"} == captured["headers"]


def test_the_protected_key_header_carries_the_regions_negotiation_version_and_a_day_of_life():
    header = json.loads(protected_key_header(b"k" * 32, region=REGIONS["EU"], now_ms=1_000_000))
    scene = header["SCENE_1"]

    assert scene["negotiationVersion"] == "1615897067573"
    assert scene["version"] == str(1_000_000 + 86_400_000)
    # 2048-bit RSA: one 256-byte block, whatever the plaintext length.
    assert len(base64.b64decode(scene["protectedKey"])) == 256


def test_the_wrapped_key_is_the_base64_text_of_the_session_key(cn_region):
    private_key, _public = _test_keypair()
    key = bytes(range(32))

    header = protected_key_header(key, region=cn_region, now_ms=1_000_000)

    assert unwrap_session_key(header, private_key) == key


def test_a_request_is_refused_when_the_key_or_iv_is_the_wrong_length():
    with pytest.raises(FirmwareInputError):
        build_update_request(
            model="PLK110",
            ota_version="PLK110_11.A.00_0001_100000000000",
            region=REGIONS["CN"],
            android_major="16",
            key=b"k" * 16,
            iv=b"v" * 16,
            now_ms=1,
        )


def test_a_response_decrypts_under_the_requests_key_and_the_responses_own_iv():
    document = captured_response("plk110")

    code, decrypted = decrypt_update_response(seal_response(document, b"k" * 32), b"k" * 32)

    assert code == 200
    assert decrypted == document


def test_a_response_that_does_not_decrypt_is_a_protocol_error_not_a_silent_empty():
    sealed = seal_response(captured_response("plk110"), b"k" * 32)

    with pytest.raises(OppoProtocolError):
        decrypt_update_response(sealed, b"wrong-key-wrong-key-wrong-key-32"[:32])


def test_an_up_to_date_code_is_returned_rather_than_raised():
    assert decrypt_update_response('{"responseCode": 2004, "errMsg": null}', b"k" * 32) == (
        2004,
        None,
    )


@pytest.mark.parametrize(
    "text", ["<html>not json</html>", '{"errMsg": "nope"}', '{"responseCode": "200"}']
)
def test_a_malformed_envelope_is_a_protocol_error(text):
    with pytest.raises(OppoProtocolError):
        decrypt_update_response(text, b"k" * 32)


def test_the_resolved_package_is_read_out_of_the_captured_200_documents():
    cdn = resolved_package(captured_response("rmx3706"), model="RMX3706")
    gate = resolved_package(captured_response("plk110"), model="PLK110")

    assert cdn.ota_version == "RMX3706_11.A.48_0480_202311201335"
    assert cdn.size == 7549973765
    assert cdn.md5 == "e027328b9c1f81382ec4d48b9e802914"
    assert cdn.url.startswith("https://gauss-compotacostauto-cn.allawnfs.com/")
    assert gate.ota_version == "PLK110_11.A.68_0680_202606250030"
    assert gate.url.startswith(f"https://{CN_GATE_HOST}/downloadCheck?")


def test_a_build_is_identified_by_real_ota_version_where_the_two_names_disagree():
    # The captured answer to `RMX3706` at its real A.48: one 7,865,286,014-byte package that
    # `otaVersion` calls A.22 and `realOtaVersion` calls C.22. The rewrite keeps the branch
    # that was ASKED about, so trusting `otaVersion` is how a C-branch build gets filed under
    # an A-branch id — which is exactly the comparison `_resolve_direct` makes.
    document = captured_response("rmx3706_chain")

    package = resolved_package(document, model="RMX3706")

    assert document["otaVersion"] == "RMX3706_11.A.22_1220_202312062317"
    assert package.ota_version == "RMX3706_11.C.22_1220_202312062317"
    assert package.size == 7865286014


def test_a_response_whose_three_names_for_one_package_disagree_is_refused():
    document = captured_response("plk110")
    document["components"][0]["componentVersion"] = "PKC110_11.C.78_1780_202607102355.97.b0884cbb"

    with pytest.raises(OppoProtocolError) as excinfo:
        resolved_package(document, model="PLK110")

    assert "disagree" in str(excinfo.value)


def test_a_response_omitting_real_ota_version_still_resolves_on_the_other_name():
    # Two of the eight captured 200s carry no `realOtaVersion` at all, and their
    # `componentVersion` sides with `otaVersion`.
    document = captured_response("plk110")
    del document["realOtaVersion"]

    assert resolved_package(document, model="PLK110").ota_version == (
        "PLK110_11.A.68_0680_202606250030"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://gauss-compotacostauto-cn.allawnfs.com/x.zip",
        "https://notallawnfs.com/x.zip",
        "https://allawnfs.com.example.net/x.zip",
        "https://example.net/x.zip",
        "ftp://gauss-compotacostauto-cn.allawnfs.com/x.zip",
    ],
)
def test_a_package_url_off_the_allowlist_is_refused(url):
    assert download_host_allowed(url) is False
    document = captured_response("plk110")
    document["components"][0]["componentPackets"]["url"] = url

    with pytest.raises(OppoProtocolError) as excinfo:
        resolved_package(document, model="PLK110")

    assert "allawn" in str(excinfo.value)


def test_the_allowlist_admits_both_shapes_the_endpoint_actually_answers():
    assert download_host_allowed("https://gauss-compotacostauto-cn.allawnfs.com/a.zip") is True
    assert download_host_allowed("https://gauss-componentotamanual.allawnofs.com/a.zip") is True
    assert download_host_allowed(f"https://{CN_GATE_HOST}/downloadCheck?c=1") is True


def test_a_resolved_build_belonging_to_another_model_is_refused():
    other = "PKC110_11.C.78_1780_202607102355"
    document = captured_response("plk110") | {"otaVersion": other, "realOtaVersion": other}
    document["components"][0]["componentVersion"] = f"{other}.97.b0884cbb"

    with pytest.raises(OppoProtocolError) as excinfo:
        resolved_package(document, model="PLK110")

    assert "PLK110" in str(excinfo.value)


def test_a_200_carrying_no_component_is_refused_rather_than_read_as_nothing_to_download():
    with pytest.raises(OppoProtocolError):
        resolved_package({"otaVersion": PLK110_BUILD, "components": []}, model="PLK110")


# --- the catalogue ------------------------------------------------------------------------


def test_only_the_oplus_rows_of_a_multi_oem_catalogue_become_refs():
    refs = parse_catalogue(catalogue(), source_url="https://example.invalid/ota")

    # 7 fixture rows, of which the Xiaomi (`ZORN`, sgp-api.buy.mi.com) and the OnePlus-NA row
    # (`CPH2655`, android.googleapis.com) are another driver's.
    assert len(refs) == 5
    assert {ref.device for ref in refs} == {"CPH2653", "PLK110", "OPD2504", "CPH2659"}
    assert all(ref.driver == "oppo" for ref in refs)


def test_every_ref_carries_the_durable_gate_and_the_published_digest():
    refs = parse_catalogue(catalogue(), source_url="https://example.invalid/ota")

    for ref in refs:
        assert ref.url.startswith("https://component-ota-")
        assert "/downloadCheck?" in ref.url
        assert ref.md5 is not None
        assert ref.android_version == "16"
        assert ref.archive_suffix == ".zip"


def test_the_newest_build_for_a_model_is_the_catalogues_clock_not_its_row_order():
    refs = parse_catalogue(catalogue(), source_url="https://example.invalid/ota")

    # OPD2504 is published twice and the catalogue lists the OLDER row last (GLO 2026-05-21
    # after EU 2026-06-05). `select_ref` reads "newest" as the last row for a device whenever
    # the build id carries no date, and no OPlus build id does.
    assert select_ref(refs, device="OPD2504").build == "OPD2504_11.A.32_0320_202606052024"
    raw = [row["ota_version"] for row in catalogue()["releases"] if row["model"] == "OPD2504"]
    assert raw[-1] == "OPD2504_11.A.31_0310_202605211741"


def test_a_named_build_still_resolves_after_the_sort():
    refs = parse_catalogue(catalogue(), source_url="https://example.invalid/ota")

    assert select_ref(refs, device="PLK110", build=PLK110_BUILD).url.startswith(
        f"https://{CN_GATE_HOST}/"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"releases": []},
        {"releases": "not a list"},
    ],
)
def test_a_catalogue_with_no_releases_is_an_error_not_an_empty_index(payload):
    with pytest.raises(EmptyFirmwareIndexError):
        parse_catalogue(payload, source_url="https://example.invalid/ota")


def test_a_catalogue_carrying_only_other_oems_is_an_error_naming_that():
    foreign = catalogue()
    foreign["releases"] = [
        row
        for row in foreign["releases"]
        if not row["source_url"].startswith("https://component-ota-")
    ]

    with pytest.raises(EmptyFirmwareIndexError) as excinfo:
        parse_catalogue(foreign, source_url="https://example.invalid/ota")

    assert "another OEM" in str(excinfo.value)


def test_a_row_that_cannot_become_a_ref_is_skipped_with_a_warning_not_a_dead_index(caplog):
    payload = catalogue()
    # 65 characters, one past what `FirmwareRef.build` accepts, on an otherwise valid OPlus row.
    payload["releases"][2]["ota_version"] = "PLK110_11.A.72_0720_202607301131" + "0" * 33

    with caplog.at_level(logging.WARNING, logger="uadclaw.drivers.oppo"):
        refs = parse_catalogue(payload, source_url="https://example.invalid/ota")

    assert {ref.device for ref in refs} == {"CPH2653", "OPD2504", "CPH2659"}
    assert "PLK110" in caplog.text


# --- the driver ---------------------------------------------------------------------------


def test_the_terms_say_reverse_engineered_and_name_both_sources():
    driver = OppoDriver(make_settings())
    terms = driver.terms()

    assert terms.risk is TermsRisk.REVERSE_ENGINEERED
    assert terms.acknowledged is True
    assert "roms.danielspringer.at" in terms.summary


async def test_the_catalogue_is_read_once_however_many_times_it_is_asked_for():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text=json.dumps(catalogue()))

    driver = OppoDriver(make_settings(), client=mock_client(handler))
    first = await driver.list_available()
    second = await driver.list_available()

    # Not tidiness: this catalogue escalates scraping to an IP-level block of its whole domain.
    assert len(calls) == 1
    assert [ref.build for ref in first] == [ref.build for ref in second]


@pytest.mark.parametrize("status", [403, 500])
async def test_a_catalogue_that_refuses_is_an_error_rather_than_an_empty_list(status):
    driver = OppoDriver(
        make_settings(), client=mock_client(lambda request: httpx.Response(status, text="no"))
    )

    with pytest.raises(FirmwareError) as excinfo:
        await driver.list_available()

    assert str(status) in str(excinfo.value)


async def test_fetch_downloads_the_endpoints_url_when_it_offers_exactly_this_build(
    cn_region, tmp_path
):
    payload = b"PK\x03\x04" + b"oppo-firmware" * 64
    # The captured PLK110 document with the build, digest and URL the catalogue row carries:
    # the endpoint trailed the catalogue on every model measured, so agreement is the case a
    # capture cannot supply.
    document = captured_response("plk110")
    document["otaVersion"] = document["realOtaVersion"] = PLK110_BUILD
    document["components"][0]["componentVersion"] = f"{PLK110_BUILD}.97.3d82867d"
    document["components"][0]["componentPackets"] = {
        "url": "https://gauss-compotacostauto-cn.allawnfs.com/component-ota/a.zip",
        "md5": hashlib.md5(payload).hexdigest(),
        "size": str(len(payload)),
    }
    seen: list[httpx.Request] = []
    driver = OppoDriver(
        make_settings(),
        client=mock_client(
            endpoint_handler(cn_region, document=document, payload=payload, seen=seen)
        ),
    )

    archive = await driver.fetch(plk110_ref(hashlib.md5(payload).hexdigest()), tmp_path)

    assert archive.path.name == f"oppo-PLK110-{PLK110_BUILD}.zip"
    assert archive.integrity_verified is True
    assert archive.path.read_bytes() == payload
    downloads = [request for request in seen if request.method == "GET"]
    assert (
        str(downloads[-1].url)
        == "https://gauss-compotacostauto-cn.allawnfs.com/component-ota/a.zip"
    )
    assert downloads[-1].headers["userId"] == GATE_USER_ID


async def test_fetch_falls_back_to_the_stored_gate_when_the_endpoint_answers_2004(
    cn_region, tmp_path, caplog
):
    payload = b"PK\x03\x04" + b"catalogue-served" * 64
    seen: list[httpx.Request] = []
    handler = endpoint_handler(
        cn_region,
        response_code=2004,
        payload=payload,
        seen=seen,
    )
    driver = OppoDriver(make_settings(), client=mock_client(handler))

    with caplog.at_level(logging.WARNING, logger="uadclaw.drivers.oppo"):
        archive = await driver.fetch(plk110_ref(hashlib.md5(payload).hexdigest()), tmp_path)

    assert archive.path.read_bytes() == payload
    assert str([r for r in seen if r.method == "GET"][-1].url) == PLK110_GATE
    # The observable: which models take this path is the only sign the `2004` hole moved.
    assert "PLK110" in caplog.text
    assert "2004" in caplog.text


async def test_fetch_falls_back_rather_than_filing_another_build_under_this_ones_id(
    cn_region, tmp_path, caplog
):
    payload = b"PK\x03\x04" + b"catalogue-served" * 64
    seen: list[httpx.Request] = []
    # The captured document as it really came back: A.68, four builds behind the catalogue's A.72.
    handler = endpoint_handler(
        cn_region, document=captured_response("plk110"), payload=payload, seen=seen
    )
    driver = OppoDriver(make_settings(), client=mock_client(handler))

    with caplog.at_level(logging.INFO, logger="uadclaw.drivers.oppo"):
        archive = await driver.fetch(plk110_ref(hashlib.md5(payload).hexdigest()), tmp_path)

    assert str([r for r in seen if r.method == "GET"][-1].url) == PLK110_GATE
    assert archive.path.read_bytes() == payload
    assert "PLK110_11.A.68_0680_202606250030" in caplog.text


async def test_an_oversized_catalogue_is_refused_before_it_is_parsed():
    # ~170 KB of real catalogue, and a worker with no memory bound of its own on the other end.
    oversized = b'{"releases": [' + b"x" * (oppo.MAX_CATALOGUE_BYTES + 1)
    driver = OppoDriver(
        make_settings(),
        client=mock_client(lambda request: httpx.Response(200, content=oversized)),
    )

    with pytest.raises(OppoProtocolError) as excinfo:
        await driver.list_available()

    assert str(oppo.MAX_CATALOGUE_BYTES) in str(excinfo.value)


async def test_an_oversized_update_response_is_refused_before_anything_decrypts_it(
    cn_region, tmp_path
):
    # The measured bodies are 3.4-13.3 KB. A response past the ceiling is a wedged or hostile
    # endpoint, and it has to stop here rather than at whatever tries to parse it.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (oppo.MAX_RESPONSE_BYTES + 1))

    driver = OppoDriver(make_settings(), client=mock_client(handler))

    with pytest.raises(OppoProtocolError) as excinfo:
        await driver.fetch(plk110_ref("0" * 32), tmp_path)

    assert str(oppo.MAX_RESPONSE_BYTES) in str(excinfo.value)


async def test_a_ref_whose_url_is_not_an_oplus_gate_never_reaches_the_update_endpoint(tmp_path):
    payload = b"PK\x03\x04" + b"elsewhere" * 64
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=payload)

    ref = FirmwareRef(
        driver="oppo",
        device="CPH2655",
        build="CPH2655_11.C.65_1651_202507071509",
        url="https://android.googleapis.com/packages/ota-api/package/a.zip",
        md5=hashlib.md5(payload).hexdigest(),
    )
    driver = OppoDriver(make_settings(), client=mock_client(handler))

    await driver.fetch(ref, tmp_path)

    assert [request.method for request in seen] == ["GET"]


async def test_the_gates_anti_leech_body_fails_on_the_digest_rather_than_landing_as_an_archive(
    cn_region, tmp_path
):
    # What the gate answers without the `userId` header: HTTP 200, a 21-byte JSON body, and no
    # error anywhere. Only the published digest turns that into a failure.
    refusal = b'{"responseCode":2306}'
    handler = endpoint_handler(cn_region, response_code=2004, payload=refusal)
    driver = OppoDriver(make_settings(), client=mock_client(handler))

    with pytest.raises(FirmwareDownloadError) as excinfo:
        await driver.fetch(plk110_ref("0" * 32), tmp_path)

    assert "md5" in str(excinfo.value)
    assert not list(tmp_path.iterdir())


async def test_fetch_refuses_a_ref_belonging_to_another_driver(tmp_path):
    driver = OppoDriver(make_settings())
    ref = replace_driver(plk110_ref("0" * 32), "xiaomi")

    with pytest.raises(FirmwareInputError):
        await driver.fetch(ref, tmp_path)


def replace_driver(ref: FirmwareRef, driver: str) -> FirmwareRef:
    return FirmwareRef(**(ref.model_dump() | {"driver": driver}))
