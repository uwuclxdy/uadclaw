"""OPPO, OnePlus and realme firmware: one catalogue for what exists, OPlus's own component-OTA
endpoint to turn one of those into bytes.

Three brands and one driver, because they share one OTA infrastructure: the same request shape,
the same four regional endpoints and the same `/downloadCheck` gate answer for all three, and a
per-brand driver would be three copies of this protocol.

**Two sources, and which one answers which question is the whole design.**

`roms.danielspringer.at/api/ota.php?latest=1` is the INDEX: it publishes one row per model and
region carrying `ota_version`, `md5`, `size`, `build_timestamp` and a `source_url` that is a
durable `/downloadCheck` gate. The OPlus endpoint is the RESOLVER: given a model it hands back
a package and a fresh URL. Neither can do the other's job, for two measured reasons.

**The endpoint answers the next build in the chain, never "the newest".** `RMX3706` asked at
its real `A.48` answered `C.22` on ColorOS 14 rather than anything on the A branch, and asked
at a synthesized `A.00` it answered `A.48` — so what comes back depends on where the question
started, which is exactly the trap `firmware.select_ref` documents from the other direction.
That answer is also where `resolved_package` gets its rule about which field names a build.
Worse, the endpoint is BEHIND the catalogue on every model where both were measured on
2026-08-12: `PLK110` A.68 against the catalogue's A.72, `PKC110` C.78 against C.79,
`RMX5010` F.61 against F.66. Walking the chain does not close that gap either: `PLK110` asked at
the endpoint's own A.68 answers `2004`, a fixed point four builds short of an A.72 whose gate
serves 9,146,672,362 bytes of zip. So "newest" is the catalogue's `build_timestamp`, and this
driver sorts its index oldest-first so `select_ref`'s row-order fallback lands on it — no OPlus
`ota_version` carries a date `firmware.build_date` can read.

**`2004` does not mean "up to date".** That same `PLK110` pair is the counterexample: `2004` at
A.68 while A.72 exists and downloads. What is proven is that five modern export models
(`CPH2797`, `CPH2659`, `CPH2653`, `RMX5011`, `RMX5210`) answer `2004` to every `otaVersion`
tried, real or synthesized, across EU, GL and IN, while older export models resolve fine on the
same endpoints. The catalogue's gates cover exactly those models, so the fallback is
load-bearing rather than a nicety — and it is never silent: falling back logs the model, the
region and the code, because a change in WHICH models take that path is the only observable that
would say the hole moved.

**The catalogue is multi-OEM and only some of it is this driver's.** Measured 2026-08-12 over the
whole 179-row response: 107 rows are OPlus `/downloadCheck` gates on the four `allawn*` hosts, 70
are Xiaomi (`sgp-api.buy.mi.com`, HyperOS build ids under codenames like `ZORN`) and 2 are
OnePlus NA packages hosted on `android.googleapis.com` with no digest published. `REGIONS` is
therefore both the region table and the row filter: a row whose gate host is not one of the four
is not this driver's, and admitting the Xiaomi rows here would file a Xiaomi build under
`driver="oppo"` and double-count that phone in `package_facts.device_count`.

**The digest a row publishes describes the zip as served.** `size` matched the `Content-Range`
total to the byte on both rows it was measured against (`PLK110`/CN 9,146,672,362 and
`CPH2653`/EU 8,304,912,951, both opening `50 4b 03 04`), and `md5` sits beside it in the same
`componentPackets` object the endpoint answers with, so the download is verified in md5 rather
than landing integrity-unverified. Only a full transfer can prove the digest itself; what this
buys today is that a gate answering its 30-byte `2306` refusal body fails loudly instead of
being saved as an archive.

**The protocol.** Request version 2 (ColorOS/RUI 2 and later) posts one AES-256-CTR blob to
`component-otapc-{sg,cn,in,eu}.allawn{os,tech}.com/update/v3`; the random key is RSA-OAEP-wrapped
under a per-region public key and travels in a `protectedKey` header, and the response body comes
back under the same key with its own IV. There is **no IMEI, no account and no device state**:
`imei` is fifteen zeroes and `deviceId` is its sha256, which is what makes a pipeline owning no
phone able to ask at all. The `otaVersion` is synthesized as `{model}_11.{branch}.00_0001_
100000000000` — the tail is load-bearing, `_0000_000000000000` answered `2100` on a device that
answered `200` to this one — and the branch letter is read off the catalogue's own `ota_version`
rather than enumerated, which is the difference between one request and four.

The download URL comes back as either a signed CDN link or another `/downloadCheck` gate. The
gate wants a `userId: oplus-ota|<anything>` header and validates only that prefix; without it it
answers `200 {"responseCode":2306}`, which is a body a downloader would otherwise save as an
archive. The signed link expires in about ten minutes and the gate does not, so what this driver
stores in a `FirmwareRef` is always the gate.

Written from the protocol and from captured request/response pairs. `R0rt1z2/realme-ota` is
GPL-3.0 and was read to understand the wire format; no code from it is here. The four public
keys, the negotiation versions and the `nvCarrier` values are OPlus's own protocol constants, the
same standing `samsung.AUTH_AES_KEY` has in this package.
"""

import base64
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uadclaw.firmware import (
    DownloadedArchive,
    EmptyFirmwareIndexError,
    FirmwareDriver,
    FirmwareError,
    FirmwareInputError,
    FirmwareRef,
    TermsPosture,
    TermsRisk,
    download_to_file,
)
from uadclaw.settings import Settings

logger = logging.getLogger(__name__)


class OppoProtocolError(FirmwareError):
    """The OPlus endpoint answered something this driver cannot read: a non-JSON envelope, a
    body that does not decrypt, or a resolved package pointing somewhere it may not follow.
    Its own type so a protocol change is distinguishable from a transport failure."""


# The gate's anti-leech check. Only the `oplus-ota|` prefix is validated (measured 2026-08-12:
# the same URL answers 302 with this header and `200 {"responseCode":2306}` without it), so the
# suffix is a client id this pipeline has no reason to spoof precisely.
GATE_USER_ID = "oplus-ota|16001011"

# No device state anywhere in the request. `deviceId` is the sha256 of that IMEI, uppercased,
# which is what the reference client sends when it has no device to ask.
IMEI = "0" * 15
DEVICE_ID = hashlib.sha256(IMEI.encode("ascii")).hexdigest().upper()

_AES_KEY_BYTES = 32
_AES_IV_BYTES = 16
# The `protectedKey` header carries an expiry a day out, as the reference client sets it.
_PROTECTED_KEY_TTL_MS = 86_400_000

# Ceilings on what a third-party server may make this worker hold in memory, enforced as the
# body ARRIVES rather than after `response.text` has already buffered whatever was sent. The
# measured update responses are 3.4-13.3 KB and the whole catalogue is ~170 KB, so both carry
# two orders of magnitude of headroom.
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CATALOGUE_BYTES = 16 * 1024 * 1024
_RESPONSE_CHUNK_BYTES = 64 * 1024

# The only code that means "here is a package". Everything else falls back to the gate.
RESPONSE_OK = 200


@dataclass(frozen=True, slots=True)
class OppoRegion:
    """One regional endpoint and everything that differs with it. `gate_host` is the host that
    region's `/downloadCheck` gates live on, which is how a catalogue row is attributed to a
    region without trusting the row's own `region` string."""

    name: str
    endpoint: str
    gate_host: str
    public_key_b64: str
    negotiation_version: str
    nv_carrier: str
    language: str


# Keyed by the `uRegion`/`trackRegion` value each one carries. `component-ota*.coloros.com` is
# NOT in here on purpose: that host set belongs to the OLD request-version-1 protocol, and
# probing it answers 404 in a way that reads as "the modern path moved".
REGIONS: dict[str, OppoRegion] = {
    "GL": OppoRegion(
        name="GL",
        endpoint="https://component-otapc-sg.allawnos.com/update/v3",
        gate_host="component-ota-sg.allawnos.com",
        public_key_b64=(
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAkA980wxi+eTGcFDiw2I6RrUeO4jL/Aj3Yw4dNuW7"
            "tYt+O1sRTHgrzxPD9SrOqzz7G0KgoSfdFHe3JVLPN+U1waK+T0HfLusVJshDaMrMiQFDUiKajb+QKr+bXQhV"
            "ofH74fjat+oRJ8vjXARSpFk4/41x5j1Bt/2bHoqtdGPcUizZ4whMwzap+hzVlZgs7BNfepo24PWPRujsN3uo"
            "pl+8u4HFpQDlQl7GdqDYDj2zNOHdFQI2UpSf0aIeKCKOpSKF72KDEESpJVQsqO4nxMwEi2jMujQeCHyTCjBZ"
            "+W35RzwT9+0pyZv8FB3c7FYY9FdF/+lvfax5mvFEBd9jO+dpMQIDAQAB"
        ),
        negotiation_version="1615895993238",
        nv_carrier="00011011",
        language="en-EN",
    ),
    "CN": OppoRegion(
        name="CN",
        endpoint="https://component-otapc-cn.allawntech.com/update/v3",
        gate_host="component-ota-cn.allawntech.com",
        public_key_b64=(
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEApXYGXQpNL7gmMzzvajHaoZIHQQvBc2cOEhJc7/ts"
            "aO4sT0unoQnwQKfNQCuv7qC1Nu32eCLuewe9LSYhDXr9KSBWjOcCFXVXteLO9WCaAh5hwnUoP/5/Wz0jJwBA"
            "+yqs3AaGLA9wJ0+B2lB1vLE4FZNE7exUfwUc03fJxHG9nCLKjIZlrnAAHjRCd8mpnADwfkCEIPIGhnwq7pdk"
            "bamZcoZfZud1+fPsELviB9u447C6bKnTU4AaMcR9Y2/uI6TJUTcgyCp+ilgU0JxemrSIPFk3jbCbzamQ6Shk"
            "w/jDRzYoXpBRg/2QDkbq+j3ljInu0RHDfOeXf3VBfHSnQ66HCwIDAQAB"
        ),
        negotiation_version="1615879139745",
        nv_carrier="10010111",
        language="zh-CN",
    ),
    "IN": OppoRegion(
        name="IN",
        endpoint="https://component-otapc-in.allawnos.com/update/v3",
        gate_host="component-ota-in.allawnos.com",
        public_key_b64=(
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwYtghkzeStC9YvAwOQmWylbp74Tj8hhi3f9IlK7A"
            "/CWrGbLgzz/BeKxNb45zBN8pgaaEOwAJ1qZQV5G4nProWCPOP1ro1PkemFJvw/vzOOT5uN0ADnHDzZkZXCU/"
            "knxqUSfLcwQlHXsYhNsAm7uOKjY9YXF4zWzYN0eFPkML3Pj/zg7hl/ov9clB2VeyI1/blMHFfcNA/fvqDTEN"
            "XcNBIhgJvXiCpLcZqp+aLZPC5AwY/sCb3j5jTWer0Rk0ZjQBZE1AncwYvUx4mA65U59cWpTyl4c47J29MsQ6"
            "6hqWv6eBHlDNZSEsQpHePUqgsf7lmO5Wd7teB8ugQki2oz1Y5QIDAQAB"
        ),
        negotiation_version="1615896309308",
        nv_carrier="00011011",
        language="en-EN",
    ),
    "EU": OppoRegion(
        name="EU",
        endpoint="https://component-otapc-eu.allawnos.com/update/v3",
        gate_host="component-ota-eu.allawnos.com",
        public_key_b64=(
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAh8/EThsK3f0WyyPgrtXb/D0Xni6UZNppaQHUqHWo"
            "976cybl92VxmehE0ISObnxERaOtrlYmTPIxkVC9MMueDvTwZ1l0KxevZVKU0sJRxNR9AFcw6D7k9fPzzpNJm"
            "hSlhpNbt3BEepdgibdRZbacF3NWy3ejOYWHgxC+I/Vj1v7QU5gD+1OhgWeRDcwuV4nGY1ln2lvkRj8EiJYXf"
            "kSq/wUI5AvPdNXdEqwou4FBcf6mD84G8pKDyNTQwwuk9lvFlcq4mRqgYaFg9DAgpDgqVK4NTJWM7tQS1GZuR"
            "A6PhupfDqnQExyBFhzCefHkEhcFywNyxlPe953NWLFWwbGvFKwIDAQAB"
        ),
        negotiation_version="1615897067573",
        nv_carrier="01000100",
        language="en-EN",
    ),
}

REGION_BY_GATE_HOST: dict[str, OppoRegion] = {
    region.gate_host: region for region in REGIONS.values()
}

# Where a resolved package is allowed to live. The response is external input that ends up as a
# download URL, so it is checked against the registrable domains OPlus serves firmware from —
# the two CDNs a signed link lands on plus the four gate hosts. Matched as `host == domain` or
# `host.endswith("." + domain)`, never as a bare suffix: `notallawnfs.com` must not pass.
ALLOWED_DOWNLOAD_DOMAINS = frozenset(
    {"allawnfs.com", "allawnofs.com", "allawnos.com", "allawntech.com"}
)

# `PLK110_11.A.72_0720_202607301131`. All 107 OPlus catalogue rows match, the major field is
# `11` on every one of them, and the branch letter runs over A, C, F, H and J — which is why the
# branch is read here rather than enumerated over the four letters a client without an index has
# to guess between.
_OTA_VERSION_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9]{2,32})_(?P<major>\d{1,3})\.(?P<branch>[A-Z0-9]{1,4})\."
    r"(?P<minor>\d{1,4})_(?P<code>\d{1,8})_(?P<stamp>\d{6,20})$"
)
# The leading integer of the catalogue's marketing version (`CPH2655_15.0.0.832(EX01)` -> 15).
_MARKETING_MAJOR_RE = re.compile(r"^[A-Za-z0-9]{2,32}_(?P<major>\d{1,3})\.")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def synthesize_ota_version(model: str, branch: str) -> str:
    """The `otaVersion` that makes the endpoint offer a FULL package rather than an incremental.

    `_0001_100000000000` is not decoration: the same request with `_0000_000000000000` answered
    `2100` on a device that answered `200` to this one (measured 2026-08-12, `RMX3301`/GL).
    """
    return f"{model}_11.{branch}.00_0001_100000000000"


def ota_version_branch(ota_version: str, *, model: str) -> str | None:
    """The branch letter of a real `ota_version`, or None when it is not one of this shape or
    does not belong to `model`."""
    match = _OTA_VERSION_RE.match(ota_version)
    if match is None or match.group("model") != model:
        return None
    return match.group("branch")


def marketing_major(version: str) -> str | None:
    """The Android major the request should claim, from the catalogue's marketing version.

    ColorOS and OxygenOS majors have tracked the Android major since ColorOS 11 (`ColorOS13` on
    the `RMX3706` that answered Android 13, `ColorOS16` on every current catalogue row), and the
    endpoint was never observed to reject a request over these two fields — they were never
    varied — so this is a plausible value rather than a proven-load-bearing one.
    """
    match = _MARKETING_MAJOR_RE.match(version)
    if match is None:
        return None
    return match.group("major")


def download_host_allowed(url: str) -> bool:
    """Whether a URL out of a decrypted response may be handed to the downloader."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in ALLOWED_DOWNLOAD_DOMAINS)


@dataclass(frozen=True, slots=True)
class ResolvedPackage:
    """What the endpoint offered: the build it named, where to get it, and what it says that
    should hash to."""

    ota_version: str
    url: str
    md5: str | None
    size: int | None


def build_update_request(
    *,
    model: str,
    ota_version: str,
    region: OppoRegion,
    android_major: str,
    key: bytes,
    iv: bytes,
    now_ms: int,
) -> tuple[str, dict[str, str]]:
    """`(wire body, headers)` for one update query. The key and IV are arguments rather than
    generated here so a test can pin the exact bytes this produces."""
    if len(key) != _AES_KEY_BYTES or len(iv) != _AES_IV_BYTES:
        raise FirmwareInputError(
            f"build_update_request: AES-256-CTR needs a {_AES_KEY_BYTES}-byte key and a "
            f"{_AES_IV_BYTES}-byte IV, got {len(key)} and {len(iv)}"
        )
    prefix = "_".join(ota_version.split("_")[:2])
    body = {
        "language": region.language,
        "romVersion": prefix,
        "otaVersion": ota_version,
        "androidVersion": f"Android{android_major}.0",
        "colorOSVersion": f"ColorOS{android_major}",
        "model": model,
        "productName": model,
        "operator": "unknown",
        "uRegion": region.name,
        "trackRegion": region.name,
        "imei": IMEI,
        "imei1": IMEI,
        "mode": "0",
        "registrationId": "unknown",
        "deviceId": DEVICE_ID,
        "version": "3",
        "type": "1",
        "otaPrefix": prefix,
        "isRealme": "1" if model.startswith("RMX") else "0",
        "time": str(now_ms),
        "canCheckSelf": "0",
    }
    encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    cipher = encryptor.update(json.dumps(body).encode("utf-8")) + encryptor.finalize()
    wire = json.dumps(
        {
            "params": json.dumps(
                {
                    "cipher": base64.b64encode(cipher).decode("ascii"),
                    "iv": base64.b64encode(iv).decode("ascii"),
                }
            )
        }
    )
    headers = {
        "language": region.language,
        "romVersion": prefix,
        "otaVersion": ota_version,
        "androidVersion": f"Android{android_major}.0",
        "colorOSVersion": f"ColorOS{android_major}",
        "model": model,
        "infVersion": "1",
        "operator": "unknown",
        "nvCarrier": region.nv_carrier,
        "uRegion": region.name,
        "trackRegion": region.name,
        "imei": IMEI,
        "imei1": IMEI,
        "deviceId": DEVICE_ID,
        "mode": "client_auto",
        "channel": "pc",
        "version": "2",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "NULL",
        "protectedKey": protected_key_header(key, region=region, now_ms=now_ms),
    }
    return wire, headers


def protected_key_header(key: bytes, *, region: OppoRegion, now_ms: int) -> str:
    """The `protectedKey` header: the session key RSA-OAEP-wrapped under the region's public key.

    What is wrapped is the key's BASE64 TEXT, not its 32 raw bytes — 44 ASCII characters — which
    is what the endpoint unwraps and what the response body is then encrypted under.
    """
    public_key = serialization.load_der_public_key(base64.b64decode(region.public_key_b64))
    wrapped = public_key.encrypt(  # type: ignore[union-attr]
        base64.b64encode(key),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None
        ),
    )
    return json.dumps(
        {
            "SCENE_1": {
                "protectedKey": base64.b64encode(wrapped).decode("ascii"),
                "version": str(now_ms + _PROTECTED_KEY_TTL_MS),
                "negotiationVersion": region.negotiation_version,
            }
        }
    )


def decrypt_update_response(text: str, key: bytes) -> tuple[int, dict[str, Any] | None]:
    """`(responseCode, decrypted document or None)`.

    A non-200 code is returned rather than raised: `2004` is a normal answer this driver acts on,
    and only a malformed envelope is a protocol failure.
    """
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OppoProtocolError(
            f"OppoDriver: the update endpoint answered non-JSON ({text[:200]!r}); the endpoint "
            "may have moved or be serving an error page"
        ) from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("responseCode"), int):
        raise OppoProtocolError(
            f"OppoDriver: the update endpoint answered no integer `responseCode` "
            f"({text[:200]!r}); the envelope shape changed"
        )
    code = int(envelope["responseCode"])
    body = envelope.get("body")
    if code != RESPONSE_OK or not isinstance(body, str):
        return code, None
    try:
        sealed = json.loads(body)
        decryptor = Cipher(
            algorithms.AES(key), modes.CTR(base64.b64decode(sealed["iv"], validate=True))
        ).decryptor()
        plain = decryptor.update(base64.b64decode(sealed["cipher"], validate=True))
        plain += decryptor.finalize()
        document = json.loads(plain)
    except (KeyError, TypeError, ValueError) as exc:
        raise OppoProtocolError(
            "OppoDriver: the update endpoint answered 200 with a body this driver could not "
            f"decrypt ({body[:120]!r}); the key wrapping or the body format changed"
        ) from exc
    if not isinstance(document, dict):
        raise OppoProtocolError(
            f"OppoDriver: a decrypted update body is not an object ({str(document)[:120]})"
        )
    return code, document


def resolved_package(document: dict[str, Any], *, model: str) -> ResolvedPackage:
    """The one package a `200` offered, validated. Every field here is external input that
    reaches a downloader or a filename, so none of it is taken on trust.

    **The identity is `realOtaVersion`, not `otaVersion`.** They disagree: `RMX3706` asked at
    `A.48` answered `otaVersion: RMX3706_11.A.22` and `realOtaVersion: RMX3706_11.C.22` for one
    7,865,286,014-byte package, and the component's own id sides with the second. The rewrite
    keeps the branch letter of the build that was ASKED about, which makes `otaVersion` the one
    field that can spell a package as something it is not — precisely the field a caller
    comparing "did I get the build I named" would reach for. Measured across 8 captured 200s:
    5 agree, 1 disagrees this way, 2 omit `realOtaVersion` and side with `otaVersion`. The
    third spelling, `componentVersion`, agrees with the chosen identity in all 8 and is checked
    here, so a response whose three names disagree is a protocol change rather than a download.
    """
    components = document.get("components")
    if not isinstance(components, list) or not components:
        raise OppoProtocolError(
            f"OppoDriver: a 200 update response for {model} carries no components; a resolved "
            "answer always carries exactly one"
        )
    component = components[0] if isinstance(components[0], dict) else {}
    packets = component.get("componentPackets")
    if not isinstance(packets, dict):
        raise OppoProtocolError(
            f"OppoDriver: the resolved component for {model} carries no `componentPackets`"
        )
    ota_version = document.get("realOtaVersion")
    if not isinstance(ota_version, str):
        ota_version = document.get("otaVersion")
    if not isinstance(ota_version, str) or ota_version_branch(ota_version, model=model) is None:
        raise OppoProtocolError(
            f"OppoDriver: a 200 update response for {model} named build {ota_version!r}, which "
            "is not an ota_version of this model; refusing to file it under this device"
        )
    component_version = component.get("componentVersion")
    if not isinstance(component_version, str) or not component_version.startswith(
        f"{ota_version}."
    ):
        raise OppoProtocolError(
            f"OppoDriver: the component offered for {ota_version} calls itself "
            f"{str(component_version)[:120]!r}; the response's names for one package disagree"
        )
    url = packets.get("url")
    if not isinstance(url, str) or not download_host_allowed(url):
        raise OppoProtocolError(
            f"OppoDriver: the package offered for {ota_version} points at {str(url)[:120]!r}, "
            f"which is not an https URL on {sorted(ALLOWED_DOWNLOAD_DOMAINS)}"
        )
    md5 = packets.get("md5")
    size = packets.get("size")
    return ResolvedPackage(
        ota_version=ota_version,
        url=url,
        md5=md5 if isinstance(md5, str) and _MD5_RE.match(md5) else None,
        size=int(size) if isinstance(size, str) and size.isdigit() else None,
    )


def parse_catalogue(payload: object, *, source_url: str) -> list[FirmwareRef]:
    """The catalogue's OPlus rows as refs, oldest build first.

    Oldest first is what makes `select_ref` correct here: no OPlus `ota_version` carries a date
    `firmware.build_date` can parse, so "newest" falls back to the last row for that device. The
    order is the catalogue's own `build_timestamp` — a model published in several regions is one
    device with several builds, and the newest of them is the newest one anywhere.
    """
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, list) or not releases:
        raise EmptyFirmwareIndexError(
            f"OppoDriver: {source_url} answered no `releases` list at all. It has published ~179 "
            "rows per read since it was first measured, so zero means the API changed or the "
            "request was refused, not that OPPO published nothing."
        )
    rows: list[tuple[tuple[str, str, str], FirmwareRef]] = []
    foreign = 0
    for release in releases:
        if not isinstance(release, dict):
            continue
        region = REGION_BY_GATE_HOST.get(_host_of(str(release.get("source_url", ""))))
        if region is None:
            foreign += 1
            continue
        ref = _ref_for(release, region=region)
        if ref is not None:
            rows.append(
                (
                    (
                        str(release.get("build_timestamp", "")),
                        str(release.get("region", "")),
                        ref.build,
                    ),
                    ref,
                )
            )
    if not rows:
        raise EmptyFirmwareIndexError(
            f"OppoDriver: {source_url} answered {len(releases)} rows and not one of them is an "
            f"OPlus release ({foreign} belong to another OEM). This catalogue carries Xiaomi and "
            "OnePlus-NA rows too, so zero OPlus rows means the gate hosts moved, not that the "
            "catalogue is empty."
        )
    logger.info(
        "oppo catalogue: %d build(s) across %d model(s); %d row(s) belong to another OEM",
        len(rows),
        len({ref.device for _key, ref in rows}),
        foreign,
    )
    rows.sort(key=lambda row: row[0])
    return [ref for _key, ref in rows]


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


async def _read_capped(response: httpx.Response, *, ceiling: int, context: str) -> str:
    """The body, refused once it passes `ceiling` rather than after it is all in memory.

    `response.text` on a streamed response buffers whatever the server chose to send, and both
    servers here are third-party and reached by a worker with no memory bound of its own.
    """
    received = 0
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes(_RESPONSE_CHUNK_BYTES):
        received += len(chunk)
        if received > ceiling:
            raise OppoProtocolError(
                f"{context}: the response passed the {ceiling}-byte ceiling after {received} "
                "bytes; refusing to read the rest of it"
            )
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _ref_for(release: dict[str, Any], *, region: OppoRegion) -> FirmwareRef | None:
    model = str(release.get("model", ""))
    ota_version = str(release.get("ota_version", ""))
    md5 = str(release.get("md5", "") or "")
    try:
        return FirmwareRef(
            driver=OppoDriver.name,
            device=model,
            build=ota_version,
            url=str(release.get("source_url", "")),
            md5=md5 if _MD5_RE.match(md5) else None,
            android_version=marketing_major(str(release.get("version", ""))),
            marketing_name=str(release.get("device", "")) or None,
        )
    except ValueError as exc:
        logger.warning(
            "oppo: catalogue row %s/%s did not validate as a ref: %s", model, ota_version, exc
        )
        return None


class OppoDriver(FirmwareDriver):
    name = "oppo"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._catalogue_url = settings.oppo_catalogue_url
        self._timeout = settings.firmware_http_timeout_seconds
        self._max_archive_bytes = settings.max_firmware_archive_bytes
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client
        # The catalogue escalates scraping to an IP-level block of the whole domain, so a driver
        # instance reads it once and every later call in the same job answers from here.
        self._refs: list[FirmwareRef] | None = None

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.REVERSE_ENGINEERED,
            source_url=self._catalogue_url,
            summary=(
                "Two sources, neither of them an OPPO/OnePlus/realme offer to distribute. The "
                "component-OTA endpoint is a private protocol reconstructed from the on-device "
                "updater, carries no credential and no device identifier, and can change without "
                "notice. The index is roms.danielspringer.at, a third-party catalogue that "
                "escalates scraping to an IP-level block of its whole domain, so this driver "
                "reads it once per job and never per device. No OPPO terms authorise automated "
                "bulk downloading."
            ),
            # Nothing to acknowledge mechanically: there is no gate an operator can accept, only
            # a posture they take by enabling the driver at all.
            acknowledged=True,
        )

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), follow_redirects=True), True

    async def list_available(self) -> list[FirmwareRef]:
        if self._refs is not None:
            return list(self._refs)
        client, owned = self._open_client()
        try:
            try:
                async with client.stream("GET", self._catalogue_url) as response:
                    if response.status_code != httpx.codes.OK:
                        await response.aread()
                        raise FirmwareError(
                            f"OppoDriver: {self._catalogue_url} answered HTTP "
                            f"{response.status_code}; expected 200. This catalogue blocks a "
                            "scraping IP at the domain level, so a 403 here is more likely a "
                            "block than an outage."
                        )
                    text = await _read_capped(
                        response, ceiling=MAX_CATALOGUE_BYTES, context="oppo catalogue"
                    )
            except httpx.HTTPError as exc:
                raise FirmwareError(
                    f"OppoDriver: {self._catalogue_url} is unreachable: {exc}"
                ) from exc
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FirmwareError(
                    f"OppoDriver: {self._catalogue_url} answered non-JSON; the API changed or an "
                    "interstitial is being served"
                ) from exc
        finally:
            if owned:
                await client.aclose()
        self._refs = parse_catalogue(payload, source_url=self._catalogue_url)
        return list(self._refs)

    async def _resolve_direct(
        self, client: httpx.AsyncClient, ref: FirmwareRef
    ) -> ResolvedPackage | None:
        """The endpoint's own URL for exactly the build `ref` names, or None to use the gate.

        Only an exact `ota_version` match is accepted. The endpoint answers the next build in a
        chain rather than a requested one, so anything else means it offered a DIFFERENT package,
        and downloading that under this ref's name would file one build's packages under
        another's build id.
        """
        region = REGION_BY_GATE_HOST.get(_host_of(ref.url))
        branch = ota_version_branch(ref.build, model=ref.device)
        if region is None or branch is None:
            logger.info(
                "oppo: %s/%s carries no OPlus gate host or branch, downloading the stored URL",
                ref.device,
                ref.build,
            )
            return None
        key = secrets.token_bytes(_AES_KEY_BYTES)
        iv = secrets.token_bytes(_AES_IV_BYTES)
        now_ms = int(time.time() * 1000)
        wire, headers = build_update_request(
            model=ref.device,
            ota_version=synthesize_ota_version(ref.device, branch),
            region=region,
            android_major=ref.android_version or "16",
            key=key,
            iv=iv,
            now_ms=now_ms,
        )
        try:
            async with client.stream(
                "POST", region.endpoint, content=wire, headers=headers
            ) as response:
                if response.status_code != httpx.codes.OK:
                    await response.aread()
                    raise FirmwareError(
                        f"OppoDriver: {region.endpoint} answered HTTP {response.status_code} "
                        f"while resolving {ref.device}/{ref.build}; expected 200"
                    )
                text = await _read_capped(
                    response,
                    ceiling=MAX_RESPONSE_BYTES,
                    context=f"oppo update[{ref.device}/{region.name}]",
                )
        except httpx.HTTPError as exc:
            raise FirmwareError(
                f"OppoDriver: {region.endpoint} is unreachable while resolving "
                f"{ref.device}/{ref.build}: {exc}"
            ) from exc
        code, document = decrypt_update_response(text, key)
        if code != RESPONSE_OK or document is None:
            # The observable for the `2004` hole, and the reason it is a warning rather than a
            # note: the question asked is always a synthesized `.00`, which is below every real
            # build, so `2004` here cannot carry its "up to date" meaning. What is left is the
            # unexplained export-model refusal or a (model, branch, region) triple this endpoint
            # does not serve — and WHICH models take this path is the only sign the hole moved.
            logger.warning(
                "oppo: %s answered responseCode %d for %s at %s; downloading the catalogue's "
                "gate instead",
                region.endpoint,
                code,
                ref.device,
                synthesize_ota_version(ref.device, branch),
            )
            return None
        package = resolved_package(document, model=ref.device)
        if package.ota_version != ref.build:
            # Routine rather than alarming: the endpoint trailed the catalogue on every model
            # where both were measured (2026-08-12, `PLK110` A.68/A.72, `PKC110` C.78/C.79,
            # `RMX5010` F.61/F.66), and asking it again at its own answer just returns `2004`,
            # so there is no walk from here to the catalogue's build.
            logger.info(
                "oppo: %s offered %s where the catalogue lists %s; downloading the catalogue's "
                "gate rather than filing another build under this one's id",
                ref.device,
                package.ota_version,
                ref.build,
            )
            return None
        if ref.md5 is not None and package.md5 is not None and package.md5 != ref.md5:
            logger.warning(
                "oppo: %s/%s resolved to md5 %s where the catalogue published %s; downloading "
                "the catalogue's gate, whose digest is the one this ref carries",
                ref.device,
                ref.build,
                package.md5,
                ref.md5,
            )
            return None
        return package

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"OppoDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        client, owned = self._open_client()
        try:
            package = await self._resolve_direct(client, ref)
            return await download_to_file(
                client,
                package.url if package is not None else ref.url,
                dest_dir / ref.archive_filename,
                # Both paths end at a `/downloadCheck` gate or at a link it minted, and the gate
                # answers a 30-byte `{"responseCode":2306}` body rather than an error without
                # this. The md5 above is what turns that body into a named failure.
                headers={"userId": GATE_USER_ID},
                expected_digest=ref.md5,
                digest_algorithm="md5",
                max_bytes=self._max_archive_bytes,
            )
        finally:
            if owned:
                await client.aclose()
